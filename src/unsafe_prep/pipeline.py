from __future__ import annotations

import ast
import json
import logging
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .utils import (
    ensure_pad_token,
    extract_category_tokens,
    load_index,
    reservoir_sample,
    resolve_local_data_dir,
    resolve_mask_index,
    should_include,
    tokenize_record,
)
from .registry import get_adapter
from .schemas import RawUnsafeRecord, TokenizedUnsafeRecord
from .prompt_alignment import PromptAligner

LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetSelection:
  source: str
  split: str = "train"
  sample_size: Optional[int] = None
  take_all: bool = False
  categories: Sequence[str] = field(default_factory=tuple)
  append_prompt: bool = False
  align_with_owt: bool = False
  align_k: int = 1
  align_encoder: Optional[str] = None
  align_prompt_source: str = "real-toxicity-prompts"
  toxicity_threshold: Optional[float] = None
  toxicity_filters: Dict[str, float] = field(default_factory=dict)
  unsafe_label_field: Optional[str] = None
  unsafe_label_values: Optional[Sequence[int]] = None
  config_name: Optional[str] = None
  generation_methods: Sequence[str] = field(default_factory=tuple)
  groups: Sequence[str] = field(default_factory=tuple)
  roberta_threshold: Optional[float] = None
  toxicity_ai_threshold: Optional[float] = None
  toxicity_human_threshold: Optional[float] = None
  toxicity_threshold_max: Optional[float] = None   # upper bound on continuation toxicity (benign RTP artifacts)
  prompt_is_safe: Optional[bool] = None            # BeaverTails: True=safe examples, False=unsafe, None=unsafe (default)
  prompt_label: Optional[int] = None               # ToxiGen: 0=non-hateful, 1=hateful, None=no filter
  data_dir: Optional[str] = None
  data_files: Optional[Dict[str, str]] = None
  streaming: bool = False
  seed: Optional[int] = None
  output_name: Optional[str] = None

  def __post_init__(self) -> None:
    if isinstance(self.sample_size, str):
      if self.sample_size.lower() == "all":
        self.sample_size = None
        self.take_all = True
      else:
        self.sample_size = int(self.sample_size)
    self.categories = tuple(str(cat).strip() for cat in self.categories if str(cat).strip())
    self.toxicity_filters = {str(k): float(v) for k, v in (self.toxicity_filters or {}).items()}
    self.generation_methods = tuple(str(m).strip().lower() for m in (self.generation_methods or []) if str(m).strip())
    self.groups = tuple(str(g).strip().lower() for g in (self.groups or []) if str(g).strip())


@dataclass
class UnsafePrepConfig:
  tokenizer_name_or_path: str
  tokenizer_alias: Optional[str] = None
  max_length: int = 512
  shard_size: int = 1024
  seed: int = 1
  output_dir: str = "artifacts/unsafe_artifacts"
  mask_token: Optional[str] = None
  datasets: Sequence[DatasetSelection] = field(default_factory=tuple)
  mix_only: bool = False
  mix_equal_datasets: bool = False
  mix_output_name: Optional[str] = None
  mix_per_dataset: Optional[int] = None
  mix_sample_size: Optional[int] = None
  mix_shuffle: bool = True


@dataclass
class StatsCollector:
  count: int = 0
  sum_lengths: float = 0.0
  sum_sq_lengths: float = 0.0
  min_length: int = field(default=int(1e9))
  max_length: int = 0

  def update(self, length: int) -> None:
    self.count += 1
    self.sum_lengths += float(length)
    self.sum_sq_lengths += float(length) * float(length)
    self.min_length = min(self.min_length, length)
    self.max_length = max(self.max_length, length)

  @property
  def mean(self) -> float:
    if self.count == 0:
      return 0.0
    return self.sum_lengths / self.count

  @property
  def std(self) -> float:
    if self.count <= 1:
      return 0.0
    mean = self.mean
    variance = max(self.sum_sq_lengths / self.count - mean * mean, 0.0)
    return math.sqrt(variance)


@dataclass
class UnsafeArtifactSummary:
  name: str
  source: str
  categories: Sequence[str]
  count: int
  shard_size: int
  num_shards: int
  mean_length: float
  std_length: float
  min_length: int
  max_length: int
  storage: Dict[str, object]
  filters: Dict[str, object]
  sample_seed: int
  sample_size_requested: Optional[int]
  category_counts: Dict[str, int] = field(default_factory=dict)


class ShardWriter:
  """Accumulates tokenized records and writes contiguous shard files.

  Each shard is saved as a torch dictionary with:
    - ``input_ids``: LongTensor [shard_size, max_length]
    - ``lengths``: LongTensor [shard_size] storing original sequence lengths
    - ``meta``: list of lightweight metadata dicts (source, category, provenance)
  """
  def __init__(self, out_dir: Path, shard_size: int, dry_run: bool, overwrite: bool):
    self.out_dir = out_dir
    self.shard_size = shard_size
    self.dry_run = dry_run
    self.overwrite = overwrite
    self.buffer_ids: List[List[int]] = []
    self.buffer_lengths: List[int] = []
    self.buffer_meta: List[Dict[str, object]] = []
    self.shard_index = 0
    self.paths: List[str] = []
    self.stats = StatsCollector()
    self.category_counter: Dict[str, int] = {}
    self.materialized_path: Optional[str] = None
    self.out_dir.mkdir(parents=True, exist_ok=True)

  def add(self, record: TokenizedUnsafeRecord, category_tokens: Sequence[str]) -> None:
    self.buffer_ids.append(record.input_ids)
    self.buffer_lengths.append(record.length)
    meta_entry = {
        "source": record.source,
        "category": record.category,
        "toxicity_score": record.toxicity_score,
        "meta": record.meta,
    }
    self.buffer_meta.append(meta_entry)
    self.stats.update(record.length)
    for token in category_tokens:
      self.category_counter[token] = self.category_counter.get(token, 0) + 1
    if len(self.buffer_ids) >= self.shard_size:
      self._flush()

  def _flush(self) -> None:
    if not self.buffer_ids:
      return
    shard_path = self.out_dir / f"shard-{self.shard_index:05d}.pt"
    stats_path = shard_path.with_suffix(".stats.json")
    shard_entry_count = len(self.buffer_ids)
    if shard_path.exists() and not self.overwrite:
      LOGGER.info("Shard %s exists; skipping write (resume mode).", shard_path)
      self.paths.append(str(shard_path.relative_to(self.out_dir)))
      if not stats_path.exists():
        LOGGER.warning("Stats file %s missing; recomputing.", stats_path)
      self.shard_index += 1
      self.buffer_ids.clear()
      self.buffer_lengths.clear()
      self.buffer_meta.clear()
      return
    if self.dry_run:
      LOGGER.debug("Dry run: skipping shard %s write.", shard_path)
      self.shard_index += 1
      self.buffer_ids.clear()
      self.buffer_lengths.clear()
      self.buffer_meta.clear()
      return

    tensor = torch.tensor(self.buffer_ids, dtype=torch.long)
    lengths = torch.tensor(self.buffer_lengths, dtype=torch.long)
    payload = {
        "input_ids": tensor,
        "lengths": lengths,
        "meta": self.buffer_meta,
    }
    torch.save(payload, shard_path)
    shard_stats = {
        "count": shard_entry_count,
        "mean_length": float(lengths.float().mean().item()),
        "max_length": int(lengths.max().item()),
        "min_length": int(lengths.min().item()),
    }
    stats_path.write_text(json.dumps(shard_stats, indent=2), encoding="utf-8")
    self.paths.append(str(shard_path.relative_to(self.out_dir)))
    self.shard_index += 1
    self.buffer_ids.clear()
    self.buffer_lengths.clear()
    self.buffer_meta.clear()

  def finalize(self) -> None:
    self._flush()

  def storage_info(self) -> Dict[str, object]:
    layout = "single_shard" if len(self.paths) == 1 else "sharded"
    materialized = self.materialized_path
    if materialized is None and len(self.paths) == 1:
      materialized = self.paths[0]
    return {
        "layout": layout,
        "paths": list(self.paths),
        "materialized_path": materialized,
    }


def _load_dataset_selection(raw_cfg: Dict[str, object]) -> DatasetSelection:
  return DatasetSelection(**raw_cfg)


def _parse_override(raw: str) -> Tuple[str, object]:
  if "=" not in raw:
    raise ValueError(f"Override '{raw}' must be in KEY=VALUE format.")
  key, value_str = raw.split("=", 1)
  key = key.strip()
  value_str = value_str.strip()
  if not key:
    raise ValueError(f"Invalid override '{raw}' (empty key).")
  try:
    value = ast.literal_eval(value_str)
  except (ValueError, SyntaxError):
    lowered = value_str.lower()
    if lowered in {"true", "false"}:
      value = lowered == "true"
    elif lowered in {"null", "none"}:
      value = None
    else:
      value = value_str
  return key, value


def load_config(path: Path, overrides: Optional[Sequence[str]] = None) -> UnsafePrepConfig:
  cfg = OmegaConf.load(path)
  if overrides:
    for item in overrides:
      key, value = _parse_override(item)
      OmegaConf.update(cfg, key, value, merge=False)
  resolved = OmegaConf.to_container(cfg, resolve=True)
  if not isinstance(resolved, dict):
    raise ValueError("Configuration root must be a mapping.")
  datasets_cfg = resolved.get("datasets", [])
  datasets: List[DatasetSelection] = []
  for entry in datasets_cfg:
    if isinstance(entry, (DictConfig, ListConfig)):
      entry = OmegaConf.to_container(entry, resolve=True)  # type: ignore
    if not isinstance(entry, dict):
      raise ValueError("Dataset entries must be mappings.")
    datasets.append(_load_dataset_selection(entry))
  return UnsafePrepConfig(
      tokenizer_name_or_path=resolved["tokenizer_name_or_path"],
      tokenizer_alias=resolved.get("tokenizer_alias"),
      max_length=int(resolved.get("max_length", 512)),
      shard_size=int(resolved.get("shard_size", 1024)),
      seed=int(resolved.get("seed", 1)),
      output_dir=resolved.get("output_dir", "artifacts/unsafe_artifacts"),
      mask_token=resolved.get("mask_token"),
      datasets=datasets,
      mix_only=bool(resolved.get("mix_only", False)),
      mix_equal_datasets=bool(resolved.get("mix_equal_datasets", False)),
      mix_output_name=resolved.get("mix_output_name"),
      mix_per_dataset=resolved.get("mix_per_dataset"),
      mix_sample_size=resolved.get("mix_sample_size"),
      mix_shuffle=bool(resolved.get("mix_shuffle", True)),
  )


def _build_adapter_kwargs(selection: DatasetSelection) -> Dict[str, object]:
  kwargs: Dict[str, object] = {
      "split": selection.split,
      "streaming": selection.streaming,
  }
  if selection.config_name:
    kwargs["config_name"] = selection.config_name
  if selection.data_dir:
    kwargs["data_dir"] = selection.data_dir
  if selection.data_files:
    kwargs["data_files"] = selection.data_files
  if selection.unsafe_label_field:
    kwargs["unsafe_label_field"] = selection.unsafe_label_field
  if selection.unsafe_label_values:
    kwargs["unsafe_label_values"] = selection.unsafe_label_values
  if selection.source == "beavertails" and selection.categories:
    kwargs["keep_categories"] = tuple(selection.categories)
  if selection.source == "beavertails" and selection.prompt_is_safe is not None:
    kwargs["prompt_is_safe"] = selection.prompt_is_safe
  if selection.source == "real-toxicity-prompts":
    if selection.toxicity_threshold is not None:
      kwargs["toxicity_threshold"] = selection.toxicity_threshold
    if selection.toxicity_threshold_max is not None:
      kwargs["toxicity_threshold_max"] = selection.toxicity_threshold_max
    if selection.toxicity_filters:
      kwargs["toxicity_filters"] = selection.toxicity_filters
  if selection.source == "toxigen":
    kwargs["config_name"] = selection.config_name or kwargs.get("config_name") or "train"
    if selection.generation_methods:
      kwargs["generation_methods"] = selection.generation_methods
    if selection.groups:
      kwargs["groups"] = selection.groups
    if selection.roberta_threshold is not None:
      kwargs["roberta_threshold"] = selection.roberta_threshold
    if selection.toxicity_ai_threshold is not None:
      kwargs["toxicity_ai_threshold"] = selection.toxicity_ai_threshold
    if selection.toxicity_human_threshold is not None:
      kwargs["toxicity_human_threshold"] = selection.toxicity_human_threshold
    if selection.prompt_label is not None:
      kwargs["prompt_label"] = selection.prompt_label
  return kwargs


def _resolve_artifact_name(selection: DatasetSelection, record_count: Optional[int], sample_size: Optional[int]) -> str:
  if selection.output_name:
    return selection.output_name
  category_suffix = f"-{'+'.join(selection.categories)}" if selection.categories else ""
  if sample_size is None:
    return f"{selection.source}{category_suffix}-all"
  count = record_count if record_count is not None else sample_size
  return f"{selection.source}{category_suffix}-{count}"


def _compute_category_totals(entries: Sequence[Dict[str, object]]) -> Dict[str, int]:
  totals: Dict[str, int] = {}
  for entry in entries:
    counts = entry.get("category_counts") if isinstance(entry, dict) else None
    if not isinstance(counts, dict):
      continue
    for category, value in counts.items():
      try:
        numeric = int(value)
      except (TypeError, ValueError):
        continue
      totals[category] = totals.get(category, 0) + numeric
  return totals


def _merge_with_existing_index(
    index_path: Path,
    new_artifacts: Sequence[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
  existing_entries: Dict[str, Dict[str, object]] = {}
  if index_path.exists():
    try:
      current = load_index(index_path)
    except Exception as exc:  # pragma: no cover - defensive
      LOGGER.warning("Failed to load existing unsafe index at %s: %s; rewriting.", index_path, exc)
    else:
      for entry in current.get("unsafe_artifacts", []):
        if isinstance(entry, dict):
          name = entry.get("name")
          if isinstance(name, str):
            existing_entries[name] = entry
  for entry in new_artifacts:
    existing_entries[entry["name"]] = entry
  merged = sorted(existing_entries.values(), key=lambda item: item.get("name", ""))
  totals = _compute_category_totals(merged)
  return merged, totals


def build_unsafe_artifacts(
    config: UnsafePrepConfig,
    output_root: Optional[Path] = None,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> Dict[str, object]:
  tokenizer = AutoTokenizer.from_pretrained(
    config.tokenizer_name_or_path,
    trust_remote_code=True,
  )
  pad_id = ensure_pad_token(tokenizer)
  mask_index = resolve_mask_index(tokenizer, config.mask_token)
  output_dir = Path(output_root or config.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  summaries: List[UnsafeArtifactSummary] = []
  aligner_cache: Dict[str, PromptAligner] = {}

  def _sample_equal_records(
      selections: Sequence[DatasetSelection],
      per_dataset: int,
  ) -> List[RawUnsafeRecord]:
    sampled: List[RawUnsafeRecord] = []
    if per_dataset <= 0:
      return sampled
    for idx, selection in enumerate(selections):
      selection_seed = selection.seed if selection.seed is not None else config.seed + idx
      selection_rng = random.Random(selection_seed)
      resolved_dir = resolve_local_data_dir(selection)
      if resolved_dir and resolved_dir != selection.data_dir:
        selection = DatasetSelection(**{**asdict(selection), "data_dir": resolved_dir})
      adapter_kwargs = _build_adapter_kwargs(selection)
      adapter = get_adapter(selection.source, **adapter_kwargs)
      base_iterator: Iterable[RawUnsafeRecord] = adapter.iter_unsafe_answers()
      categories_filter = {cat.lower() for cat in selection.categories}
      if categories_filter:
        base_iterator = (
            record
            for record in base_iterator
            if categories_filter.intersection(extract_category_tokens(record))
        )
      iterator_to_close = None
      if selection.streaming:
        record_iter = base_iterator
      else:
        record_iter = tqdm(
            base_iterator,
            desc=f"mix:{selection.source}:{selection.split}",
            unit="records",
        )
        iterator_to_close = record_iter if hasattr(record_iter, "close") else None
      sampled_records, total_seen = reservoir_sample(iter(record_iter), per_dataset, selection_rng)
      if iterator_to_close:
        iterator_to_close.close()
      if not sampled_records:
        LOGGER.warning("Equal-mix: no records selected for %s; skipping.", selection.source)
        continue
      if selection.align_with_owt:
        align_key = f"{selection.align_prompt_source}|{selection.align_encoder or config.tokenizer_name_or_path}"
        if align_key not in aligner_cache:
          aligner_cache[align_key] = PromptAligner(
              encoder_name=selection.align_encoder or config.tokenizer_name_or_path,
              prompt_source=selection.align_prompt_source,
          )
        aligner = aligner_cache[align_key]
        aligned_records = aligner.align_records(
            sampled_records,
            k=selection.align_k or 1,
            source_prefix=selection.source,
        )
        if not aligned_records:
          LOGGER.warning("Equal-mix: alignment produced no records for %s; skipping.", selection.source)
          continue
        sampled_records = aligned_records
      if len(sampled_records) < per_dataset:
        LOGGER.warning(
            "Equal-mix: %s yielded %d/%d records (seen=%d).",
            selection.source,
            len(sampled_records),
            per_dataset,
            total_seen,
        )
      sampled.extend(sampled_records)
    return sampled

  if not config.mix_only:
    for idx, selection in enumerate(config.datasets):
      if not should_include(selection, include, exclude):
        continue
      selection_seed = selection.seed if selection.seed is not None else config.seed + idx
      selection_rng = random.Random(selection_seed)
      LOGGER.info("Processing dataset '%s' (split=%s).", selection.source, selection.split)
      resolved_dir = resolve_local_data_dir(selection)
      if resolved_dir and resolved_dir != selection.data_dir:
        LOGGER.info("  Using local dataset directory: %s", resolved_dir)
        selection = DatasetSelection(**{**asdict(selection), "data_dir": resolved_dir})

      adapter_kwargs = _build_adapter_kwargs(selection)
      adapter = get_adapter(selection.source, **adapter_kwargs)
      base_iterator: Iterable[RawUnsafeRecord] = adapter.iter_unsafe_answers()
      categories_filter = {cat.lower() for cat in selection.categories}
      if categories_filter:
        base_iterator = (
            record
            for record in base_iterator
            if categories_filter.intersection(extract_category_tokens(record))
        )

      iterator_to_close = None
      if selection.streaming:
        record_iter = base_iterator
      else:
        record_iter = tqdm(
            base_iterator,
            desc=f"{selection.source}:{selection.split}",
            unit="records",
        )
        iterator_to_close = record_iter if hasattr(record_iter, "close") else None

      sample_size = None if selection.take_all or selection.sample_size is None else int(selection.sample_size)
      total_seen = 0
      records_to_process: Iterable[RawUnsafeRecord]
      if sample_size is not None:
        sampled_records, total_seen = reservoir_sample(iter(record_iter), sample_size, selection_rng)
        if iterator_to_close:
          iterator_to_close.close()
          iterator_to_close = None
        if not sampled_records:
          LOGGER.warning("No records selected for %s; skipping.", selection.source)
          continue
        records_to_process = sampled_records
        artifact_name = _resolve_artifact_name(selection, len(sampled_records), sample_size)
      else:
        records_to_process = record_iter
        artifact_name = _resolve_artifact_name(selection, None, None)

      # Optional OWT-aligned prompt replacement
      if selection.align_with_owt:
        align_key = f"{selection.align_prompt_source}|{selection.align_encoder or config.tokenizer_name_or_path}"
        if align_key not in aligner_cache:
          aligner_cache[align_key] = PromptAligner(
              encoder_name=selection.align_encoder or config.tokenizer_name_or_path,
              prompt_source=selection.align_prompt_source,
          )
        aligner = aligner_cache[align_key]
        record_list = list(records_to_process)
        aligned_records = aligner.align_records(
            record_list,
            k=selection.align_k or 1,
            source_prefix=selection.source,
        )
        if not aligned_records:
          LOGGER.warning("Alignment produced no records for %s; skipping.", selection.source)
          continue
        records_to_process = aligned_records
        # If no explicit output_name was provided, append an alignment suffix to keep names unique.
        if not selection.output_name:
          artifact_name = f"{artifact_name}-knn-owt"

      shard_writer = ShardWriter(
          out_dir=output_dir / artifact_name,
          shard_size=config.shard_size,
          dry_run=dry_run,
          overwrite=overwrite,
      )
      for record in records_to_process:
        tokenized = tokenize_record(
            tokenizer=tokenizer,
            pad_id=pad_id,
            mask_index=mask_index,
            max_length=config.max_length,
            record=record,
            append_prompt=bool(getattr(selection, "append_prompt", False)),
        )
        shard_writer.add(tokenized, extract_category_tokens(record))
      shard_writer.finalize()
      if iterator_to_close:
        iterator_to_close.close()

      count = shard_writer.stats.count
      if count == 0:
        LOGGER.warning("No records materialized for %s; skipping.", selection.source)
        continue

      summary = UnsafeArtifactSummary(
          name=artifact_name,
          source=selection.source,
          categories=tuple(sorted(shard_writer.category_counter.keys())),
          count=count,
          shard_size=config.shard_size,
          num_shards=len(shard_writer.paths),
          mean_length=shard_writer.stats.mean,
          std_length=shard_writer.stats.std,
          min_length=shard_writer.stats.min_length if shard_writer.stats.count else 0,
          max_length=shard_writer.stats.max_length,
          storage=shard_writer.storage_info(),
          filters={
              "categories": selection.categories,
              "toxicity_threshold": selection.toxicity_threshold,
          },
          sample_seed=selection_seed,
          sample_size_requested=sample_size,
          category_counts=dict(shard_writer.category_counter),
      )
      summaries.append(summary)

      if sample_size is None:
        LOGGER.info(
            "Finished '%s': %d records across %d shards.",
            artifact_name,
            count,
            len(shard_writer.paths),
        )
      else:
        LOGGER.info(
            "Finished '%s': sampled %d of %d records (%d shards).",
            artifact_name,
            count,
            total_seen,
            len(shard_writer.paths),
        )
  elif config.mix_equal_datasets:
    LOGGER.info("mix_only enabled: skipping per-dataset artifacts.")

  if config.mix_equal_datasets:
    mix_selections = [s for s in config.datasets if should_include(s, include, exclude)]
    if not mix_selections:
      LOGGER.warning("Equal-mix requested but no datasets matched include/exclude filters.")
    else:
      if config.mix_per_dataset is None:
        if config.mix_sample_size is None:
          raise ValueError("mix_equal_datasets requires mix_per_dataset or mix_sample_size.")
        per_dataset = int(config.mix_sample_size) // max(len(mix_selections), 1)
      else:
        per_dataset = int(config.mix_per_dataset)
      if per_dataset <= 0:
        LOGGER.warning("Equal-mix per-dataset size is %d; skipping mixed artifact.", per_dataset)
      else:
        mix_name = config.mix_output_name or "unsafe_mixed_equal"
        LOGGER.info(
            "Building equal-mix artifact '%s' with %d samples per dataset (%d datasets).",
            mix_name,
            per_dataset,
            len(mix_selections),
        )
        mixed_records = _sample_equal_records(mix_selections, per_dataset)
        if mixed_records:
          if config.mix_shuffle:
            mix_rng = random.Random(config.seed)
            mix_rng.shuffle(mixed_records)
          append_prompt_by_source = {
              s.source: bool(getattr(s, "append_prompt", False)) for s in mix_selections
          }
          shard_writer = ShardWriter(
              out_dir=output_dir / mix_name,
              shard_size=config.shard_size,
              dry_run=dry_run,
              overwrite=overwrite,
          )
          for record in mixed_records:
            tokenized = tokenize_record(
                tokenizer=tokenizer,
                pad_id=pad_id,
                mask_index=mask_index,
                max_length=config.max_length,
                record=record,
                append_prompt=append_prompt_by_source.get(record.source, False),
            )
            shard_writer.add(tokenized, extract_category_tokens(record))
          shard_writer.finalize()
          count = shard_writer.stats.count
          if count > 0:
            summary = UnsafeArtifactSummary(
                name=mix_name,
                source="mixed",
                categories=tuple(sorted(shard_writer.category_counter.keys())),
                count=count,
                shard_size=config.shard_size,
                num_shards=len(shard_writer.paths),
                mean_length=shard_writer.stats.mean,
                std_length=shard_writer.stats.std,
                min_length=shard_writer.stats.min_length if shard_writer.stats.count else 0,
                max_length=shard_writer.stats.max_length,
                storage=shard_writer.storage_info(),
                filters={
                    "mix_sources": [s.source for s in mix_selections],
                    "mix_per_dataset": per_dataset,
                    "mix_shuffle": config.mix_shuffle,
                },
                sample_seed=config.seed,
                sample_size_requested=per_dataset * len(mix_selections),
                category_counts=dict(shard_writer.category_counter),
            )
            summaries.append(summary)
        else:
          LOGGER.warning("Equal-mix produced no records; mixed artifact not written.")

  artifacts_payload = [
      {
          "name": summary.name,
          "source": summary.source,
          "categories": summary.categories,
          "count": summary.count,
          "num_shards": summary.num_shards,
          "mean_length": summary.mean_length,
          "std_length": summary.std_length,
          "min_length": summary.min_length,
          "max_length": summary.max_length,
          "storage": summary.storage,
          "filters": summary.filters,
          "sample_seed": summary.sample_seed,
          "sample_size_requested": summary.sample_size_requested,
          "category_counts": summary.category_counts,
      }
      for summary in summaries
  ]

  index = {
      "built_at": datetime.now(timezone.utc).isoformat(),
      "tokenizer": config.tokenizer_name_or_path,
      "tokenizer_alias": config.tokenizer_alias or "",
      "max_length": config.max_length,
      "mask_index": mask_index,
      "pad_id": int(pad_id),
      "tokenizer_name_or_path": config.tokenizer_name_or_path,
      "tokenizer_len": int(len(tokenizer)),
      "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", len(tokenizer))),
      "eos_id": int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
      "shard_size": config.shard_size,
      "dry_run": dry_run,
      "unsafe_artifacts": artifacts_payload,
      "category_totals": _compute_category_totals(artifacts_payload),
  }

  if not dry_run:
    index_path = output_dir / "index.json"
    merged_artifacts, merged_totals = _merge_with_existing_index(index_path, artifacts_payload)
    index.update(
        {
            "unsafe_artifacts": merged_artifacts,
            "category_totals": merged_totals,
        }
    )
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    LOGGER.info("Updated unsafe artifact index at %s", index_path)
  else:
    LOGGER.info("Dry run complete; index not written.")
  return index
