from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch
from transformers import PreTrainedTokenizerBase

from .schemas import RawUnsafeRecord, TokenizedUnsafeRecord

if TYPE_CHECKING:
  from .pipeline import DatasetSelection


LOGGER = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "pad": "<|pad|>",
    "pad_candidates": ("<|pad|>", "<pad>", "[PAD]"),
    "mask_candidates": ("[MASK]", "MASK", "<mask>"),
}


def ensure_pad_token(tokenizer: PreTrainedTokenizerBase, eos_token_id: Optional[int] = None) -> int:
  """
  Ensure the tokenizer has a dedicated padding token that is not a normal vocab token.

  The pad token must:
    - exist
    - not equal EOS
    - use a dedicated control-looking string (one of pad_candidates)
  """
  eos_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id

  def _is_control_pad(tok_str: Optional[str], tok_id: Optional[int]) -> bool:
    if tok_str is None or tok_id is None:
      return False
    if tok_id == tokenizer.unk_token_id:
      return False
    if eos_id is not None and tok_id == eos_id:
      return False
    return tok_str in SPECIAL_TOKENS["pad_candidates"]

  # If an acceptable pad token already exists, keep it.
  if _is_control_pad(getattr(tokenizer, "pad_token", None), tokenizer.pad_token_id):
    return int(tokenizer.pad_token_id)

  # Otherwise, force-add a dedicated pad token.
  for candidate in SPECIAL_TOKENS["pad_candidates"]:
    tokenizer.add_special_tokens({"pad_token": candidate})
    pad_id = tokenizer.convert_tokens_to_ids(candidate)
    if eos_id is not None and pad_id == eos_id:
      continue
    tokenizer.pad_token = candidate
    tokenizer.pad_token_id = pad_id
    LOGGER.warning(
        "Added dedicated pad token to tokenizer: %s (id=%s)",
        tokenizer.pad_token,
        tokenizer.pad_token_id,
    )
    return int(tokenizer.pad_token_id)

  raise RuntimeError("Failed to assign a dedicated pad token.")


def resolve_mask_index(tokenizer: PreTrainedTokenizerBase, mask_token: Optional[str]) -> Optional[int]:
  if tokenizer.mask_token_id is not None:
    return tokenizer.mask_token_id
  if mask_token:
    token_id = tokenizer.convert_tokens_to_ids(mask_token)
    if token_id != tokenizer.unk_token_id:
      return token_id
  for candidate in SPECIAL_TOKENS["mask_candidates"]:
    token_id = tokenizer.convert_tokens_to_ids(candidate)
    if token_id != tokenizer.unk_token_id:
      return token_id
  LOGGER.warning("No mask token resolved; caller must supply mask_token explicitly.")
  return None


def normalise_category_tokens(category: str) -> Sequence[str]:
  tokens = set()
  if not category:
    return ("unspecified",)
  separators = ["|", ",", ";", "/"]
  working = category
  for sep in separators:
    working = working.replace(sep, "|")
  for chunk in working.split("|"):
    chunk = chunk.strip()
    if not chunk:
      continue
    tokens.add(chunk.lower())
  return tuple(sorted(tokens)) if tokens else ("unspecified",)


def extract_category_tokens(record: RawUnsafeRecord) -> Sequence[str]:
  meta_categories = record.meta.get("raw_categories") if record.meta else None
  if isinstance(meta_categories, (list, tuple)):
    tokens = set()
    for item in meta_categories:
      if isinstance(item, str):
        tokens.update(normalise_category_tokens(item))
    if tokens:
      return tuple(sorted(tokens))
  if isinstance(meta_categories, str):
    return normalise_category_tokens(meta_categories)
  return normalise_category_tokens(record.category)


def reservoir_sample(
    iterator: Iterator[RawUnsafeRecord],
    sample_size: int,
    rng: random.Random,
) -> Tuple[List[RawUnsafeRecord], int]:
  reservoir: List[RawUnsafeRecord] = []
  total = 0
  for total, record in enumerate(iterator, start=1):
    if len(reservoir) < sample_size:
      reservoir.append(record)
    else:
      idx = rng.randint(0, total - 1)
      if idx < sample_size:
        reservoir[idx] = record
  return reservoir, total


def tokenize_record(
    tokenizer: PreTrainedTokenizerBase,
    pad_id: int,
    mask_index: Optional[int],
    max_length: int,
    record: RawUnsafeRecord,
    append_prompt: bool = False,
) -> TokenizedUnsafeRecord:
  text = record.answer_text
  if append_prompt:
    prompt_text = None
    if record.meta:
      prompt_text = record.meta.get("prompt_text") or record.meta.get("prompt")
      if isinstance(prompt_text, dict):
        prompt_text = prompt_text.get("text") or prompt_text.get("prompt") or prompt_text.get("content")
    if isinstance(prompt_text, str) and prompt_text.strip():
      text = f"{prompt_text.strip()} {record.answer_text}"

  encoded = tokenizer.encode(
      text,
      add_special_tokens=False,
      max_length=max_length,
      truncation=True,
  )
  truncated = encoded[:max_length]
  length = len(truncated)
  if length < max_length:
    truncated = truncated + [pad_id] * (max_length - length)
  return TokenizedUnsafeRecord(
      source=record.source,
      category=record.category,
      answer_text=record.answer_text,
      toxicity_score=record.toxicity_score,
      meta=record.meta,
      input_ids=truncated,
      length=length,
      mask_index=mask_index,
  )


def _candidate_local_dirs(source: str) -> Sequence[str]:
  mapping = {
      "beavertails": ["PKU-Alignment__BeaverTails", "BeaverTails"],
      "real-toxicity-prompts": ["allenai__real-toxicity-prompts", "real-toxicity-prompts"],
      "toxigen": ["toxigen__toxigen-data", "toxigen-data"],
  }
  return mapping.get(source, [])


def resolve_local_data_dir(selection: "DatasetSelection") -> Optional[str]:
  if selection.data_dir:
    return str(Path(selection.data_dir).expanduser())
  cache_root = os.getenv("HF_DATASETS_CACHE")
  if not cache_root:
    return None
  for candidate in _candidate_local_dirs(selection.source):
    candidate_path = Path(cache_root).expanduser() / candidate
    if candidate_path.exists():
      return str(candidate_path)
  return None


def should_include(
    selection: "DatasetSelection",
    include: Optional[Sequence[str]],
    exclude: Optional[Sequence[str]],
) -> bool:
  if include:
    if selection.source not in include and (selection.output_name or selection.source) not in include:
      return False
  if exclude and selection.source in exclude:
    return False
  return True


def load_index(path: Path) -> Dict[str, object]:
  if path.is_dir():
    index_path = path / "index.json"
  else:
    index_path = path
  if not index_path.exists():
    raise FileNotFoundError(f"Unsafe artifact index not found at {index_path}")
  raw = index_path.read_text(encoding="utf-8")
  return json.loads(raw)


def _resolve_storage(entry: Dict[str, object], root: Path) -> Dict[str, object]:
  storage = entry.get("storage")
  if not isinstance(storage, dict):
    raise ValueError("Unsafe artifact entry missing 'storage' description.")

  paths = storage.get("paths") or []
  resolved_paths = []
  for shard in paths:
    shard_path = Path(shard)
    if not shard_path.is_absolute():
      shard_path = (root / shard_path).resolve()
    resolved_paths.append(str(shard_path))
  storage["paths"] = resolved_paths
  materialized = storage.get("materialized_path")
  if materialized:
    materialized_path = Path(materialized)
    if not materialized_path.is_absolute():
      materialized_path = (root / materialized_path).resolve()
    storage["materialized_path"] = str(materialized_path)
  return storage


def find_unsafe_artifact(root: Path, name: Optional[str]) -> Dict[str, object]:
  if root.is_file():
    return {
        "path": str(root.parent),
        "name": root.stem,
        "storage": {
            "layout": "single_file",
            "paths": [],
            "materialized_path": str(root.resolve()),
        },
    }
  if (root / "shard-00000.pt").exists() or root.name.startswith("shard-"):
    storage = {
        "layout": "single_shard" if len(list(root.glob("shard-*.pt"))) == 1 else "sharded",
        "paths": sorted(p.name for p in root.glob("shard-*.pt")),
        "materialized_path": str((root / "unsafe_reference.pt").resolve()) if (root / "unsafe_reference.pt").exists() else None,
    }
    return {
        "path": str(root.resolve()),
        "name": root.name,
        "storage": storage,
    }

  index = load_index(root)
  artifacts = index.get("unsafe_artifacts") or []
  if not artifacts:
    raise ValueError(f"No unsafe artifacts recorded under {root}.")
  if name is None:
    if len(artifacts) == 1:
      entry = dict(artifacts[0])
    else:
      raise ValueError(
          "Multiple unsafe artifacts available; specify --unsafe-artifact-name. "
          f"Available: {[artifact['name'] for artifact in artifacts]}"
      )
  else:
    entry = next((dict(artifact) for artifact in artifacts if artifact.get("name") == name), None)
    if entry is None:
      raise ValueError(f"Unsafe artifact '{name}' not found in {root}.")

  artifact_root = (root / entry["name"]).resolve() if (root / entry["name"]).exists() else root.resolve()
  storage = _resolve_storage(entry, artifact_root)
  entry["storage"] = storage
  entry["path"] = str(artifact_root)
  return entry


def materialize_artifact(artifact_dir: Path, storage: Dict[str, object], overwrite: bool = False) -> Path:
  artifact_dir = artifact_dir.resolve()
  target = storage.get("materialized_path")
  if target:
    target_path = Path(target)
    if target_path.exists() and not overwrite:
      return target_path
    target = str(target_path)
  else:
    target_path = artifact_dir / "unsafe_reference.pt"
    target = str(target_path)

  if Path(target).exists() and not overwrite:
    return Path(target)

  shards = storage.get("paths") or []
  if not shards:
    raise ValueError(f"No shard paths provided for artifact at {artifact_dir}")
  tensors = []
  for shard in sorted(shards):
    shard_path = Path(shard)
    if not shard_path.exists():
      raise FileNotFoundError(f"Shard missing: {shard_path}")
    payload = torch.load(shard_path, map_location="cpu")
    if isinstance(payload, dict) and "input_ids" in payload:
      tensors.append(payload["input_ids"].long())
    elif isinstance(payload, torch.Tensor):
      tensors.append(payload.long())
    else:
      raise ValueError(f"Unsupported shard payload format: {shard_path}")
  if not tensors:
    raise ValueError(f"No shards found in {artifact_dir}")
  combined = torch.cat(tensors, dim=0)
  target_path = Path(target)
  target_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(combined, target_path)
  LOGGER.info("Materialized unsafe tensor at %s", target_path)
  return target_path
