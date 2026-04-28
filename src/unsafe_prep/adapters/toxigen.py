from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

from datasets import load_dataset

from . import UnsafeDatasetAdapter, register_adapter
from .utils import coerce_float, safe_float
from ..schemas import RawUnsafeRecord, ToxigenTrainRecord, ToxigenAnnotatedRecord


def _normalise_sequence(values: Sequence[str]) -> Tuple[str, ...]:
  return tuple(str(v).strip().lower() for v in values if str(v).strip())


def _coerce_prompt_label(value: object) -> Optional[int]:
  if value is None:
    return None
  if isinstance(value, bool):
    return 1 if value else 0
  if isinstance(value, int):
    if value in (0, 1):
      return value
  if isinstance(value, float):
    if value in (0.0, 1.0):
      return int(value)
  if isinstance(value, str):
    lowered = value.strip().lower()
    if lowered in {"1", "true", "unsafe", "yes"}:
      return 1
    if lowered in {"0", "false", "benign", "safe", "no"}:
      return 0
    if lowered in {"none", "null"}:
      return None
  raise SystemExit("prompt_label must be 0/1 or boolean when set.")


def _record_prompt_label(value: object) -> Optional[int]:
  if value is None:
    return None
  if isinstance(value, bool):
    return 1 if value else 0
  if isinstance(value, int):
    return value if value in (0, 1) else None
  if isinstance(value, float):
    if value in (0.0, 1.0):
      return int(value)
    return None
  if isinstance(value, str):
    lowered = value.strip().lower()
    if lowered in {"0", "1"}:
      return int(lowered)
  return None

@dataclass
class ToxigenAdapter(UnsafeDatasetAdapter):
  name: str = "toxigen"
  split: str = "train"
  config_name: str = "train"
  data_dir: Optional[str] = None
  data_files: Optional[dict] = None
  streaming: bool = False
  generation_methods: Tuple[str, ...] = field(default_factory=tuple)
  groups: Tuple[str, ...] = field(default_factory=tuple)
  roberta_threshold: Optional[float] = None
  toxicity_ai_threshold: Optional[float] = None
  toxicity_human_threshold: Optional[float] = None
  prompt_label: Optional[int] = None

  def __post_init__(self) -> None:
    self.generation_methods = _normalise_sequence(self.generation_methods)
    self.groups = _normalise_sequence(self.groups)
    self.prompt_label = _coerce_prompt_label(self.prompt_label)

  def iter_unsafe_answers(self) -> Iterator[RawUnsafeRecord]:
    roberta_thresh = coerce_float(self.roberta_threshold, "roberta_threshold", allow_none=True)
    tox_ai_thresh = coerce_float(self.toxicity_ai_threshold, "toxicity_ai_threshold", allow_none=True)
    tox_human_thresh = coerce_float(self.toxicity_human_threshold, "toxicity_human_threshold", allow_none=True)
    dataset_path = self.data_dir or "toxigen"
    load_kwargs = {
        "split": self.split,
        "streaming": self.streaming,
        "name": self.config_name,
    }
    if self.data_files:
      load_kwargs["data_files"] = self.data_files
    if self.data_dir and not Path(self.data_dir).is_absolute():
      load_kwargs["data_dir"] = self.data_dir
    dataset = load_dataset(dataset_path, **load_kwargs)
    iterator = dataset if self.streaming else dataset  # type: ignore
    config = (self.config_name or "train").lower()

    for row in iterator:
      if config == "train":
        record = ToxigenTrainRecord(**row)
        text = record.generation
        if not isinstance(text, str) or not text.strip():
          continue
        roberta_pred = record.roberta_prediction
        if roberta_thresh is not None:
          roberta_val = safe_float(roberta_pred)
          if roberta_val is None or roberta_val < roberta_thresh:
            continue
        method = record.generation_method
        if self.generation_methods and str(method).strip().lower() not in self.generation_methods:
          continue
        group = record.group
        group_token = str(group).strip().lower() if group else ""
        if self.groups and group_token not in self.groups:
          continue
        prompt_label = record.prompt_label
        if self.prompt_label is not None:
          label_val = _record_prompt_label(prompt_label)
          if label_val is None or label_val != self.prompt_label:
            continue
        meta = {
            "prompt": record.prompt,
            "generation_method": method,
            "group": group,
            "prompt_label": prompt_label,
            "roberta_prediction": roberta_pred,
        }
        toxicity_ai = record.toxicity_ai
        toxicity_human = record.toxicity_human
        toxicity_score = None
        for candidate in (roberta_pred, toxicity_ai, toxicity_human):
          toxicity_score = safe_float(candidate)
          if toxicity_score is not None:
            break
        category = group_token or "toxicity"
      elif config == "annotated":
        record = ToxigenAnnotatedRecord(**row)
        text = record.text
        if not isinstance(text, str) or not text.strip():
          continue
        toxicity_ai = record.toxicity_ai
        toxicity_human = record.toxicity_human
        if tox_ai_thresh is not None:
          tox_ai_val = safe_float(toxicity_ai)
          if tox_ai_val is None or tox_ai_val < tox_ai_thresh:
            continue
        if tox_human_thresh is not None:
          tox_human_val = safe_float(toxicity_human)
          if tox_human_val is None or tox_human_val < tox_human_thresh:
            continue
        group = record.target_group or record.group
        group_token = str(group).strip().lower() if group else ""
        if self.groups and group_token not in self.groups:
          continue
        meta = {
            "target_group": group,
            "toxicity_ai": toxicity_ai,
            "toxicity_human": toxicity_human,
            "predicted_author": record.predicted_author,
            "generation_method": record.actual_method,
        }
        toxicity_score = None
        for candidate in (toxicity_ai, toxicity_human):
          toxicity_score = safe_float(candidate)
          if toxicity_score is not None:
            break
        category = group_token or "toxicity"
      else:
        continue

      yield RawUnsafeRecord(
          source=self.name,
          category=category,
          answer_text=text.strip(),
          toxicity_score=toxicity_score,
          meta=meta,
      )


def factory(**kwargs) -> ToxigenAdapter:
  return ToxigenAdapter(**kwargs)


register_adapter("toxigen", factory)
