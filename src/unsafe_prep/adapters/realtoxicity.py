from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional

from datasets import load_dataset

from . import UnsafeDatasetAdapter, register_adapter
from .utils import coerce_float, safe_float
from ..schemas import RawUnsafeRecord, RealToxicityPromptRecord
from ..constants import (
    RTP_PROMPT_FIELD,
    RTP_CONTINUATION_FIELD,
    RTP_TEXT_FIELD,
    RTP_SCORE_FIELDS,
)


@dataclass
class RealToxicityAdapter(UnsafeDatasetAdapter):
  name: str = "real-toxicity-prompts"
  split: str = "train"
  toxicity_threshold: float = 0.5
  toxicity_threshold_max: Optional[float] = None   # upper bound on continuation toxicity (for benign artifacts)
  toxicity_filters: Dict[str, float] = field(default_factory=dict)
  prompt_score_min: Optional[float] = None
  prompt_score_max: Optional[float] = None
  prompt_score_filters: Dict[str, float] = field(default_factory=dict)
  prompt_variant: Optional[str] = None
  data_dir: Optional[str] = None
  data_files: Optional[dict] = None
  streaming: bool = False

  def iter_unsafe_answers(self) -> Iterator[RawUnsafeRecord]:
    base_threshold = coerce_float(self.toxicity_threshold, "toxicity_threshold")
    base_threshold_max = coerce_float(self.toxicity_threshold_max, "toxicity_threshold_max", allow_none=True)
    dataset_path = self.data_dir or "real-toxicity-prompts"
    load_kwargs = {
        "split": self.split,
        "streaming": self.streaming,
    }
    if self.data_files:
      load_kwargs["data_files"] = self.data_files
    if self.data_dir and not Path(self.data_dir).is_absolute():
      load_kwargs["data_dir"] = self.data_dir
    dataset = load_dataset(dataset_path, **load_kwargs)
    iterator = dataset if self.streaming else dataset  # type: ignore
    for row in iterator:
      model = RealToxicityPromptRecord(**row)
      prompt_dict = model.prompt
      prompt_text = prompt_dict.get(RTP_TEXT_FIELD) if isinstance(prompt_dict, dict) else None
      prompt_scores: Dict[str, float] = {}
      if isinstance(prompt_dict, dict):
        for field in RTP_SCORE_FIELDS:
          value = prompt_dict.get(field)
          value_f = safe_float(value)
          if value_f is not None:
            prompt_scores[field] = value_f
      prompt_max = max(prompt_scores.values()) if prompt_scores else None

      if self.prompt_score_min is not None:
        min_val = coerce_float(self.prompt_score_min, "prompt_score_min")
        if prompt_max is None or prompt_max < min_val:
          continue
      if self.prompt_score_max is not None:
        max_val = coerce_float(self.prompt_score_max, "prompt_score_max")
        if prompt_max is None or prompt_max > max_val:
          continue

      if self.prompt_score_filters:
        prompt_filters_ok = True
        for field, threshold in self.prompt_score_filters.items():
          prompt_val = prompt_scores.get(field)
          if prompt_val is None and isinstance(prompt_dict, dict):
            raw_val = prompt_dict.get(field) or prompt_dict.get(field.replace(".", "_"))
            prompt_val = safe_float(raw_val)
          if prompt_val is None:
            prompt_filters_ok = False
            break
          if prompt_val < coerce_float(threshold, f"prompt_score_filters['{field}']"):
            prompt_filters_ok = False
            break
        if not prompt_filters_ok:
          continue

      continuation = model.continuation
      if not isinstance(continuation, dict):
        continue
      text = continuation.get(RTP_TEXT_FIELD)
      if not isinstance(text, str) or not text.strip():
        continue
      score_map: Dict[str, float] = {}
      for field in RTP_SCORE_FIELDS:
        value = continuation.get(field)
        if value is None and isinstance(model.prompt, dict):
          value = model.prompt.get(field)
        value_f = safe_float(value)
        if value_f is not None:
          score_map[field] = value_f
      if not score_map:
        continue
      max_score = max(score_map.values())
      if max_score < base_threshold:
        continue
      if base_threshold_max is not None and max_score > base_threshold_max:
        continue

      filters_ok = True
      for field, threshold in self.toxicity_filters.items():
        value_f = score_map.get(field)
        if value_f is None:
          value = continuation.get(field)
          if value is None:
            value = continuation.get(field.replace(".", "_"))
          if value is None and isinstance(model.prompt, dict):
            value = model.prompt.get(field)
          if value is None:
            filters_ok = False
            break
          value_f = safe_float(value)
        if value_f is None:
          filters_ok = False
          break
        threshold_value = coerce_float(threshold, f"toxicity_filters['{field}']")
        if value_f < threshold_value:
          filters_ok = False
          break
      if not filters_ok:
        continue

      meta = {
          "prompt_id": model.prompt.get("id") if isinstance(model.prompt, dict) else None,
          "prompt_text": prompt_text,
          "continuation_meta": {k: continuation.get(k) for k in continuation.keys() if k != RTP_TEXT_FIELD},
          "prompt_scores": prompt_scores,
          "prompt_max_score": prompt_max,
          "prompt_score_filters": self.prompt_score_filters,
          "prompt_variant": self.prompt_variant,
          "toxicity_filters": self.toxicity_filters,
          "toxicity_scores": score_map,
      }
      yield RawUnsafeRecord(
          source=self.name,
          category="toxicity",
          answer_text=text.strip(),
          toxicity_score=max_score,
          meta=meta,
      )


def factory(**kwargs) -> RealToxicityAdapter:
  return RealToxicityAdapter(**kwargs)


register_adapter("real-toxicity-prompts", factory)
