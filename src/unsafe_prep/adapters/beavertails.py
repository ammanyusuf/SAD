from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Any

from datasets import load_dataset

from . import UnsafeDatasetAdapter, register_adapter
from ..schemas import RawUnsafeRecord, BeaverTailsRecord


def _parse_category_field(value: Any) -> Tuple[List[str], bool]:
  if isinstance(value, str):
    try:
      data = json.loads(value)
    except json.JSONDecodeError:
      data = {value: True}
  elif isinstance(value, dict):
    data = value
  else:
    data = {}
  categories: List[str] = []
  for key, flag in data.items():
    if bool(flag):
      categories.append(str(key))
  return categories, bool(categories)


@dataclass
class BeaverTailsAdapter(UnsafeDatasetAdapter):
  name: str = "beavertails"
  split: str = "train"
  data_dir: Optional[str] = None
  data_files: Optional[dict] = None
  streaming: bool = False
  keep_categories: Optional[Sequence[str]] = None
  prompt_is_safe: Optional[bool] = None

  def iter_unsafe_answers(self) -> Iterator[RawUnsafeRecord]:
    dataset_path = self.data_dir or "BeaverTails"
    load_kwargs = {
        "split": self.split,
        "streaming": self.streaming,
    }
    if self.data_files:
        load_kwargs["data_files"] = self.data_files
    if self.data_dir and not Path(self.data_dir).is_absolute():
        load_kwargs["data_dir"] = self.data_dir
    dataset = load_dataset(dataset_path, **load_kwargs)
    iterator: Iterable = dataset if self.streaming else dataset  # type: ignore
    for row in iterator:
      record = BeaverTailsRecord(**row)
      if self.prompt_is_safe is None:
        # Default behavior: only use unsafe examples.
        if record.is_safe:
          continue
      elif record.is_safe != self.prompt_is_safe:
        # Filter to either safe or unsafe prompts depending on prompt_is_safe flag.
        continue
      categories, has_category = _parse_category_field(record.category)
      if self.keep_categories:
        selected = {cat for cat in categories}
        if not selected.intersection(self.keep_categories):
          continue
      answer = record.response
      prompt = record.prompt
      category = "|".join(categories) if categories else "unspecified"
      meta = {
          "prompt": prompt,
          "raw_categories": categories,
          "example_id": row.get("id") or row.get("response_id"),
          "preference": row.get("preference") or row.get("tag"),
          "is_safe": record.is_safe,
      }
      yield RawUnsafeRecord(
          source=self.name,
          category=category,
          answer_text=answer.strip(),
          toxicity_score=None,
          meta=meta,
      )


def factory(**kwargs) -> BeaverTailsAdapter:
  return BeaverTailsAdapter(**kwargs)


register_adapter("beavertails", factory)
