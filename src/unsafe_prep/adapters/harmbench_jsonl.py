from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Tuple

from . import UnsafeDatasetAdapter, register_adapter
from ..schemas import RawUnsafeRecord

LOGGER = logging.getLogger(__name__)


def _to_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value).expanduser()


def _normalize_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        maybe = value.strip()
        if not maybe:
            return None
        if maybe.isdigit():
            return int(maybe)
        try:
            return int(float(maybe))
        except ValueError:
            return None
    return None


def _resolve_jsonl_path(split: str, data_dir: Optional[str], data_files: Optional[dict]) -> Path:
    if isinstance(data_files, dict) and split in data_files:
        return Path(data_files[split]).expanduser()

    data_dir_path = _to_path(data_dir)
    if data_dir_path is None:
        raise ValueError("harmbench_jsonl requires data_files[split] or data_dir.")

    if data_dir_path.is_file() and data_dir_path.suffix == ".jsonl":
        return data_dir_path

    if data_dir_path.is_dir():
        candidates = sorted(data_dir_path.glob("*.jsonl"))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError(f"No .jsonl files found in directory: {data_dir_path}")
        raise ValueError(
            f"Multiple .jsonl files found in {data_dir_path}; specify data_files. "
            f"Candidates: {[p.name for p in candidates]}"
        )

    raise FileNotFoundError(f"JSONL path not found: {data_dir_path}")


@dataclass
class HarmbenchJsonlAdapter(UnsafeDatasetAdapter):
    name: str = "harmbench_jsonl"
    split: str = "train"
    data_dir: Optional[str] = None
    data_files: Optional[dict] = None
    streaming: bool = False
    unsafe_label_field: str = "label"
    unsafe_label_values: Optional[Sequence[int]] = None

    def iter_unsafe_answers(self) -> Iterator[RawUnsafeRecord]:
        jsonl_path = _resolve_jsonl_path(self.split, self.data_dir, self.data_files)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"HarmBench JSONL not found: {jsonl_path}")

        allowed_values: Optional[set[int]] = None
        if self.unsafe_label_values:
            allowed_values = set(self.unsafe_label_values)

        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    LOGGER.warning("Skipping malformed JSONL line %d: %s", line_num, exc)
                    continue

                if not isinstance(record, dict):
                    LOGGER.warning("Skipping non-mapping JSONL line %d", line_num)
                    continue

                category = record.get("category")
                answer_text = record.get("answer_text")
                meta = record.get("meta") or {}
                
                # Priority: 
                # 1. unsafe_label_field (if it exists in record or meta)
                # 2. "label" at root
                label_candidate = record.get(self.unsafe_label_field) if self.unsafe_label_field else None
                if label_candidate is None:
                    label_candidate = meta.get(self.unsafe_label_field) if self.unsafe_label_field else None
                if label_candidate is None:
                    label_candidate = record.get("label")

                label_val = _normalize_label(label_candidate)
                if allowed_values and (label_val not in allowed_values):
                    continue
                if not isinstance(category, str) or not isinstance(answer_text, str):
                    LOGGER.warning("Skipping line %d missing category/answer_text", line_num)
                    continue

                yield RawUnsafeRecord(
                    source="harmbench",
                    category=category,
                    answer_text=answer_text,
                    toxicity_score=record.get("toxicity_score"),
                    meta={**meta, "label": label_val},
                )


def factory(**kwargs) -> HarmbenchJsonlAdapter:
    return HarmbenchJsonlAdapter(**kwargs)


register_adapter("harmbench_jsonl", factory)
