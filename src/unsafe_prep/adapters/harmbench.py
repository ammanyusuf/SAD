from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Union

from . import UnsafeDatasetAdapter, register_adapter
from ..schemas import RawUnsafeRecord


def _to_path(value: Union[str, Path]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    return path


@dataclass
class HarmbenchAdapter(UnsafeDatasetAdapter):
    """Adapter that reads HarmBench `test_cases.json` style files."""

    name: str = "harmbench"
    test_cases_path: str = ""
    text_key: str = "test_case"
    completion_key: Optional[str] = "completion"

    def iter_unsafe_answers(self) -> Iterator[RawUnsafeRecord]:
        if not self.test_cases_path:
            raise ValueError("HarmbenchAdapter requires `test_cases_path` pointing to test_cases.json.")
        path = _to_path(self.test_cases_path)
        if not path.exists():
            raise FileNotFoundError(f"HarmBench test_cases file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"Expected mapping at {path}; found {type(data).__name__}")
        for behavior_id, entries in data.items():
            entries_list = entries if isinstance(entries, list) else [entries]
            for idx, entry in enumerate(entries_list):
                prompt_text: Optional[str] = None
                completion_text: Optional[str] = None
                metadata: Dict[str, object] = {
                    "behavior_id": behavior_id,
                    "entry_index": idx,
                }
                if isinstance(entry, str):
                    prompt_text = entry
                elif isinstance(entry, dict):
                    # Prefer explicit text_key, but fall back to common aliases.
                    prompt_text = entry.get(self.text_key) or entry.get("prompt") or entry.get("instruction")
                    if self.completion_key:
                        completion_text = entry.get(self.completion_key)
                    if not completion_text:
                        for key in ("generation", "response", "completion_text"):
                            val = entry.get(key)
                            if isinstance(val, str) and val.strip():
                                completion_text = val
                                break
                    metadata.update({k: v for k, v in entry.items() if k not in {self.text_key}})
                else:
                    continue
                if not isinstance(prompt_text, str) or not prompt_text.strip():
                    continue
                answer = completion_text.strip() if isinstance(completion_text, str) else ""
                yield RawUnsafeRecord(
                    source=self.name,
                    category=str(behavior_id),
                    answer_text=answer,
                    toxicity_score=None,
                    meta=metadata,
                )


def factory(**kwargs) -> HarmbenchAdapter:
    return HarmbenchAdapter(**kwargs)


register_adapter("harmbench", factory)
