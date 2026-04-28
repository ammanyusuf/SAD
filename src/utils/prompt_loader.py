from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from omegaconf import DictConfig, OmegaConf

from sampling.sample_text import PromptRecord
from unsafe_prep.registry import AdapterNotFoundError, available_adapters, get_adapter
from unsafe_prep.schemas import RawUnsafeRecord

LOGGER = logging.getLogger(__name__)

_PROMPT_FIELDS = (
    "prompt",
    "prompt_text",
    "test_case",
    "text",
    "question",
    "input",
    "instruction",
    "query",
    "source",
)
_COMPLETION_FIELDS = (
    "response",
    "completion",
    "continuation",
    "answer",
    "generation",
    "target",
    "output",
)


def _extract_text_blob(value: Any) -> Optional[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, dict):
        for key in ("text", "prompt", "content", "value"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _slice_metadata_payload(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        cleaned = {
            key: val
            for key, val in value.items()
            if key not in {"text", "prompt", "content", "value"}
        }
        return cleaned or None
    return None


def _parse_prompt_entry(entry: Dict[str, Any], default_id: str) -> PromptRecord:
    prompt_text: Optional[str] = None
    used_field: Optional[str] = None
    prompt_field_payload: Any = None
    for field in _PROMPT_FIELDS:
        if field in entry:
            payload = entry[field]
            prompt_text = _extract_text_blob(payload)
            if prompt_text:
                used_field = field
                prompt_field_payload = payload
                break
    if prompt_text is None:
        raise TypeError(f"Prompt entry {default_id} is missing a valid text field.")
    prompt_id = str(entry.get("prompt_id") or entry.get("id") or default_id)
    completion_field: Optional[str] = None
    completion_payload: Any = None
    reference_completion: Optional[str] = None
    for completion_field in _COMPLETION_FIELDS:
        if completion_field in entry:
            payload = entry[completion_field]
            reference_completion = _extract_text_blob(payload)
            if reference_completion:
                completion_payload = payload
                break
    metadata: Dict[str, Any] = {}
    for key, value in entry.items():
        if key in {"prompt_id", "id"}:
            continue
        if used_field and key == used_field:
            prompt_meta = _slice_metadata_payload(prompt_field_payload)
            if prompt_meta:
                metadata[f"{used_field}_meta"] = prompt_meta
            continue
        if completion_field and key == completion_field:
            completion_meta = _slice_metadata_payload(completion_payload)
            if completion_meta:
                metadata[f"{completion_field}_meta"] = completion_meta
            continue
        metadata[key] = value
    if reference_completion:
        metadata["reference_completion"] = reference_completion
    return PromptRecord(prompt_id=prompt_id, prompt=prompt_text, metadata=metadata)


def _load_prompts_from_json(path: Path) -> List[PromptRecord]:
    raw_text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse dataset JSON at {path}: {exc}") from exc

    records: List[PromptRecord] = []

    def handle_entry(entry: Any, default_id: str) -> None:
        if isinstance(entry, dict):
            records.append(_parse_prompt_entry(entry, default_id))
        elif isinstance(entry, str):
            records.append(PromptRecord(prompt_id=default_id, prompt=entry, metadata={}))
        else:
            raise TypeError(f"Unsupported prompt entry type {type(entry).__name__} in {path}.")

    if isinstance(data, list):
        for idx, entry in enumerate(data):
            handle_entry(entry, f"{path.stem}:{idx}")
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    handle_entry(item, f"{key}:{idx}")
            else:
                handle_entry(value, str(key))
    else:
        raise SystemExit(f"Unsupported dataset format in {path}. Expected list or object.")

    if not records:
        raise SystemExit(f"No prompts found in dataset: {path}")
    return records


def _extract_prompt_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    for key in _PROMPT_FIELDS + ("behavior", "behavior_text"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _records_from_adapter(name: str, params: Dict[str, Any]) -> List[PromptRecord]:
    try:
        adapter = get_adapter(name, **params)
    except AdapterNotFoundError as exc:
        raise SystemExit(
            f"Prompt source '{name}' is not registered. "
            f"Available adapters: {', '.join(available_adapters())}"
        ) from exc
    prompts: List[PromptRecord] = []
    iterator = adapter.iter_unsafe_answers()
    for idx, record in enumerate(iterator, start=1):
        prompt_text = _extract_prompt_from_meta(record.meta)
        if not prompt_text:
            continue
        prompt_id = str(
            record.meta.get("prompt_id")
            or record.meta.get("example_id")
            or f"{record.source}:{record.meta.get('behavior_id', '')}:{idx}"
        )
        metadata = dict(record.meta)
        metadata.setdefault("source", record.source)
        metadata["category"] = record.category
        metadata["toxicity_score"] = record.toxicity_score
        if record.answer_text:
            metadata["reference_completion"] = record.answer_text
        prompts.append(PromptRecord(prompt_id=prompt_id, prompt=prompt_text, metadata=metadata))
    if not prompts:
        LOGGER.warning("Adapter '%s' produced zero prompts with the provided filters.", name)
    return prompts


def load_prompt_records(
    dataset_path: Optional[Path],
    prompt_source_cfg: Optional[DictConfig],
) -> List[PromptRecord]:
    """
    Load prompt records either from a JSON file or via a registered HF adapter.

    Args:
        dataset_path: Optional path to a JSON/JSONL dataset.
        prompt_source_cfg: Hydra config describing the adapter source.
    """
    if dataset_path is not None:
        return _load_prompts_from_json(dataset_path)
    if prompt_source_cfg:
        source_name = prompt_source_cfg.get("name")
        if source_name:
            params_cfg = prompt_source_cfg.get("params") or {}
            params = OmegaConf.to_container(params_cfg, resolve=True) if isinstance(params_cfg, DictConfig) else params_cfg
            if params is None:
                params = {}
            if not isinstance(params, dict):
                raise SystemExit("data.prompt_source.params must be a mapping of adapter kwargs.")
            return _records_from_adapter(str(source_name), params)
    return []
