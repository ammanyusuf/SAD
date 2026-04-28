"""Utilities for constructing capability evaluation plans."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from omegaconf import DictConfig, ListConfig, OmegaConf

LOGGER = logging.getLogger(__name__)


def _as_list(value: Optional[Sequence[Any]]) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _stringify(value: object) -> str:
    return str(value)


def _bool_flag(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return default


def _looks_like_chat_or_instruct(*values: Optional[str]) -> bool:
    for value in values:
        if not value:
            continue
        lowered = value.lower()
        if "instruct" in lowered or "chat" in lowered:
            return True
    return False


def _deep_merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)  # type: ignore[return-value]
    if isinstance(value, dict):
        return copy.deepcopy(value)
    raise SystemExit("Expected a mapping when constructing dataset metadata.")


def _dataset_entries(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, (DictConfig, ListConfig)):
        raw = OmegaConf.to_container(raw, resolve=True)
    entries: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                entries.append(copy.deepcopy(item))
            elif isinstance(item, str):
                entries.append({"use": item})
    elif isinstance(raw, dict):
        for name, payload in raw.items():
            if isinstance(payload, dict):
                copied = copy.deepcopy(payload)
                copied.setdefault("name", name)
                entries.append(copied)
    else:
        raise SystemExit("datasets config must be a list or mapping.")
    return entries


def _load_pipeline_config(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise SystemExit(f"Pipeline config at {path} must resolve to a mapping.")
    return cfg


def _find_repo_root(reference: Path) -> Path:
    current = reference if reference.is_dir() else reference.parent
    for path in [current, *current.parents]:
        if (path / ".git").exists():
            return path
    return reference if reference.is_dir() else reference.parent


def _resolve_subconfig_path(cfg_path: Path, rel_path: str) -> Path:
    target = Path(rel_path)
    if target.is_absolute():
        return target
    repo_root = _find_repo_root(cfg_path)
    candidate = (repo_root / target).resolve()
    if not candidate.exists():
        raise SystemExit(f"Data config '{candidate}' (from '{rel_path}') was not found.")
    return candidate


def _load_dataset_catalog(cfg_path: Path, catalog_value: Optional[object]) -> Dict[str, Dict[str, Any]]:
    if catalog_value in (None, "", "null"):
        return {}
    catalog_name = _stringify(catalog_value).strip()
    if not catalog_name:
        return {}
    catalog_path = _resolve_subconfig_path(cfg_path, catalog_name)
    if not catalog_path.exists():
        raise SystemExit(f"Data config '{catalog_path}' (from '{catalog_name}') was not found.")
    catalog_cfg = _load_pipeline_config(catalog_path)
    if "sets" not in catalog_cfg:
        raise SystemExit(f"Data config '{catalog_path}' must define a 'sets' mapping.")
    entries: Dict[str, Dict[str, Any]] = {}
    for key, payload in catalog_cfg.sets.items():  # type: ignore[attr-defined]
        data = _to_plain_dict(payload)
        data.setdefault("name", key)
        entries[key] = data
    if not entries:
        raise SystemExit(f"Data config '{catalog_path}' did not define any dataset entries.")
    return entries


def _sweep_list(value: Optional[Sequence[object]], default: Sequence[Optional[object]]) -> List[Optional[object]]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [value]


def _resolve_run_dir(base_dir: Path, slug: str, run_id: str) -> Path:
    return (base_dir / slug / run_id).resolve()


def _parse_tasks(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, (list, tuple, ListConfig)):
        return [str(item).strip() for item in raw if str(item).strip()]
    raise SystemExit("tasks must be a string or list.")


def _parse_arg_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, (list, tuple, ListConfig)):
        return [str(item).strip() for item in raw if str(item).strip()]
    raise SystemExit("model_args/eval_args must be a string or list.")


@dataclass
class CapabilityPlan:
    dataset: str
    label: str
    description: str
    run_id: str
    run_dir: Path
    variant: str
    metadata: Dict[str, Any]


@dataclass
class DatasetSpec:
    name: str
    description: str
    slug: str
    run_base_id: str
    tasks: List[str]
    eval_args: List[str]
    model_args: List[str]
    batch_size: Optional[int]
    num_fewshot: Optional[int]
    apply_chat_template: Optional[bool]
    confirm_run_unsafe_code: Optional[bool]
    log_samples: Optional[bool]
    allow_code_eval: Optional[bool]
    backend: Optional[str]
    slurm_overrides: Dict[str, Any]
    safety_cfg: Optional[Dict[str, Any]]
    skip_baseline: bool
    model_checkpoint: Optional[str]
    model_family: Optional[str]
    model_variant: Optional[str]
    model_name: Optional[str]
    tokenizer_name: Optional[str]

    def base_metadata(self) -> Dict[str, Any]:
        return {
            "tasks": list(self.tasks),
            "eval_args": list(self.eval_args),
            "model_args": list(self.model_args),
            "batch_size": self.batch_size,
            "num_fewshot": self.num_fewshot,
            "apply_chat_template": self.apply_chat_template,
            "confirm_run_unsafe_code": self.confirm_run_unsafe_code,
            "log_samples": self.log_samples,
            "allow_code_eval": self.allow_code_eval,
            "backend": self.backend,
            "slurm": self.slurm_overrides,
            "model_checkpoint": self.model_checkpoint,
            "model_family": self.model_family,
            "model_variant": self.model_variant,
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
        }


class CapabilityPlanBuilder:
    def __init__(
        self,
        cfg: DictConfig,
        cfg_path: Path,
        shared_datasets: Optional[Dict[str, Dict[str, Any]]] = None,
        timestamp_override: Optional[str] = None,
    ):
        if "run" not in cfg:
            raise SystemExit("Pipeline config must include a 'run' section.")
        self._raw_cfg = cfg
        self.cfg_path = cfg_path
        self.shared_datasets = shared_datasets or {}
        run_cfg = cfg.run  # type: ignore[attr-defined]
        output_root = getattr(run_cfg, "output_root", None)
        if not output_root:
            raise SystemExit("run.output_root must be specified.")
        self.output_root = Path(str(output_root)).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.slug_prefix = _stringify(getattr(run_cfg, "slug_prefix", "capability"))
        self.run_prefix = _stringify(getattr(run_cfg, "run_id_prefix", "capability"))

        self.base_model_args = _parse_arg_list(getattr(run_cfg, "base_model_args", None))
        self.base_eval_args = _parse_arg_list(getattr(run_cfg, "base_eval_args", None))
        self.default_batch_size = getattr(run_cfg, "batch_size", None)
        self.default_num_fewshot = getattr(run_cfg, "num_fewshot", None)
        self.default_apply_chat_template = getattr(run_cfg, "apply_chat_template", None)
        self.default_confirm_run_unsafe = getattr(run_cfg, "confirm_run_unsafe_code", None)
        self.default_log_samples = getattr(run_cfg, "log_samples", None)
        self.default_allow_code_eval = getattr(run_cfg, "allow_code_eval", None)
        self.default_backend = getattr(run_cfg, "backend", None)

        self.model_checkpoint_default = None
        self.model_family_default = None
        self.model_variant_default = None
        self.model_name_default = None
        self.tokenizer_name_default = None
        model_path = getattr(run_cfg, "model_path", None)
        model_checkpoint = getattr(run_cfg, "model_checkpoint", None)
        model_family = getattr(run_cfg, "model_family", None)
        model_variant = getattr(run_cfg, "model_variant", None)
        model_name = getattr(run_cfg, "model_name", None)
        tokenizer_name = getattr(run_cfg, "tokenizer_name", None)
        if model_checkpoint not in (None, "", "null"):
            self.model_checkpoint_default = _stringify(model_checkpoint)
        elif model_path not in (None, "", "null"):
            self.model_checkpoint_default = _stringify(model_path)
        if model_name not in (None, "", "null"):
            self.model_name_default = _stringify(model_name)
        if tokenizer_name not in (None, "", "null"):
            self.tokenizer_name_default = _stringify(tokenizer_name)
        if model_family not in (None, "", "null"):
            self.model_family_default = _stringify(model_family)
        if model_variant not in (None, "", "null"):
            self.model_variant_default = _stringify(model_variant)

        self.baseline_only = _bool_flag(getattr(run_cfg, "baseline_only", False), False)
        default_etas = [0.25, 0.5, 1, 2, 4]
        run_safety_etas = getattr(run_cfg, "safety_etas", None)
        legacy_scales = getattr(run_cfg, "safety_scales", None)
        if run_safety_etas is not None:
            self.safety_etas_default = list(run_safety_etas)
        elif legacy_scales is not None:
            self.safety_etas_default = list(legacy_scales)
        else:
            if self.baseline_only:
                self.safety_etas_default = []
            else:
                self.safety_etas_default = default_etas
                LOGGER.warning(
                    "run.safety_etas is not set; using default etas: %s", default_etas
                )
        t_start_default = getattr(run_cfg, "t_start", None)
        self.t_start_default: List[Optional[object]] = _sweep_list(t_start_default, default=[None])
        t_end_default = getattr(run_cfg, "t_end", None)
        self.t_end_default: List[Optional[object]] = _sweep_list(t_end_default, default=[None])
        self.default_safety_cfg = _to_plain_dict(run_cfg.safety) if "safety" in run_cfg else None

        self.timestamp = timestamp_override or datetime.now().strftime("%Y%m%d%H%M%S")

    def build(self, restrict_to: Optional[Sequence[str]]) -> List[CapabilityPlan]:
        datasets_raw = getattr(self._raw_cfg, "datasets", None)
        entries = _dataset_entries(datasets_raw)
        if not entries:
            raise SystemExit("Pipeline config did not declare any datasets.")
        requested = {item.strip() for item in (restrict_to or []) if item.strip()}
        plans: List[CapabilityPlan] = []
        for raw_entry in entries:
            resolved_dict = self._resolve_dataset_entry(raw_entry)
            entry_cfg = OmegaConf.create(resolved_dict)
            spec = self._parse_dataset_spec(entry_cfg)
            if requested and spec.name not in requested:
                continue
            plans.extend(self._build_dataset_plans(spec))
        if not plans:
            raise SystemExit("No capability plans were created. Check the dataset filters.")
        LOGGER.info("Scheduled %d capability jobs from %s.", len(plans), self.cfg_path)
        return plans

    def _resolve_dataset_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        data = copy.deepcopy(entry)
        reference = data.pop("use", data.pop("dataset_ref", None))
        if not reference:
            return data
        ref_key = _stringify(reference)
        base_spec = self.shared_datasets.get(ref_key)
        if not base_spec:
            raise SystemExit(f"Dataset reference '{ref_key}' was not found in the data config.")
        merged = _deep_merge_dicts(base_spec, data)
        if "name" not in merged or not merged["name"]:
            merged["name"] = ref_key
        return merged

    def _parse_dataset_spec(self, entry: DictConfig) -> DatasetSpec:
        if "name" not in entry:
            raise SystemExit("Each dataset entry must include a 'name'.")
        name = _stringify(entry.name)
        description = _stringify(getattr(entry, "description", ""))
        slug = _stringify(getattr(entry, "experiment_slug", f"{self.slug_prefix}_{name}"))
        run_base_id = _stringify(getattr(entry, "run_id", f"{self.run_prefix}-{name}-{self.timestamp}"))
        tasks = _parse_tasks(getattr(entry, "tasks", None))
        if not tasks:
            raise SystemExit(f"Dataset '{name}' must define tasks.")
        eval_args = self.base_eval_args + _parse_arg_list(getattr(entry, "eval_args", None))
        model_args = self.base_model_args + _parse_arg_list(getattr(entry, "model_args", None))
        batch_size = getattr(entry, "batch_size", self.default_batch_size)
        num_fewshot = getattr(entry, "num_fewshot", self.default_num_fewshot)
        apply_chat_template = getattr(entry, "apply_chat_template", self.default_apply_chat_template)
        confirm_run_unsafe = getattr(entry, "confirm_run_unsafe_code", self.default_confirm_run_unsafe)
        log_samples = getattr(entry, "log_samples", self.default_log_samples)
        allow_code_eval = getattr(entry, "allow_code_eval", self.default_allow_code_eval)
        backend = getattr(entry, "backend", self.default_backend)

        slurm_overrides = _to_plain_dict(entry.slurm) if "slurm" in entry else {}
        safety_cfg = _to_plain_dict(entry.safety) if "safety" in entry else None
        if self.default_safety_cfg is not None:
            if safety_cfg is None:
                safety_cfg = copy.deepcopy(self.default_safety_cfg)
            else:
                safety_cfg = _deep_merge_dicts(copy.deepcopy(self.default_safety_cfg), safety_cfg)
        skip_baseline = _bool_flag(entry.skip_baseline, False) if "skip_baseline" in entry else False

        model_checkpoint = self.model_checkpoint_default
        model_family = self.model_family_default
        model_variant = self.model_variant_default
        model_name = self.model_name_default
        tokenizer_name = self.tokenizer_name_default
        if "model_path" in entry and entry.model_path not in (None, "", "null"):
            model_checkpoint = _stringify(entry.model_path)
        if "model_checkpoint" in entry and entry.model_checkpoint not in (None, "", "null"):
            model_checkpoint = _stringify(entry.model_checkpoint)
        if "model_name" in entry and entry.model_name not in (None, "", "null"):
            model_name = _stringify(entry.model_name)
        if "tokenizer_name" in entry and entry.tokenizer_name not in (None, "", "null"):
            tokenizer_name = _stringify(entry.tokenizer_name)
        if "model_family" in entry and entry.model_family not in (None, "", "null"):
            model_family = _stringify(entry.model_family)
        if "model_variant" in entry and entry.model_variant not in (None, "", "null"):
            model_variant = _stringify(entry.model_variant)
        if apply_chat_template is None:
            apply_chat_template = _looks_like_chat_or_instruct(
                model_name, model_variant, model_checkpoint, tokenizer_name
            )
            if apply_chat_template:
                LOGGER.info(
                    "Dataset '%s': inferred apply_chat_template=true from model identifiers.",
                    name,
                )

        return DatasetSpec(
            name=name,
            description=description,
            slug=slug,
            run_base_id=run_base_id,
            tasks=tasks,
            eval_args=eval_args,
            model_args=model_args,
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            apply_chat_template=apply_chat_template,
            confirm_run_unsafe_code=confirm_run_unsafe,
            log_samples=log_samples,
            allow_code_eval=allow_code_eval,
            backend=backend,
            slurm_overrides=slurm_overrides,
            safety_cfg=safety_cfg,
            skip_baseline=skip_baseline,
            model_checkpoint=model_checkpoint,
            model_family=model_family,
            model_variant=model_variant,
            model_name=model_name,
            tokenizer_name=tokenizer_name,
        )

    def _build_dataset_plans(self, spec: DatasetSpec) -> List[CapabilityPlan]:
        plans: List[CapabilityPlan] = []
        base_metadata = spec.base_metadata()

        if not spec.skip_baseline:
            baseline_metadata = {"safety_enabled": False, **base_metadata}
            plans.append(
                self._make_plan(
                    spec=spec,
                    label="baseline",
                    variant="baseline",
                    metadata_override=baseline_metadata,
                    description_suffix=f"{spec.description} (baseline)".strip(),
                )
            )

        safety_cfg = None if self.baseline_only else spec.safety_cfg
        if isinstance(safety_cfg, dict):
            artifact_root = safety_cfg.get("artifact_root")
            artifact_names = safety_cfg.get("artifact_names") or safety_cfg.get("artifacts")
            unsafe_artifacts = safety_cfg.get("unsafe_artifacts")
            auto_build_unsafe = safety_cfg.get("auto_build_unsafe_artifacts")
            if auto_build_unsafe in (None, "", "null"):
                auto_build_unsafe = safety_cfg.get("auto_generate_artifact")
            t_start_list: List[Optional[object]] = _sweep_list(
                safety_cfg.get("t_start"), self.t_start_default
            )
            t_end_list: List[Optional[object]] = _sweep_list(
                safety_cfg.get("t_end"), self.t_end_default
            )
            if auto_build_unsafe not in (None, "", "null"):
                auto_build_unsafe = _bool_flag(auto_build_unsafe, False)
            if unsafe_artifacts:
                unsafe_artifacts = _stringify(unsafe_artifacts)
            if artifact_root and artifact_names:
                artifact_root = _stringify(artifact_root)
                artifact_names_list = [_stringify(item) for item in artifact_names]
                etas = safety_cfg.get("etas", safety_cfg.get("scales", self.safety_etas_default))
                for artifact in artifact_names_list:
                    for eta in etas:
                        for t_start in t_start_list:
                            for t_end in t_end_list:
                                t_start_val = int(t_start) if t_start is not None else None
                                t_end_val = int(t_end) if t_end is not None else None
                                if (
                                    t_start_val is not None
                                    and t_end_val is not None
                                    and (t_end_val <= t_start_val)
                                ):
                                    LOGGER.warning(
                                        "Skipping invalid timestep range t_start=%s, t_end=%s for artifact %s.",
                                        t_start,
                                        t_end,
                                        artifact,
                                    )
                                    continue
                                meta_override = {
                                    "safety_enabled": True,
                                    "artifact_name": artifact,
                                    "safety_eta": eta,
                                    "artifact_root": artifact_root,
                                    "t_start": t_start,
                                    "t_end": t_end,
                                }
                                if auto_build_unsafe not in (None, "", "null"):
                                    meta_override["auto_build_unsafe_artifacts"] = auto_build_unsafe
                                label = f"{artifact}-eta{eta}"
                                if t_start is not None or t_end is not None:
                                    label = (
                                        f"{label}-ts{t_start_val if t_start is not None else 'n'}_"
                                        f"{t_end_val if t_end is not None else 'n'}"
                                    )
                                plans.append(
                                    self._make_plan(
                                        spec=spec,
                                        label=label,
                                        variant="safe",
                                        metadata_override={**meta_override, **base_metadata},
                                        description_suffix=(
                                            f"{spec.description} (artifact={artifact}, eta={eta})"
                                        ).strip(),
                                    )
                                )
            elif unsafe_artifacts:
                etas = safety_cfg.get("etas", safety_cfg.get("scales", self.safety_etas_default))
                for eta in etas:
                    for t_start in t_start_list:
                        for t_end in t_end_list:
                            t_start_val = int(t_start) if t_start is not None else None
                            t_end_val = int(t_end) if t_end is not None else None
                            if (
                                t_start_val is not None
                                and t_end_val is not None
                                and (t_end_val <= t_start_val)
                            ):
                                LOGGER.warning(
                                    "Skipping invalid timestep range t_start=%s, t_end=%s for unsafe_artifacts.",
                                    t_start,
                                    t_end,
                                )
                                continue
                            meta_override = {
                                "safety_enabled": True,
                                "safety_eta": eta,
                                "unsafe_artifacts": unsafe_artifacts,
                                "t_start": t_start,
                                "t_end": t_end,
                            }
                            if auto_build_unsafe not in (None, "", "null"):
                                meta_override["auto_build_unsafe_artifacts"] = auto_build_unsafe
                            label = f"unsafe_artifacts-eta{eta}"
                            if t_start is not None or t_end is not None:
                                label = (
                                    f"{label}-ts{t_start_val if t_start is not None else 'n'}_"
                                    f"{t_end_val if t_end is not None else 'n'}"
                                )
                            plans.append(
                                self._make_plan(
                                    spec=spec,
                                    label=label,
                                    variant="safe",
                                    metadata_override={**meta_override, **base_metadata},
                                    description_suffix=(
                                        f"{spec.description} (unsafe_artifacts, eta={eta})"
                                    ).strip(),
                                )
                            )
            elif artifact_root or artifact_names:
                raise SystemExit(
                    f"Dataset '{spec.name}' safety config must include both artifact_root and artifact_names."
                )
        return plans

    def _make_plan(
        self,
        spec: DatasetSpec,
        label: str,
        variant: str,
        metadata_override: Optional[Dict[str, Any]],
        description_suffix: str,
    ) -> CapabilityPlan:
        run_id = f"{spec.run_base_id}-{label}"
        run_dir = _resolve_run_dir(self.output_root, spec.slug, run_id)
        plan_metadata = spec.base_metadata()
        if metadata_override:
            plan_metadata.update(metadata_override)
        return CapabilityPlan(
            dataset=spec.name,
            label=label,
            description=description_suffix or spec.description,
            run_id=run_id,
            run_dir=run_dir,
            variant=variant,
            metadata=plan_metadata,
        )


def build_capability_plans(
    cfg_path: Path,
    restrict_to: Optional[Sequence[str]],
    timestamp_override: Optional[str] = None,
) -> List[CapabilityPlan]:
    cfg = _load_pipeline_config(cfg_path)
    shared_datasets = _load_dataset_catalog(cfg_path, getattr(cfg, "data_catalog", None))
    builder = CapabilityPlanBuilder(
        cfg=cfg,
        cfg_path=cfg_path,
        shared_datasets=shared_datasets,
        timestamp_override=timestamp_override,
    )
    return builder.build(restrict_to)
