"""Utilities for constructing multi-dataset jailbreak evaluation plans."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from omegaconf import DictConfig, ListConfig, OmegaConf


LOGGER = logging.getLogger(__name__)


def _as_list(value: Optional[Sequence[str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


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


def _parse_variants(raw: Any) -> List["JailbreakVariant"]:
    if raw is None:
        return [JailbreakVariant(name="", label="", description="", overrides=[], metadata={})]
    if isinstance(raw, (DictConfig, ListConfig)):
        raw = OmegaConf.to_container(raw, resolve=True)
    variants: List[JailbreakVariant] = []

    def _build_variant(item: Dict[str, Any], default_name: str) -> JailbreakVariant:
        name = _stringify(item.get("name", default_name))
        label = _stringify(item.get("label", name))
        description = _stringify(item.get("description", ""))
        overrides = _as_list(item.get("overrides", []))
        metadata = _to_plain_dict(item.get("metadata", {})) if isinstance(item, dict) else {}
        attack_method = item.get("attack_method")
        defense_method = item.get("defense_method")
        output_name = item.get("output_name")
        if attack_method not in (None, "", "null"):
            overrides.append(f"jailbreak.attack_method={_stringify(attack_method)}")
            metadata["attack_method"] = _stringify(attack_method)
        if defense_method not in (None, "", "null"):
            overrides.append(f"jailbreak.defense_method={_stringify(defense_method)}")
            metadata["defense_method"] = _stringify(defense_method)
        if output_name not in (None, "", "null"):
            overrides.append(f"jailbreak.output_name={_stringify(output_name)}")
            metadata["output_name"] = _stringify(output_name)
        return JailbreakVariant(
            name=name,
            label=label,
            description=description,
            overrides=overrides,
            metadata=metadata,
        )

    if isinstance(raw, list):
        for idx, item in enumerate(raw):
            if isinstance(item, dict):
                variants.append(_build_variant(item, f"variant_{idx}"))
            elif isinstance(item, str):
                variants.append(
                    _build_variant({"attack_method": item, "name": item}, item)
                )
    elif isinstance(raw, dict):
        for name, payload in raw.items():
            if isinstance(payload, dict):
                payload_copy = copy.deepcopy(payload)
                payload_copy.setdefault("name", name)
                variants.append(_build_variant(payload_copy, name))
    else:
        raise SystemExit("jailbreak_variants config must be a list or mapping.")
    return variants or [JailbreakVariant(name="", label="", description="", overrides=[], metadata={})]


def _parse_defense_variants(raw: Any) -> List["DefenseVariant"]:
    if raw is None:
        return []
    if isinstance(raw, (DictConfig, ListConfig)):
        raw = OmegaConf.to_container(raw, resolve=True)
    variants: List[DefenseVariant] = []

    def _build_variant(item: Dict[str, Any], default_name: str) -> DefenseVariant:
        name = _stringify(item.get("name", default_name))
        label = _stringify(item.get("label", name))
        description = _stringify(item.get("description", ""))
        overrides = _as_list(item.get("overrides", []))
        metadata = _to_plain_dict(item.get("metadata", {})) if isinstance(item, dict) else {}
        defense_method = item.get("defense_method")
        output_name = item.get("output_name")
        if defense_method not in (None, "", "null"):
            overrides.append(f"jailbreak.defense_method={_stringify(defense_method)}")
            metadata["defense_method"] = _stringify(defense_method)
        if output_name not in (None, "", "null"):
            overrides.append(f"jailbreak.output_name={_stringify(output_name)}")
            metadata["output_name"] = _stringify(output_name)
        return DefenseVariant(
            name=name,
            label=label,
            description=description,
            overrides=overrides,
            metadata=metadata,
        )

    if isinstance(raw, list):
        for idx, item in enumerate(raw):
            if isinstance(item, dict):
                variants.append(_build_variant(item, f"defense_{idx}"))
            elif isinstance(item, str):
                variants.append(
                    _build_variant({"defense_method": item, "name": item}, item)
                )
    elif isinstance(raw, dict):
        for name, payload in raw.items():
            if isinstance(payload, dict):
                payload_copy = copy.deepcopy(payload)
                payload_copy.setdefault("name", name)
                variants.append(_build_variant(payload_copy, name))
    else:
        raise SystemExit("jailbreak_defense_variants config must be a list or mapping.")
    return variants


def _sweep_list(value: Optional[Sequence[object]], default: Sequence[Optional[object]]) -> List[Optional[object]]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [value]


def _resolve_run_dir(base_dir: Path, slug: str, run_id: str) -> Path:
    return (base_dir / slug / run_id).resolve()


@dataclass
class JailbreakVariant:
    name: str
    label: str
    description: str
    overrides: List[str]
    metadata: Dict[str, Any]

    @property
    def slug(self) -> str:
        return self.label or self.name


@dataclass
class DefenseVariant:
    name: str
    label: str
    description: str
    overrides: List[str]
    metadata: Dict[str, Any]

    @property
    def slug(self) -> str:
        return self.label or self.name


@dataclass
class JailbreakPlan:
    dataset: str
    label: str
    description: str
    hydra_config: str
    overrides: List[str]
    experiment_slug: str
    run_id: str
    run_dir: Path
    variant: str
    jailbreak_variant: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class DatasetSpec:
    name: str
    description: str
    slug: str
    run_base_id: str
    config_name: Optional[str]
    dataset_json: str
    prompt_limit: Optional[int]
    overrides: List[str]
    slurm_overrides: Dict[str, Any]
    safety_cfg: Optional[Dict[str, Any]]
    skip_baseline: bool
    jailbreak_variants: List[JailbreakVariant]
    defense_variants: List[DefenseVariant]
    model_checkpoint: Optional[str]
    model_family: Optional[str]
    model_variant: Optional[str]
    model_name: Optional[str]
    tokenizer_name: Optional[str]
    alt_prompts: Dict[str, str]

    def base_metadata(self) -> Dict[str, Any]:
        return {
            "dataset_json": self.dataset_json,
            "prompt_limit": self.prompt_limit,
            "slurm": self.slurm_overrides,
            "model_checkpoint": self.model_checkpoint,
            "model_family": self.model_family,
            "model_variant": self.model_variant,
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "alt_prompts": dict(self.alt_prompts),
        }


class JailbreakPlanBuilder:
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
        if "config_name" not in run_cfg:
            raise SystemExit("run.config_name must be provided so Hydra knows which config to load.")
        self.hydra_config = _stringify(run_cfg.config_name)  # type: ignore[attr-defined]
        output_root = getattr(run_cfg, "output_root", None)
        if not output_root:
            raise SystemExit("run.output_root must be specified.")
        self.output_root = Path(str(output_root)).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.slug_prefix = _stringify(getattr(run_cfg, "slug_prefix", "jailbreak"))
        self.run_prefix = _stringify(getattr(run_cfg, "run_id_prefix", "jailbreak"))
        base_overrides = getattr(run_cfg, "base_overrides", [])
        self.base_overrides = _as_list(base_overrides)

        run_jailbreak = getattr(run_cfg, "jailbreak", None)
        if run_jailbreak:
            for key, value in _to_plain_dict(run_jailbreak).items():
                if value not in (None, "", "null"):
                    self.base_overrides.append(f"jailbreak.{key}={_stringify(value)}")

        run_gen = getattr(run_cfg, "gen", None)
        if run_gen:
            for key, value in _to_plain_dict(run_gen).items():
                if value in (None, "", "null"):
                    continue
                mapped_key = None
                if key == "sampling_steps":
                    mapped_key = "steps"
                elif key == "max_new_tokens":
                    mapped_key = "gen_length"
                elif key == "block_length":
                    mapped_key = "block_length"
                elif key == "temperature":
                    mapped_key = "temperature"
                if mapped_key:
                    self.base_overrides.append(f"jailbreak.{mapped_key}={_stringify(value)}")

        output_name = getattr(run_cfg, "output_name", None)
        if output_name not in (None, "", "null"):
            self.base_overrides.append(f"jailbreak.output_name={_stringify(output_name)}")

        self.default_attack_prompt = None
        attack_prompt = getattr(run_cfg, "attack_prompt", None)
        if attack_prompt not in (None, "", "null"):
            self.default_attack_prompt = _stringify(attack_prompt)

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

        # These flags affect whether safety sweeps will be constructed at all.
        self.baseline_only = _bool_flag(getattr(run_cfg, "baseline_only", False), False)
        self.apply_defense_variants_to_baseline = _bool_flag(
            getattr(run_cfg, "apply_defense_variants_to_baseline", False),
            False,
        )

        default_etas = [0.25, 0.5, 1, 2, 4]
        run_safety_etas = getattr(run_cfg, "safety_etas", None)
        legacy_scales = getattr(run_cfg, "safety_scales", None)
        if run_safety_etas is not None:
            self.safety_etas_default = list(run_safety_etas)
        elif legacy_scales is not None:
            self.safety_etas_default = list(legacy_scales)
        else:
            # When baseline_only is enabled we will not build safety sweeps, so
            # avoid default etas (and the warning) that can be confusing.
            if self.baseline_only:
                self.safety_etas_default = []
            else:
                self.safety_etas_default = default_etas
                LOGGER.warning(
                    "run.safety_etas is not set; using default etas: %s", default_etas
                )
        t_start_default = getattr(run_cfg, "t_start", None)
        self.t_start_default: List[Optional[object]] = _sweep_list(
            t_start_default, default=[None]
        )
        t_end_default = getattr(run_cfg, "t_end", None)
        self.t_end_default: List[Optional[object]] = _sweep_list(
            t_end_default, default=[None]
        )

        self.default_variants = _parse_variants(getattr(run_cfg, "jailbreak_variants", None))
        self.default_defense_variants = _parse_defense_variants(
            getattr(run_cfg, "jailbreak_defense_variants", None)
        )
        self.timestamp = timestamp_override or datetime.now().strftime("%Y%m%d%H%M%S")

    def build(self, restrict_to: Optional[Sequence[str]]) -> List[JailbreakPlan]:
        datasets_raw = getattr(self._raw_cfg, "datasets", None)
        entries = _dataset_entries(datasets_raw)
        if not entries:
            if self.default_attack_prompt:
                entries = [
                    {
                        "name": "default",
                        "dataset_json": self.default_attack_prompt,
                        "description": "default attack prompt",
                    }
                ]
            else:
                raise SystemExit("Pipeline config did not declare any datasets.")
        requested = {item.strip() for item in (restrict_to or []) if item.strip()}
        plans: List[JailbreakPlan] = []
        for raw_entry in entries:
            resolved_dict = self._resolve_dataset_entry(raw_entry)
            entry_cfg = OmegaConf.create(resolved_dict)
            spec = self._parse_dataset_spec(entry_cfg)
            if requested and spec.name not in requested:
                continue
            plans.extend(self._build_dataset_plans(spec))
        if not plans:
            raise SystemExit("No jailbreak plans were created. Check the dataset filters.")
        LOGGER.info("Scheduled %d jailbreak jobs from %s.", len(plans), self.cfg_path)
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
        dataset_json = ""
        if "dataset_json" in entry:
            value = entry.dataset_json
            if value not in (None, "", "null"):
                dataset_json = _stringify(value)
        if not dataset_json and self.default_attack_prompt:
            dataset_json = self.default_attack_prompt
        if not dataset_json:
            raise SystemExit(f"Dataset '{name}' must provide data.dataset_json or run.attack_prompt.")
        prompt_limit: Optional[int] = None
        if "prompt_limit" in entry:
            limit_val = entry.prompt_limit
            if limit_val not in (None, "", "null"):
                prompt_limit = int(limit_val)
        elif "limit" in entry:
            limit_val = entry.limit
            if limit_val not in (None, "", "null"):
                prompt_limit = int(limit_val)
        alt_prompts: Dict[str, str] = {}
        if "dija_prompt" in entry and entry.dija_prompt not in (None, "", "null"):
            alt_prompts["dija_prompt"] = _stringify(entry.dija_prompt)
        if "dija_attack_prompt" in entry and entry.dija_attack_prompt not in (None, "", "null"):
            alt_prompts["dija_attack_prompt"] = _stringify(entry.dija_attack_prompt)
        for key in ("autodan_attack_prompt", "autodan_prompt", "gcg_attack_prompt", "gcg_prompt"):
            if key in entry and entry[key] not in (None, "", "null"):
                alt_prompts[_stringify(key)] = _stringify(entry[key])
        for key in entry.keys():
            if not isinstance(key, str):
                continue
            if key.endswith("_attack_prompt") or key.endswith("_prompt"):
                value = entry[key]
                if value not in (None, "", "null"):
                    alt_prompts.setdefault(_stringify(key), _stringify(value))
        config_name: Optional[str] = None
        if "config_name" in entry:
            raw_config_name = entry.config_name
            if raw_config_name not in (None, "", "null"):
                config_name = _stringify(raw_config_name)
        description = _stringify(getattr(entry, "description", ""))
        slug = _stringify(getattr(entry, "experiment_slug", f"{self.slug_prefix}_{name}"))
        run_base_id = _stringify(getattr(entry, "run_id", f"{self.run_prefix}-{name}-{self.timestamp}"))
        dataset_overrides = _as_list(getattr(entry, "overrides", []))
        slurm_overrides = _to_plain_dict(entry.slurm) if "slurm" in entry else {}
        safety_cfg = _to_plain_dict(entry.safety) if "safety" in entry else None
        skip_baseline = _bool_flag(entry.skip_baseline, False) if "skip_baseline" in entry else False
        variants_raw = None
        if "jailbreak_variants" in entry:
            variants_raw = entry.jailbreak_variants
        elif "variants" in entry:
            variants_raw = entry.variants
        elif "prompt_variants" in entry:
            variants_raw = entry.prompt_variants
        jailbreak_variants = _parse_variants(variants_raw) if variants_raw is not None else self.default_variants
        defense_variants_raw = None
        if "jailbreak_defense_variants" in entry:
            defense_variants_raw = entry.jailbreak_defense_variants
        elif "defense_variants" in entry:
            defense_variants_raw = entry.defense_variants
        defense_variants = (
            _parse_defense_variants(defense_variants_raw)
            if defense_variants_raw is not None
            else self.default_defense_variants
        )

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

        return DatasetSpec(
            name=name,
            description=description,
            slug=slug,
            run_base_id=run_base_id,
            config_name=config_name,
            dataset_json=dataset_json,
            prompt_limit=prompt_limit,
            overrides=dataset_overrides,
            slurm_overrides=slurm_overrides,
            safety_cfg=safety_cfg,
            skip_baseline=skip_baseline,
            jailbreak_variants=jailbreak_variants,
            defense_variants=defense_variants,
            model_checkpoint=model_checkpoint,
            model_family=model_family,
            model_variant=model_variant,
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            alt_prompts=alt_prompts,
        )

    def _build_dataset_plans(self, spec: DatasetSpec) -> List[JailbreakPlan]:
        plans: List[JailbreakPlan] = []
        for jb_variant in spec.jailbreak_variants:
            variant_overrides = list(self.base_overrides) + list(spec.overrides) + list(jb_variant.overrides)
            base_metadata = spec.base_metadata()
            if jb_variant.metadata:
                base_metadata.update(jb_variant.metadata)
            selected_prompt = spec.dataset_json
            attack_method = jb_variant.metadata.get("attack_method") if jb_variant.metadata else None
            attack_method_key = _stringify(attack_method).strip().lower() if attack_method else ""
            variant_key_raw = jb_variant.name or jb_variant.label or ""
            variant_key = _stringify(variant_key_raw).strip().lower().replace("-", "_").replace(" ", "_")
            if attack_method_key == "dija":
                selected_prompt = (
                    spec.alt_prompts.get("dija_prompt")
                    or spec.alt_prompts.get("dija_attack_prompt")
                    or selected_prompt
                )
            elif attack_method_key:
                selected_prompt = (
                    spec.alt_prompts.get(f"{attack_method_key}_attack_prompt")
                    or spec.alt_prompts.get(f"{attack_method_key}_prompt")
                    or selected_prompt
                )
            if variant_key:
                selected_prompt = (
                    spec.alt_prompts.get(f"{variant_key}_attack_prompt")
                    or spec.alt_prompts.get(f"{variant_key}_prompt")
                    or selected_prompt
                )
            jb_variant_name = jb_variant.name or jb_variant.label or None
            description_base = spec.description
            if jb_variant.description or jb_variant.label:
                desc_suffix = jb_variant.description or jb_variant.label
                description_base = f"{spec.description} ({desc_suffix})".strip()
            variant_run_base = spec.run_base_id
            if jb_variant.slug:
                variant_run_base = f"{spec.run_base_id}-{jb_variant.slug}"
            attack_method_val = None
            if jb_variant.metadata:
                attack_method_val = jb_variant.metadata.get("attack_method")
                if attack_method_val in (None, "", "null"):
                    attack_method_val = None
            meta_common: Dict[str, Any] = {"dataset_json": selected_prompt}
            if attack_method_val is not None:
                meta_common["attack_method"] = _stringify(attack_method_val)

            if not spec.skip_baseline:
                baseline_metadata = {"safety_enabled": False, **meta_common}
                plans.append(
                    self._make_plan(
                        spec=spec,
                        label="baseline",
                        variant="baseline",
                        overrides=variant_overrides,
                        suffix_overrides=["safety.enabled=false"],
                        metadata_override=baseline_metadata,
                        description_suffix=f"{description_base} (baseline)".strip(),
                        run_base_id=variant_run_base,
                        jailbreak_variant=jb_variant_name,
                    )
                )
                if self.apply_defense_variants_to_baseline and spec.defense_variants:
                    for defense in spec.defense_variants:
                        defense_overrides = list(defense.overrides)
                        defense_meta = dict(defense.metadata)
                        defense_label = defense.slug or defense.name or "defense"
                        defense_suffix = f"baseline-def-{defense_label}"
                        plans.append(
                            self._make_plan(
                                spec=spec,
                                label=defense_suffix,
                                variant="baseline",
                                overrides=variant_overrides + defense_overrides,
                                suffix_overrides=["safety.enabled=false"],
                                metadata_override={**baseline_metadata, **defense_meta},
                                description_suffix=(
                                    f"{description_base} (baseline, defense={defense_label})".strip()
                                ),
                                run_base_id=variant_run_base,
                                jailbreak_variant=jb_variant_name,
                            )
                        )

            safety_cfg = None if self.baseline_only else spec.safety_cfg
            if isinstance(safety_cfg, dict):
                artifact_root = safety_cfg.get("artifact_root")
                artifact_names = safety_cfg.get("artifact_names") or safety_cfg.get("artifacts")
                unsafe_artifacts = safety_cfg.get("unsafe_artifacts")
                t_start_list: List[Optional[object]] = _sweep_list(
                    safety_cfg.get("t_start"), self.t_start_default
                )
                t_end_list: List[Optional[object]] = _sweep_list(
                    safety_cfg.get("t_end"), self.t_end_default
                )
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
                                    if (t_start_val is not None and t_end_val is not None) and (t_end_val <= t_start_val):
                                        LOGGER.warning(
                                            "Skipping invalid timestep range t_start=%s, t_end=%s for artifact %s.",
                                            t_start, t_end, artifact
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
                                    suffixes = [
                                        "safety.enabled=true",
                                        f"safety.eta={eta}",
                                        f"safety.unsafe_artifact_root={artifact_root}",
                                        f"safety.unsafe_artifact_name={artifact}",
                                    ]
                                    if t_start is not None:
                                        suffixes.append(f"safety.t_start={t_start_val if t_start_val is not None else t_start}")
                                    if t_end is not None:
                                        suffixes.append(f"safety.t_end={t_end_val if t_end_val is not None else t_end}")
                                    label = f"{artifact}-eta{eta}"
                                    if t_start is not None or t_end is not None:
                                        label = f"{label}-ts{t_start_val if t_start is not None else 'n'}_{t_end_val if t_end is not None else 'n'}"
                                    base_safe_plan = self._make_plan(
                                        spec=spec,
                                        label=label,
                                        variant="safe",
                                        overrides=variant_overrides,
                                        suffix_overrides=suffixes,
                                        metadata_override={**meta_override, **meta_common},
                                        description_suffix=f"{description_base} (artifact={artifact}, eta={eta})".strip(),
                                        run_base_id=variant_run_base,
                                        jailbreak_variant=jb_variant_name,
                                    )
                                    plans.append(base_safe_plan)
                                    for defense in spec.defense_variants or []:
                                        defense_overrides = list(defense.overrides)
                                        defense_meta = dict(defense.metadata)
                                        defense_label = defense.slug or defense.name
                                        defense_suffix = f"{label}-def-{defense_label}" if defense_label else f"{label}-def"
                                        plans.append(
                                            self._make_plan(
                                                spec=spec,
                                                label=defense_suffix,
                                                variant="safe",
                                                overrides=variant_overrides + defense_overrides,
                                                suffix_overrides=suffixes,
                                                metadata_override={**meta_override, **defense_meta, **meta_common},
                                                description_suffix=f"{description_base} (artifact={artifact}, eta={eta}, defense={defense_label})".strip(),
                                                run_base_id=variant_run_base,
                                                jailbreak_variant=jb_variant_name,
                                            )
                                        )
                elif unsafe_artifacts:
                    etas = safety_cfg.get("etas", safety_cfg.get("scales", self.safety_etas_default))
                    for eta in etas:
                        for t_start in t_start_list:
                            for t_end in t_end_list:
                                t_start_val = int(t_start) if t_start is not None else None
                                t_end_val = int(t_end) if t_end is not None else None
                                if (t_start_val is not None and t_end_val is not None) and (t_end_val <= t_start_val):
                                    LOGGER.warning(
                                        "Skipping invalid timestep range t_start=%s, t_end=%s for unsafe_artifacts.",
                                        t_start, t_end
                                    )
                                    continue
                                meta_override = {
                                    "safety_enabled": True,
                                    "safety_eta": eta,
                                    "unsafe_artifacts": unsafe_artifacts,
                                    "t_start": t_start,
                                    "t_end": t_end,
                                }
                                suffixes = [
                                    "safety.enabled=true",
                                    f"safety.eta={eta}",
                                    f"safety.unsafe_artifacts={unsafe_artifacts}",
                                ]
                                if t_start is not None:
                                    suffixes.append(f"safety.t_start={t_start_val if t_start_val is not None else t_start}")
                                if t_end is not None:
                                    suffixes.append(f"safety.t_end={t_end_val if t_end_val is not None else t_end}")
                                label = f"unsafe_artifacts-eta{eta}"
                                if t_start is not None or t_end is not None:
                                    label = f"{label}-ts{t_start_val if t_start is not None else 'n'}_{t_end_val if t_end is not None else 'n'}"
                                base_safe_plan = self._make_plan(
                                    spec=spec,
                                    label=label,
                                    variant="safe",
                                    overrides=variant_overrides,
                                    suffix_overrides=suffixes,
                                    metadata_override={**meta_override, **meta_common},
                                    description_suffix=f"{description_base} (unsafe_artifacts, eta={eta})".strip(),
                                    run_base_id=variant_run_base,
                                    jailbreak_variant=jb_variant_name,
                                )
                                plans.append(base_safe_plan)
                                for defense in spec.defense_variants or []:
                                    defense_overrides = list(defense.overrides)
                                    defense_meta = dict(defense.metadata)
                                    defense_label = defense.slug or defense.name
                                    defense_suffix = f"{label}-def-{defense_label}" if defense_label else f"{label}-def"
                                    plans.append(
                                    self._make_plan(
                                        spec=spec,
                                        label=defense_suffix,
                                        variant="safe",
                                        overrides=variant_overrides + defense_overrides,
                                        suffix_overrides=suffixes,
                                        metadata_override={**meta_override, **defense_meta, **meta_common},
                                        description_suffix=f"{description_base} (unsafe_artifacts, eta={eta}, defense={defense_label})".strip(),
                                        run_base_id=variant_run_base,
                                        jailbreak_variant=jb_variant_name,
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
        overrides: List[str],
        suffix_overrides: Iterable[str],
        metadata_override: Optional[Dict[str, Any]],
        description_suffix: str,
        run_base_id: Optional[str] = None,
        jailbreak_variant: Optional[str] = None,
    ) -> JailbreakPlan:
        run_id_base = run_base_id or spec.run_base_id
        run_id = f"{run_id_base}-{label}"
        merged_overrides = list(overrides) + list(suffix_overrides)
        run_dir = _resolve_run_dir(self.output_root, spec.slug, run_id)
        plan_metadata = spec.base_metadata()
        plan_metadata["dataset_json"] = spec.dataset_json
        if metadata_override:
            plan_metadata.update(metadata_override)
        hydra_config = spec.config_name or self.hydra_config
        return JailbreakPlan(
            dataset=spec.name,
            label=label,
            description=description_suffix or spec.description,
            hydra_config=hydra_config,
            overrides=merged_overrides,
            experiment_slug=spec.slug,
            run_id=run_id,
            run_dir=run_dir,
            variant=variant,
            jailbreak_variant=jailbreak_variant,
            metadata=plan_metadata,
        )


def build_jailbreak_plans(
    cfg_path: Path,
    restrict_to: Optional[Sequence[str]],
    timestamp_override: Optional[str] = None,
) -> List[JailbreakPlan]:
    cfg = _load_pipeline_config(cfg_path)
    shared_datasets = _load_dataset_catalog(cfg_path, getattr(cfg, "data_catalog", None))
    builder = JailbreakPlanBuilder(
        cfg=cfg,
        cfg_path=cfg_path,
        shared_datasets=shared_datasets,
        timestamp_override=timestamp_override,
    )
    return builder.build(restrict_to)
