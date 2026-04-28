"""Utilities for constructing multi-dataset generation and scoring plans."""

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


def _resolve_run_dir(base_dir: Path, slug: str, run_id: str) -> Path:
    return (base_dir / slug / run_id).resolve()


def _merge_dicts(primary: Optional[Dict[str, object]], secondary: Optional[Dict[str, object]]) -> Dict[str, object]:
    merged: Dict[str, object] = {}
    if isinstance(primary, dict):
        merged.update(primary)
    if isinstance(secondary, dict):
        merged.update(secondary)
    return merged


def _deep_merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


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


def _stringify(value: object) -> str:
    return str(value)


def _prompt_source_overrides(prompt_source: Dict[str, Any]) -> List[str]:
    overrides: List[str] = []
    name = prompt_source.get("name")
    if name:
        overrides.append(f"++data.prompt_source.name={_stringify(name)}")
    params = prompt_source.get("params")
    if isinstance(params, dict):
        for key, value in params.items():
            if value in (None, "", "null"):
                continue
            overrides.append(f"++data.prompt_source.params.{key}={_stringify(value)}")
    return overrides


def _score_settings_dict(score_cfg: Dict[str, object]) -> Dict[str, object]:
    cfg = dict(score_cfg)
    cfg.pop("config_name", None)
    return {
        "track": cfg.get("track", "safety"),
        "model": cfg.get("model", "mdlm-0p5b"),
        "classifier": cfg.get("classifier", "llamaguard"),
        "classifier_model": cfg.get("classifier_model"),
        "behaviors_csv": cfg.get("behaviors_csv"),
        "indexes_dir": cfg.get("indexes_dir"),
        "batch_size": cfg.get("batch_size", 4),
        "max_new_tokens": cfg.get("max_new_tokens", 32),
        "force": _bool_flag(cfg.get("force", True), True),
        "dry_run": _bool_flag(cfg.get("dry_run", False), False),
        "compute_perplexity": _bool_flag(cfg.get("compute_perplexity", False), False),
        "perplexity_model_name": cfg.get("perplexity_model_name", "gpt2-large"),
        "perplexity_model_path_overwrite": cfg.get("perplexity_model_path_overwrite"),
        "perplexity_model": cfg.get("perplexity_model", "gpt2-large"),
        "perplexity_batch_size": cfg.get("perplexity_batch_size", 8),
        "perplexity_max_length": cfg.get("perplexity_max_length", 1024),
        "compute_hygiene_metrics": _bool_flag(cfg.get("compute_hygiene_metrics", True), True),
        "compute_lexical_metrics": _bool_flag(cfg.get("compute_lexical_metrics", True), True),
        "overlap_ns": cfg.get("overlap_ns", [1, 2, 3, 4]),
        "distinct_ns": cfg.get("distinct_ns", [1, 2, 3, 4]),
        "fuzzy_overlap_ngram": cfg.get("fuzzy_overlap_ngram", 10),
        "fuzzy_max_samples": cfg.get("fuzzy_max_samples", 50),
        "compute_bertscore": _bool_flag(cfg.get("compute_bertscore", False), False),
        "bertscore_model": cfg.get("bertscore_model", "microsoft/deberta-xlarge-mnli"),
        "bertscore_batch_size": cfg.get("bertscore_batch_size", 8),
        "compute_mauve": _bool_flag(cfg.get("compute_mauve", False), False),
        "mauve_model_name": cfg.get("mauve_model_name", "gpt2"),
        "mauve_max_texts": cfg.get("mauve_max_texts", 5000),
        "mauve_max_text_length": cfg.get("mauve_max_text_length", 256),
        "mauve_seed": cfg.get("mauve_seed", 0),
        "compute_refusal_metrics": _bool_flag(cfg.get("compute_refusal_metrics", True), True),
        "refusal_max_chars": cfg.get("refusal_max_chars", 200),
        "refusal_max_tokens": cfg.get("refusal_max_tokens", 40),
        "refusal_content_ratio_threshold": cfg.get("refusal_content_ratio_threshold", 0.2),
        "non_answer_content_ratio_threshold": cfg.get("non_answer_content_ratio_threshold", 0.12),
        "compute_degeneration_metrics": _bool_flag(cfg.get("compute_degeneration_metrics", True), True),
        "degeneration_max_span_threshold": cfg.get("degeneration_max_span_threshold", 50),
        "degeneration_distinct2_threshold": cfg.get("degeneration_distinct2_threshold", 0.10),
        "degeneration_repeat2_threshold": cfg.get("degeneration_repeat2_threshold", 0.30),
        "degeneration_include_early_stop": _bool_flag(cfg.get("degeneration_include_early_stop", True), True),
        "compute_distribution_mmd": _bool_flag(cfg.get("compute_distribution_mmd", True), True),
        "mmd_split_half_trials": cfg.get("mmd_split_half_trials", 5),
        "text_field": cfg.get("text_field", "completion"),
    }


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


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)  # type: ignore[return-value]
    if isinstance(value, dict):
        return copy.deepcopy(value)
    raise SystemExit("Expected a mapping when constructing dataset metadata.")


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


def _parse_prompt_variants(raw: Any, dataset_name: str) -> List[PromptVariant]:
    if raw is None:
        return [PromptVariant(name="", label="", description="", overrides=[], metadata={})]
    if isinstance(raw, (DictConfig, ListConfig)):
        raw = OmegaConf.to_container(raw, resolve=True)
    variants: List[PromptVariant] = []

    def _build_variant(item: Dict[str, Any], default_name: str) -> PromptVariant:
        name = _stringify(item.get("name", default_name))
        label = _stringify(item.get("label", name))
        description = _stringify(item.get("description", ""))
        overrides = _as_list(item.get("overrides", []))
        metadata = _to_plain_dict(item.get("metadata", {})) if isinstance(item, dict) else {}
        return PromptVariant(name=name, label=label, description=description, overrides=overrides, metadata=metadata)

    if isinstance(raw, list):
        for idx, item in enumerate(raw):
            if isinstance(item, dict):
                variants.append(_build_variant(item, default_name=f"{dataset_name}_variant_{idx}"))
            elif isinstance(item, str):
                variants.append(_build_variant({"name": item}, default_name=item))
            else:
                raise SystemExit("prompt_variants entries must be strings or mappings.")
    elif isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                payload = dict(value)
                payload.setdefault("name", key)
                variants.append(_build_variant(payload, default_name=str(key)))
            elif isinstance(value, str):
                variants.append(_build_variant({"name": value}, default_name=str(key)))
            else:
                raise SystemExit("prompt_variants mapping values must be strings or mappings.")
    else:
        raise SystemExit("prompt_variants must be a list or mapping.")
    return variants or [PromptVariant(name="", label="", description="", overrides=[], metadata={})]


def _sweep_list(value: Any, default: Sequence[Optional[object]]) -> List[Optional[object]]:
    """
    Normalize a sweepable config entry into a list of values.

    Rules:
      - None -> copy of default
      - list/tuple -> list(value)
      - otherwise -> [value]
    """
    if value is None:
        return list(default)
    if isinstance(value, ListConfig):
        value = list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


@dataclass
class ScorePlan:
    hydra_config: str
    overrides: List[str]
    output_dir: Path


@dataclass
class PromptVariant:
    name: str
    label: str
    description: str
    overrides: List[str]
    metadata: Dict[str, Any]

    @property
    def slug(self) -> str:
        return self.label or self.name


@dataclass
class GenerationPlan:
    dataset: str
    label: str
    description: str
    hydra_config: str
    overrides: List[str]
    experiment_slug: str
    run_id: str
    run_dir: Path
    variant: str
    prompt_variant: Optional[str]
    metadata: Dict[str, Any]
    score: Optional[ScorePlan]


@dataclass
class DatasetSpec:
    name: str
    description: str
    slug: str
    track_name: str
    run_base_id: str
    config_name: Optional[str]
    dataset_json: str
    prompt_source: Optional[Dict[str, Any]]
    model_family: Optional[str]
    model_variant: Optional[str]
    limit: Optional[int]
    unconditional_samples: Optional[int]
    overrides: List[str]
    slurm_overrides: Dict[str, Any]
    score_cfg: Dict[str, Any]
    safety_cfg: Optional[Dict[str, Any]]
    skip_baseline: bool
    prompt_variants: List[PromptVariant]

    def shared_overrides(self, base_overrides: List[str], output_root: Path) -> List[str]:
        overrides = list(base_overrides) + list(self.overrides)
        overrides.extend(
            [
                f"io.base_dir={output_root}",
                f"io.track_name={self.track_name}",
                f"io.experiment_slug={self.slug}",
            ]
        )
        if self.dataset_json:
            overrides.append(f"data.dataset_json={self.dataset_json}")
        if self.limit is not None:
            overrides.append(f"data.limit={self.limit}")
        if self.unconditional_samples is not None:
            overrides.append(f"gen.unconditional_samples={self.unconditional_samples}")
        if self.prompt_source:
            overrides.extend(_prompt_source_overrides(self.prompt_source))
        if self.model_family:
            overrides.append(f"model.family={self.model_family}")
        if self.model_variant:
            overrides.append(f"model.variant={self.model_variant}")
        return overrides

    def base_metadata(self, score_settings: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dataset_json": self.dataset_json,
            "prompt_limit": self.limit,
            "unconditional_samples": self.unconditional_samples,
            "track_name": self.track_name,
            "score_settings": score_settings,
            "slurm": self.slurm_overrides,
            "safety_artifact_root": None,
            "prompt_source": self.prompt_source,
            "prompt_variant": None,
            "model_family": self.model_family,
            "model_variant": self.model_variant,
        }


class ExperimentPlanBuilder:
    def __init__(
        self,
        cfg: DictConfig,
        disable_scoring: bool,
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
        self.score_config = self.hydra_config
        if "score" in run_cfg:
            score_defaults = _to_plain_dict(run_cfg.score)  # type: ignore[attr-defined]
        else:
            score_defaults = {}
        if "config_name" in score_defaults:
            self.score_config = _stringify(score_defaults.pop("config_name"))
        self.score_defaults = score_defaults
        output_root = getattr(run_cfg, "output_root", None)
        if not output_root:
            raise SystemExit("run.output_root must be specified.")
        self.output_root = Path(str(output_root)).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.slug_prefix = _stringify(getattr(run_cfg, "slug_prefix", "prompt"))
        self.run_prefix = _stringify(getattr(run_cfg, "run_id_prefix", "prompt"))
        base_overrides = getattr(run_cfg, "base_overrides", [])
        self.base_overrides = _as_list(base_overrides)
        self.model_family_default = _stringify(getattr(run_cfg, "model_family", "")) or None
        self.model_variant_default = _stringify(getattr(run_cfg, "model_variant", "")) or None
        gen_cfg = getattr(run_cfg, "gen", None)
        if gen_cfg:
            for key, value in _to_plain_dict(gen_cfg).items():
                if value not in (None, "", "null"):
                    self.base_overrides.append(f"gen.{key}={_stringify(value)}")
        model_checkpoint = getattr(run_cfg, "model_checkpoint", None)
        model_name = getattr(run_cfg, "model_name", None)
        tokenizer_name = getattr(run_cfg, "tokenizer_name", None)
        if model_checkpoint:
            self.base_overrides.append(f"model.checkpoint={model_checkpoint}")
        if model_name:
            self.base_overrides.append(f"model.model_name={model_name}")
        if tokenizer_name:
            self.base_overrides.append(f"model.tokenizer_name={tokenizer_name}")
        if self.model_variant_default:
            self.base_overrides.append(f"model.variant={self.model_variant_default}")
        default_etas = [0, 0.25, 0.5, 1, 2, 4, 8]
        run_safety_etas = getattr(run_cfg, "safety_etas", None)
        legacy_scales = getattr(run_cfg, "safety_scales", None)
        if run_safety_etas is not None:
            self.safety_etas_default = list(run_safety_etas)
        elif legacy_scales is not None:
            self.safety_etas_default = list(legacy_scales)
        else:
            self.safety_etas_default = default_etas
            LOGGER.warning(
                "run.safety_etas is not set; using default etas: %s", default_etas
            )
        crit_default = getattr(run_cfg, "critical_steps", None)
        self.critical_steps_default: List[Optional[object]] = _sweep_list(
            crit_default, default=[None]
        )
        t_start_default = getattr(run_cfg, "t_start", None)
        self.t_start_default: List[Optional[object]] = _sweep_list(
            t_start_default, default=[None]
        )
        t_end_default = getattr(run_cfg, "t_end", None)
        self.t_end_default: List[Optional[object]] = _sweep_list(
            t_end_default, default=[None]
        )
        self.semantic_weight_default = getattr(run_cfg, "semantic_weight", None)
        self.semantic_temp_default = getattr(run_cfg, "semantic_temp", None)
        self.semantic_sigma_default = getattr(run_cfg, "semantic_sigma", None)
        self.track_name_default = _stringify(getattr(run_cfg, "track_name", "safety"))
        self.disable_scoring = disable_scoring
        self.timestamp = timestamp_override or datetime.now().strftime("%Y%m%d%H%M%S")
        # Filtering baselines
        self.posthoc_filter_n = int(getattr(run_cfg, "posthoc_filter_n", 8))
        self.best_of_n_n = int(getattr(run_cfg, "best_of_n_n", 8))
        self.fk_k_particles = int(getattr(run_cfg, "fk_k_particles", 8))
        # Which extra baseline variants to generate (list of strings, e.g. ["posthoc_filter", "best_of_n", "fk_steering"])
        extra_baselines_raw = getattr(run_cfg, "extra_baselines", None)
        if extra_baselines_raw is None:
            self.extra_baselines: List[str] = []
        elif isinstance(extra_baselines_raw, str):
            self.extra_baselines = [extra_baselines_raw]
        else:
            self.extra_baselines = [_stringify(x) for x in extra_baselines_raw]

    def build(self, restrict_to: Optional[Sequence[str]]) -> List[GenerationPlan]:
        datasets_raw = getattr(self._raw_cfg, "datasets", None)
        entries = _dataset_entries(datasets_raw)
        if not entries:
            raise SystemExit("Pipeline config did not declare any datasets.")
        requested = {item.strip() for item in (restrict_to or []) if item.strip()}
        plans: List[GenerationPlan] = []
        for raw_entry in entries:
            resolved_dict = self._resolve_dataset_entry(raw_entry)
            entry_cfg = OmegaConf.create(resolved_dict)
            spec = self._parse_dataset_spec(entry_cfg)
            if requested and spec.name not in requested:
                continue
            plans.extend(self._build_dataset_plans(spec))
        if not plans:
            raise SystemExit("No generation plans were created. Check the dataset filters.")
        LOGGER.info("Scheduled %d generation jobs from %s.", len(plans), self.cfg_path)
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
        prompt_source: Optional[Dict[str, Any]] = None
        if "prompt_source" in entry:
            prompt_source = _to_plain_dict(entry.prompt_source)
        if not dataset_json and not prompt_source:
            raise SystemExit(f"Dataset '{name}' must provide data.dataset_json or data.prompt_source.")
        limit: Optional[int] = None
        if "prompt_limit" in entry:
            limit_val = entry.prompt_limit
            if limit_val not in (None, "", "null"):
                limit = int(limit_val)
        elif "limit" in entry:
            limit_val = entry.limit
            if limit_val not in (None, "", "null"):
                limit = int(limit_val)
        unconditional_samples: Optional[int] = None
        if "unconditional_samples" in entry:
            raw_value = entry.unconditional_samples
            if raw_value not in (None, "", "null"):
                unconditional_samples = int(raw_value)
        config_name: Optional[str] = None
        if "config_name" in entry:
            raw_config_name = entry.config_name
            if raw_config_name not in (None, "", "null"):
                config_name = _stringify(raw_config_name)
        description = _stringify(getattr(entry, "description", ""))
        slug = _stringify(getattr(entry, "experiment_slug", f"{self.slug_prefix}_{name}"))
        run_base_id = _stringify(getattr(entry, "run_id", f"{self.run_prefix}-{name}-{self.timestamp}"))
        track_name = _stringify(getattr(entry, "track_name", self.track_name_default))
        dataset_overrides = _as_list(getattr(entry, "overrides", []))
        slurm_overrides = _to_plain_dict(entry.slurm) if "slurm" in entry else {}
        score_cfg = _merge_dicts(self.score_defaults, _to_plain_dict(entry.score) if "score" in entry else {})
        if self.disable_scoring:
            score_cfg["enabled"] = False
        if "config_name" in score_cfg:
            score_cfg.pop("config_name")
        safety_cfg = _to_plain_dict(entry.safety) if "safety" in entry else None
        skip_baseline = _bool_flag(entry.skip_baseline, False) if "skip_baseline" in entry else False
        prompt_variants_raw = None
        if "prompt_variants" in entry:
            prompt_variants_raw = entry.prompt_variants
        elif "prompt_variant" in entry:
            prompt_variants_raw = [entry.prompt_variant]
        prompt_variants = _parse_prompt_variants(prompt_variants_raw, name)
        model_family = self.model_family_default
        model_variant = self.model_variant_default
        if "model_family" in entry and entry.model_family not in (None, "", "null"):
            model_family = _stringify(entry.model_family)
        if "model_variant" in entry and entry.model_variant not in (None, "", "null"):
            model_variant = _stringify(entry.model_variant)
        return DatasetSpec(
            name=name,
            description=description,
            slug=slug,
            track_name=track_name,
            run_base_id=run_base_id,
            config_name=config_name,
            dataset_json=dataset_json,
            prompt_source=prompt_source,
            model_family=model_family,
            model_variant=model_variant,
            limit=limit,
            unconditional_samples=unconditional_samples,
            overrides=dataset_overrides,
            slurm_overrides=slurm_overrides,
            score_cfg=score_cfg,
            safety_cfg=safety_cfg,
            skip_baseline=skip_baseline,
            prompt_variants=prompt_variants,
        )

    def _build_dataset_plans(self, spec: DatasetSpec) -> List[GenerationPlan]:
        plans: List[GenerationPlan] = []
        score_settings = _score_settings_dict(spec.score_cfg)
        for prompt_variant in spec.prompt_variants:
            shared_overrides = spec.shared_overrides(self.base_overrides, self.output_root)
            variant_overrides = shared_overrides + list(prompt_variant.overrides)
            base_metadata = spec.base_metadata(score_settings)
            prompt_variant_name = prompt_variant.name or prompt_variant.label or None
            if prompt_variant_name:
                token = f"data.prompt_variant={prompt_variant_name}"
                if token not in variant_overrides:
                    variant_overrides.append(token)
            base_metadata["prompt_variant"] = prompt_variant_name
            if prompt_variant.metadata:
                base_metadata["prompt_variant_metadata"] = prompt_variant.metadata
            description_base = spec.description
            if prompt_variant.description or prompt_variant.label:
                desc_suffix = prompt_variant.description or prompt_variant.label
                description_base = f"{spec.description} ({desc_suffix})".strip()
            variant_run_base = spec.run_base_id
            if prompt_variant.slug:
                variant_run_base = f"{spec.run_base_id}-{prompt_variant.slug}"

            if not spec.skip_baseline:
                plans.append(
                    self._make_plan(
                        spec=spec,
                        label="baseline",
                        variant="baseline",
                        shared_overrides=variant_overrides,
                        score_cfg=spec.score_cfg,
                        suffix_overrides=["safety.enabled=false"],
                        metadata_override={"safety_enabled": False},
                        description_suffix=f"{description_base} (baseline)".strip(),
                        base_metadata=base_metadata,
                        run_base_id=variant_run_base,
                        prompt_variant=prompt_variant_name,
                    )
                )

            # --- Extra filtering baselines ---
            model_family = spec.model_family or self.model_family_default or "mdlm"
            for extra_variant in self.extra_baselines:
                extra_variant = extra_variant.lower()
                # FK Steering only makes sense for MDLM
                if extra_variant == "fk_steering" and "mdlm" not in model_family.lower():
                    LOGGER.info(
                        "Skipping fk_steering for non-MDLM family '%s' (dataset=%s).",
                        model_family, spec.name,
                    )
                    continue

                if extra_variant == "posthoc_filter":
                    n = self.posthoc_filter_n
                    extra_suffix_overrides = [
                        "safety.enabled=false",
                        f"model.variant={extra_variant}",
                    ]
                    extra_meta = {
                        "safety_enabled": False,
                        "model_variant": extra_variant,
                        "n_per_prompt": n,
                    }
                    plans.append(
                        self._make_plan(
                            spec=spec,
                            label="posthoc_filter",
                            variant="posthoc_filter",
                            shared_overrides=variant_overrides,
                            score_cfg=spec.score_cfg,
                            suffix_overrides=extra_suffix_overrides,
                            metadata_override=extra_meta,
                            description_suffix=f"{description_base} (posthoc_filter n={n})".strip(),
                            base_metadata=base_metadata,
                            run_base_id=variant_run_base,
                            prompt_variant=prompt_variant_name,
                        )
                    )
                elif extra_variant == "best_of_n":
                    n = self.best_of_n_n
                    extra_suffix_overrides = [
                        "safety.enabled=false",
                        f"model.variant={extra_variant}",
                    ]
                    extra_meta = {
                        "safety_enabled": False,
                        "model_variant": extra_variant,
                        "n_per_prompt": n,
                    }
                    plans.append(
                        self._make_plan(
                            spec=spec,
                            label="best_of_n",
                            variant="best_of_n",
                            shared_overrides=variant_overrides,
                            score_cfg=spec.score_cfg,
                            suffix_overrides=extra_suffix_overrides,
                            metadata_override=extra_meta,
                            description_suffix=f"{description_base} (best_of_n n={n})".strip(),
                            base_metadata=base_metadata,
                            run_base_id=variant_run_base,
                            prompt_variant=prompt_variant_name,
                        )
                    )
                elif extra_variant == "fk_steering":
                    k = self.fk_k_particles
                    extra_suffix_overrides = [
                        "safety.enabled=false",
                        f"model.variant={extra_variant}",
                    ]
                    extra_meta = {
                        "safety_enabled": False,
                        "model_variant": extra_variant,
                        "fk_k_particles": k,
                    }
                    plans.append(
                        self._make_plan(
                            spec=spec,
                            label="fk_steering",
                            variant="fk_steering",
                            shared_overrides=variant_overrides,
                            score_cfg=spec.score_cfg,
                            suffix_overrides=extra_suffix_overrides,
                            metadata_override=extra_meta,
                            description_suffix=f"{description_base} (fk_steering k={k})".strip(),
                            base_metadata=base_metadata,
                            run_base_id=variant_run_base,
                            prompt_variant=prompt_variant_name,
                        )
                    )

            safety_cfg = spec.safety_cfg
            if isinstance(safety_cfg, dict):
                artifact_root = safety_cfg.get("artifact_root")
                artifact_names = safety_cfg.get("artifact_names") or safety_cfg.get("artifacts")
                prototype_root = safety_cfg.get("prototype_root") or safety_cfg.get("prototypes_root")
                prototype_names = safety_cfg.get("prototype_names") or safety_cfg.get("prototypes")
                semantic_ref_root = safety_cfg.get("semantic_ref_root")
                semantic_ref_paths = safety_cfg.get("semantic_ref_paths")
                semantic_use = _bool_flag(safety_cfg.get("use_semantic_gating"), False)
                semantic_cache = _bool_flag(safety_cfg.get("cache_semantic_ref"), False)
                semantic_weight = safety_cfg.get("semantic_weight", self.semantic_weight_default)
                semantic_temp = safety_cfg.get("semantic_temp", self.semantic_temp_default)
                semantic_sigma = safety_cfg.get("semantic_sigma", self.semantic_sigma_default)
                semantic_checkpoint = safety_cfg.get("semantic_checkpoint", None)
                semantic_embed_attr = safety_cfg.get("semantic_embed_attr", None)
                critical_steps_list: List[Optional[object]] = _sweep_list(
                    safety_cfg.get("critical_steps"), self.critical_steps_default
                )
                t_start_list: List[Optional[object]] = _sweep_list(
                    safety_cfg.get("t_start"), self.t_start_default
                )
                t_end_list: List[Optional[object]] = _sweep_list(
                    safety_cfg.get("t_end"), self.t_end_default
                )
                if artifact_root and artifact_names:
                    artifact_root = _stringify(artifact_root)
                    base_metadata["safety_artifact_root"] = artifact_root
                    artifact_names_list = [_stringify(item) for item in artifact_names]
                    semantic_ref_paths_list: List[Optional[str]] = []
                    if semantic_ref_paths:
                        semantic_ref_paths_list = [
                            _stringify(path) for path in _as_list(semantic_ref_paths)
                        ]
                    semantic_ref_root_str = _stringify(semantic_ref_root) if semantic_ref_root else None
                    etas = safety_cfg.get("etas", safety_cfg.get("scales", self.safety_etas_default))
                    for idx, artifact in enumerate(artifact_names_list):
                        for eta in etas:
                            for crit in critical_steps_list:
                                if crit is not None:
                                    crit_suffix = "-cs" + "_".join(str(x) for x in crit) if isinstance(crit, (list, tuple)) else f"-cs{crit}"
                                else:
                                    crit_suffix = ""
                                for t_start in t_start_list:
                                    for t_end in t_end_list:
                                        t_start_val = int(t_start) if t_start is not None else None
                                        t_end_val = int(t_end) if t_end is not None else None
                                        if (t_start_val is not None and t_end_val is not None) and (t_end_val <= t_start_val):
                                            LOGGER.warning(
                                                "Skipping invalid or degenerate timestep range t_start=%s, t_end=%s for artifact %s.",
                                                t_start, t_end, artifact
                                            )
                                            continue
                                        semantic_path = None
                                        if semantic_ref_paths_list and idx < len(semantic_ref_paths_list):
                                            semantic_path = semantic_ref_paths_list[idx]
                                        elif semantic_ref_root_str:
                                            semantic_path = str(Path(semantic_ref_root_str).expanduser() / f"semantic_ref_embeddings_{artifact}.pt")

                                        meta_override = {
                                            "safety_enabled": True,
                                            "artifact_name": artifact,
                                            "safety_eta": eta,
                                            "artifact_root": artifact_root,
                                            "critical_steps": crit,
                                            "t_start": t_start,
                                            "t_end": t_end,
                                            "prompt_variant": prompt_variant_name,
                                        }
                                        if semantic_path:
                                            meta_override.update(
                                                {
                                                    "semantic_ref_path": semantic_path,
                                                    "use_semantic_gating": semantic_use,
                                                    "semantic_weight": semantic_weight,
                                                    "semantic_temp": semantic_temp,
                                                    "semantic_sigma": semantic_sigma,
                                                    "cache_semantic_ref": semantic_cache,
                                                    "semantic_checkpoint": semantic_checkpoint,
                                                    "semantic_embed_attr": semantic_embed_attr,
                                                }
                                            )
                                        suffixes = [
                                            "safety.enabled=true",
                                            f"safety.eta={eta}",
                                            f"safety.unsafe_artifact_root={artifact_root}",
                                            f"safety.unsafe_artifact_name={artifact}",
                                        ]
                                        if semantic_path:
                                            suffixes.append("safety.use_semantic_gating=true" if semantic_use else "safety.use_semantic_gating=false")
                                            if semantic_weight is not None:
                                                suffixes.append(f"safety.semantic_weight={semantic_weight}")
                                            if semantic_temp is not None:
                                                suffixes.append(f"safety.semantic_temp={semantic_temp}")
                                            if semantic_sigma is not None:
                                                suffixes.append(f"safety.semantic_sigma={semantic_sigma}")
                                            suffixes.append(f"safety.cache_semantic_ref={semantic_cache}")
                                            suffixes.append(f"safety.semantic_ref_path={semantic_path}")
                                            if semantic_checkpoint:
                                                suffixes.append(f"safety.semantic_checkpoint={semantic_checkpoint}")
                                            if semantic_embed_attr:
                                                suffixes.append(f"safety.semantic_embed_attr={semantic_embed_attr}")
                                        if crit is not None:
                                            suffixes.append(f"safety.critical_steps={crit}")
                                        if t_start is not None:
                                            suffixes.append(f"safety.t_start={t_start_val if t_start_val is not None else t_start}")
                                        if t_end is not None:
                                            suffixes.append(f"safety.t_end={t_end_val if t_end_val is not None else t_end}")
                                        if prompt_variant_name:
                                            suffixes.append(f"data.prompt_variant={prompt_variant_name}")
                                        t_suffix = ""
                                        if t_start is not None or t_end is not None:
                                            t_suffix = f"-ts{t_start_val if t_start is not None else 'n'}_{t_end_val if t_end_val is not None else 'n'}"
                                        plans.append(
                                            self._make_plan(
                                                spec=spec,
                                                label=f"{artifact}-eta{eta}{crit_suffix}{t_suffix}",
                                                variant="safe",
                                                shared_overrides=variant_overrides,
                                                score_cfg=spec.score_cfg,
                                                suffix_overrides=suffixes,
                                                metadata_override=meta_override,
                                                description_suffix=f"{description_base} (artifact={artifact}, eta={eta}{' cs='+str(crit) if crit is not None else ''}{' t=' + str(t_start) + '-' + str(t_end) if (t_start is not None or t_end is not None) else ''})".strip(),
                                                base_metadata=base_metadata,
                                                run_base_id=variant_run_base,
                                                prompt_variant=prompt_variant_name,
                                            )
                                        )
                elif artifact_root or artifact_names:
                    raise SystemExit(
                        f"Dataset '{spec.name}' safety config must include both artifact_root and artifact_names."
                    )
                if prototype_root and prototype_names:
                    prototype_root = _stringify(prototype_root)
                    base_metadata["prototype_root"] = prototype_root
                    prototype_names_list = [_stringify(item) for item in prototype_names]
                    etas = safety_cfg.get("etas", safety_cfg.get("scales", self.safety_etas_default))
                    for proto in prototype_names_list:
                        for eta in etas:
                            for crit in critical_steps_list:
                                crit_suffix = f"-cs{crit}" if crit is not None else ""
                                proto_path = (Path(prototype_root).expanduser() / proto).expanduser()
                                for t_start in t_start_list:
                                    for t_end in t_end_list:
                                        t_start_val = int(t_start) if t_start is not None else None
                                        t_end_val = int(t_end) if t_end is not None else None
                                        if (t_start_val is not None and t_end_val is not None) and (t_end_val < t_start_val):
                                            LOGGER.warning(
                                                "Skipping invalid timestep range t_start=%s, t_end=%s for prototype %s.",
                                                t_start, t_end, proto
                                            )
                                            continue
                                        meta_override = {
                                            "safety_enabled": True,
                                            "prototype_name": proto,
                                            "safety_eta": eta,
                                            "prototype_root": prototype_root,
                                            "critical_steps": crit,
                                            "t_start": t_start,
                                            "t_end": t_end,
                                            "prototype_path": proto_path,
                                            "prompt_variant": prompt_variant_name,
                                        }
                                        suffixes = [
                                            "safety.enabled=true",
                                            f"safety.eta={eta}",
                                            f"safety.unsafe_prototypes={proto_path}",
                                        ]
                                        if crit is not None:
                                            suffixes.append(f"safety.critical_steps={crit}")
                                        if t_start is not None:
                                            suffixes.append(f"safety.t_start={t_start_val if t_start_val is not None else t_start}")
                                        if t_end is not None:
                                            suffixes.append(f"safety.t_end={t_end_val if t_end_val is not None else t_end}")
                                        if prompt_variant_name:
                                            suffixes.append(f"data.prompt_variant={prompt_variant_name}")
                                        t_suffix = ""
                                        if t_start is not None or t_end is not None:
                                            t_suffix = f"-ts{t_start_val if t_start is not None else 'n'}-{t_end_val if t_end is not None else 'n'}"
                                        plans.append(
                                            self._make_plan(
                                                spec=spec,
                                                label=f"{proto}-proto-eta{eta}{crit_suffix}{t_suffix}",
                                                variant="safe",
                                                shared_overrides=variant_overrides,
                                                score_cfg=spec.score_cfg,
                                                suffix_overrides=suffixes,
                                                metadata_override=meta_override,
                                                description_suffix=f"{description_base} (prototype={proto}, eta={eta}{' cs='+str(crit) if crit is not None else ''}{' t=' + str(t_start) + '-' + str(t_end) if (t_start is not None or t_end is not None) else ''})".strip(),
                                                base_metadata=base_metadata,
                                                run_base_id=variant_run_base,
                                                prompt_variant=prompt_variant_name,
                                            )
                                        )
        return plans

    def _make_plan(
        self,
        spec: DatasetSpec,
        label: str,
        variant: str,
        shared_overrides: List[str],
        score_cfg: Dict[str, object],
        suffix_overrides: Iterable[str],
        metadata_override: Optional[Dict[str, Any]],
        description_suffix: str,
        base_metadata: Dict[str, Any],
        run_base_id: Optional[str] = None,
        prompt_variant: Optional[str] = None,
    ) -> GenerationPlan:
        run_id_base = run_base_id or spec.run_base_id
        run_id = f"{run_id_base}-{label}"
        overrides = list(shared_overrides) + list(suffix_overrides) + [f"io.run_id={run_id}"]
        run_dir = _resolve_run_dir(self.output_root, spec.slug, run_id)
        baseline_run_dir: Optional[Path] = None
        if variant != "baseline":
            suffix = f"-{label}"
            base_id = run_id[: -len(suffix)] if label and run_id.endswith(suffix) else run_id
            baseline_run_id = f"{base_id}-baseline"
            baseline_run_dir = run_dir.parent / baseline_run_id
        score_plan = _build_score_plan(run_dir, label, score_cfg, self.score_config, baseline_run_dir)
        plan_metadata = dict(base_metadata)
        plan_metadata["score_cfg"] = score_cfg
        if metadata_override:
            plan_metadata.update(metadata_override)
        hydra_config = spec.config_name or self.hydra_config
        return GenerationPlan(
            dataset=spec.name,
            label=label,
            description=description_suffix or spec.description,
            hydra_config=hydra_config,
            overrides=overrides,
            experiment_slug=spec.slug,
            run_id=run_id,
            run_dir=run_dir,
            variant=variant,
            prompt_variant=prompt_variant,
            metadata=plan_metadata,
            score=score_plan,
        )


def _build_score_plan(
    run_dir: Path,
    label: str,
    score_cfg: Dict[str, object],
    hydra_config: str,
    baseline_run_dir: Optional[Path] = None,
) -> Optional[ScorePlan]:
    if not _bool_flag(score_cfg.get("enabled", True), True):
        return None
    score_settings = _score_settings_dict(score_cfg)
    score_overrides = [
        f"score.track={score_settings['track']}",
        f"score.model={score_settings['model']}",
        f"score.run_dir={run_dir}",
        f"score.classifier={score_settings['classifier']}",
        f"score.batch_size={score_settings['batch_size']}",
        f"score.max_new_tokens={score_settings['max_new_tokens']}",
        f"score.force={str(score_settings['force']).lower()}",
        f"score.dry_run={str(score_settings['dry_run']).lower()}",
    ]
    if baseline_run_dir is not None:
        score_overrides.append(f"score.baseline_run_dir={baseline_run_dir}")
    if score_settings.get("compute_perplexity"):
        score_overrides.extend(
            [
                "score.compute_perplexity=true",
                f"score.perplexity_model={score_settings['perplexity_model']}",
                f"score.perplexity_batch_size={score_settings['perplexity_batch_size']}",
                f"score.perplexity_max_length={score_settings['perplexity_max_length']}",
                f"score.text_field={score_settings['text_field']}",
            ]
        )
    optional_fields = {
        "score.classifier_model": score_settings.get("classifier_model"),
        "score.behaviors_csv": score_settings.get("behaviors_csv"),
        "score.indexes_dir": score_settings.get("indexes_dir"),
        "score.perplexity_model_name": score_settings.get("perplexity_model_name"),
        "score.perplexity_model_path_overwrite": score_settings.get("perplexity_model_path_overwrite"),
    }
    for key, value in optional_fields.items():
        if value:
            score_overrides.append(f"{key}={value}")
    score_output_dir = run_dir / "scores" / label
    score_overrides.append(f"io.output_dir={score_output_dir}")
    return ScorePlan(hydra_config=hydra_config, overrides=score_overrides, output_dir=score_output_dir)


def build_generation_plans(
    cfg_path: Path,
    restrict_to: Optional[Sequence[str]],
    dry_run: bool,
    disable_scoring: bool,
    timestamp_override: Optional[str] = None,
) -> List[GenerationPlan]:
    cfg = _load_pipeline_config(cfg_path)
    shared_datasets = _load_dataset_catalog(cfg_path, getattr(cfg, "data_catalog", None))
    builder = ExperimentPlanBuilder(
        cfg=cfg,
        disable_scoring=disable_scoring,
        cfg_path=cfg_path,
        shared_datasets=shared_datasets,
        timestamp_override=timestamp_override,
    )
    plans = builder.build(restrict_to)
    if dry_run:
        for plan in plans:
            LOGGER.info("  - %s:%s -> %s", plan.dataset, plan.label, plan.run_dir)
    return plans
