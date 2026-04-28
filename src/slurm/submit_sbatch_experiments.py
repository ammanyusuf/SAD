#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import re
import json
import csv
import statistics

from types import SimpleNamespace

from omegaconf import DictConfig, OmegaConf

from src.utils.experiment_setup import (
    GenerationPlan,
    build_generation_plans,
    _score_settings_dict,
    _resolve_subconfig_path,
)

MODEL_CONFIGS = {
    "mdlm": {
        "env": {
            "CHECKPOINT_PATH": os.path.expanduser("~/scratch/models/text-diffusion/mdlm.ckpt"),
            "TOKENIZER_PATH": os.path.expanduser("~/scratch/hf_models/gpt2-large"),
        },
    },
    "llada": {
        "env": {
            "CHECKPOINT_PATH": os.path.expanduser("~/scratch/hf_models/LLaDA-8B-Base"),
            "TOKENIZER_PATH": os.path.expanduser("~/scratch/hf_models/LLaDA-8B-Base"),
        },
    },
}

def _create_config_snapshot(repo_root: Path, spec_root: Path, timestamp: str) -> Optional[Path]:
    """
    Creates a snapshot of the configs directory to ensure job consistency.
    Returns the path to the snapshot directory (containing the 'configs' folder).
    """
    snapshot_dir = spec_root / f"configs_{timestamp}"
    if snapshot_dir.exists():
        return snapshot_dir

    src_configs = repo_root / "configs"
    if not src_configs.exists():
        print(f"[warning] No configs directory found at {src_configs}; skipping snapshot.")
        return None

    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        dst_configs = snapshot_dir / "configs"
        shutil.copytree(src_configs, dst_configs)
        print(f"[info] Created config snapshot at {dst_configs}")
        return snapshot_dir
    except Exception as e:
        print(f"[warning] Failed to create config snapshot: {e}")
        return None


def _validate_environment(plans: List[GenerationPlan]) -> None:
    """
    Validates that the current environment variables match the expected configuration
    for the models being used in the plans.
    """
    for plan in plans:
        model_family = (plan.metadata.get("model_family") or "mdlm").lower()
        # Fallback for mdlm variations if any
        if "mdlm" in model_family:
            model_key = "mdlm"
        elif "llada" in model_family:
            model_key = "llada"
        else:
            continue

        config = MODEL_CONFIGS.get(model_key)
        if not config:
            continue

        expected_env = config["env"]
        # Check if current environment matches expected values for this model
        for key, expected_value in expected_env.items():
            current_value = os.environ.get(key)
            if current_value:
                 current_value = os.path.expanduser(current_value)

            # We check if the environment variable IS set to the expected value.
            # If the plan uses a specific checkpoint that matches expectation,
            # we must ensure the environment supports it (user intent).
            if current_value != expected_value:
                 print(f"[WARN] Environment mismatch for model '{model_family}':")
                 print(f"  Expected {key}='{expected_value}'")
                 print(f"  Found    {key}='{current_value}'")
                 print(f"  Please source the correct environment for {model_family} before submitting.")
                 # We could raise SystemExit here if strict enforcement is required
                 raise SystemExit(f"Aborting due to environment mismatch for {model_family}.")


@dataclass
class PairedRunInfo:
    baseline_run_dir: Optional[Path] = None
    baseline_run_id: Optional[str] = None
    safe_run_dir: Optional[Path] = None
    safe_run_id: Optional[str] = None


def _env_copy() -> Dict[str, str]:
    return dict(os.environ)


def _bool_to_flag(value: bool) -> str:
    return "1" if value else "0"


def _chunk_list(items: Sequence, size: int) -> Iterable[Sequence]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _count_gen_submissions(
    gen_jobs: Sequence[Tuple[Any, Dict[str, str], str, str, str]],
    batch_size: int,
    gpus_per_node: str,
) -> int:
    if batch_size <= 1:
        return len(gen_jobs)
    grouped: Dict[Tuple[str, str, str, str, str], List[int]] = defaultdict(list)
    for plan, _, array_range, wall_time, mem in gen_jobs:
        variant_mode = "safe" if getattr(plan, "variant", None) != "baseline" else "baseline"
        key = (variant_mode, array_range, wall_time, mem, gpus_per_node)
        grouped[key].append(1)
    total = 0
    for items in grouped.values():
        total += (len(items) + batch_size - 1) // batch_size
    return total


def _count_score_submissions(
    score_jobs: Sequence[Tuple[Any, Dict[str, str], str, str, str]],
    batch_size: int,
    gpus_per_node: str,
) -> int:
    if batch_size <= 1:
        return len(score_jobs)
    grouped: Dict[Tuple[str, str, str, str, Tuple[Tuple[str, Optional[str]], ...]], List[int]] = defaultdict(list)
    for _, env, array_range, wall_time, mem in score_jobs:
        sig = _score_env_signature(env)
        key = (array_range, wall_time, mem, gpus_per_node, sig)
        grouped[key].append(1)
    total = 0
    for items in grouped.values():
        total += (len(items) + batch_size - 1) // batch_size
    return total


def _env_to_spec(env: Dict[str, str]) -> str:
    keys = [
        "GEN_CONFIG_NAME",
        "EXPERIMENT_SLUG",
        "RUN_ID",
        "PROMPT_VARIANT",
        "PROMPT_SOURCE_NAME",
        "TRACK_NAME",
        "RESULTS_ROOT",
        "DATASET_JSON",
        "PROMPT_LIMIT",
        "UNCONDITIONAL_SAMPLES",
        "SAFETY_ENABLED",
        "SAFETY_ETA",
        "SAFETY_SCALE",
        "SAFETY_SEMANTIC_WEIGHT",
        "SAFETY_SEMANTIC_TEMP",
        "SAFETY_SEMANTIC_SIGMA",
        "SAFETY_SEMANTIC_REF_PATH",
        "UNSAFE_ARTIFACT_NAME",
        "UNSAFE_ARTIFACT_ROOT",
        "UNSAFE_ARTIFACTS",
        "UNSAFE_PROTOTYPES",
        "UNSAFE_PROTOTYPE_ROOT",
        "CRITICAL_STEPS",
        "SAFETY_T_START",
        "SAFETY_T_END",
        "SAMPLING_STEPS",
        "MAX_NEW_TOKENS",
        "BLOCK_LENGTH",
        "TEMPERATURE",
        "ADD_BOS",
        "ADD_EOS",
        "DRY_RUN",
        "GEN_BATCH_SIZE",
        "CHECKPOINT_PATH",
        "TOKENIZER_PATH",
        "MODEL_FAMILY",
        "MODEL_NAME",
        "MODEL_VARIANT",
        "MDLM_EMBED_ATTR",
        "USE_SEMANTIC_GATING",
        "GEN_SEED",
        "RUN_SUBDIR",
        "N_PER_PROMPT",
        "FK_K_PARTICLES",
        "LLAMAGUARD_CHECKPOINT_PATH",
        "FK_ROBERTA_CHECKPOINT_PATH",
    ]
    parts: List[str] = []
    for key in keys:
        value = env.get(key)
        if value not in (None, "", "null"):
            parts.append(f"{key}={shlex.quote(str(value))}")
    for key, value in sorted(env.items()):
        if not key.startswith("PROMPT_SOURCE_PARAM_"):
            continue
        if value in (None, "", "null"):
            continue
        parts.append(f"{key}={shlex.quote(str(value))}")
    return " ".join(parts)


def _score_env_to_spec(env: Dict[str, str]) -> str:
    keys = [
        "SCORE_CONFIG_NAME",
        "SCORE_BATCH_SIZE",
        "MAX_NEW_TOKENS",
        "TRACK",
        "MODEL",
        "CLASSIFIER",
        "CLASSIFIER_MODEL",
        "BEHAVIORS_CSV",
        "INDEXES_DIR",
        "SCORE_PPL_MODEL_NAME",
        "SCORE_PPL_MODEL_PATH_OVERWRITE",
        "SCORE_COMPUTE_PERPLEXITY",
        "SCORE_COMPUTE_HYGIENE_METRICS",
        "SCORE_COMPUTE_LEXICAL_METRICS",
        "SCORE_PPL_BATCH_SIZE",
        "SCORE_PPL_MAX_LENGTH",
        "SCORE_OVERLAP_NS",
        "SCORE_DISTINCT_NS",
        "SCORE_FUZZY_OVERLAP_NGRAM",
        "SCORE_FUZZY_MAX_SAMPLES",
        "SCORE_COMPUTE_BERTSCORE",
        "SCORE_BERTSCORE_MODEL",
        "SCORE_BERTSCORE_BATCH_SIZE",
        "SCORE_COMPUTE_MAUVE",
        "SCORE_MAUVE_MODEL_NAME",
        "SCORE_MAUVE_MAX_TEXTS",
        "SCORE_MAUVE_MAX_TEXT_LENGTH",
        "SCORE_MAUVE_SEED",
        "SCORE_COMPUTE_REFUSAL_METRICS",
        "SCORE_REFUSAL_MAX_CHARS",
        "SCORE_REFUSAL_MAX_TOKENS",
        "SCORE_REFUSAL_CONTENT_RATIO_THRESHOLD",
        "SCORE_NON_ANSWER_CONTENT_RATIO_THRESHOLD",
        "SCORE_COMPUTE_DEGENERATION_METRICS",
        "SCORE_DEGEN_MAX_SPAN_THRESHOLD",
        "SCORE_DEGEN_DISTINCT2_THRESHOLD",
        "SCORE_DEGEN_REPEAT2_THRESHOLD",
        "SCORE_DEGEN_INCLUDE_EARLY_STOP",
        "SCORE_COMPUTE_DISTRIBUTION_MMD",
        "SCORE_MMD_SPLIT_HALF_TRIALS",
        "SCORE_CONTINUE_ON_ERROR",
        "RUN_DIR",
        "RESULTS_ROOT",
        "REPO_ROOT",
        "CONFIG_SNAPSHOT_PATH",
        "BASELINE_RUN_DIR",
    ]
    parts: List[str] = []
    for key in keys:
        value = env.get(key)
        if value not in (None, "", "null"):
            parts.append(f"{key}={shlex.quote(str(value))}")
    return " ".join(parts)


def _validate_hydra_config(repo_root: Path, config_name: str, overrides: Sequence[str]) -> None:
    """
    Compose a Hydra config to catch schema/override errors before submission.
    """
    try:
        from hydra import initialize_config_dir, compose
    except ImportError:
        print("[warning] hydra is not installed; skipping hydra validation.")
        return
    config_dir = repo_root / "configs"
    try:
        with initialize_config_dir(config_dir=str(config_dir), job_name="validate", version_base=None):
            compose(config_name=config_name, overrides=list(overrides))
    except Exception as exc:  # pylint: disable=broad-except
        raise SystemExit(f"[error] Hydra validation failed for {config_name} with overrides {overrides}:\n{exc}")


def _get_paired_run_info(
    plan: Any, lookup: Dict[Tuple[str, str, Optional[str], Optional[int]], PairedRunInfo]
) -> Optional[PairedRunInfo]:
    dataset = getattr(plan, "dataset", None)
    slug = getattr(plan, "experiment_slug", None)
    prompt_variant = getattr(plan, "prompt_variant", None)
    seed = getattr(plan, "metadata", {}).get("seed") if hasattr(plan, "metadata") else None
    if dataset is None:
        return None
    return lookup.get((dataset, slug, prompt_variant, seed))


def _run_key(plan: Any) -> Tuple[str, str, Optional[str], Optional[int]]:
    dataset = getattr(plan, "dataset", None)
    slug = getattr(plan, "experiment_slug", None)
    prompt_variant = getattr(plan, "prompt_variant", None)
    seed = getattr(plan, "metadata", {}).get("seed") if hasattr(plan, "metadata") else None
    return (dataset, slug, prompt_variant, seed)


def _job_key(run_id: str, seed: Optional[int]) -> str:
    return f"{run_id}::seed={seed}" if seed is not None else run_id


def _append_seed_override(overrides: List[str], seed: Optional[int]) -> List[str]:
    """
    Ensure gen.seed matches the requested seed. Existing gen.seed overrides are
    replaced rather than left intact so multi-seed submissions vary the RNG.
    """
    if seed is None:
        return list(overrides)
    updated: List[str] = []
    replaced = False
    for token in overrides:
        if token.startswith("gen.seed="):
            updated.append(f"gen.seed={seed}")
            replaced = True
        else:
            updated.append(token)
    if not replaced:
        updated.append(f"gen.seed={seed}")
    return updated


def _update_score_overrides(
    overrides: List[str],
    run_dir: Path,
    baseline_run_dir: Optional[Path],
) -> List[str]:
    updated: List[str] = []
    for token in overrides:
        if token.startswith("score.run_dir="):
            updated.append(f"score.run_dir={run_dir}")
        elif token.startswith("score.baseline_run_dir=") and baseline_run_dir is not None:
            updated.append(f"score.baseline_run_dir={baseline_run_dir}")
        else:
            updated.append(token)
    return updated


def _infer_baseline_pair(plan: Any) -> tuple[str, Path]:
    """
    Derive the baseline run_id and run_dir for a safe plan by stripping the
    plan label suffix and appending '-baseline'.
    """
    label = getattr(plan, "label", "") or ""
    run_id = getattr(plan, "run_id", "")
    base_id = run_id
    suffix = f"-{label}"
    if label and run_id.endswith(suffix):
        base_id = run_id[: -len(suffix)]
    baseline_run_id = f"{base_id}-baseline"
    run_dir: Path = getattr(plan, "run_dir")
    seed_component: Optional[str] = None
    if run_dir.name.startswith("seed="):
        seed_component = run_dir.name
        run_dir = run_dir.parent
    baseline_dir = run_dir.parent / baseline_run_id
    if seed_component:
        baseline_dir = baseline_dir / seed_component
    return baseline_run_id, baseline_dir


def _score_env_signature(env: Dict[str, str]) -> Tuple[Tuple[str, Optional[str]], ...]:
    keys = [
        "TRACK",
        "MODEL",
        "CLASSIFIER",
        "CLASSIFIER_MODEL",
        "BEHAVIORS_CSV",
        "INDEXES_DIR",
        "SCORE_CONFIG_NAME",
        "SCORE_BATCH_SIZE",
        "MAX_NEW_TOKENS",
        "FORCE",
        "DRY_RUN",
        "SCORE_PPL_MODEL_NAME",
        "SCORE_PPL_MODEL_PATH_OVERWRITE",
        "RESULTS_ROOT",
        "BASELINE_RUN_DIR",
    ]
    return tuple((key, env.get(key)) for key in keys)

def _infer_score_only_baseline(run_dir: Path) -> Optional[Path]:
    run_id = run_dir.name
    if run_id.endswith("-baseline"):
        return None
    tokens = run_id.split("-")
    ts_idx = None
    for idx, token in enumerate(tokens):
        if re.fullmatch(r"\d{14}", token):
            ts_idx = idx
            break
    if ts_idx is None:
        return None
    base_tokens = tokens[: ts_idx + 1]
    if ts_idx + 1 < len(tokens):
        base_tokens.append(tokens[ts_idx + 1])
    baseline_id = "-".join(base_tokens) + "-baseline"
    return run_dir.parent / baseline_id


def _determine_slurm_params(
    plan: GenerationPlan,
    mode: str,
    default_array: str,
    default_time: str,
    default_mem: str = "8G",
) -> Tuple[str, str, str]:
    slurm_cfg = plan.metadata.get("slurm")
    array_val = None
    time_val = None
    mem_val = None
    if isinstance(slurm_cfg, dict):
        variant_cfg = slurm_cfg.get(mode, {})
        if isinstance(variant_cfg, dict):
            array_val = variant_cfg.get("array")
            time_val = variant_cfg.get("time")
            mem_val = variant_cfg.get("mem")
        if array_val is None:
            array_val = slurm_cfg.get("array")
        if time_val is None:
            time_val = slurm_cfg.get("time")
        if mem_val is None:
            mem_val = slurm_cfg.get("mem")
    array_range = str(array_val or default_array)
    wall_time = str(time_val or default_time)
    mem = str(mem_val or default_mem)
    return array_range, wall_time, mem


def _build_generation_env(
    plan: GenerationPlan,
    repo_root: Path,
    results_root: Optional[Path],
    safe_artifact_root: Optional[str],
    checkpoint_path: Optional[str],
    tokenizer_path: Optional[str],
    config_snapshot_path: Optional[Path] = None,
    seed: Optional[int] = None,
    run_subdir: Optional[str] = None,
) -> Dict[str, str]:
    env = _env_copy()
    env["REPO_ROOT"] = str(repo_root)
    if config_snapshot_path:
        env["CONFIG_SNAPSHOT_PATH"] = str(config_snapshot_path)
    env["GEN_CONFIG_NAME"] = plan.hydra_config
    env["EXPERIMENT_SLUG"] = plan.experiment_slug
    env["RUN_ID"] = plan.run_id
    if plan.prompt_variant:
        env["PROMPT_VARIANT"] = str(plan.prompt_variant)
    model_family = plan.metadata.get("model_family")
    if model_family:
        env["MODEL_FAMILY"] = str(model_family)
    model_name = plan.metadata.get("model_name")
    if model_name:
        env["MODEL_NAME"] = str(model_name)
    model_variant = plan.metadata.get("model_variant")
    if model_variant:
        env["MODEL_VARIANT"] = str(model_variant)
    model_checkpoint = plan.metadata.get("model_checkpoint")
    if model_checkpoint:
        env["CHECKPOINT_PATH"] = str(model_checkpoint)
    tokenizer_name = plan.metadata.get("tokenizer_name")
    if tokenizer_name:
        env["TOKENIZER_PATH"] = str(tokenizer_name)
    base_dir = results_root or plan.run_dir.parent.parent
    base_dir = _derive_results_root(plan.run_dir, results_root)
    env["RESULTS_ROOT"] = str(base_dir)
    track_name = plan.metadata.get("track_name")
    if track_name:
        env["TRACK_NAME"] = str(track_name)
    dataset_json = plan.metadata.get("dataset_json")
    if dataset_json:
        env["DATASET_JSON"] = str(dataset_json)
    prompt_limit = plan.metadata.get("prompt_limit")
    if prompt_limit not in (None, "", "null"):
        env["PROMPT_LIMIT"] = str(prompt_limit)
    unconditional_samples = plan.metadata.get("unconditional_samples")
    if unconditional_samples not in (None, "", "null"):
        env["UNCONDITIONAL_SAMPLES"] = str(unconditional_samples)
    prompt_source = plan.metadata.get("prompt_source")
    if isinstance(prompt_source, dict):
        name = prompt_source.get("name")
        if name not in (None, "", "null"):
            env.setdefault("PROMPT_SOURCE_NAME", str(name))
        params = prompt_source.get("params")
        if isinstance(params, dict):
            for key, value in params.items():
                if value in (None, "", "null"):
                    continue
                env.setdefault(f"PROMPT_SOURCE_PARAM_{key}", str(value))
    for token in plan.overrides or []:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if value in ("", "null", "None"):
            continue
        cleaned_key = key.lstrip("+")
        # Wire new Hydra overrides through sbatch by mapping them into env vars here and
        # consuming them in generate_array.sh before calling tools.generate.
        if cleaned_key == "data.prompt_source.name":
            env["PROMPT_SOURCE_NAME"] = value
        elif cleaned_key.startswith("data.prompt_source.params."):
            param_key = cleaned_key[len("data.prompt_source.params.") :]
            if param_key:
                env[f"PROMPT_SOURCE_PARAM_{param_key}"] = value
        elif cleaned_key == "data.prompt_variant":
            env["PROMPT_VARIANT"] = value
        elif cleaned_key == "model.family":
            env["MODEL_FAMILY"] = value
        elif cleaned_key == "model.model_name":
            env["MODEL_NAME"] = value
        elif cleaned_key == "model.variant":
            env["MODEL_VARIANT"] = value
        elif cleaned_key == "gen.sampling_steps":
            env["SAMPLING_STEPS"] = value
        elif cleaned_key == "gen.max_new_tokens":
            env["MAX_NEW_TOKENS"] = value
        elif cleaned_key == "gen.block_length":
            env["BLOCK_LENGTH"] = value
        elif cleaned_key == "gen.temperature":
            env["TEMPERATURE"] = value
        elif cleaned_key == "gen.batch_size":
            env["GEN_BATCH_SIZE"] = value
        elif cleaned_key == "safety.use_semantic_gating":
            env["USE_SEMANTIC_GATING"] = value
    # Filtering baselines: posthoc_filter, best_of_n, fk_steering
    if plan.variant in ("posthoc_filter", "best_of_n"):
        env["SAFETY_ENABLED"] = "0"
        env["MODEL_VARIANT"] = plan.variant
        n_per_prompt = plan.metadata.get("n_per_prompt", 8)
        env["N_PER_PROMPT"] = str(n_per_prompt)
        llamaguard_path = os.getenv("LLAMAGUARD_CHECKPOINT_PATH", "")
        if llamaguard_path:
            env["LLAMAGUARD_CHECKPOINT_PATH"] = llamaguard_path
        for key in (
            "SAFETY_ETA", "SAFETY_SCALE", "UNSAFE_ARTIFACT_NAME", "UNSAFE_ARTIFACT_ROOT",
            "UNSAFE_ARTIFACTS", "UNSAFE_PROTOTYPES", "UNSAFE_PROTOTYPE_ROOT",
            "CRITICAL_STEPS", "SAFETY_T_START", "SAFETY_T_END", "SAFETY_SEMANTIC_REF_PATH",
            "SAFETY_SEMANTIC_WEIGHT", "SAFETY_SEMANTIC_TEMP", "SAFETY_SEMANTIC_SIGMA", "MDLM_EMBED_ATTR",
        ):
            env.pop(key, None)
    elif plan.variant == "fk_steering":
        env["SAFETY_ENABLED"] = "0"
        env["MODEL_VARIANT"] = "fk_steering"
        k_particles = plan.metadata.get("fk_k_particles", 8)
        env["FK_K_PARTICLES"] = str(k_particles)
        fk_roberta_path = os.getenv("FK_ROBERTA_CHECKPOINT_PATH", "")
        if fk_roberta_path:
            env["FK_ROBERTA_CHECKPOINT_PATH"] = fk_roberta_path
        for key in (
            "SAFETY_ETA", "SAFETY_SCALE", "UNSAFE_ARTIFACT_NAME", "UNSAFE_ARTIFACT_ROOT",
            "UNSAFE_ARTIFACTS", "UNSAFE_PROTOTYPES", "UNSAFE_PROTOTYPE_ROOT",
            "CRITICAL_STEPS", "SAFETY_T_START", "SAFETY_T_END", "SAFETY_SEMANTIC_REF_PATH",
            "SAFETY_SEMANTIC_WEIGHT", "SAFETY_SEMANTIC_TEMP", "SAFETY_SEMANTIC_SIGMA", "MDLM_EMBED_ATTR",
        ):
            env.pop(key, None)
    elif plan.variant == "safe":
        env["SAFETY_ENABLED"] = "1"
        eta_value = plan.metadata.get("safety_eta", plan.metadata.get("safety_scale", 1.0))
        env["SAFETY_ETA"] = str(eta_value)
        env["SAFETY_SCALE"] = str(plan.metadata.get("safety_scale", ""))
        artifact_name = plan.metadata.get("artifact_name")
        proto_name = plan.metadata.get("prototype_name")
        proto_path = plan.metadata.get("prototype_path")
        if artifact_name:
            env["UNSAFE_ARTIFACT_NAME"] = str(artifact_name)
        artifact_root = (
            plan.metadata.get("artifact_root")
            or plan.metadata.get("safety_artifact_root")
            or safe_artifact_root
        )
        if artifact_name:
            if not artifact_root:
                raise SystemExit(f"Safe run '{plan.dataset}:{plan.label}' is missing a safety artifact root.")
            env["UNSAFE_ARTIFACT_ROOT"] = str(artifact_root)
            if "UNSAFE_SEMANTIC_ROOT" not in env:
                env["UNSAFE_SEMANTIC_ROOT"] = str(Path(artifact_root) / "semantic_refs")
        if proto_name:
            if proto_path:
                env["UNSAFE_PROTOTYPES"] = str(proto_path)
            else:
                # fall back to joining with prototype_root if present
                proto_root = plan.metadata.get("prototype_root") or ""
                env["UNSAFE_PROTOTYPES"] = str(Path(proto_root) / proto_name) if proto_root else str(proto_name)
            if plan.metadata.get("prototype_root"):
                env["UNSAFE_PROTOTYPE_ROOT"] = str(plan.metadata["prototype_root"])
        if plan.metadata.get("critical_steps") is not None:
            env["CRITICAL_STEPS"] = str(plan.metadata["critical_steps"])
        if plan.metadata.get("t_start") is not None:
            env["SAFETY_T_START"] = str(plan.metadata["t_start"])
        if plan.metadata.get("t_end") is not None:
            env["SAFETY_T_END"] = str(plan.metadata["t_end"])
        if plan.metadata.get("semantic_ref_path"):
            env["SAFETY_SEMANTIC_REF_PATH"] = str(plan.metadata["semantic_ref_path"])
        else:
            env.pop("SAFETY_SEMANTIC_REF_PATH", None)
        if plan.metadata.get("semantic_weight") is not None:
            env["SAFETY_SEMANTIC_WEIGHT"] = str(plan.metadata["semantic_weight"])
        else:
            env.pop("SAFETY_SEMANTIC_WEIGHT", None)
        if plan.metadata.get("semantic_temp") is not None:
            env["SAFETY_SEMANTIC_TEMP"] = str(plan.metadata["semantic_temp"])
        else:
            env.pop("SAFETY_SEMANTIC_TEMP", None)
        if plan.metadata.get("semantic_sigma") is not None:
            env["SAFETY_SEMANTIC_SIGMA"] = str(plan.metadata["semantic_sigma"])
        else:
            env.pop("SAFETY_SEMANTIC_SIGMA", None)
        if plan.metadata.get("semantic_embed_attr"):
            env["MDLM_EMBED_ATTR"] = str(plan.metadata["semantic_embed_attr"])
        else:
            env.pop("MDLM_EMBED_ATTR", None)
        if plan.metadata.get("use_semantic_gating") is not None:
            env["USE_SEMANTIC_GATING"] = _bool_to_flag(plan.metadata["use_semantic_gating"])
    else:
        env["SAFETY_ENABLED"] = "0"
        for key in (
            "SAFETY_ETA",
            "SAFETY_SCALE",
            "UNSAFE_ARTIFACT_NAME",
            "UNSAFE_ARTIFACT_ROOT",
            "UNSAFE_ARTIFACTS",
            "UNSAFE_PROTOTYPES",
            "UNSAFE_PROTOTYPE_ROOT",
            "CRITICAL_STEPS",
            "SAFETY_T_START",
            "SAFETY_T_END",
            "SAFETY_SEMANTIC_REF_PATH",
            "SAFETY_SEMANTIC_WEIGHT",
            "SAFETY_SEMANTIC_TEMP",
            "SAFETY_SEMANTIC_SIGMA",
            "MDLM_EMBED_ATTR",
        ):
            env.pop(key, None)
    if checkpoint_path:
        env["CHECKPOINT_PATH"] = str(checkpoint_path)
    if tokenizer_path:
        env["TOKENIZER_PATH"] = str(tokenizer_path)
    if seed is not None:
        env["GEN_SEED"] = str(seed)
    if run_subdir:
        env["RUN_SUBDIR"] = run_subdir
    return env


def _build_score_env(
    plan: GenerationPlan,
    repo_root: Path,
    results_root: Optional[Path],
    config_snapshot_path: Optional[Path] = None,
) -> Optional[Dict[str, str]]:
    if plan.score is None:
        return None
    score_settings = plan.metadata.get("score_settings") or {}
    raw_score_cfg = plan.metadata.get("score_cfg")
    if raw_score_cfg is not None:
        merged_score_cfg: Dict[str, Any] = {}
        base_defaults = _load_base_score_defaults(
            config_snapshot_path,
            repo_root,
            plan.score.hydra_config,
        )
        merged_score_cfg.update(base_defaults)
        merged_score_cfg.update(_to_plain_dict(raw_score_cfg))
        score_settings = _score_settings_dict(merged_score_cfg)
    if not score_settings:
        return None
    env = _env_copy()
    env["REPO_ROOT"] = str(repo_root)
    if config_snapshot_path:
        env["CONFIG_SNAPSHOT_PATH"] = str(config_snapshot_path)
    base_dir = _derive_results_root(plan.run_dir, results_root)
    env["RESULTS_ROOT"] = str(base_dir)
    env["RUN_DIR"] = str(plan.run_dir)
    env["TRACK"] = str(score_settings.get("track", "safety"))
    env["MODEL"] = str(score_settings.get("model", "mdlm-0p5b"))
    env["CLASSIFIER"] = str(score_settings.get("classifier", "llamaguard"))
    classifier_model = score_settings.get("classifier_model")
    if classifier_model:
        env["CLASSIFIER_MODEL"] = str(classifier_model)
    else:
        env.pop("CLASSIFIER_MODEL", None)
    behaviors_csv = score_settings.get("behaviors_csv")
    if behaviors_csv:
        env["BEHAVIORS_CSV"] = str(behaviors_csv)
    indexes_dir = score_settings.get("indexes_dir")
    if indexes_dir:
        env["INDEXES_DIR"] = str(indexes_dir)
    env["SCORE_CONFIG_NAME"] = plan.score.hydra_config
    env["SCORE_BATCH_SIZE"] = str(score_settings.get("batch_size", 16))
    env["SCORE_CONTINUE_ON_ERROR"] = _bool_to_flag(bool(score_settings.get("continue_on_error", False)))
    env["MAX_NEW_TOKENS"] = str(score_settings.get("max_new_tokens", 32))
    env["FORCE"] = _bool_to_flag(bool(score_settings.get("force", True)))
    env["DRY_RUN"] = _bool_to_flag(bool(score_settings.get("dry_run", False)))
    compute_ppl = bool(score_settings.get("compute_perplexity", True))
    env["SCORE_COMPUTE_PERPLEXITY"] = _bool_to_flag(compute_ppl)
    env["SCORE_COMPUTE_HYGIENE_METRICS"] = _bool_to_flag(
        bool(score_settings.get("compute_hygiene_metrics", True))
    )
    env["SCORE_COMPUTE_LEXICAL_METRICS"] = _bool_to_flag(
        bool(score_settings.get("compute_lexical_metrics", True))
    )
    overlap_ns = score_settings.get("overlap_ns")
    distinct_ns = score_settings.get("distinct_ns")
    if overlap_ns:
        env["SCORE_OVERLAP_NS"] = ",".join(str(x) for x in overlap_ns)
    if distinct_ns:
        env["SCORE_DISTINCT_NS"] = ",".join(str(x) for x in distinct_ns)
    env["SCORE_FUZZY_OVERLAP_NGRAM"] = str(score_settings.get("fuzzy_overlap_ngram", 10))
    env["SCORE_FUZZY_MAX_SAMPLES"] = str(score_settings.get("fuzzy_max_samples", 50))
    env["SCORE_COMPUTE_BERTSCORE"] = _bool_to_flag(bool(score_settings.get("compute_bertscore", False)))
    bertscore_model = score_settings.get("bertscore_model")
    if bertscore_model:
        env["SCORE_BERTSCORE_MODEL"] = str(bertscore_model)
    env["SCORE_BERTSCORE_BATCH_SIZE"] = str(score_settings.get("bertscore_batch_size", 8))

    env["SCORE_COMPUTE_MAUVE"] = _bool_to_flag(bool(score_settings.get("compute_mauve", False)))
    env["SCORE_MAUVE_MODEL_NAME"] = str(score_settings.get("mauve_model_name", "gpt2"))
    env["SCORE_MAUVE_MAX_TEXTS"] = str(score_settings.get("mauve_max_texts", 5000))
    env["SCORE_MAUVE_MAX_TEXT_LENGTH"] = str(score_settings.get("mauve_max_text_length", 256))
    env["SCORE_MAUVE_SEED"] = str(score_settings.get("mauve_seed", 0))

    env["SCORE_COMPUTE_REFUSAL_METRICS"] = _bool_to_flag(
        bool(score_settings.get("compute_refusal_metrics", True))
    )
    env["SCORE_REFUSAL_MAX_CHARS"] = str(score_settings.get("refusal_max_chars", 200))
    env["SCORE_REFUSAL_MAX_TOKENS"] = str(score_settings.get("refusal_max_tokens", 40))
    env["SCORE_REFUSAL_CONTENT_RATIO_THRESHOLD"] = str(
        score_settings.get("refusal_content_ratio_threshold", 0.2)
    )
    env["SCORE_NON_ANSWER_CONTENT_RATIO_THRESHOLD"] = str(
        score_settings.get("non_answer_content_ratio_threshold", 0.12)
    )

    env["SCORE_COMPUTE_DEGENERATION_METRICS"] = _bool_to_flag(
        bool(score_settings.get("compute_degeneration_metrics", True))
    )
    env["SCORE_DEGEN_MAX_SPAN_THRESHOLD"] = str(score_settings.get("degeneration_max_span_threshold", 50))
    env["SCORE_DEGEN_DISTINCT2_THRESHOLD"] = str(
        score_settings.get("degeneration_distinct2_threshold", 0.10)
    )
    env["SCORE_DEGEN_REPEAT2_THRESHOLD"] = str(
        score_settings.get("degeneration_repeat2_threshold", 0.30)
    )
    env["SCORE_DEGEN_INCLUDE_EARLY_STOP"] = _bool_to_flag(
        bool(score_settings.get("degeneration_include_early_stop", True))
    )
    env["SCORE_COMPUTE_DISTRIBUTION_MMD"] = _bool_to_flag(
        bool(score_settings.get("compute_distribution_mmd", True))
    )
    env["SCORE_MMD_SPLIT_HALF_TRIALS"] = str(score_settings.get("mmd_split_half_trials", 5))
    if compute_ppl:
        env["SCORE_PPL_MODEL_NAME"] = str(score_settings.get("perplexity_model_name", "gpt2-large"))
        env["SCORE_PPL_MODEL_PATH_OVERWRITE"] = str(
            score_settings.get("perplexity_model_path_overwrite") or ""
        )
        env["SCORE_PPL_MODEL"] = str(score_settings.get("perplexity_model", "gpt2-large"))
        env["SCORE_PPL_BATCH_SIZE"] = str(score_settings.get("perplexity_batch_size", 8))
        env["SCORE_PPL_MAX_LENGTH"] = str(score_settings.get("perplexity_max_length", 1024))
    overrides = getattr(plan.score, "overrides", None)
    if overrides:
        for token in overrides:
            if not token.startswith("score.baseline_run_dir="):
                continue
            _, value = token.split("=", 1)
            if value:
                env["BASELINE_RUN_DIR"] = value
            break
    return env


def _submit_sbatch(
    cmd: Sequence[str],
    env: Dict[str, str],
    dry_run: bool,
    integration_test: bool = False,
) -> str:
    printable = shlex.join(cmd)
    print(f"[sbatch] {printable}")
    if dry_run:
        return "dry-run"
    if integration_test:
        script_idx = next((i for i, part in enumerate(cmd) if part.endswith(".sh")), None)
        if script_idx is None:
            raise RuntimeError(f"Could not locate script path in command: {printable}")
        script_and_args = list(cmd[script_idx:])
        print(f"[integration-test] running script directly: {shlex.join(script_and_args)}")
        local_cmd = ["bash"] + script_and_args
        local_env = dict(env)
        local_env.setdefault("SLURM_JOB_ID", "local")
        local_env.setdefault("SLURM_ARRAY_JOB_ID", local_env["SLURM_JOB_ID"])
        local_env.setdefault("SLURM_ARRAY_TASK_ID", "0")
        local_env.setdefault("SLURM_ARRAY_TASK_MIN", local_env["SLURM_ARRAY_TASK_ID"])
        local_env.setdefault("SLURM_ARRAY_TASK_MAX", local_env["SLURM_ARRAY_TASK_ID"])
        local_env.setdefault("SLURM_ARRAY_TASK_STEP", "1")
        if not local_env.get("SLURM_TMPDIR"):
            tmpdir = Path.cwd() / ".slurm_tmp_local"
            tmpdir.mkdir(parents=True, exist_ok=True)
            local_env["SLURM_TMPDIR"] = str(tmpdir)
        print(f"[integration-test] running locally: {shlex.join(local_cmd)}")
        print(f"[integration-test] with env vars: {_env_to_spec(local_env)}")
        print("[integration-test] starting subprocess...")
        completed = subprocess.run(local_cmd, env=local_env, check=False)
        print(f"[integration-test] subprocess finished with returncode={completed.returncode}")
        if completed.returncode != 0:
            raise SystemExit(f"Integration test failed (exit={completed.returncode}) for {local_cmd}")
        return "local-run"
    time.sleep(2)
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        job_id = completed.stdout.strip().split("\n")[-1]
        print(f"  -> job {job_id}")
        return job_id
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] sbatch failed (returncode={exc.returncode})")
        if exc.stdout:
            print(f"[stdout] {exc.stdout.strip()}")
        if exc.stderr:
            print(f"[stderr] {exc.stderr.strip()}")
        raise


def _extract_timestamp(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"(\d{14})", text)
    if match:
        return match.group(1)
    return None


def _to_plain_dict(payload: Optional[object]) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, DictConfig):
        return OmegaConf.to_container(payload, resolve=True)  # type: ignore[return-value]
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def _load_base_score_defaults(
    snapshot_path: Optional[Path],
    repo_root: Path,
    config_name: str,
) -> Dict[str, Any]:
    candidates = []
    if snapshot_path:
        candidates.append(snapshot_path / "configs" / f"{config_name}.yaml")
    candidates.append(repo_root / "configs" / f"{config_name}.yaml")
    for path in candidates:
        if path.exists():
            cfg = OmegaConf.load(path)
            return _to_plain_dict(getattr(cfg, "score", None))
    return {}


def _derive_results_root(run_dir: Path, explicit_root: Optional[Path]) -> Path:
    """
    Determine RESULTS_ROOT for staging. Handles seed subdirectories without
    altering the pre-existing layout.
    """
    if explicit_root:
        return explicit_root
    # Expected structures:
    #   .../<results_root>/<slug>/<run_id>
    #   .../<results_root>/<slug>/<run_id>/seed=<n>
    parts_up = 3 if run_dir.name.startswith("seed=") else 2
    try:
        return run_dir.parents[parts_up - 1]
    except IndexError:
        return run_dir.parent



def _is_metric_json(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".json") and any(
        token in name for token in ("summary", "metrics", "mauve", "bertscore", "refusal", "degeneration", "hygiene")
    )


def _is_metric_csv(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".csv") and "summary" in name


def _flatten_numeric(obj: Any, prefix: str = "") -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            metrics.update(_flatten_numeric(value, next_prefix))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]"
            metrics.update(_flatten_numeric(value, next_prefix))
    else:
        if isinstance(obj, bool):
            return metrics
        if isinstance(obj, (int, float)) and prefix:
            metrics[prefix] = float(obj)
    return metrics


def _collect_json_metrics(path: Path, base_key: str) -> Dict[str, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    flat = _flatten_numeric(payload)
    return {f"{base_key}:{k}": v for k, v in flat.items()}


def _collect_csv_metrics(path: Path, base_key: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except Exception:
        return metrics
    if not rows or not reader.fieldnames:
        return metrics
    id_field = reader.fieldnames[0]
    for idx, row in enumerate(rows):
        row_label = row.get(id_field) or str(idx)
        for key, value in row.items():
            if key == id_field:
                continue
            try:
                num = float(value)
            except (TypeError, ValueError):
                continue
            metrics[f"{base_key}:{row_label}:{key}"] = num
    return metrics


def _discover_seed_dirs(run_dir: Path, seeds: Optional[Sequence[int]]) -> Tuple[List[Tuple[int, Path]], List[int]]:
    discovered: List[Tuple[int, Path]] = []
    missing: List[int] = []
    if seeds:
        for seed in seeds:
            seed_dir = run_dir / f"seed={seed}"
            if seed_dir.exists():
                discovered.append((seed, seed_dir))
            else:
                missing.append(seed)
        if not discovered and len(seeds) == 1 and run_dir.exists():
            discovered.append((seeds[0], run_dir))
    else:
        for child in sorted(run_dir.glob("seed=*")):
            if child.is_dir():
                seed_token = child.name.split("=", 1)[-1]
                try:
                    seed_val = int(seed_token)
                except ValueError:
                    continue
                discovered.append((seed_val, child))
        if not discovered and run_dir.exists():
            discovered.append((None, run_dir))
    return discovered, missing


def aggregate_seed_runs(run_dir: Path, seeds: Optional[Sequence[int]] = None) -> Path:
    seed_dirs, missing = _discover_seed_dirs(run_dir, seeds)
    if not seed_dirs:
        raise SystemExit(f"No seed directories found under {run_dir}")
    if missing:
        print(f"[warning] Missing seeds: {missing}")
    metrics_by_key: Dict[str, List[float]] = defaultdict(list)
    for seed, seed_dir in seed_dirs:
        rel_prefix = f"seed={seed}" if seed is not None else seed_dir.name
        for path in seed_dir.rglob("*"):
            if path.is_dir():
                continue
            rel_path = f"{rel_prefix}/{path.relative_to(seed_dir)}"
            if _is_metric_json(path):
                seed_metrics = _collect_json_metrics(path, rel_path)
            elif _is_metric_csv(path):
                seed_metrics = _collect_csv_metrics(path, rel_path)
            else:
                continue
            for key, value in seed_metrics.items():
                metrics_by_key[key].append(value)
    summary_dir = run_dir / "aggregate"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, Union[float, int]]] = {}
    for key, values in metrics_by_key.items():
        if not values:
            continue
        mean_val = statistics.fmean(values)
        std_val = statistics.pstdev(values) if len(values) > 1 else 0.0
        summary[key] = {"mean": mean_val, "std": std_val, "count": len(values)}
    summary_json = summary_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_csv = summary_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std", "count"])
        for metric, stats in sorted(summary.items()):
            writer.writerow([metric, stats["mean"], stats["std"], stats["count"]])
    summary_md = summary_dir / "summary.md"
    with summary_md.open("w", encoding="utf-8") as handle:
        handle.write("| metric | mean | std | count |\n")
        handle.write("| --- | --- | --- | --- |\n")
        for metric, stats in sorted(summary.items()):
            handle.write(f"| {metric} | {stats['mean']:.6g} | {stats['std']:.6g} | {stats['count']} |\n")
    print(f"[aggregate] Wrote aggregate summaries to {summary_dir}")
    return summary_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit prompt elicitation sweeps to Slurm.")
    parser.add_argument("--config", type=Path, required=True, help="Pipeline config (YAML/JSON).")
    parser.add_argument("--only", nargs="*", default=None, help="Subset of datasets to submit.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--gen-script", type=Path, default=Path("src/slurm/generate_array.sh"))
    parser.add_argument("--score-script", type=Path, default=Path("src/slurm/score_array.sh"))
    parser.add_argument("--hazard-script", type=Path, default=Path("src/slurm/hazard_report.sh"))
    parser.add_argument("--account", default=None, help="Slurm account to charge (e.g., rrg-<your-PI>_gpu).")
    parser.add_argument("--baseline-array", default="0-3")
    parser.add_argument("--baseline-time", default="0-00:20")
    parser.add_argument("--safe-array", default="0-3")
    parser.add_argument("--safe-time", default="0-00:40")
    parser.add_argument("--gpus-per-node", default="a100:1")
    parser.add_argument("--score-gpus-per-node", default="a100:1")
    parser.add_argument("--score-array", default="0-0")
    parser.add_argument("--score-time", default="0-01:00")
    parser.add_argument("--hazard-time", default="0-01:00")
    parser.add_argument("--hazard-examples-per-hazard", type=int, default=5)
    parser.add_argument(
        "--hazard-examples-per-metric",
        type=int,
        default=5,
        help="Examples to keep for top/bottom per-metric slices in hazard report.",
    )
    parser.add_argument(
        "--hazard-examples-per-transition",
        type=int,
        default=5,
        help="Examples to keep for safety transition slices (baseline vs safe).",
    )
    parser.add_argument(
        "--hazard-after-score",
        dest="hazard_after_score",
        action="store_true",
        help="Submit a combined hazard report after scoring completes (default).",
    )
    parser.add_argument(
        "--no-hazard-after-score",
        dest="hazard_after_score",
        action="store_false",
        help="Skip hazard report submission after scoring.",
    )
    parser.set_defaults(hazard_after_score=True)
    parser.add_argument("--gen-batch-size", type=int, default=1, help="Configs per generation sbatch.")
    parser.add_argument("--score-batch-size", type=int, default=1, help="Runs per scoring sbatch.")
    parser.add_argument(
        "--score-continue-on-error",
        action="store_true",
        help="Continue scoring other runs in a batch even if one run fails.",
    )
    parser.add_argument("--safe-artifact-root", default=None)
    parser.add_argument("--checkpoint-path", default=None, help="Absolute path to model checkpoint to stage.")
    parser.add_argument("--tokenizer-path", default=None, help="Absolute path to tokenizer to stage.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Override single RNG seed for generation.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="List of RNG seeds for repeated runs, e.g., --seeds 1 2 3.",
    )
    parser.add_argument(
        "--validate-hydra",
        action="store_true",
        help="Compose Hydra configs locally (first few jobs) to catch errors before submission.",
    )
    parser.add_argument(
        "--trillium",
        action="store_true",
        help="Target Trillium cluster: skip --mem in sbatch (enforced by cluster) and allow explicit --nodes.",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Number of nodes to request in sbatch submissions (useful for Trillium).",
    )
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument(
        "--score-only-dir",
        nargs="+",
        type=Path,
        default=None,
        help="Rescore existing runs under these directories (skip generation submission).",
    )
    parser.add_argument(
        "--hazard-only-dir",
        nargs="+",
        type=Path,
        default=None,
        help="Rebuild hazard report over existing run directories (skip generation/scoring submission).",
    )
    parser.add_argument(
        "--integration-test",
        action="store_true",
        help="Run scripts directly (no sbatch) for interactive testing.",
    )
    parser.add_argument(
        "--safe-debug",
        action="store_true",
        help="Enable safe debugging output.",
    )
    parser.add_argument(
        "--refresh-configs",
        action="store_true",
        help="Delete previous config snapshots before creating a new one.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Run aggregation over completed seed runs and exit (no submissions).",
    )
    parser.add_argument(
        "--aggregate-run-dir",
        type=Path,
        default=None,
        help="Parent run directory containing seed=*/ outputs for aggregation.",
    )
    parser.add_argument(
        "--auto-aggregate",
        action="store_true",
        help="After score jobs, submit a dependent job to aggregate seed metrics per run.",
    )
    args = parser.parse_args()

    if args.safe_debug:
        os.environ["SAFE_REPELLENCY_DEBUG"] = "1"

    if args.aggregate_only:
        if args.aggregate_run_dir is None:
            raise SystemExit("--aggregate-only requires --aggregate-run-dir")
        seeds_arg: Optional[List[int]] = None
        if args.seeds:
            seeds_arg = list(dict.fromkeys(args.seeds))
        elif args.seed is not None:
            seeds_arg = [args.seed]
        aggregate_seed_runs(args.aggregate_run_dir.resolve(), seeds_arg)
        return

    print("[DEBUG] Starting main...")
    repo_root = args.repo_root.resolve()
    gen_script_path = (args.gen_script if args.gen_script.is_absolute() else (repo_root / args.gen_script)).resolve()
    score_script_path = (args.score_script if args.score_script.is_absolute() else (repo_root / args.score_script)).resolve()
    hazard_script_path = (args.hazard_script if args.hazard_script.is_absolute() else (repo_root / args.hazard_script)).resolve()
    results_root = args.results_root.resolve() if args.results_root else None
    safe_artifact_root = (
        str(Path(args.safe_artifact_root).resolve()) if args.safe_artifact_root else None
    )
    seeds_arg: Optional[List[int]] = None
    if args.seeds:
        seeds_arg = list(dict.fromkeys(args.seeds))
    elif args.seed is not None:
        seeds_arg = [args.seed]
    multi_seed_mode = bool(seeds_arg) and len(seeds_arg) > 1
    spec_root = (results_root or repo_root) / ".slurm_specs"
    score_only_dirs = [p.resolve() for p in args.score_only_dir] if args.score_only_dir else []
    hazard_only_dirs_raw = [p.resolve() for p in args.hazard_only_dir] if args.hazard_only_dir else []
    hazard_only_dirs: List[Path] = []
    for p in hazard_only_dirs_raw:
        if p.exists() and p.is_dir():
            hazard_only_dirs.append(p)
            continue
        pattern = p.name
        parent = p.parent if p.parent != Path("") else Path(".")
        for cand in parent.glob(f"{pattern}*"):
            if cand.is_dir():
                hazard_only_dirs.append(cand.resolve())
    hazard_only_dirs = sorted({p for p in hazard_only_dirs})
    cfg = OmegaConf.load(args.config)
    catalog_value = getattr(cfg, "data_catalog", None)
    resolved_catalog = None
    if catalog_value not in (None, "", "null"):
        try:
            resolved_catalog = _resolve_subconfig_path(args.config, str(catalog_value))
        except Exception:
            resolved_catalog = None

    # Create config snapshot for this submission batch immediately to avoid race conditions
    submission_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if args.refresh_configs and spec_root.exists():
        for candidate in spec_root.glob("configs_*"):
            if candidate.is_dir():
                shutil.rmtree(candidate, ignore_errors=True)
    snapshot_path = _create_config_snapshot(repo_root, spec_root, submission_timestamp)
    if snapshot_path:
        print(f"[INFO] Using config snapshot at {snapshot_path}")

    plans: List[GenerationPlan] = []
    paired_run_info: Dict[Tuple[str, str, Optional[str], Optional[int]], PairedRunInfo] = {}
    if not score_only_dirs and not hazard_only_dirs:
        plans = build_generation_plans(
            cfg_path=args.config,
            restrict_to=args.only,
            dry_run=False,
            disable_scoring=args.skip_scoring,
            timestamp_override=submission_timestamp,
        )
        print(f"[DEBUG] Built {len(plans)} plans. Validating environment...")
        _validate_environment(plans)
        print("[DEBUG] Environment validation passed.")

        if seeds_arg:
            expanded: List[GenerationPlan] = []
            for plan in plans:
                for seed_val in seeds_arg:
                    use_seed_subdir = len(seeds_arg) > 1
                    seed_run_dir = plan.run_dir / f"seed={seed_val}" if use_seed_subdir else plan.run_dir
                    metadata = dict(plan.metadata)
                    metadata["seed"] = seed_val
                    seed_overrides = _append_seed_override(plan.overrides, seed_val)
                    score_plan = plan.score
                    baseline_dir_override: Optional[Path] = None
                    if plan.variant == "safe":
                        _, baseline_base = _infer_baseline_pair(plan)
                        baseline_dir_override = baseline_base / f"seed={seed_val}" if use_seed_subdir else baseline_base
                    if plan.score:
                        score_overrides = _update_score_overrides(
                            plan.score.overrides,
                            seed_run_dir,
                            baseline_dir_override,
                        )
                        score_plan = replace(
                            plan.score,
                            overrides=score_overrides,
                            output_dir=seed_run_dir / "scores" / plan.label,
                        )
                    expanded.append(
                        replace(
                            plan,
                            overrides=seed_overrides,
                            run_dir=seed_run_dir,
                            metadata=metadata,
                            score=score_plan,
                        )
                    )
            plans = expanded

        for plan in plans:
            run_key = _run_key(plan)
            info = paired_run_info.setdefault(run_key, PairedRunInfo())
            if plan.variant == "baseline":
                info.baseline_run_dir = plan.run_dir
                info.baseline_run_id = plan.run_id
            elif plan.variant == "safe":
                info.safe_run_dir = plan.run_dir
                info.safe_run_id = plan.run_id

    all_score_jobs: list[str] = []
    all_run_dirs: list[str] = []
    all_run_ids: list[str] = []
    hazard_env: Optional[Dict[str, str]] = None
    gen_jobs: List[Tuple[GenerationPlan, Dict[str, str], str, str, str]] = []
    score_jobs: List[Tuple[GenerationPlan, Dict[str, str], str, str, str]] = []

    if score_only_dirs:
        # prepare score-only jobs
        score_cfg = cfg.run.get("score") if hasattr(cfg, "run") else None
        if score_cfg is None:
            raise SystemExit("Config is missing run.score settings required for score-only mode.")
        hydra_score_config = score_cfg.get("config_name", cfg.run.get("config_name", "config"))
        base_score_cfg = _load_base_score_defaults(snapshot_path, repo_root, hydra_score_config)
        merged_score_cfg: Dict[str, Any] = {}
        merged_score_cfg.update(base_score_cfg)
        merged_score_cfg.update(_to_plain_dict(score_cfg))
        score_settings = _score_settings_dict(merged_score_cfg)

        for run_dir in score_only_dirs:
            if not run_dir.exists():
                print(f"[warning] skipping score-only for {run_dir} (directory does not exist)")
                continue
            pseudo_plan = SimpleNamespace(
                run_id=run_dir.name,
                run_dir=run_dir,
                score=SimpleNamespace(hydra_config=hydra_score_config),
                metadata={"score_settings": score_settings},
            )
            env = _build_score_env(pseudo_plan, repo_root, results_root, snapshot_path)
            if env is None:
                print(f"[warning] skipping score-only for {run_dir} (missing score settings)")
                continue
            if args.score_continue_on_error:
                env["SCORE_CONTINUE_ON_ERROR"] = "1"
            if "BASELINE_RUN_DIR" not in env:
                baseline_guess = _infer_score_only_baseline(run_dir)
                if baseline_guess is not None:
                    env["BASELINE_RUN_DIR"] = str(baseline_guess)
            score_jobs.append(
                (
                    pseudo_plan,
                    env,
                    args.score_array,
                    args.score_time,
                    "16G",
                )
            )
            all_run_dirs.append(str(run_dir))
            all_run_ids.append(run_dir.name)
    elif hazard_only_dirs:
        for run_dir in hazard_only_dirs:
            if not run_dir.exists():
                print(f"[warning] skipping hazard-only for {run_dir} (directory does not exist)")
                continue
            all_run_dirs.append(str(run_dir))
            all_run_ids.append(run_dir.name)
    else:
        # prepare generation and scoring jobs
        for plan in plans:
            prompt_variant = getattr(plan, "prompt_variant", None)
            prompt_tag = f" prompt_variant={prompt_variant}" if prompt_variant else ""
            seed_val = plan.metadata.get("seed") if hasattr(plan, "metadata") else None
            seed_tag = f" seed={seed_val}" if seed_val is not None else ""
            print(f"\n[dataset={plan.dataset} label={plan.label} variant={plan.variant}{prompt_tag}{seed_tag}]")
            is_safe_variant = plan.variant != "baseline"
            array_default = args.safe_array if is_safe_variant else args.baseline_array
            time_default = args.safe_time if is_safe_variant else args.baseline_time
            mode = "safe" if is_safe_variant else "baseline"
            array_range, wall_time, mem = _determine_slurm_params(plan, mode, array_default, time_default)
            gen_env = _build_generation_env(
                plan=plan,
                repo_root=repo_root,
                results_root=results_root,
                safe_artifact_root=safe_artifact_root,
                checkpoint_path=args.checkpoint_path,
                tokenizer_path=args.tokenizer_path,
                config_snapshot_path=snapshot_path,
                seed=seed_val,
                run_subdir=(f"seed={seed_val}" if multi_seed_mode else None),
            )

            # Skip MDLM runs if temperature is set > 0.0 (as it doesn't use it)
            model_family_chk = gen_env.get("MODEL_FAMILY", "").lower()
            temperature_chk = gen_env.get("TEMPERATURE")
            if "mdlm" in model_family_chk and temperature_chk is not None:
                try:
                    t_val = float(temperature_chk)
                except ValueError:
                    t_val = 0.0
                if t_val > 0.0:
                    print(f"  [skip] Skipping MDLM run for {plan.run_id} because temperature={t_val} is set (MDLM does not support sampling temperature).")
                    continue

            gen_config_name = gen_env.get("GEN_CONFIG_NAME")
            if gen_config_name:
                print(f"  [generate] hydra config = {gen_config_name}")
            prompt_meta = plan.metadata.get("prompt_source") or {}
            if isinstance(prompt_meta, dict):
                prompt_name = prompt_meta.get("name")
                params = prompt_meta.get("params")
                data_dir = params.get("data_dir") if isinstance(params, dict) else None
                if prompt_name:
                    if data_dir:
                        print(f"  [generate] prompt_source = {prompt_name} (data_dir={data_dir})")
                    else:
                        print(f"  [generate] prompt_source = {prompt_name}")

            gen_jobs.append((plan, gen_env, array_range, wall_time, mem))

            score_env = _build_score_env(plan, repo_root, results_root, snapshot_path)
            if score_env:
                if args.score_continue_on_error:
                    score_env["SCORE_CONTINUE_ON_ERROR"] = "1"
                plan_variant = getattr(plan, "variant", None)
                if plan_variant == "safe":
                    pair_info = _get_paired_run_info(plan, paired_run_info)
                    baseline_dir = pair_info.baseline_run_dir if pair_info else None
                    if not baseline_dir:
                        _, baseline_dir = _infer_baseline_pair(plan)
                    if baseline_dir:
                        score_env["BASELINE_RUN_DIR"] = str(baseline_dir)
                    print(f"  [score] baseline run dir = {score_env.get('BASELINE_RUN_DIR', 'N/A')}")
                score_array, score_time, score_mem = _determine_slurm_params(
                    plan,
                    mode="score",
                    default_array=args.score_array,
                    default_time=args.score_time,
                )
                score_jobs.append((plan, score_env, score_array, score_time, score_mem))
                all_run_dirs.append(str(plan.run_dir))
                seed_val = plan.metadata.get("seed") if hasattr(plan, "metadata") else None
                run_id_with_seed = f"{plan.run_id} (seed={seed_val})" if seed_val is not None else plan.run_id
                all_run_ids.append(run_id_with_seed)

    print("\n[summary] planned submissions:")
    env_echo_keys = [
        "REPO_ROOT",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "HF_MODELS_CACHE",
        "CHECKPOINT_PATH",
        "TOKENIZER_PATH",
        "SKIP_PIP_UPGRADE",
        "PIP_INSTALL_ARGS",
        "EXTERNAL_VENV_ACTIVATE",
        "PYTHONPATH",
        "MODEL_CONFIG_PATH",
    ]
    print("  environment (effective at submission):")
    for key in env_echo_keys:
        val = os.environ.get(key)
        print(f"    {key}={val if val is not None else '<unset>'}")
    if gen_jobs:
        print("  generation env preview (up to 3 jobs):")
        for plan, gen_env, _, _, _ in gen_jobs[:3]:
            print(f"    {plan.run_id}: {_env_to_spec(gen_env)}")
    if score_jobs and not args.skip_scoring:
        print("  scoring env preview (up to 3 jobs):")
        for plan, score_env, _, _, _ in score_jobs[:3]:
            print(f"    {plan.run_id}: {_score_env_to_spec(score_env)}")
    if catalog_value not in (None, "", "null"):
        print(f"  data_catalog: {catalog_value} -> {resolved_catalog}")
    print(f"  generation jobs: {len(gen_jobs)}")
    if gen_jobs:
        print("  generation resources (first job):")
        print(f"    time: {gen_jobs[0][3]}")
        print(f"    mem:  {gen_jobs[0][4]}")
    gen_submit_count = _count_gen_submissions(gen_jobs, args.gen_batch_size, args.gpus_per_node)
    print(f"  generation sbatch submissions (post-batching): {gen_submit_count}")
    if not args.skip_scoring:
        print(f"  scoring jobs:    {len(score_jobs)}")
        if score_jobs:
            print("  scoring resources (first job):")
            print(f"    time: {score_jobs[0][3]}")
            print(f"    mem:  {score_jobs[0][4]}")
        score_submit_count = _count_score_submissions(
            score_jobs,
            args.score_batch_size,
            args.score_gpus_per_node,
        )
        print(f"  scoring sbatch submissions (post-batching):    {score_submit_count}")
    else:
        print("  scoring jobs:    skipped (skip-scoring)")
    hazard_only_count = len(hazard_only_dirs)
    if hazard_only_count:
        print(f"  hazard-only dirs: {hazard_only_count}")
    print(f"  total run dirs:  {len(all_run_dirs)}")
    total_submit = gen_submit_count + (score_submit_count if not args.skip_scoring else 0)
    print(f"  total jobs to submit: {len(gen_jobs) + len(score_jobs)}")
    print(f"  total sbatch submissions (post-batching): {total_submit}")

    hazard_only_mode = bool(hazard_only_dirs)
    hazard_planned = bool(all_run_dirs) and (hazard_only_mode or (not args.skip_scoring and args.hazard_after_score))
    if hazard_planned:
        hazard_suffix_preview = _extract_timestamp(all_run_ids[0]) or submission_timestamp
        hazard_output_dir_preview = (
            results_root / f"hazard_report_combined_{hazard_suffix_preview}"
            if results_root
            else _derive_results_root(Path(all_run_dirs[0]), None) / f"hazard_report_combined_{hazard_suffix_preview}"
        )
        dep_batches = len(score_jobs) if (not hazard_only_mode and not args.skip_scoring) else 0
        print(f"  hazard report: planned -> {hazard_output_dir_preview}")
        if dep_batches:
            print(f"    dependencies: {dep_batches} score batch(es)")
        else:
            print("    dependencies: none (hazard-only or scoring disabled)")
        print(f"    hazard run dirs: {len(all_run_dirs)}")

    aggregate_planned = bool(args.auto_aggregate) and bool(all_run_dirs)
    if aggregate_planned:
        aggregate_bases_preview = sorted({(Path(rd).parent if Path(rd).name.startswith("seed=") else Path(rd)) for rd in all_run_dirs})
        dep_batches = len(score_jobs) if not args.skip_scoring else 0
        print(f"  aggregate reports: planned bases = {len(aggregate_bases_preview)}")
        if dep_batches:
            print(f"    dependencies: {dep_batches} score batch(es)")
        else:
            print("    dependencies: none (auto-aggregate without scoring deps)")
        preview_list = aggregate_bases_preview[:3]
        for base in preview_list:
            print(f"    base: {base}")
        if len(aggregate_bases_preview) > len(preview_list):
            print(f"    ... + {len(aggregate_bases_preview) - len(preview_list)} more")

    if args.validate_hydra and gen_jobs:
        preview = gen_jobs[: min(3, len(gen_jobs))]
        print(f"\n[validate] Composing Hydra configs for {len(preview)} generation job(s) to catch errors...")
        for plan, _, _, _, _ in preview:
            print(f"  - {plan.run_id} ({plan.hydra_config})")
            _validate_hydra_config(repo_root, plan.hydra_config, plan.overrides or [])
        print("[validate] Hydra composition succeeded for preview jobs.")
    if not args.dry_run and not args.integration_test:
        try:
            user_input = input("Proceed with submission? [y/N]: ").strip().lower()
        except EOFError:
            user_input = "n"
        if user_input not in {"y", "yes"}:
            print("Aborting submission.")
            return

    gen_job_ids: Dict[str, Optional[str]] = {}

    print(f"[DEBUG] Starting generation job submission (count={len(gen_jobs)})...")
    if args.gen_batch_size <= 1:
        for plan, gen_env, array_range, wall_time, mem in gen_jobs:
            seed_val = plan.metadata.get("seed") if hasattr(plan, "metadata") else None
            gen_cmd = [
                "sbatch",
                "--parsable",
                *(["--account", args.account] if args.account else []),
                *(["--nodes", str(args.nodes)] if args.nodes else []),
                f"--time={wall_time}",
                *([] if args.trillium else [f"--mem={mem}"]),
                f"--array={array_range}",
                f"--gpus-per-node={args.gpus_per_node}",
                str(gen_script_path),
            ]
            gen_job = _submit_sbatch(
                gen_cmd,
                gen_env,
                args.dry_run,
                args.integration_test,
            )
            gen_job_ids[_job_key(plan.run_id, seed_val)] = None if gen_job in ("dry-run", "local-run") else gen_job
    else:
        if gen_jobs:
            spec_root.mkdir(parents=True, exist_ok=True)
        grouped_gen: Dict[
            Tuple[str, str, str, str, str], List[Tuple[GenerationPlan, Dict[str, str], str, str, str]]
        ] = {}
        for item in gen_jobs:
            plan, env, array_range, wall_time, mem = item
            variant_mode = "safe" if plan.variant != "baseline" else "baseline"
            key = (variant_mode, array_range, wall_time, mem, args.gpus_per_node)
            grouped_gen.setdefault(key, []).append(item)

        batch_idx = 0
        for key, jobs in grouped_gen.items():
            variant_mode, array_range, wall_time, mem, gpus_per_node = key
            for batch in _chunk_list(jobs, args.gen_batch_size):
                batch_file = spec_root / f"gen_batch_{variant_mode}_{batch_idx}.envlist"
                with batch_file.open("w") as f:
                    for plan, env, _, _, _ in batch:
                        f.write(_env_to_spec(env) + "\n")

                batch_env = _env_copy()
                batch_env["REPO_ROOT"] = str(repo_root)
                if results_root:
                    batch_env["RESULTS_ROOT"] = str(results_root)
                if args.checkpoint_path:
                    batch_env["CHECKPOINT_PATH"] = str(args.checkpoint_path)
                if args.tokenizer_path:
                    batch_env["TOKENIZER_PATH"] = str(args.tokenizer_path)
                batch_env["CONFIG_BATCH_FILE"] = str(batch_file)

                gen_cmd = [
                    "sbatch",
                    "--parsable",
                    *(["--account", args.account] if args.account else []),
                    *(["--nodes", str(args.nodes)] if args.nodes else []),
                    f"--time={wall_time}",
                    *([] if args.trillium else [f"--mem={mem}"]),
                    f"--array={array_range}",
                    f"--gpus-per-node={gpus_per_node}",
                    str(gen_script_path),
                ]
                gen_job = _submit_sbatch(
                    gen_cmd,
                    batch_env,
                    args.dry_run,
                    args.integration_test,
                )
                for plan, _, _, _, _ in batch:
                    seed_val = plan.metadata.get("seed") if hasattr(plan, "metadata") else None
                    gen_job_ids[_job_key(plan.run_id, seed_val)] = None if gen_job in ("dry-run", "local-run") else gen_job
                batch_idx += 1

    if args.score_batch_size <= 1:
        print(f"[DEBUG] Starting score job submission (count={len(score_jobs)})...")
        for plan, score_env, score_array, score_time, score_mem in score_jobs:
            seed_val = plan.metadata.get("seed") if hasattr(plan, "metadata") else None
            dep_job = gen_job_ids.get(_job_key(plan.run_id, seed_val))
            dep_ids: List[str] = []
            if dep_job:
                dep_ids.append(dep_job)
            plan_variant = getattr(plan, "variant", None)
            if plan_variant == "safe":
                pair_info = _get_paired_run_info(plan, paired_run_info)
                baseline_run_id = None
                if pair_info and pair_info.baseline_run_id:
                    baseline_run_id = pair_info.baseline_run_id
                else:
                    baseline_run_id, _ = _infer_baseline_pair(plan)
                if baseline_run_id:
                    baseline_dep = gen_job_ids.get(_job_key(baseline_run_id, seed_val))
                    if baseline_dep:
                        dep_ids.append(baseline_dep)
            score_cmd: List[str] = [
                "sbatch",
                "--parsable",
                *(["--account", args.account] if args.account else []),
                *(["--nodes", str(args.nodes)] if args.nodes else []),
                f"--array={score_array}",
                f"--time={score_time}",
                *([] if args.trillium else [f"--mem={score_mem}"]),
                f"--gpus-per-node={args.score_gpus_per_node}",
                str(score_script_path),
                str(repo_root),
            ]
            if dep_ids:
                joined = ":".join(dict.fromkeys(dep_ids))
                score_cmd.insert(2, f"--dependency=afterok:{joined}")
            score_job = _submit_sbatch(
                score_cmd,
                score_env,
                args.dry_run,
                args.integration_test,
            )
            if score_job not in ("dry-run", "local-run"):
                all_score_jobs.append(score_job)
    else:
        if score_jobs:
            spec_root.mkdir(parents=True, exist_ok=True)
        grouped_scores: Dict[
            Tuple[str, str, str, str, Tuple[Tuple[str, Optional[str]], ...]],
            List[Tuple[GenerationPlan, Dict[str, str], str, str, str]],
        ] = {}
        for item in score_jobs:
            plan, env, array_range, wall_time, mem = item
            sig = _score_env_signature(env)
            key = (array_range, wall_time, mem, args.score_gpus_per_node, sig)
            grouped_scores.setdefault(key, []).append(item)

        score_batch_idx = 0
        for key, jobs in grouped_scores.items():
            score_array, score_time, mem, gpus_per_node, _ = key
            for batch in _chunk_list(jobs, args.score_batch_size):
                score_list_file = spec_root / f"score_batch_{score_batch_idx}.txt"
                with score_list_file.open("w") as f:
                    for plan, _, _, _, _ in batch:
                        f.write(f"{plan.run_dir}\n")

                batch_env = _env_copy()
                batch_env["REPO_ROOT"] = str(repo_root)
                batch_env["SCORE_RUN_LIST_FILE"] = str(score_list_file)
                # Propagate shared score env fields from the first entry.
                for k, v in batch[0][1].items():
                    if k in ("RUN_DIR", "REPO_ROOT"):
                        continue
                    batch_env[k] = v

                score_cmd: List[str] = [
                    "sbatch",
                    "--parsable",
                    *(["--account", args.account] if args.account else []),
                    *(["--nodes", str(args.nodes)] if args.nodes else []),
                    f"--array={score_array}",
                    f"--time={score_time}",
                    *([] if args.trillium else [f"--mem={mem}"]),
                    f"--gpus-per-node={gpus_per_node}",
                    str(score_script_path),
                    str(repo_root),
                ]
                dep_ids: List[str] = []
                for plan, _, _, _, _ in batch:
                    seed_val = plan.metadata.get("seed") if hasattr(plan, "metadata") else None
                    job_id = gen_job_ids.get(_job_key(plan.run_id, seed_val))
                    if job_id and job_id not in dep_ids:
                        dep_ids.append(job_id)
                # Add baseline deps for each safe plan in this batch.
                for plan, _, _, _, _ in batch:
                    plan_variant = getattr(plan, "variant", None)
                    if plan_variant != "safe":
                        continue
                    seed_val = plan.metadata.get("seed") if hasattr(plan, "metadata") else None
                    pair_info = _get_paired_run_info(plan, paired_run_info)
                    baseline_run_id = None
                    if pair_info and pair_info.baseline_run_id:
                        baseline_run_id = pair_info.baseline_run_id
                    else:
                        baseline_run_id, _ = _infer_baseline_pair(plan)
                    if baseline_run_id:
                        baseline_job_id = gen_job_ids.get(_job_key(baseline_run_id, seed_val))
                        if baseline_job_id and baseline_job_id not in dep_ids:
                            dep_ids.append(baseline_job_id)
                if dep_ids:
                    joined = ":".join(dep_ids)
                    score_cmd.insert(2, f"--dependency=afterok:{joined}")

                score_job = _submit_sbatch(
                    score_cmd,
                    batch_env,
                    args.dry_run,
                    args.integration_test,
                )
                if score_job not in ("dry-run", "local-run"):
                    all_score_jobs.append(score_job)
                score_batch_idx += 1

    hazard_only_mode = bool(hazard_only_dirs)
    # Submit a single hazard report over all runs (scored or provided directly).
    if all_run_dirs and (hazard_only_mode or not args.skip_scoring):
        hazard_env = _env_copy()
        hazard_env["REPO_ROOT"] = str(repo_root)
        if snapshot_path:
            hazard_env["CONFIG_SNAPSHOT_PATH"] = str(snapshot_path)
        
        hazard_suffix = _extract_timestamp(all_run_ids[0]) or datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Write run dirs to file to avoid Argument list too long (E2BIG)
        spec_root.mkdir(parents=True, exist_ok=True)
        hazard_dirs_file = spec_root / f"hazard_dirs_{hazard_suffix}.txt"
        print(f"  [hazard] writing {len(all_run_dirs)} run dirs to {hazard_dirs_file} (avoiding E2BIG)")
        with hazard_dirs_file.open("w") as f:
            for rd in all_run_dirs:
                f.write(f"{rd}\n")
        hazard_env["RUN_DIRS_FILE"] = str(hazard_dirs_file)

        hazard_base = (
            results_root / f"hazard_report_combined_{hazard_suffix}"
            if results_root
            else _derive_results_root(Path(all_run_dirs[0]), None) / f"hazard_report_combined_{hazard_suffix}"
        )
        hazard_env["OUTPUT_DIR"] = str(hazard_base)
        hazard_env["HAZARD_TIMESTAMP"] = hazard_suffix
        hazard_env["EXAMPLES_PER_HAZARD"] = str(args.hazard_examples_per_hazard)
        hazard_env["EXAMPLES_PER_METRIC"] = str(args.hazard_examples_per_metric)
        hazard_env["EXAMPLES_PER_TRANSITION"] = str(args.hazard_examples_per_transition)

        hazard_cmd: Sequence[str] = [
            "sbatch",
            "--parsable",
            *(["--account", args.account] if args.account else []),
            *(["--nodes", str(args.nodes)] if args.nodes else []),
            f"--time={args.hazard_time}",
            str(hazard_script_path),
        ]
        if all_score_jobs:
            # Large dependency lists can exceed argument length limits. If the
            # dependency string is too long, submit lightweight barrier jobs
            # that each depend on a chunk of score jobs, then depend on those
            # barriers instead.
            deps_list = list(all_score_jobs)
            barrier_jobs: list[str] = []
            max_dep_chars = 4000
            max_chunk_size = 100
            joined = ":".join(deps_list)
            if len(joined) > max_dep_chars:
                for chunk in _chunk_list(deps_list, max_chunk_size):
                    chunk_dep = ":".join(chunk)
                    barrier_cmd = [
                        "sbatch",
                        "--parsable",
                        *(["--account", args.account] if args.account else []),
                        f"--dependency=afterany:{chunk_dep}",
                        "--time=00:10:00",
                        "--wrap",
                        "sleep 1",
                    ]
                    barrier_job = _submit_sbatch(
                        barrier_cmd,
                        _env_copy(),
                        args.dry_run,
                        args.integration_test,
                        confirm_integration=False,
                    )
                    if barrier_job not in ("dry-run", "local-run"):
                        barrier_jobs.append(barrier_job)
                deps_list = barrier_jobs
            deps = ":".join(deps_list)
            hazard_cmd = list(hazard_cmd)
            hazard_cmd.insert(2, f"--dependency=afterany:{deps}")

        print(
            f"\n[hazard] {'integration-test' if args.integration_test else 'submitting'} combined report over {len(all_run_dirs)} runs -> "
            f"{hazard_env['OUTPUT_DIR']}"
        )
        if all_score_jobs:
            print(f"[hazard] dependencies (score jobs): {len(all_score_jobs)}")
        else:
            print("[hazard] no score-job dependencies; submitting immediately.")

        hazard_job = _submit_sbatch(
            hazard_cmd,
            hazard_env,
            args.dry_run,
            args.integration_test,
        )
        print(f"[hazard] submitted job: {hazard_job} -> {hazard_env['OUTPUT_DIR']}")

    if args.auto_aggregate and all_run_dirs:
        dep_ids = all_score_jobs if all_score_jobs else [jid for jid in gen_job_ids.values() if jid]
        if dep_ids:
            aggregate_bases = sorted({(Path(rd).parent if Path(rd).name.startswith("seed=") else Path(rd)) for rd in all_run_dirs})
            seeds_part: List[str] = []
            if seeds_arg:
                seeds_part = ["--seeds", *[str(s) for s in seeds_arg]]
            aggregate_jobs: List[str] = []
            for base_dir in aggregate_bases:
                agg_dep_ids = list(dep_ids)
                agg_dep_joined = ":".join(agg_dep_ids)
                if len(agg_dep_joined) > 4000:
                    barrier_jobs: List[str] = []
                    for chunk in _chunk_list(agg_dep_ids, 100):
                        chunk_dep = ":".join(chunk)
                        barrier_cmd = [
                            "sbatch",
                            "--parsable",
                            *(["--account", args.account] if args.account else []),
                            f"--dependency=afterok:{chunk_dep}",
                            "--time=00:10:00",
                            "--wrap",
                            "sleep 1",
                        ]
                        barrier_job = _submit_sbatch(
                            barrier_cmd,
                            _env_copy(),
                            args.dry_run,
                            args.integration_test,
                            confirm_integration=False,
                        )
                        if barrier_job not in ("dry-run", "local-run"):
                            barrier_jobs.append(barrier_job)
                    agg_dep_ids = barrier_jobs
                    agg_dep_joined = ":".join(agg_dep_ids)
                agg_cmd = [
                    "sbatch",
                    "--parsable",
                    f"--dependency=afterok:{agg_dep_joined}",
                    *(["--account", args.account] if args.account else []),
                    f"--time={args.score_time}",
                    *([] if args.trillium else [f"--mem=8G"]),
                    "--wrap",
                    f"cd {repo_root} && python src/slurm/submit_sbatch_experiments.py --aggregate-only --aggregate-run-dir {shlex.quote(str(base_dir))} {' '.join(seeds_part)}",
                ]
                agg_job = _submit_sbatch(
                    agg_cmd,
                    _env_copy(),
                    args.dry_run,
                    args.integration_test,
                )
                aggregate_jobs.append(agg_job)
            print(
                "\n[aggregate] submitted "
                f"{len(aggregate_jobs)} aggregate job(s) with dependency on scoring/gen completion:"
            )
            for job, base_dir in zip(aggregate_jobs, aggregate_bases):
                print(f"  job {job}: aggregate {base_dir}")

    print("\n[info] Prompt pipeline submissions queued.")


if __name__ == "__main__":
    main()
