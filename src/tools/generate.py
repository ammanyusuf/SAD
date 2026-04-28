#!/usr/bin/env python3

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import pynvml
import json

from utils.prompt_loader import load_prompt_records
from sampling.sample_text import (
    GenerationRun,
    GenerationResult,
    GenerationSettings,
    ModelSettings,
    PromptRecord,
    SafetySettings,
    resolve_eta_config,
    run_generation,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class GenerationMetadata:
    created_at: str
    model: Dict[str, Any]
    data: Dict[str, Any]
    io: Dict[str, Any]
    sharding: Dict[str, Any]
    timings: Dict[str, Any]
    run_id: Optional[str]
    generation: Dict[str, Any]
    safety: Dict[str, Any]
    telemetry: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _resolve_path(path_value: Optional[str]) -> Optional[Path]:
    if path_value in (None, "", "null"):
        return None
    path_str = str(path_value)
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(to_absolute_path(path_str))


def _select_prompt_source(cfg: DictConfig) -> Tuple[Optional[DictConfig], Optional[str]]:
    """Return the configured prompt source (if any)."""
    primary = cfg.data.get("prompt_source")
    if primary and primary.get("name"):
        return primary, "data.prompt_source"
    return primary, None


def _compute_contiguous_slice(length: int, parts: int, index: int) -> tuple[int, int]:
    if parts <= 0:
        raise SystemExit(f"num_shards must be positive (got {parts}).")
    if index < 0 or index >= parts:
        raise SystemExit(f"Index {index} is outside valid range [0, {parts}).")
    base = length // parts
    remainder = length % parts
    start = index * base + min(index, remainder)
    size = base + (1 if index < remainder else 0)
    end = min(length, start + size)
    return start, end


def _select_shard(
    records: Sequence[PromptRecord],
    shard_id: int,
    num_shards: int,
) -> List[PromptRecord]:
    start, end = _compute_contiguous_slice(len(records), num_shards, shard_id)
    return list(records[start:end])


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _serialize_results(run: GenerationRun) -> Iterable[Dict[str, Any]]:
    for item in run.results:
        yield {
            "prompt_id": item.prompt_id,
            "prompt": item.prompt,
            "completion": item.completion,
            "full_text": item.full_text,
            "token_ids": item.token_ids,
            "prompt_length": item.prompt_length,
            "metadata": item.metadata,
        }


def _query_gpu_busy_percent() -> Optional[float]:
    if pynvml is None:
        return None
    device_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        pynvml.nvmlInit()
        gpu_index = int(device_env.split(",")[0]) if device_env else 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(utilization.gpu)
    except (pynvml.NVMLError, ValueError):
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _git_commit_hash() -> Optional[str]:
    repo_root = Path(__file__).resolve().parents[2]
    head_path = repo_root / ".git" / "HEAD"
    if not head_path.exists():
        return None
    try:
        head_ref = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if head_ref.startswith("ref:"):
        ref = head_ref.split(" ", 1)[1].strip()
        ref_path = repo_root / ".git" / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip() or None
        return None
    return head_ref or None


def _write_slurm_meta(output_dir: Path) -> None:
    meta = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
    }
    (output_dir / "slurm_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def _has_override(task_overrides: Sequence[str], key: str) -> bool:
    """Return True if Hydra overrides include the given key (ignoring leading +)."""
    for override in task_overrides:
        normalized = override.lstrip("+")
        if normalized.startswith(f"{key}=") or normalized == key:
            return True
    return False


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    LOGGER.info(cfg)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    LOGGER.info("Starting generation run with Hydra config.")
    LOGGER.info("Config snapshot:\n%s", OmegaConf.to_yaml(cfg, resolve=False))
    model_family = str(getattr(cfg.model, "family", "mdlm")).lower()
    dataset_path = _resolve_path(cfg.data.dataset_json)
    prompt_source_cfg, prompt_source_origin = _select_prompt_source(cfg)
    prompt_source_active = bool(prompt_source_cfg and prompt_source_cfg.get("name"))
    if prompt_source_origin and prompt_source_origin != "data.prompt_source":
        params_obj = prompt_source_cfg.get("params") if prompt_source_cfg else None
        if isinstance(params_obj, DictConfig):
            data_dir = params_obj.get("data_dir")
        elif isinstance(params_obj, dict):
            data_dir = params_obj.get("data_dir")
        else:
            data_dir = None
        if data_dir:
            LOGGER.info(
                "Using prompt source from %s (data_dir=%s).",
                prompt_source_origin,
                data_dir,
            )
        else:
            LOGGER.info("Using prompt source from %s.", prompt_source_origin)
    checkpoint_path = _resolve_path(cfg.model.checkpoint)
    tokenizer_path = _resolve_path(cfg.model.tokenizer_name)
    checkpoint_exists = checkpoint_path is not None and checkpoint_path.exists()
    tokenizer_exists = tokenizer_path is not None and tokenizer_path.exists()
    checkpoint_value = checkpoint_path if checkpoint_exists else cfg.model.checkpoint
    tokenizer_value = tokenizer_path if tokenizer_exists else cfg.model.tokenizer_name
    unsafe_artifacts = _resolve_path(cfg.safety.unsafe_artifacts)
    unsafe_artifact_root = _resolve_path(cfg.safety.unsafe_artifact_root)

    if not dataset_path and not prompt_source_active and cfg.gen.unconditional_samples <= 0:
        raise SystemExit(
            "Provide data.dataset_json, configure data.prompt_source, "
            "or request gen.unconditional_samples > 0."
        )
    if checkpoint_path is None or (not checkpoint_exists and model_family != "llada"):
        raise SystemExit(f"Checkpoint path does not exist: {cfg.model.checkpoint}")
    if tokenizer_path is None or (not tokenizer_exists and model_family != "llada"):
        raise SystemExit(f"Tokenizer path does not exist: {cfg.model.tokenizer_name}")
    auto_build_unsafe = bool(cfg.safety.get("auto_build_unsafe_artifacts", False))
    if unsafe_artifacts and not unsafe_artifacts.exists() and not auto_build_unsafe:
        raise SystemExit(f"Unsafe artifact path does not exist: {unsafe_artifacts}")
    if unsafe_artifact_root and not unsafe_artifact_root.exists() and not auto_build_unsafe:
        raise SystemExit(f"Unsafe artifact path does not exist: {unsafe_artifact_root}")

    run_root = Path(cfg.io.output_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_root / "config_merged.yaml", resolve=False)
    _write_slurm_meta(run_root)

    records: List[PromptRecord] = []
    if dataset_path is not None and not dataset_path.exists():
        raise SystemExit(f"Dataset file not found: {dataset_path}")
    if dataset_path is not None or prompt_source_active:
        records = load_prompt_records(dataset_path, prompt_source_cfg)
    total_prompts = len(records) if records else cfg.data.total_prompts

    shard_records: List[PromptRecord] = []
    if cfg.data.limit == 0 and cfg.gen.unconditional_samples > 0:
        records = []
        LOGGER.info(
            "data.limit=0 with unconditional_samples>0; skipping conditioned prompts."
        )
    slice_bounds: Optional[tuple[int, int]] = None
    if records:
        if (cfg.sharding.range_start is None) ^ (cfg.sharding.range_end is None):
            raise SystemExit(
                "Provide both sharding.range_start and sharding.range_end, or neither."
            )
        if cfg.sharding.range_start is not None and cfg.sharding.range_end is not None:
            start = max(0, cfg.sharding.range_start)
            end = min(len(records), cfg.sharding.range_end)
            if end <= start:
                raise SystemExit(
                    f"Invalid slice [{cfg.sharding.range_start}, {cfg.sharding.range_end}) "
                    f"for dataset of size {len(records)}."
                )
            slice_bounds = (start, end)
            shard_records = list(records[start:end])
        else:
            shard_records = _select_shard(
                records,
                cfg.sharding.shard_id,
                cfg.sharding.num_shards,
            )
            slice_bounds = _compute_contiguous_slice(
                len(records),
                cfg.sharding.num_shards,
                cfg.sharding.shard_id,
            )

        if cfg.data.limit:
            shard_records = shard_records[: max(cfg.data.limit, 0)]

        LOGGER.info(
            "Loaded %d prompts; slice %s yields %d prompts.",
            len(records),
            slice_bounds if slice_bounds else f"{cfg.sharding.shard_id}/{cfg.sharding.num_shards}",
            len(shard_records),
        )
    else:
        LOGGER.info(
            "No prompt dataset provided; shard %d/%d will generate %d unconditional samples.",
            cfg.sharding.shard_id,
            cfg.sharding.num_shards,
            cfg.gen.unconditional_samples,
        )

    model_settings = ModelSettings(
        model_name=cfg.model.model_name,
        checkpoint_path=checkpoint_value,
        tokenizer_name=str(tokenizer_value),
        precision=cfg.model.precision,
        variant=cfg.model.get("variant"),
    )
    generation_settings = GenerationSettings(
        max_new_tokens=cfg.gen.max_new_tokens,
        prefix_length=cfg.data.prefix_length,
        sampling_steps=cfg.gen.sampling_steps,
        batch_size=cfg.gen.batch_size,
        seed=cfg.gen.seed,
        add_bos=cfg.gen.add_bos,
        add_eos=cfg.gen.add_eos,
        unconditional_samples=max(cfg.gen.unconditional_samples, 0),
        auto_batch=cfg.io.auto_batch,
        auto_batch_target_pct=cfg.io.target_vram_pct,
        max_auto_batch_size=None,
        auto_batch_warmup_prompts=cfg.io.auto_batch_warmup_prompts,
        precision=cfg.model.precision,
        block_length=cfg.gen.get("block_length"),
        transfer_schedule=cfg.gen.get("transfer_schedule"),
    )
    eta_value, eta_from_scale = resolve_eta_config(cfg.safety)
    safety_settings = SafetySettings(
        enabled=cfg.safety.enabled,
        eta=eta_value,
        eta_from_scale=eta_from_scale,
        weight_mode=cfg.safety.get("weight_mode", "eta_beta_hat"),
        beta_hat_mode=cfg.safety.get("beta_hat_mode", "mc_mean"),
        beta_hat_clip_min=cfg.safety.get("beta_hat_clip_min"),
        beta_hat_clip_max=cfg.safety.get("beta_hat_clip_max"),
        schedule_mode=cfg.safety.get("schedule_mode", "hard_window"),
        scale=cfg.safety.get("scale"),
        unsafe_artifacts=unsafe_artifacts,
        unsafe_artifact_root=unsafe_artifact_root,
        unsafe_artifact_name=cfg.safety.unsafe_artifact_name,
        unsafe_prototypes=(
            Path(cfg.safety.unsafe_prototypes).expanduser()
            if cfg.safety.unsafe_prototypes
            else None
        ),
        critical_steps=cfg.safety.get("critical_steps"),
        t_start=cfg.safety.get("t_start"),
        t_end=cfg.safety.get("t_end"),
        use_semantic_gating=cfg.safety.get("use_semantic_gating", False),
        semantic_weight=float(cfg.safety.get("semantic_weight", 0.0)),
        semantic_temp=float(cfg.safety.get("semantic_temp", 1.0)),
        semantic_sigma=cfg.safety.get("semantic_sigma"),
        cache_semantic_ref=cfg.safety.get("cache_semantic_ref", False),
        semantic_ref_path=(
            Path(cfg.safety.semantic_ref_path).expanduser()
            if cfg.safety.get("semantic_ref_path")
            else None
        ),
        semantic_checkpoint=(
            Path(cfg.safety.semantic_checkpoint).expanduser()
            if cfg.safety.get("semantic_checkpoint")
            else None
        ),
        semantic_embed_attr=cfg.safety.get("semantic_embed_attr"),
        auto_build_unsafe_artifacts=auto_build_unsafe,
        tokenizer_name_or_path=str(tokenizer_value) if tokenizer_value else None,
    )
    shard_metadata = {
        "track": cfg.io.track_name,
        "experiment_slug": cfg.io.experiment_slug,
        "run_id": cfg.io.run_id,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "slice_start": slice_bounds[0] if slice_bounds else None,
        "slice_end": slice_bounds[1] if slice_bounds else None,
        "total_prompts": total_prompts,
    }

    if cfg.gen.dry_run:
        LOGGER.info("Dry run requested; skipping model execution.")
        base_records = shard_records or [
            PromptRecord(prompt_id=f"dry-run:{idx}", prompt="", metadata={})
            for idx in range(max(1, cfg.gen.unconditional_samples))
        ]
        stub_results = [
            GenerationResult(
                prompt_id=record.prompt_id,
                prompt=record.prompt,
                completion="[dry-run]",
                full_text="[dry-run]",
                token_ids=[],
                prompt_length=0,
                prompt_mask=[],
                metadata={**record.metadata, "dry_run": True},
            )
            for record in base_records
        ]
        run = GenerationRun(
            results=stub_results,
            timings={
                "load_seconds": 0.0,
                "generation_seconds": 0.0,
                "total_seconds": 0.0,
            },
            resolved_config={},
        )
    else:
        _filter_variants = {"posthoc_filter", "best_of_n", "fk_steering"}
        _model_variant = str(model_settings.variant or "").lower()
        if model_family == "llada":
            # lazy import to avoid extra deps
            from sampling.llada_engine import run_llada_generation
            run = run_llada_generation(
                prompts=shard_records if shard_records else None,
                model=model_settings,
                generation=generation_settings,
                safety=safety_settings,
                shard_metadata=shard_metadata,
            )
        elif _model_variant in _filter_variants:
            # Filtering baseline variants (posthoc_filter, best_of_n, fk_steering)
            # dispatch through the backend registry with the correct model family.
            import lightning as _lightning
            from sampling.backends.registry import get_backend as _get_backend
            if generation_settings.seed is not None:
                _lightning.seed_everything(generation_settings.seed)
            _backend = _get_backend(model_family, _model_variant)
            _backend.load(model_settings=model_settings, device=None)
            import torch as _torch
            with _torch.inference_mode():
                run = _backend.generate_batch(
                    prompts=shard_records if shard_records else None,
                    generation=generation_settings,
                    safety=safety_settings,
                    shard_metadata=shard_metadata,
                )
        else:
            run = run_generation(
                prompts=shard_records if shard_records else None,
                model=model_settings,
                generation=generation_settings,
                safety=safety_settings,
                shard_metadata=shard_metadata,
            )

    shard_id = cfg.sharding.shard_id
    shard_dir = run_root / f"shard_{shard_id:05d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    generations_path = shard_dir / "generations.jsonl"
    _write_jsonl(generations_path, _serialize_results(run))

    gpu_busy = _query_gpu_busy_percent()
    peak_vram_bytes = run.timings.get("peak_vram_bytes")
    peak_vram_gb = round(peak_vram_bytes / (1024**3), 4) if peak_vram_bytes else None
    gen_seconds = run.timings.get("generation_seconds") or 0.0
    sequences_per_second = (len(run.results) / gen_seconds) if gen_seconds > 0 else None

    metadata = GenerationMetadata(
        created_at=datetime.now(timezone.utc).isoformat(),
        model={
            "name": cfg.model.model_name,
            "checkpoint": str(checkpoint_value),
            "tokenizer": str(tokenizer_value),
            "precision": cfg.model.precision,
            "family": getattr(cfg.model, "family", "mdlm"),
        },
        data={
            "dataset_path": str(dataset_path) if dataset_path else None,
            "total_prompts": total_prompts,
            "prefix_length": cfg.data.prefix_length,
            "limit": cfg.data.limit,
            "prompt_variant": getattr(cfg.data, "prompt_variant", None),
        },
        io={
            "output_dir": str(run_root),
            "experiment_slug": cfg.io.experiment_slug,
            "run_id": cfg.io.run_id,
            "track_name": cfg.io.track_name,
        },
        sharding={
            "slice_start": slice_bounds[0] if slice_bounds else None,
            "slice_end": slice_bounds[1] if slice_bounds else None,
            "shard_id": cfg.sharding.shard_id,
            "num_shards": cfg.sharding.num_shards,
        },
        timings=run.timings,
        run_id=cfg.io.run_id,
        generation={
            "max_new_tokens": cfg.gen.max_new_tokens,
            "batch_size": cfg.gen.batch_size,
            "sampling_steps": cfg.gen.sampling_steps,
            "seed": cfg.gen.seed,
            "add_bos": cfg.gen.add_bos,
            "add_eos": cfg.gen.add_eos,
            "precision": cfg.model.precision,
            "auto_batch": cfg.io.auto_batch,
            "auto_batch_target_pct": cfg.io.target_vram_pct,
            "auto_batch_warmup_prompts": cfg.io.auto_batch_warmup_prompts,
        },
        safety={
            "enabled": safety_settings.enabled,
            "scale": safety_settings.scale,
            "unsafe_artifacts": str(unsafe_artifacts) if unsafe_artifacts else None,
            "unsafe_artifact_root": str(unsafe_artifact_root) if unsafe_artifact_root else None,
            "unsafe_artifact_name": cfg.safety.unsafe_artifact_name,
        },
        telemetry={
            "sequences_per_second": sequences_per_second,
            "peak_vram_gb": peak_vram_gb,
            "gpu_busy_pct": gpu_busy,
        },
    )
    (shard_dir / "run_metadata.json").write_text(
        json.dumps(metadata.to_dict(), indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Wrote %d records to %s", len(run.results), generations_path)


if __name__ == "__main__":
    main()
