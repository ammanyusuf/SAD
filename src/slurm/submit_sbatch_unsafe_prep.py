#!/usr/bin/env python3
"""
Submit unsafe-prep tensor builds (and optional semantic caches) to Slurm.

Splits the artifact sweep across a requested number of jobs, submits
`generate_unsafe_tensors.sh` for each chunk, and (optionally) chains
`generate_semantic_cache.sh` with an afterok dependency on the tensor job.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Optional

from omegaconf import OmegaConf


def _chunk(items: List[str], parts: int) -> Iterable[List[str]]:
    if parts <= 0:
        raise SystemExit("jobs must be a positive integer")
    size = max(1, (len(items) + parts - 1) // parts)
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _load_artifact_names(sweep_path: Path) -> List[str]:
    cfg = OmegaConf.load(sweep_path)
    cfg_data = OmegaConf.to_container(cfg, resolve=True)
    datasets = cfg_data.get("datasets") if isinstance(cfg_data, dict) else None
    if not datasets or not isinstance(datasets, list):
        raise SystemExit(f"Unsafe sweep config at {sweep_path} is missing a datasets list.")
    names: List[str] = []
    for entry in datasets:
        if not isinstance(entry, dict):
            continue
        name = entry.get("output_name") or entry.get("name")
        if name:
            names.append(str(name))
    if not names:
        raise SystemExit(f"No output_name entries discovered in {sweep_path}.")
    return names


def _env_copy() -> Dict[str, str]:
    return dict(os.environ)


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


def _submit_sbatch(cmd: Sequence[str], env: Dict[str, str], dry_run: bool, integration_test: bool) -> str:
    printable = shlex.join(cmd)
    print(f"[sbatch] {printable}")
    if dry_run:
        return "dry-run"
    if integration_test:
        script_idx = next((i for i, part in enumerate(cmd) if part.endswith(".sh")), None)
        if script_idx is None:
            raise RuntimeError(f"Could not locate script path in command: {printable}")
        script_and_args = list(cmd[script_idx:])
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
        completed = subprocess.run(local_cmd, env=local_env, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"Integration test failed (exit={completed.returncode}) for {local_cmd}")
        return "local-run"
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        last_line = completed.stdout.strip().split("\n")[-1]
        match = re.search(r"(\d+)$", last_line)
        job_id = match.group(1) if match else last_line
        print(f"  -> job {job_id}")
        return job_id
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] sbatch failed (returncode={exc.returncode})")
        if exc.stdout:
            print(f"[stdout] {exc.stdout.strip()}")
        if exc.stderr:
            print(f"[stderr] {exc.stderr.strip()}")
        raise


def _resolve_path(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit unsafe-prep sweep jobs to Slurm.")
    parser.add_argument("--config", type=Path, required=True, help="Unsafe-prep submission config.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--tensors-script", type=Path, default=Path("src/slurm/generate_unsafe_tensors.sh"))
    parser.add_argument("--semantic-script", type=Path, default=Path("src/slurm/generate_semantic_cache.sh"))
    parser.add_argument("--account", default=None, help="Slurm account to charge (e.g., rrg-<your-PI>_gpu).")
    parser.add_argument(
        "--gpus-per-node",
        default="a100:1",
        help="GPU resource string for semantic cache jobs (e.g., h100:1).",
    )
    parser.add_argument(
        "--tensor-gpus-per-node",
        default=None,
        help="GPU resource string for tensor jobs; omit to request CPUs only.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting.")
    parser.add_argument(
        "--integration-test",
        action="store_true",
        help="Execute the scripts locally (no sbatch) to validate wiring.",
    )
    parser.add_argument(
        "--semantic-only",
        action="store_true",
        help="Skip tensor builds and only run semantic cache jobs for existing artifacts.",
    )

    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    cfg_path = _resolve_path(args.config, repo_root)
    cfg = OmegaConf.load(cfg_path)

    sweep_cfg_value = cfg.get("sweep_config")
    if sweep_cfg_value is None:
        raise SystemExit("sweep_config must point to an unsafe prompt sweep YAML.")
    sweep_path = _resolve_path(Path(str(sweep_cfg_value)), repo_root)
    if not sweep_path.exists():
        raise SystemExit(f"Unsafe sweep config not found: {sweep_path}")
    artifact_names = _load_artifact_names(sweep_path)
    jobs = int(cfg.get("jobs", 1))
    output_root_val = cfg.get("output_root")
    if output_root_val is None:
        raise SystemExit("output_root must be provided in the unsafe-prep submission config.")
    tokenizer_val = cfg.get("tokenizer")
    if tokenizer_val is None:
        raise SystemExit("tokenizer must be provided in the unsafe-prep submission config.")
    output_root = _resolve_path(Path(str(output_root_val)), repo_root)
    tokenizer_path = _resolve_path(Path(str(tokenizer_val)), repo_root)

    submission_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    spec_root = output_root / ".slurm_specs"
    snapshot_path = _create_config_snapshot(repo_root, spec_root, submission_timestamp)

    base_env = _env_copy()
    base_env["REPO_ROOT"] = str(repo_root)
    if snapshot_path:
        base_env["CONFIG_SNAPSHOT_PATH"] = str(snapshot_path)
    
    try:
        rel_sweep = sweep_path.relative_to(repo_root)
        base_env["UNSAFE_CONFIG"] = str(rel_sweep)
    except ValueError:
        base_env["UNSAFE_CONFIG"] = str(sweep_path)

    base_env["UNSAFE_OUTPUT_ROOT"] = str(output_root)
    base_env["TOKENIZER_NAME_OR_PATH"] = str(tokenizer_path)
    max_length = cfg.get("max_length")
    shard_size = cfg.get("shard_size")
    if max_length is not None:
        base_env["UNSAFE_MAX_LENGTH"] = str(max_length)
    if shard_size is not None:
        base_env["UNSAFE_SHARD_SIZE"] = str(shard_size)
    python_bin = str(cfg.get("python", "python"))
    base_env["PYTHON_BIN"] = python_bin

    semantic_cfg = cfg.get("semantic_cache") or {}
    semantic_enabled = bool(semantic_cfg.get("enabled", False))

    tensors_script = _resolve_path(args.tensors_script, repo_root)
    semantic_script = _resolve_path(args.semantic_script, repo_root)

    planned_batches = list(_chunk(artifact_names, jobs))
    if not args.dry_run:
        print("[confirm] planned submissions:")
        print(f"  tensor max_length={max_length if max_length is not None else 'default'}")
        print(f"  tensor shard_size={shard_size if shard_size is not None else 'default'}")
        for idx, batch in enumerate(planned_batches):
            print(f"  batch {idx}: {', '.join(batch)}")
        proceed = input("Submit these jobs? [y/N]: ").strip().lower()
        if proceed not in {"y", "yes"}:
            print("Aborted.")
            return

    for idx, batch in enumerate(planned_batches):
        env = dict(base_env)
        env["ARTIFACT_NAMES"] = " ".join(batch)

        tensor_job = None
        if not args.semantic_only:
            print(f"[submit] batch {idx}: tensors for {', '.join(batch)}")
            tensor_cmd = [
                "sbatch",
                *(["--account", args.account] if args.account else []),
                *(["--gpus-per-node=" + args.tensor_gpus_per_node] if args.tensor_gpus_per_node else []),
                str(tensors_script),
            ]
            tensor_job = _submit_sbatch(tensor_cmd, env, args.dry_run, args.integration_test)

        if not semantic_enabled:
            continue

        semantic_env = dict(env)
        semantic_env["ARTIFACT_ROOT"] = str(output_root)
        provider = semantic_cfg.get("provider", "mdlm")
        semantic_env["PROVIDER"] = str(provider)
        print(f"[submit] batch {idx}: semantic caches for {', '.join(batch)} (provider={provider})")
        if semantic_cfg.get("checkpoint") is not None:
            semantic_env["CHECKPOINT_PATH"] = str(
                _resolve_path(Path(str(semantic_cfg.checkpoint)), repo_root)
            )
        if semantic_cfg.get("tokenizer") is not None:
            tokenizer_override = _resolve_path(Path(str(semantic_cfg.tokenizer)), repo_root)
            semantic_env["TOKENIZER_PATH"] = str(tokenizer_override)
            semantic_env["TOKENIZER_OVERRIDE"] = str(tokenizer_override)
        if semantic_cfg.get("encoder") is not None:
            semantic_env["ENCODER_NAME"] = str(
                _resolve_path(Path(str(semantic_cfg.encoder)), repo_root)
            )
        if semantic_cfg.get("model_config") is not None:
            semantic_env["MODEL_CONFIG_PATH"] = str(
                _resolve_path(Path(str(semantic_cfg.model_config)), repo_root)
            )
        if semantic_cfg.get("mdlm_embed_attr") is not None:
            semantic_env["MDLM_EMBED_ATTR"] = str(semantic_cfg.mdlm_embed_attr)
        if semantic_cfg.get("batch_size") is not None:
            semantic_env["BATCH_SIZE"] = str(semantic_cfg.batch_size)
        if semantic_cfg.get("device") is not None:
            semantic_env["DEVICE"] = str(semantic_cfg.device)

        semantic_cmd = ["sbatch"]
        if tensor_job not in {"dry-run", "local-run", None}:
            semantic_cmd.append(f"--dependency=afterok:{tensor_job}")
        if args.account:
            semantic_cmd.extend(["--account", args.account])
        semantic_cmd.append(f"--gpus-per-node={args.gpus_per_node}")
        semantic_cmd.append(str(semantic_script))
        _submit_sbatch(semantic_cmd, semantic_env, args.dry_run, args.integration_test)


if __name__ == "__main__":
    main()
