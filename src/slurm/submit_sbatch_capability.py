#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
import signal
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf

from src.utils.capability_experiment_setup import CapabilityPlan, build_capability_plans


DEFAULT_SLURM_ACCOUNT = ""  # [Compute Canada] set via --account or SLURM_ACCOUNT env var
MMLU_GROUP_TASKS = (
    "mmlu_stem",
    "mmlu_social_sciences",
    "mmlu_humanities",
    "mmlu_other",
)
MMLU_SUBJECT_TASKS = (
    "mmlu_abstract_algebra",
    "mmlu_anatomy",
    "mmlu_astronomy",
    "mmlu_business_ethics",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_chemistry",
    "mmlu_college_computer_science",
    "mmlu_college_mathematics",
    "mmlu_college_medicine",
    "mmlu_college_physics",
    "mmlu_computer_security",
    "mmlu_conceptual_physics",
    "mmlu_econometrics",
    "mmlu_electrical_engineering",
    "mmlu_elementary_mathematics",
    "mmlu_formal_logic",
    "mmlu_global_facts",
    "mmlu_high_school_biology",
    "mmlu_high_school_chemistry",
    "mmlu_high_school_computer_science",
    "mmlu_high_school_european_history",
    "mmlu_high_school_geography",
    "mmlu_high_school_government_and_politics",
    "mmlu_high_school_macroeconomics",
    "mmlu_high_school_mathematics",
    "mmlu_high_school_microeconomics",
    "mmlu_high_school_physics",
    "mmlu_high_school_psychology",
    "mmlu_high_school_statistics",
    "mmlu_high_school_us_history",
    "mmlu_high_school_world_history",
    "mmlu_human_aging",
    "mmlu_human_sexuality",
    "mmlu_international_law",
    "mmlu_jurisprudence",
    "mmlu_logical_fallacies",
    "mmlu_machine_learning",
    "mmlu_management",
    "mmlu_marketing",
    "mmlu_medical_genetics",
    "mmlu_miscellaneous",
    "mmlu_moral_disputes",
    "mmlu_moral_scenarios",
    "mmlu_nutrition",
    "mmlu_philosophy",
    "mmlu_prehistory",
    "mmlu_professional_accounting",
    "mmlu_professional_law",
    "mmlu_professional_medicine",
    "mmlu_professional_psychology",
    "mmlu_public_relations",
    "mmlu_security_studies",
    "mmlu_sociology",
    "mmlu_us_foreign_policy",
    "mmlu_virology",
    "mmlu_world_religions",
)


def _env_copy() -> Dict[str, str]:
    return dict(os.environ)


def _bool_to_flag(value: bool) -> str:
    return "1" if value else "0"


def _env_to_spec(env: Dict[str, str]) -> str:
    keys = [
        "REPO_ROOT",
        "OUTPUT_DIR",
        "MODEL_PATH",
        "MODEL_FAMILY",
        "MODEL_VARIANT",
        "MODEL_NAME",
        "CAP_BACKEND",
        "CAP_TASKS",
        "CAP_MODEL_ARGS",
        "CAP_EVAL_ARGS",
        "CAP_BATCH_SIZE",
        "CAP_NUM_FEWSHOT",
        "CAP_APPLY_CHAT_TEMPLATE",
        "CAP_CONFIRM_RUN_UNSAFE_CODE",
        "CAP_LOG_SAMPLES",
        "CAP_ALLOW_CODE_EVAL",
        "SAFETY_ENABLED",
        "SAFETY_ETA",
        "SAFETY_SCALE",
        "SAFETY_AUTO_BUILD_UNSAFE_ARTIFACTS",
        "UNSAFE_ARTIFACT_ROOT",
        "UNSAFE_ARTIFACT_NAME",
        "UNSAFE_ARTIFACTS",
        "SAFETY_T_START",
        "SAFETY_T_END",
    ]
    parts: List[str] = []
    for key in keys:
        value = env.get(key)
        if value not in (None, "", "null"):
            parts.append(f"{key}={shlex.quote(str(value))}")
    return " ".join(parts)


def _determine_slurm_params(
    plan: CapabilityPlan,
    cfg,
    mode: str,
    default_array: str,
    default_time: str,
    default_mem: str = "8G",
) -> Tuple[str, str, str]:
    slurm_cfg = plan.metadata.get("slurm")
    cfg_slurm = cfg.get("slurm", {}) if cfg is not None else {}
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
    if array_val is None:
        array_val = cfg_slurm.get("array")
    if time_val is None:
        time_val = cfg_slurm.get("time")
    if mem_val is None:
        mem_val = cfg_slurm.get("mem")
    array_range = str(array_val or default_array)
    wall_time = str(time_val or default_time)
    mem = str(mem_val or default_mem)
    return array_range, wall_time, mem


def _build_env(
    plan: CapabilityPlan,
    repo_root: Path,
    results_root: Optional[Path] = None,
    model_path_override: Optional[str] = None,
) -> Dict[str, str]:
    env = _env_copy()
    env["REPO_ROOT"] = str(repo_root)
    output_dir = plan.run_dir
    if results_root is not None:
        output_dir = results_root / plan.run_dir.relative_to(plan.run_dir.parents[1])
    env["OUTPUT_DIR"] = str(output_dir)

    model_path = plan.metadata.get("model_checkpoint")
    if model_path_override:
        model_path = model_path_override
    if model_path:
        env["MODEL_PATH"] = str(model_path)

    if plan.metadata.get("model_family"):
        env["MODEL_FAMILY"] = str(plan.metadata.get("model_family"))
    if plan.metadata.get("model_variant"):
        env["MODEL_VARIANT"] = str(plan.metadata.get("model_variant"))
    if plan.metadata.get("model_name"):
        env["MODEL_NAME"] = str(plan.metadata.get("model_name"))

    backend = plan.metadata.get("backend") or plan.metadata.get("model_family")
    if backend:
        env["CAP_BACKEND"] = str(backend)

    tasks = plan.metadata.get("tasks") or []
    if tasks:
        env["CAP_TASKS"] = ",".join(str(t) for t in tasks)
    model_args = plan.metadata.get("model_args") or []
    if model_args:
        env["CAP_MODEL_ARGS"] = ",".join(str(t) for t in model_args)
    eval_args = plan.metadata.get("eval_args") or []
    if eval_args:
        env["CAP_EVAL_ARGS"] = ",".join(str(t) for t in eval_args)

    if plan.metadata.get("batch_size") is not None:
        env["CAP_BATCH_SIZE"] = str(plan.metadata.get("batch_size"))
    if plan.metadata.get("num_fewshot") is not None:
        env["CAP_NUM_FEWSHOT"] = str(plan.metadata.get("num_fewshot"))

    for key, env_key in (
        ("apply_chat_template", "CAP_APPLY_CHAT_TEMPLATE"),
        ("confirm_run_unsafe_code", "CAP_CONFIRM_RUN_UNSAFE_CODE"),
        ("log_samples", "CAP_LOG_SAMPLES"),
        ("allow_code_eval", "CAP_ALLOW_CODE_EVAL"),
    ):
        value = plan.metadata.get(key)
        if value is not None:
            env[env_key] = _bool_to_flag(bool(value))

    if plan.variant == "safe":
        env["SAFETY_ENABLED"] = "1"
        if plan.metadata.get("safety_eta") is not None:
            env["SAFETY_ETA"] = str(plan.metadata.get("safety_eta"))
        if plan.metadata.get("safety_scale") is not None:
            env["SAFETY_SCALE"] = str(plan.metadata.get("safety_scale"))
        if plan.metadata.get("auto_build_unsafe_artifacts") is not None:
            env["SAFETY_AUTO_BUILD_UNSAFE_ARTIFACTS"] = _bool_to_flag(
                bool(plan.metadata.get("auto_build_unsafe_artifacts"))
            )
        if plan.metadata.get("unsafe_artifacts"):
            env["UNSAFE_ARTIFACTS"] = str(plan.metadata.get("unsafe_artifacts"))
        if plan.metadata.get("artifact_root"):
            env["UNSAFE_ARTIFACT_ROOT"] = str(plan.metadata.get("artifact_root"))
        if plan.metadata.get("artifact_name"):
            env["UNSAFE_ARTIFACT_NAME"] = str(plan.metadata.get("artifact_name"))
        if plan.metadata.get("t_start") is not None:
            env["SAFETY_T_START"] = str(plan.metadata.get("t_start"))
        if plan.metadata.get("t_end") is not None:
            env["SAFETY_T_END"] = str(plan.metadata.get("t_end"))
    else:
        env["SAFETY_ENABLED"] = "0"

    return env


def _coerce_env_map(raw_env: Any) -> Dict[str, str]:
    if raw_env is None:
        return {}
    if isinstance(raw_env, DictConfig):
        raw_env = OmegaConf.to_container(raw_env, resolve=True)
    if isinstance(raw_env, dict):
        env: Dict[str, str] = {}
        for key, value in raw_env.items():
            if value in (None, "", "null"):
                continue
            env[str(key)] = str(value)
        return env
    raise SystemExit("run.env must be a mapping when provided.")


def _build_sbatch_command(
    script_path: Path,
    cfg,
    account: Optional[str] = None,
    gpus_per_node: Optional[str] = None,
    cpus_per_task: Optional[str] = None,
    mem: Optional[str] = None,
    time_limit: Optional[str] = None,
    array: Optional[str] = None,
    nodes: Optional[int] = None,
    partition: Optional[str] = None,
) -> list[str]:
    slurm_cfg = cfg.get("slurm", {}) or {}
    cmd = ["sbatch", "--parsable"]
    time_value = time_limit or slurm_cfg.get("time")
    mem_value = mem or slurm_cfg.get("mem")
    gpus_value = gpus_per_node or slurm_cfg.get("gpus_per_node")
    cpus_value = cpus_per_task or slurm_cfg.get("cpus_per_task")
    account_value = account or slurm_cfg.get("account") or DEFAULT_SLURM_ACCOUNT
    array_value = array or slurm_cfg.get("array")
    partition_value = partition or slurm_cfg.get("partition")
    if time_value:
        cmd.extend(["--time", str(time_value)])
    if mem_value:
        cmd.extend(["--mem", str(mem_value)])
    if gpus_value:
        cmd.extend(["--gpus-per-node", str(gpus_value)])
    if cpus_value:
        cmd.extend(["--cpus-per-task", str(cpus_value)])
    if account_value:
        cmd.extend(["--account", str(account_value)])
    if array_value:
        cmd.extend(["--array", str(array_value)])
    if nodes is not None:
        cmd.extend(["--nodes", str(nodes)])
    if partition_value:
        cmd.extend(["--partition", str(partition_value)])
    cmd.append(str(script_path))
    return cmd


def _chunk_list(items: Sequence[Any], chunk_size: int) -> Iterable[List[Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for idx in range(0, len(items), chunk_size):
        yield list(items[idx : idx + chunk_size])


def _parse_task_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _split_mmlu_subject_jobs(
    env: Dict[str, str], chunk_size: int
) -> List[Dict[str, str]]:
    base_output = Path(env["OUTPUT_DIR"])
    jobs: List[Dict[str, str]] = []
    for idx, chunk in enumerate(_chunk_list(MMLU_SUBJECT_TASKS, chunk_size)):
        split_env = dict(env)
        split_env["CAP_TASKS"] = ",".join(chunk)
        if chunk_size == 1:
            split_env["OUTPUT_DIR"] = str(base_output / chunk[0])
        else:
            split_env["OUTPUT_DIR"] = str(base_output / f"mmlu_subjects_{idx:02d}")
        jobs.append(split_env)
    return jobs


def _submit_sbatch(
    cmd: Sequence[str],
    env: Dict[str, str],
    dry_run: bool,
    integration_test: bool,
    confirm_integration: bool = True,
) -> str:
    printable = shlex.join(cmd)
    print(f"[sbatch] {printable}")
    if dry_run:
        return "dry-run"
    if integration_test:
        env.setdefault("CAP_TASKS", "mmlu")
        env.setdefault("CAP_BATCH_SIZE", "1")
        if confirm_integration:
            try:
                user_input = input("Proceed with integration-test run? [y/N]: ").strip().lower()
            except EOFError:
                user_input = "n"
            if user_input not in {"y", "yes"}:
                print("[integration-test] aborted.")
                return "integration-abort"
        local_cmd = ["bash", str(cmd[-1])]
        print(f"[integration-test] running: {shlex.join(local_cmd)}")
        proc: Optional[subprocess.Popen[str]] = None

        def _terminate_process_group(p: Optional[subprocess.Popen[str]]) -> None:
            if not p or p.poll() is not None:
                return
            try:
                os.killpg(p.pid, signal.SIGINT)
            except Exception:
                pass
            try:
                p.wait(timeout=2)
                return
            except Exception:
                pass
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:
                pass
            try:
                p.wait(timeout=2)
                return
            except Exception:
                pass
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass

        def _signal_handler(signum, _frame) -> None:
            print(f"\n[integration-test] received signal {signum}; terminating child process.")
            _terminate_process_group(proc)
            raise SystemExit(128 + signum)

        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
        try:
            proc = subprocess.Popen(local_cmd, env=env, preexec_fn=os.setsid)
            proc.wait()
        except KeyboardInterrupt:
            print("\n[integration-test] interrupted; stopping further runs.")
            _terminate_process_group(proc)
            raise SystemExit(130)
        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)
        return "local-run"
    result = subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE, text=True)
    job_id = result.stdout.strip()
    print(f"[sbatch] job_id={job_id}")
    return job_id


def _normalize_job_id(job_id: str) -> str:
    return job_id.split(";", 1)[0].strip()


def _cancel_jobs(job_ids: Sequence[str]) -> None:
    normalized = [_normalize_job_id(job_id) for job_id in job_ids if job_id]
    if not normalized:
        return
    print(f"[submit] cancelling {len(normalized)} job(s): {', '.join(normalized)}")
    subprocess.run(["scancel", *normalized], check=False)


def _summary_dir(plans: Sequence[CapabilityPlan], results_root: Optional[Path], timestamp: str) -> Path:
    if results_root is not None:
        return results_root / f"capability_summary_{timestamp}"
    if not plans:
        return Path(".") / f"capability_summary_{timestamp}"
    # run_dir = output_root/slug/run_id, so parent.parent is output_root
    return plans[0].run_dir.parent.parent / f"capability_summary_{timestamp}"


def _write_summary(
    plans: Sequence[CapabilityPlan],
    summary_dir: Path,
    cfg_path: Path,
    timestamp: str,
) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "summary.json"
    plans_path = summary_dir / "plans.jsonl"
    summary = {
        "timestamp": timestamp,
        "config_path": str(cfg_path),
        "total_runs": len(plans),
        "runs": [],
    }
    for plan in plans:
        meta = plan.metadata
        summary["runs"].append(
            {
                "dataset": plan.dataset,
                "label": plan.label,
                "variant": plan.variant,
                "run_id": plan.run_id,
                "run_dir": str(plan.run_dir),
                "backend": meta.get("backend") or meta.get("model_family"),
                "tasks": meta.get("tasks"),
                "model_name": meta.get("model_name"),
                "model_checkpoint": meta.get("model_checkpoint"),
                "safety_enabled": meta.get("safety_enabled", False),
                "safety_eta": meta.get("safety_eta"),
                "t_start": meta.get("t_start"),
                "t_end": meta.get("t_end"),
                "auto_build_unsafe_artifacts": meta.get("auto_build_unsafe_artifacts"),
                "artifact_root": meta.get("artifact_root"),
                "artifact_name": meta.get("artifact_name"),
                "unsafe_artifacts": meta.get("unsafe_artifacts"),
            }
        )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with plans_path.open("w", encoding="utf-8") as fh:
        for plan in summary["runs"]:
            fh.write(json.dumps(plan) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit capability eval jobs to slurm.")
    parser.add_argument("--config", required=True, help="Path to the sbatch config yaml.")
    parser.add_argument("--repo-root", required=True, help="Path to repo root containing src/.")
    parser.add_argument("--only", nargs="*", default=None, help="Restrict to dataset names.")
    parser.add_argument("--results-root", default=None, help="Override output root for run dirs.")
    parser.add_argument("--model-path", default=None, help="Override model checkpoint path.")
    parser.add_argument("--account", default=None, help="Slurm account to charge (e.g., rrg-<your-PI>_gpu).")
    parser.add_argument("--gpus-per-node", default=None, help="Override gpus-per-node (e.g., a100:1).")
    parser.add_argument("--cpus-per-task", default=None, help="Override cpus-per-task.")
    parser.add_argument("--mem", default=None, help="Override memory (e.g., 32G).")
    parser.add_argument("--time", dest="time_limit", default=None, help="Override time (e.g., 04:00:00).")
    parser.add_argument("--array", default=None, help="Override Slurm array range.")
    parser.add_argument("--baseline-array", default="0-0", help="Default baseline array range.")
    parser.add_argument("--safe-array", default="0-0", help="Default safe array range.")
    parser.add_argument("--baseline-time", default="04:00:00", help="Default baseline time limit.")
    parser.add_argument("--safe-time", default="04:00:00", help="Default safe time limit.")
    parser.add_argument("--gen-batch-size", type=int, default=1, help="Batch runs into CONFIG_BATCH_FILE groups.")
    parser.add_argument("--nodes", type=int, default=None, help="Override number of nodes.")
    parser.add_argument("--partition", default=None, help="Override Slurm partition.")
    parser.add_argument(
        "--resume-from-timestamp",
        default=None,
        help="Reuse an existing run timestamp for output dirs (e.g., 20240201235959).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--integration-test", action="store_true")
    parser.add_argument(
        "--split-mmlu",
        action="store_true",
        help="Submit MMLU as multiple jobs (mmlu_stem, mmlu_social_sciences, mmlu_humanities, mmlu_other).",
    )
    parser.add_argument(
        "--split-mmlu-subjects",
        action="store_true",
        help="Submit MMLU as multiple subject jobs (one or more subjects per job).",
    )
    parser.add_argument(
        "--mmlu-subject-batch-size",
        type=int,
        default=1,
        help="Number of MMLU subjects to include per job when using --split-mmlu-subjects.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = OmegaConf.load(cfg_path)
    repo_root = Path(args.repo_root).resolve()
    run_cfg = cfg.run
    if not run_cfg.get("output_root"):
        raise SystemExit("run.output_root must be set in the config.")
    script_name = run_cfg.get("script", "eval_capability.sh")
    script_path = repo_root / "src" / "slurm" / script_name
    if not script_path.exists():
        raise SystemExit(f"Script not found: {script_path}")

    timestamp = (
        str(args.resume_from_timestamp)
        if args.resume_from_timestamp
        else datetime.now().strftime("%Y%m%d%H%M%S")
    )
    results_root = Path(args.results_root).resolve() if args.results_root else None
    spec_root = (results_root or repo_root) / ".slurm_specs"

    plans = build_capability_plans(
        cfg_path=cfg_path,
        restrict_to=args.only,
        timestamp_override=timestamp,
    )
    summary_dir = _summary_dir(plans, results_root, timestamp)
    _write_summary(plans, summary_dir, cfg_path, timestamp)
    print(f"[summary] capability summary folder: {summary_dir}")

    jobs: List[Tuple[CapabilityPlan, Dict[str, str], str, str, str]] = []
    for plan in plans:
        is_safe = plan.variant == "safe"
        mode = "safe" if is_safe else "baseline"
        array_default = args.safe_array if is_safe else args.baseline_array
        time_default = args.safe_time if is_safe else args.baseline_time
        array_range, wall_time, mem = _determine_slurm_params(
            plan,
            cfg=cfg,
            mode=mode,
            default_array=array_default,
            default_time=time_default,
        )
        if args.array:
            array_range = args.array
        if args.time_limit:
            wall_time = args.time_limit
        if args.mem:
            mem = args.mem
        env = _build_env(
            plan=plan,
            repo_root=repo_root,
            results_root=results_root,
            model_path_override=args.model_path,
        )
        run_env = _coerce_env_map(run_cfg.get("env"))
        for key, value in run_env.items():
            env.setdefault(key, value)
        tasks = _parse_task_list(env.get("CAP_TASKS"))
        if args.split_mmlu_subjects and tasks == ["mmlu"]:
            for split_env in _split_mmlu_subject_jobs(
                env, chunk_size=max(1, args.mmlu_subject_batch_size)
            ):
                jobs.append((plan, split_env, array_range, wall_time, mem))
        elif args.split_mmlu and tasks == ["mmlu"]:
            base_output = Path(env["OUTPUT_DIR"])
            for group in MMLU_GROUP_TASKS:
                split_env = dict(env)
                split_env["CAP_TASKS"] = group
                split_env["OUTPUT_DIR"] = str(base_output / group)
                jobs.append((plan, split_env, array_range, wall_time, mem))
        else:
            jobs.append((plan, env, array_range, wall_time, mem))

    print("\n[summary] planned submissions:")
    print(f"  total jobs: {len(jobs)}")
    if jobs:
        task_counts: Dict[str, int] = {}
        for _, env, _, _, _ in jobs:
            tasks = _parse_task_list(env.get("CAP_TASKS"))
            if not tasks:
                task_counts["<none>"] = task_counts.get("<none>", 0) + 1
            else:
                for task in tasks:
                    task_counts[task] = task_counts.get(task, 0) + 1
        print("  jobs by task:")
        for task in sorted(task_counts):
            print(f"    {task}: {task_counts[task]}")
    if jobs:
        print("  env preview (up to 3 jobs):")
        for plan, env, _, _, _ in jobs[:3]:
            print(f"    {plan.run_id}: {_env_to_spec(env)}")

    try:
        user_input = input("Proceed with sbatch submissions? [y/N]: ").strip().lower()
    except EOFError:
        user_input = "n"
    if user_input not in {"y", "yes"}:
        print("[submit] aborted.")
        return

    submitted_job_ids: List[str] = []
    try:
        if args.gen_batch_size <= 1:
            for plan, env, array_range, wall_time, mem in jobs:
                cmd = _build_sbatch_command(
                    script_path,
                    cfg,
                    account=args.account,
                    gpus_per_node=args.gpus_per_node,
                    cpus_per_task=args.cpus_per_task,
                    mem=mem,
                    time_limit=wall_time,
                    array=array_range,
                    nodes=args.nodes,
                    partition=args.partition,
                )
                print("[info] sbatch command:\n", " ".join(shlex.quote(c) for c in cmd))
                print("  env preview:")
                print(f"    {_env_to_spec(env)}")
                job_id = _submit_sbatch(
                    cmd,
                    env,
                    args.dry_run,
                    args.integration_test,
                    confirm_integration=False if args.integration_test else True,
                )
                if job_id not in {"dry-run", "integration-abort", "local-run"}:
                    submitted_job_ids.append(job_id)
        else:
            if jobs:
                spec_root.mkdir(parents=True, exist_ok=True)
            grouped: Dict[Tuple[str, str, str, str, str], List[Tuple[CapabilityPlan, Dict[str, str], str, str, str]]] = {}
            gpus_per_node = args.gpus_per_node or (cfg.get("slurm", {}) or {}).get("gpus_per_node")
            for item in jobs:
                plan, env, array_range, wall_time, mem = item
                variant_mode = "safe" if plan.variant == "safe" else "baseline"
                key = (variant_mode, array_range, wall_time, mem, str(gpus_per_node or ""))
                grouped.setdefault(key, []).append(item)

            batch_idx = 0
            for key, batch_jobs in grouped.items():
                variant_mode, array_range, wall_time, mem, gpus_value = key
                for batch in _chunk_list(batch_jobs, args.gen_batch_size):
                    batch_file = spec_root / f"capability_gen_batch_{variant_mode}_{batch_idx}.envlist"
                    with batch_file.open("w") as f:
                        for plan, env, _, _, _ in batch:
                            f.write(_env_to_spec(env) + "\n")
                    batch_env = _env_copy()
                    batch_env["CONFIG_BATCH_FILE"] = str(batch_file)
                    cmd = _build_sbatch_command(
                        script_path,
                        cfg,
                        account=args.account,
                        gpus_per_node=gpus_value,
                        cpus_per_task=args.cpus_per_task,
                        mem=mem,
                        time_limit=wall_time,
                        array=array_range,
                        nodes=args.nodes,
                        partition=args.partition,
                    )
                    print("[info] sbatch command:\n", " ".join(shlex.quote(c) for c in cmd))
                    print("  env preview:")
                    print(f"    CONFIG_BATCH_FILE={batch_env['CONFIG_BATCH_FILE']}")
                    job_id = _submit_sbatch(
                        cmd,
                        batch_env,
                        args.dry_run,
                        args.integration_test,
                        confirm_integration=False if args.integration_test else True,
                    )
                    if job_id not in {"dry-run", "integration-abort", "local-run"}:
                        submitted_job_ids.append(job_id)
                    batch_idx += 1
    except KeyboardInterrupt:
        print("\n[submit] interrupted; stopping remaining submissions.")
        _cancel_jobs(submitted_job_ids)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
