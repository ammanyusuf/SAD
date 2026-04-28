"""
submit.py — Simple Slurm submitter for memorization extraction jobs.

Reads memorization/config.yaml, expands experiments x shards, and submits
one sbatch job per (experiment, shard).  Optionally chains evaluate.py jobs
with --dependency=afterok.

Usage:
  python memorization/submit.py [--config memorization/config.yaml] [--dry-run]
  python memorization/submit.py --evaluate --decode-tokenizer /path/to/tok
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
  import yaml
except ImportError:
  print("[ERROR] PyYAML is required: pip install pyyaml", file=sys.stderr)
  raise


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def _expand_vars(value: Any) -> Any:
  """Recursively expand environment variables in string values."""
  if isinstance(value, str):
    return os.path.expandvars(value)
  if isinstance(value, dict):
    return {k: _expand_vars(v) for k, v in value.items()}
  if isinstance(value, list):
    return [_expand_vars(v) for v in value]
  return value


def load_config(path: str) -> Dict[str, Any]:
  with open(path, encoding="utf-8") as fh:
    raw = yaml.safe_load(fh)
  return _expand_vars(raw)


# ─────────────────────────────────────────────────────────────────────────────
# sbatch submission
# ─────────────────────────────────────────────────────────────────────────────

def _sbatch_script_path() -> str:
  return str(Path(__file__).parent / "sbatch_extract.sh")


def build_extract_env(
  exp: Dict[str, Any],
  shard_id: int,
  output_dir: str,
) -> Dict[str, str]:
  """Build the --export env dict for one (experiment, shard) job."""
  output_jsonl = os.path.join(
    output_dir,
    f"{exp['model']}_shard{shard_id:02d}.jsonl",
  )
  env: Dict[str, str] = {
    "MEM_MODEL":          exp.get("model", "llada-instruct"),
    "MEM_CHECKPOINT":     exp["checkpoint"],
    "MEM_TOKENIZER":      exp.get("tokenizer", exp["checkpoint"]),
    "MEM_PROMPTS_JSON":   exp["prompts_json"],
    "MEM_OUTPUT_JSONL":   output_jsonl,
    "MEM_STEPS":          str(exp.get("steps", 128)),
    "MEM_MAX_NEW_TOKENS": str(exp.get("max_new_tokens", 256)),
    "MEM_BATCH_SIZE":     str(exp.get("batch_size", 4)),
    "MEM_TEMPERATURE":    str(exp.get("temperature", 0.0)),
    "MEM_PRECISION":      exp.get("precision", "bf16"),
    "MEM_SEED":           str(exp.get("seed", 1)),
    "MEM_SHARD_ID":       str(shard_id),
    "MEM_NUM_SHARDS":     str(exp.get("num_shards", 1)),
  }
  if exp.get("prompt_limit") is not None:
    env["MEM_PROMPT_LIMIT"] = str(exp["prompt_limit"])
  return env


def build_evaluate_env(
  exp: Dict[str, Any],
  shard_id: int,
  output_dir: str,
  signals: str,
  decode_tokenizer: Optional[str],
  llamaguard_model: Optional[str],
  perplexity_model: str,
) -> Dict[str, str]:
  extract_jsonl = os.path.join(output_dir, f"{exp['model']}_shard{shard_id:02d}.jsonl")
  eval_jsonl = os.path.join(output_dir, f"{exp['model']}_shard{shard_id:02d}_eval.jsonl")
  env: Dict[str, str] = {
    "MEM_EVAL_INPUT_JSONL":   extract_jsonl,
    "MEM_EVAL_OUTPUT_JSONL":  eval_jsonl,
    "MEM_EVAL_SIGNALS":       signals,
    "MEM_EVAL_PPL_MODEL":     perplexity_model,
  }
  if decode_tokenizer:
    env["MEM_EVAL_DECODE_TOKENIZER"] = decode_tokenizer
  if llamaguard_model:
    env["MEM_EVAL_LLAMAGUARD_MODEL"] = llamaguard_model
  return env


def _format_export(env: Dict[str, str]) -> str:
  """Build the --export=ALL,K=V,... string for sbatch."""
  pairs = ",".join(f"{k}={v}" for k, v in env.items())
  return f"ALL,{pairs}"


def submit_job(
  script: str,
  job_name: str,
  slurm_cfg: Dict[str, Any],
  export_env: Dict[str, str],
  dependency: Optional[str] = None,
  dry_run: bool = False,
) -> Optional[str]:
  """Submit an sbatch job; return the job ID string (or None on dry run)."""
  log_dir = slurm_cfg.get("log_dir", "/tmp/slurm_logs")
  os.makedirs(log_dir, exist_ok=True)

  cmd = [
    "sbatch",
    f"--job-name={job_name}",
    f"--account={slurm_cfg.get('account', 'aip-mijungp')}",
    f"--time={slurm_cfg.get('time', '02:00:00')}",
    f"--mem={slurm_cfg.get('mem', '32G')}",
    f"--gpus-per-node={slurm_cfg.get('gpus', 'l40s:1')}",
    f"--cpus-per-task={slurm_cfg.get('cpus_per_task', 1)}",
    f"--output={log_dir}/{job_name}_%j.out",
    f"--error={log_dir}/{job_name}_%j.err",
    f"--export={_format_export(export_env)}",
  ]
  if dependency:
    cmd.append(f"--dependency={dependency}")
  cmd.append(script)

  if dry_run:
    print("[DRY RUN]", " ".join(cmd))
    return None

  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
    print(f"[ERROR] sbatch failed:\n{result.stderr}", file=sys.stderr)
    sys.exit(1)

  job_id = result.stdout.strip().split()[-1]
  print(f"[INFO] Submitted {job_name} -> job_id={job_id}")
  return job_id


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate sbatch (inline script — no separate template needed)
# ─────────────────────────────────────────────────────────────────────────────

_EVALUATE_INLINE_SCRIPT = """\
#!/bin/bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$HOME/repos/safe-text-diffusion}"
EXTERNAL_VENV_ACTIVATE="${EXTERNAL_VENV_ACTIVATE:-$HOME/repos/safe-text-diffusion/.env-gpu/bin/activate}"
if command -v module >/dev/null 2>&1; then
  module purge
  module load StdEnv/2023 cuda/12.2 python/3.11 gcc arrow/21.0.0 scipy-stack || true
fi
if [[ -f "${EXTERNAL_VENV_ACTIVATE}" ]]; then
  source "${EXTERNAL_VENV_ACTIVATE}"
fi
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}/src/third_party/mdlm:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
DECODE_TOK_ARG=()
if [[ -n "${MEM_EVAL_DECODE_TOKENIZER:-}" ]]; then
  DECODE_TOK_ARG=(--decode-tokenizer "${MEM_EVAL_DECODE_TOKENIZER}")
fi
LG_ARG=()
if [[ -n "${MEM_EVAL_LLAMAGUARD_MODEL:-}" ]]; then
  LG_ARG=(--llamaguard-model "${MEM_EVAL_LLAMAGUARD_MODEL}")
fi
python memorization/evaluate.py \
  --input-jsonl  "${MEM_EVAL_INPUT_JSONL}" \
  --output-jsonl "${MEM_EVAL_OUTPUT_JSONL}" \
  --signals      "${MEM_EVAL_SIGNALS:-mask_frac,perplexity}" \
  --perplexity-model "${MEM_EVAL_PPL_MODEL:-gpt2-large}" \
  "${DECODE_TOK_ARG[@]}" \
  "${LG_ARG[@]}"
echo "[INFO] Evaluation complete: ${MEM_EVAL_OUTPUT_JSONL}"
"""


def _write_evaluate_script(tmp_dir: str) -> str:
  """Write the inline evaluate script to a temp file; return its path."""
  script_path = os.path.join(tmp_dir, "sbatch_evaluate_inline.sh")
  with open(script_path, "w") as fh:
    fh.write(_EVALUATE_INLINE_SCRIPT)
  os.chmod(script_path, 0o755)
  return script_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
  p = argparse.ArgumentParser(
    description="Submit memorization extraction (and optional evaluation) jobs to Slurm.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  p.add_argument(
    "--config",
    default=str(Path(__file__).parent / "config.yaml"),
    help="Path to config.yaml.",
  )
  p.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting.")
  p.add_argument("--evaluate", action="store_true",
                 help="Also submit evaluate.py jobs chained via --dependency=afterok.")
  p.add_argument("--signals", default="mask_frac,perplexity",
                 help="Comma-separated evaluation signals (used only with --evaluate).")
  p.add_argument("--decode-tokenizer", default=None,
                 help="Path to tokenizer for decoding in evaluate.py.")
  p.add_argument("--llamaguard-model", default=None,
                 help="Path to LlamaGuard model (required if 'llamaguard' in --signals).")
  p.add_argument("--perplexity-model", default="gpt2-large",
                 help="HuggingFace model name for perplexity evaluation.")
  p.add_argument("--experiment-filter", default=None,
                 help="Only submit experiments whose 'model' tag matches this string.")
  return p.parse_args(argv)


def main(argv=None) -> None:
  args = parse_args(argv)
  cfg = load_config(args.config)

  output_dir: str = cfg.get("output_dir", os.path.expandvars("${RESULTS_ROOT:-/tmp}/memorization"))
  slurm_cfg: Dict[str, Any] = cfg.get("slurm", {})
  experiments: List[Dict[str, Any]] = cfg.get("experiments", [])

  if not experiments:
    print("[ERROR] No experiments defined in config.yaml.", file=sys.stderr)
    sys.exit(1)

  extract_script = _sbatch_script_path()
  import tempfile
  tmp_dir = tempfile.mkdtemp(prefix="mem_submit_")
  eval_script = _write_evaluate_script(tmp_dir) if args.evaluate else None

  n_jobs = 0
  for exp in experiments:
    model_tag = exp.get("model", "llada-instruct")
    if args.experiment_filter and args.experiment_filter not in model_tag:
      continue

    num_shards = int(exp.get("num_shards", 1))
    exp_output_dir = os.path.join(output_dir, model_tag)
    os.makedirs(exp_output_dir, exist_ok=True)

    for shard_id in range(num_shards):
      job_name = f"mem_{model_tag}_s{shard_id:02d}"

      extract_env = build_extract_env(exp, shard_id, exp_output_dir)
      extract_job_id = submit_job(
        script=extract_script,
        job_name=job_name,
        slurm_cfg=slurm_cfg,
        export_env=extract_env,
        dry_run=args.dry_run,
      )
      n_jobs += 1

      if args.evaluate and eval_script is not None:
        eval_env = build_evaluate_env(
          exp=exp,
          shard_id=shard_id,
          output_dir=exp_output_dir,
          signals=args.signals,
          decode_tokenizer=args.decode_tokenizer,
          llamaguard_model=args.llamaguard_model,
          perplexity_model=args.perplexity_model,
        )
        dep = f"afterok:{extract_job_id}" if extract_job_id else None
        eval_job_name = f"mem_eval_{model_tag}_s{shard_id:02d}"
        # Evaluate jobs use the same Slurm resources but no GPU needed for mask_frac only.
        eval_slurm = dict(slurm_cfg)
        if "llamaguard" not in args.signals and "perplexity" not in args.signals:
          eval_slurm["gpus"] = "0"
        submit_job(
          script=eval_script,
          job_name=eval_job_name,
          slurm_cfg=eval_slurm,
          export_env=eval_env,
          dependency=dep,
          dry_run=args.dry_run,
        )
        n_jobs += 1

  suffix = " (dry run)" if args.dry_run else ""
  print(f"[INFO] Submitted {n_jobs} job(s){suffix}.")


if __name__ == "__main__":
  main()
