#!/bin/bash
# memorization/sbatch_extract.sh
# Sbatch template for denoising-trajectory extraction jobs.
#
# Environment variables (set by submit.py via --export):
#   REPO_ROOT, RESULTS_ROOT, HF_MODELS_CACHE, HF_HOME, HF_DATASETS_CACHE
#   JAILBREAK_DATA_ROOT
#   MEM_CHECKPOINT, MEM_TOKENIZER, MEM_PROMPTS_JSON, MEM_OUTPUT_JSONL
#   MEM_STEPS, MEM_MAX_NEW_TOKENS, MEM_BATCH_SIZE, MEM_TEMPERATURE
#   MEM_PRECISION, MEM_PROMPT_LIMIT, MEM_SHARD_ID, MEM_NUM_SHARDS
#   MEM_SEED, MEM_MODEL, MEM_NO_CHAT_TEMPLATE
#   EXTERNAL_VENV_ACTIVATE
#
# SBATCH directives are set dynamically by submit.py via --job-name,
# --account, --time, --mem, --gpus-per-node, --output, --error.
# The #SBATCH lines below are defaults overridden by submit.py.
#SBATCH --job-name=mem_extract
#SBATCH --account=aip-mijungp
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=32G
#SBATCH --mail-user=ammany01@cs.ubc.ca
#SBATCH --mail-type=FAIL,TIME_LIMIT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"

echo "[INFO] Job ${SLURM_JOB_ID:-local} starting on $(hostname)"
echo "[INFO] REPO_ROOT=${REPO_ROOT}"

# ── Stage repo to SLURM_TMPDIR ──────────────────────────────────────────────
SLURM_TMPDIR="${SLURM_TMPDIR:-/tmp/mem_extract_$$}"
mkdir -p "${SLURM_TMPDIR}"
TMP_REPO="${SLURM_TMPDIR}/repo"
echo "[INFO] Staging repo to ${TMP_REPO}"
rsync -a \
  --exclude=".git" \
  --exclude=".env" \
  --exclude=".env-gpu" \
  --exclude=".env-jailbreak" \
  --exclude="*.pyc" \
  --exclude="__pycache__" \
  "${REPO_ROOT}/" "${TMP_REPO}/"

# ── HF caches ────────────────────────────────────────────────────────────────
SRC_HF_HOME="${HF_HOME:-$HOME/scratch/hf_home}"
SRC_HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/scratch/hf_datasets}"
SRC_HF_MODELS_CACHE="${HF_MODELS_CACHE:-$HOME/scratch/hf_models}"

export HF_HOME="${SLURM_TMPDIR}/hf_home"
export HF_DATASETS_CACHE="${SLURM_TMPDIR}/hf_datasets"
export HF_MODELS_CACHE="${SLURM_TMPDIR}/hf_models"
export TRANSFORMERS_CACHE="${HF_MODELS_CACHE}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_MODELS_CACHE}"
rsync -a "${SRC_HF_HOME}/"         "${HF_HOME}/"         || true
rsync -a "${SRC_HF_DATASETS_CACHE}/" "${HF_DATASETS_CACHE}/" || true
echo "[INFO] HF caches staged."

# ── Python environment ───────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
  module purge
  module load StdEnv/2023 cuda/12.2 python/3.11 gcc arrow/21.0.0 scipy-stack || true
fi

EXTERNAL_VENV_ACTIVATE="${EXTERNAL_VENV_ACTIVATE:-$HOME/repos/safe-text-diffusion/.env-gpu/bin/activate}"
if [[ -f "${EXTERNAL_VENV_ACTIVATE}" ]]; then
  echo "[INFO] Activating venv: ${EXTERNAL_VENV_ACTIVATE}"
  # shellcheck disable=SC1090
  source "${EXTERNAL_VENV_ACTIVATE}"
else
  echo "[WARN] Venv not found at ${EXTERNAL_VENV_ACTIVATE}; using system Python." >&2
fi

export PYTHONPATH="${TMP_REPO}:${TMP_REPO}/src:${TMP_REPO}/src/third_party/mdlm:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "[INFO] Python: $(python --version)"
echo "[INFO] PYTHONPATH: ${PYTHONPATH}"

# ── Resolve output path ───────────────────────────────────────────────────────
RESULTS_ROOT="${RESULTS_ROOT:-${SCRATCH:-$HOME}/results}"
MEM_OUTPUT_JSONL="${MEM_OUTPUT_JSONL:-${RESULTS_ROOT}/memorization/shard_${MEM_SHARD_ID:-0}.jsonl}"
mkdir -p "$(dirname "${MEM_OUTPUT_JSONL}")"

TMP_OUTPUT="${SLURM_TMPDIR}/outputs/output_shard_${MEM_SHARD_ID:-0}.jsonl"
mkdir -p "${SLURM_TMPDIR}/outputs"

# ── Resolve model path ────────────────────────────────────────────────────────
# Prefer staged model in SLURM_TMPDIR; fall back to original cache path.
_resolve_model_path() {
  local src="$1"
  if [[ -d "${src}" ]]; then
    echo "${src}"
  elif [[ -d "${HF_MODELS_CACHE}/${src}" ]]; then
    echo "${HF_MODELS_CACHE}/${src}"
  elif [[ -d "${SRC_HF_MODELS_CACHE}/${src}" ]]; then
    echo "${SRC_HF_MODELS_CACHE}/${src}"
  else
    echo "${src}"
  fi
}

RESOLVED_CHECKPOINT="$(_resolve_model_path "${MEM_CHECKPOINT:-}")"
RESOLVED_TOKENIZER="$(_resolve_model_path "${MEM_TOKENIZER:-${MEM_CHECKPOINT:-}}")"
echo "[INFO] Checkpoint: ${RESOLVED_CHECKPOINT}"
echo "[INFO] Tokenizer:  ${RESOLVED_TOKENIZER}"

# ── Build extract.py arguments ────────────────────────────────────────────────
EXTRACT_ARGS=(
  --model       "${MEM_MODEL:-llada-instruct}"
  --checkpoint  "${RESOLVED_CHECKPOINT}"
  --tokenizer   "${RESOLVED_TOKENIZER}"
  --prompts-json "${MEM_PROMPTS_JSON}"
  --output-jsonl "${TMP_OUTPUT}"
  --steps       "${MEM_STEPS:-128}"
  --max-new-tokens "${MEM_MAX_NEW_TOKENS:-256}"
  --batch-size  "${MEM_BATCH_SIZE:-4}"
  --temperature "${MEM_TEMPERATURE:-0.0}"
  --precision   "${MEM_PRECISION:-bf16}"
  --seed        "${MEM_SEED:-42}"
  --shard-id    "${MEM_SHARD_ID:-0}"
  --num-shards  "${MEM_NUM_SHARDS:-1}"
)
if [[ -n "${MEM_PROMPT_LIMIT:-}" ]]; then
  EXTRACT_ARGS+=(--prompt-limit "${MEM_PROMPT_LIMIT}")
fi
if [[ "${MEM_NO_CHAT_TEMPLATE:-0}" == "1" ]]; then
  EXTRACT_ARGS+=(--no-chat-template)
fi

echo "[INFO] Running extraction..."
(
  set -x
  cd "${TMP_REPO}"
  python memorization/extract.py "${EXTRACT_ARGS[@]}"
)

echo "[INFO] Extraction complete. Staging output to ${MEM_OUTPUT_JSONL}"
rsync -a "${TMP_OUTPUT}" "${MEM_OUTPUT_JSONL}"
echo "[INFO] Done. Output at ${MEM_OUTPUT_JSONL}"
