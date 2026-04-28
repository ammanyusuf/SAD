#!/bin/bash
#SBATCH --job-name=wiki_mem
#SBATCH --account=rrg-<your-PI>   # [Compute Canada] replace with your allocation
#SBATCH --time=0-06:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mem=32G
#SBATCH --mail-user=${USER}@<your-institution.ca>  # [Compute Canada] set to your email
#SBATCH --mail-type=FAIL,TIME_LIMIT,END
#SBATCH --output=/scratch/%u/logs/safe-text-diffusion/%x_%j.out
#SBATCH --error=/scratch/%u/logs/safe-text-diffusion/%x_%j.err

# ---------------------------------------------------------------------------
# Measure verbatim memorization of WikiText-103 sequences in MDLM vs GPT-2.
#
# Submit from the repo root:
#   source scripts/env_profile.sh dlm-memorization
#   sbatch src/slurm/measure_wiki_memorization.sh
#
# Optional overrides (export before sbatch):
#   export WIKI_MODEL_NAME=mdlm-wiki      # "mdlm-wiki", "gpt2-large", or unset for all
#   export WIKI_MASKING=both              # "random", "contiguous", or "both"
#   export WIKI_SUFFIX_TOKENS=3           # suffix length for contiguous mode (default 3)
#   export WIKI_MASK_RATIO=0.2            # mask ratio for random mode (default 0.2)
#   export WIKI_N_SAMPLES=1000            # number of documents to evaluate
#   export WIKI_R=512                     # MC trajectories per sample
#   export WIKI_DEBUG=0                   # set to 1 for 20-example smoke test
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$HOME/scratch/models/text-diffusion/702-1250000.ckpt}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$HOME/scratch/hf_models/gpt2-large}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${REPO_ROOT}/src/third_party/mdlm/configs/config.yaml}"
MEMORIZATION_DATA_DIR="${MEMORIZATION_DATA_DIR:-$HOME/scratch/data/memorization}"
MEMORIZATION_RESULTS_DIR="${MEMORIZATION_RESULTS_DIR:-$HOME/scratch/results/memorization}"
HF_MODELS_CACHE="${HF_MODELS_CACHE:-$HOME/scratch/hf_models}"

# Experiment knobs
WIKI_MODEL_NAME="${WIKI_MODEL_NAME:-}"             # empty = run all models in config
WIKI_MASKING="${WIKI_MASKING:-both}"
WIKI_SUFFIX_TOKENS="${WIKI_SUFFIX_TOKENS:-3}"
WIKI_MASK_RATIO="${WIKI_MASK_RATIO:-0.2}"
WIKI_N_SAMPLES="${WIKI_N_SAMPLES:-1000}"
WIKI_R="${WIKI_R:-512}"
WIKI_DEBUG="${WIKI_DEBUG:-0}"

OUT_DIR="${MEMORIZATION_RESULTS_DIR}/wiki_memorization/${SLURM_JOB_ID}"

# ---------------------------------------------------------------------------
# Validate required inputs
# ---------------------------------------------------------------------------
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] CHECKPOINT_PATH not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
mkdir -p "$(dirname "/scratch/%u/logs/safe-text-diffusion/x")"

echo "[INFO] Job ${SLURM_JOB_ID} starting on $(hostname)"
echo "[INFO] REPO_ROOT:              ${REPO_ROOT}"
echo "[INFO] CHECKPOINT_PATH:        ${CHECKPOINT_PATH}"
echo "[INFO] WIKI_MODEL_NAME:        ${WIKI_MODEL_NAME:-<all>}"
echo "[INFO] WIKI_MASKING:           ${WIKI_MASKING}"
echo "[INFO] WIKI_SUFFIX_TOKENS:     ${WIKI_SUFFIX_TOKENS}"
echo "[INFO] WIKI_MASK_RATIO:        ${WIKI_MASK_RATIO}"
echo "[INFO] WIKI_N_SAMPLES:         ${WIKI_N_SAMPLES}"
echo "[INFO] WIKI_R:                 ${WIKI_R}"
echo "[INFO] WIKI_DEBUG:             ${WIKI_DEBUG}"
echo "[INFO] OUT_DIR:                ${OUT_DIR}"

# ---------------------------------------------------------------------------
# Stage repo into SLURM_TMPDIR (fast local SSD)
# ---------------------------------------------------------------------------
TMP_REPO="${SLURM_TMPDIR}/repo"
mkdir -p "${TMP_REPO}"

echo "[INFO] Staging repo → ${TMP_REPO}"
rsync -a \
  --exclude=".git" \
  --exclude=".env*" \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  "${REPO_ROOT}/" "${TMP_REPO}/"

# Stage HF caches offline
export HF_HOME="${SLURM_TMPDIR}/hf_home"
export HF_DATASETS_CACHE="${SLURM_TMPDIR}/hf_datasets"
export HF_MODELS_CACHE="${SLURM_TMPDIR}/hf_models"
export TRANSFORMERS_CACHE="${HF_MODELS_CACHE}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_MODELS_CACHE}"
rsync -a "$HOME/scratch/hf_home/"     "${HF_HOME}/"          || true
rsync -a "$HOME/scratch/hf_datasets/" "${HF_DATASETS_CACHE}/" || true
rsync -a "${TOKENIZER_PATH}/"         "${HF_MODELS_CACHE}/$(basename "${TOKENIZER_PATH}")/" || true

# Override HF_MODELS_CACHE env var used by config to point at staged copy
export HF_MODELS_CACHE="${SLURM_TMPDIR}/hf_models"

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
echo "[INFO] Loading modules"
module purge
module load StdEnv/2023 cuda/12.2 python/3.11 gcc arrow/21.0.0 scipy-stack faiss rust opencv

VENV_ACTIVATE="${REPO_ROOT}/.env-gpu/bin/activate"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  echo "[INFO] Activating existing venv: ${VENV_ACTIVATE}"
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
else
  echo "[ERROR] .env-gpu venv not found at ${VENV_ACTIVATE}" >&2
  exit 1
fi

export PYTHONPATH="${TMP_REPO}:${TMP_REPO}/src:${TMP_REPO}/src/third_party/mdlm:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1

# ---------------------------------------------------------------------------
# Memorization env vars (read by wiki_memorization.yaml via ${VAR} expansion)
# ---------------------------------------------------------------------------
export CHECKPOINT_PATH="${CHECKPOINT_PATH}"
export TOKENIZER_PATH="${HF_MODELS_CACHE}/$(basename "${TOKENIZER_PATH}")"
export MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH}"
export MDLM_CONFIG_OVERRIDES=""
export MEMORIZATION_DATA_DIR="${MEMORIZATION_DATA_DIR}"
export MEMORIZATION_RESULTS_DIR="${OUT_DIR}"   # write results into job-specific subdir

# ---------------------------------------------------------------------------
# Build CLI args
# ---------------------------------------------------------------------------
PYTHON_ARGS=(
  --config "${TMP_REPO}/memorization/configs/wiki_memorization.yaml"
  --masking "${WIKI_MASKING}"
  --suffix-tokens "${WIKI_SUFFIX_TOKENS}"
  --mask-ratio "${WIKI_MASK_RATIO}"
  --output-dir "${OUT_DIR}"
)

if [[ -n "${WIKI_MODEL_NAME}" ]]; then
  PYTHON_ARGS+=(--model-name "${WIKI_MODEL_NAME}")
fi

if [[ "${WIKI_DEBUG}" == "1" ]]; then
  PYTHON_ARGS+=(--debug)
fi

# Override n_samples and R via config by patching the yaml is messy;
# instead pass as env vars and let the script pick them up via argparse
# (we do this via a small wrapper below if needed, or just rely on config defaults)
# For now: if non-default, write a temp config patch
if [[ "${WIKI_N_SAMPLES}" != "1000" || "${WIKI_R}" != "512" ]]; then
  TMP_CFG="${SLURM_TMPDIR}/wiki_memorization_override.yaml"
  cp "${TMP_REPO}/memorization/configs/wiki_memorization.yaml" "${TMP_CFG}"
  # Patch n_samples and R using sed
  sed -i "s/^  n_samples:.*/  n_samples: ${WIKI_N_SAMPLES}/" "${TMP_CFG}"
  sed -i "s/^  R:.*/  R: ${WIKI_R}/" "${TMP_CFG}"
  PYTHON_ARGS[1]="${TMP_CFG}"   # replace --config arg
  echo "[INFO] Patched config: n_samples=${WIKI_N_SAMPLES}, R=${WIKI_R}"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "[INFO] Launching: python experiments/measure_wiki_memorization.py ${PYTHON_ARGS[*]}"
cd "${TMP_REPO}"
python experiments/measure_wiki_memorization.py "${PYTHON_ARGS[@]}" \
  2>&1 | tee "${SLURM_TMPDIR}/wiki_mem.log"

# ---------------------------------------------------------------------------
# Sync log back to output dir
# ---------------------------------------------------------------------------
cp "${SLURM_TMPDIR}/wiki_mem.log" "${OUT_DIR}/"

echo "[INFO] Results at: ${OUT_DIR}"
ls "${OUT_DIR}/"
echo "[INFO] Job ${SLURM_JOB_ID} done."
