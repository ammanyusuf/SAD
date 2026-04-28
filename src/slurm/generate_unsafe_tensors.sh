#!/bin/bash
#SBATCH --job-name=sb_unsafe_tensors
#SBATCH --account=def-<your-PI>   # [Compute Canada] replace with your allocation
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-user=${USER}@<your-institution.ca>  # [Compute Canada] set to your email
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=/scratch/%u/logs/safe-text-diffusion/unsafe_tensors_%j.out
#SBATCH --error=/scratch/%u/logs/safe-text-diffusion/unsafe_tensors_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT=${REPO_ROOT:-${DEFAULT_REPO_ROOT}}

CONFIG_PATH=${UNSAFE_CONFIG:-${REPO_ROOT}/configs/unsafe_prep/unsafe_prompt_sweep.yaml}
OUTPUT_ROOT=${UNSAFE_OUTPUT_ROOT:-${SCRATCH:-$HOME}/safe-text-diffusion/artifacts/unsafe_artifacts}
TOKENIZER_PATH=${TOKENIZER_NAME_OR_PATH:-/scratch/${USER}/hf_models/gpt2-large}
UNSAFE_MAX_LENGTH=${UNSAFE_MAX_LENGTH:-1024}
UNSAFE_SHARD_SIZE=${UNSAFE_SHARD_SIZE:-1024}
PYTHON_BIN=${PYTHON_BIN:-python}
# ARTIFACT_NAMES=${ARTIFACT_NAMES:-"beavertails-0100 beavertails-0500 beavertails-1000 beavertails-prompt-and-continuations-0100 beavertails-prompt-and-continuations-0500 beavertails-prompt-and-continuations-1000 real-toxicity-prompts-0100 real-toxicity-prompts-0500 real-toxicity-prompts-1000 real-toxicity-prompts-prompt-and-continuations-0100 real-toxicity-prompts-prompt-and-continuations-0500 real-toxicity-prompts-prompt-and-continuations-1000 toxigen-0100 toxigen-0500 toxigen-1000 toxigen-prompt-and-continuations-0100 toxigen-prompt-and-continuations-0500 toxigen-prompt-and-continuations-1000 beavertails-knn-owt-prompt-continuation-0100 beavertails-knn-owt-prompt-continuation-0500 beavertails-knn-owt-prompt-continuation-1000 toxigen-knn-owt-prompt-continuation-0100 toxigen-knn-owt-prompt-continuation-0500 toxigen-knn-owt-prompt-continuation-1000"}
# ARTIFACT_NAMES=${ARTIFACT_NAMES:-"beavertails-knn-owt-prompt-continuation-0100 beavertails-knn-owt-prompt-continuation-0500 beavertails-knn-owt-prompt-continuation-1000 toxigen-knn-owt-prompt-continuation-0100 toxigen-knn-owt-prompt-continuation-0500 toxigen-knn-owt-prompt-continuation-1000"}
ARTIFACT_NAMES=${ARTIFACT_NAMES:-"beavertails-0100 beavertails-0500 beavertails-1000 real-toxicity-prompts-0100 real-toxicity-prompts-0500 real-toxicity-prompts-1000 toxigen-0100 toxigen-0500 toxigen-1000"}

LOG_DIR="${SLURM_TMPDIR}/logs"
mkdir -p "${LOG_DIR}"

module purge
module load StdEnv/2023 python/3.11 gcc arrow/21.0.0 scipy-stack

echo "[INFO] Job ${SLURM_JOB_ID} starting on $(hostname)"
TMP_REPO="${SLURM_TMPDIR}/repo"
mkdir -p "${TMP_REPO}"
echo "[INFO] Staging repository to ${TMP_REPO}"
rsync -a --exclude=".git" --exclude=".env" --exclude=".env-gpu" --exclude=".env-jailbreak" "${REPO_ROOT}/" "${TMP_REPO}/"
echo "[INFO] Repository staged."

if [[ -n "${CONFIG_SNAPSHOT_PATH:-}" && -d "${CONFIG_SNAPSHOT_PATH}" ]]; then
  echo "[INFO] Overwriting staged configs with snapshot from ${CONFIG_SNAPSHOT_PATH}/configs/"
  rsync -a "${CONFIG_SNAPSHOT_PATH}/configs/" "${TMP_REPO}/configs/"
fi

for cache_var in HF_HOME HF_DATASETS_CACHE; do
  if [[ -z "${!cache_var:-}" || ! -d "${!cache_var}" ]]; then
    echo "[ERROR] ${cache_var} must point to an existing directory." >&2
    exit 1
  fi
done
if [[ -n "${HF_MODELS_CACHE:-}" && -d "${HF_MODELS_CACHE}" ]]; then
  SRC_HF_MODELS_CACHE=${HF_MODELS_CACHE}
elif [[ -n "${TRANSFORMERS_CACHE:-}" && -d "${TRANSFORMERS_CACHE}" ]]; then
  SRC_HF_MODELS_CACHE=${TRANSFORMERS_CACHE}
else
  echo "[ERROR] Provide HF_MODELS_CACHE (preferred) or TRANSFORMERS_CACHE for model weights." >&2
  exit 1
fi
SRC_HF_HOME=${HF_HOME}
SRC_HF_DATASETS_CACHE=${HF_DATASETS_CACHE}

USING_EXTERNAL_ENV=0
echo "[INFO] Setting up Python environment"
if [[ -n "${EXTERNAL_VENV_ACTIVATE:-}" ]]; then
  if [[ -f "${EXTERNAL_VENV_ACTIVATE}" ]]; then
    # shellcheck disable=SC1090
    source "${EXTERNAL_VENV_ACTIVATE}"
    USING_EXTERNAL_ENV=1
  else
    echo "[WARN] EXTERNAL_VENV_ACTIVATE='${EXTERNAL_VENV_ACTIVATE}' not found; falling back to local venv." >&2
  fi
fi
if [[ "${USING_EXTERNAL_ENV}" -ne 1 ]]; then
  ${PYTHON_BIN} -m venv "${SLURM_TMPDIR}/venv"
  # shellcheck disable=SC1090
  source "${SLURM_TMPDIR}/venv/bin/activate"
  if [[ "${SKIP_PIP_UPGRADE:-0}" != "1" ]]; then
    python -m pip install --upgrade pip
  fi
  if [[ -n "${PIP_INSTALL_ARGS:-}" ]]; then
    python -m pip install ${PIP_INSTALL_ARGS} -r "${TMP_REPO}/src/requirements-cc.txt"
  else
    python -m pip install -r "${TMP_REPO}/src/requirements-cc.txt"
  fi
fi
echo "Python environment ready."

echo "[INFO] Setting up HuggingFace caches in SLURM_TMPDIR"
export HF_HOME=${SLURM_TMPDIR}/hf_home
export HF_DATASETS_CACHE=${SLURM_TMPDIR}/hf_datasets
export HF_MODELS_CACHE=${SLURM_TMPDIR}/hf_models
export TRANSFORMERS_CACHE=${HF_MODELS_CACHE}
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_MODELS_CACHE}"
rsync -a "${SRC_HF_HOME}/" "${HF_HOME}/" || true
rsync -a "${SRC_HF_DATASETS_CACHE}/" "${HF_DATASETS_CACHE}/" || true
export REALTOXICITY_DATA_DIR=${REALTOXICITY_DATA_DIR:-${HF_DATASETS_CACHE}/real-toxicity-prompts}
echo "[INFO] Refreshing staged RealToxicityPrompts cache at ${REALTOXICITY_DATA_DIR}"
rm -rf "${HF_DATASETS_CACHE}/real-toxicity-prompts/default"
rsync -a --delete "${REALTOXICITY_DATA_DIR}/" "${HF_DATASETS_CACHE}/real-toxicity-prompts/" || true
# rsync -a "${SRC_HF_MODELS_CACHE}/" "${HF_MODELS_CACHE}/" || true
echo "[INFO] HuggingFace caches set up at ${HF_HOME}, ${HF_DATASETS_CACHE}, ${HF_MODELS_CACHE}"

stage_model_path() {
  local src_path="$1"
  if [[ -z "${src_path}" ]]; then
    return
  fi
  if [[ -f "${src_path}" || -d "${src_path}" ]]; then
    echo "${src_path}"
    return
  fi
  if [[ -d "${SRC_HF_MODELS_CACHE}/${src_path}" ]]; then
    echo "${SRC_HF_MODELS_CACHE}/${src_path}"
    return
  fi
}

stage_tokenizer="${TOKENIZER_PATH:-${TOKENIZER_NAME_OR_PATH}}"
tok_source="$(stage_model_path "${stage_tokenizer}")"
if [[ -n "${tok_source}" ]]; then
  base_name="$(basename "${tok_source}")"
  dest="${HF_MODELS_CACHE}/${base_name}"
  echo "[INFO] Staging tokenizer/model ${base_name} to ${dest}"
  rsync -a --progress --human-readable "${tok_source}/" "${dest}/" || true
else
  echo "[WARN] Could not locate tokenizer/model '${stage_tokenizer}' to stage; relying on cache." >&2
fi

export PYTHONPATH="${TMP_REPO}/src:${TMP_REPO}/src/third_party:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1
export NVIDIA_TF32_OVERRIDE=1
export CUDA_LAUNCH_BLOCKING=0

mkdir -p "${OUTPUT_ROOT}"

COMMON_OVERRIDES=(
  "tokenizer_name_or_path=${TOKENIZER_PATH}"
  "max_length=${UNSAFE_MAX_LENGTH}"
  "shard_size=${UNSAFE_SHARD_SIZE}"
  "output_dir=${OUTPUT_ROOT}"
)

echo "[INFO] Beginning unsafe tensor build for artifacts: ${ARTIFACT_NAMES}"
for artifact in ${ARTIFACT_NAMES}; do
  echo "[INFO] Building unsafe tensor ${artifact}"
  CMD=(
    "${PYTHON_BIN}"
    -m
    unsafe_prep.build_unsafe_artifacts
    --config "${CONFIG_PATH}"
    --out "${OUTPUT_ROOT}"
    --force
    --include "${artifact}"
  )
  for override in "${COMMON_OVERRIDES[@]}"; do
    CMD+=(--set "${override}")
  done
  rtp_local_cache="${HF_DATASETS_CACHE}/real-toxicity-prompts/default"
  if [[ -d "${rtp_local_cache}" ]]; then
    echo "[INFO] Removing staged real-toxicity-prompts cache at ${rtp_local_cache}"
    rm -rf "${rtp_local_cache}"
  fi
  echo "[INFO] $(printf '%q ' "${CMD[@]}")"
  "${CMD[@]}"
done

echo "[INFO] Unsafe tensor build complete. Index located at ${OUTPUT_ROOT}/index.json"
