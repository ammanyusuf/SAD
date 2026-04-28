#!/bin/bash
#SBATCH --job-name=sb_unsafe_protos
#SBATCH --account=rrg-<your-PI>   # [Compute Canada] replace with your allocation
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-user=${USER}@<your-institution.ca>  # [Compute Canada] set to your email
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=/scratch/%u/logs/safe-text-diffusion/unsafe_protos_%j.out
#SBATCH --error=/scratch/%u/logs/safe-text-diffusion/unsafe_protos_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT=${REPO_ROOT:-${DEFAULT_REPO_ROOT}}

ARTIFACT_ROOT=${UNSAFE_ARTIFACT_ROOT:-${SCRATCH:-$HOME}/safe-text-diffusion/artifacts/unsafe_artifacts}
PROTOTYPE_OUTPUT_ROOT=${PROTOTYPE_OUTPUT_ROOT:-${ARTIFACT_ROOT}/prototypes}
NUM_PROTOTYPES=${NUM_PROTOTYPES:-64}
TOKENIZER_PATH=${TOKENIZER_NAME_OR_PATH:-/scratch/${USER}/hf_models/gpt2-large}
PROTOTYPE_MAX_LENGTH=${PROTOTYPE_MAX_LENGTH:-}
VENV_ACTIVATE=${EXTERNAL_VENV_ACTIVATE:-}
PYTHON_BIN=${PYTHON_BIN:-python}
ARTIFACT_NAMES=${ARTIFACT_NAMES:-""}

LOG_DIR="${SLURM_TMPDIR}/logs"
mkdir -p "${LOG_DIR}"

module purge
module load StdEnv/2023 python/3.11 gcc arrow/21.0.0 scipy-stack

echo "[INFO] Job ${SLURM_JOB_ID} starting on $(hostname)"
TMP_REPO="${SLURM_TMPDIR}/repo"
mkdir -p "${TMP_REPO}"
echo "[INFO] Staging repository to ${TMP_REPO}"
rsync -a --exclude=".git" --exclude=".env" --exclude=".env-gpu" --exclude=".env-jailbreak"  "${REPO_ROOT}/" "${TMP_REPO}/"
echo "[INFO] Repository staged."

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
if [[ -n "${VENV_ACTIVATE}" ]]; then
  if [[ -f "${VENV_ACTIVATE}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_ACTIVATE}"
    USING_EXTERNAL_ENV=1
  else
    echo "[WARN] EXTERNAL_VENV_ACTIVATE='${VENV_ACTIVATE}' not found; falling back to local venv." >&2
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
export TOKENIZERS_PARALLELISM=false
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1
export NVIDIA_TF32_OVERRIDE=1
export CUDA_LAUNCH_BLOCKING=0
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_MODELS_CACHE}"
rsync -a "${SRC_HF_HOME}/" "${HF_HOME}/" || true
rsync -a "${SRC_HF_DATASETS_CACHE}/" "${HF_DATASETS_CACHE}/" || true
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

mkdir -p "${PROTOTYPE_OUTPUT_ROOT}"

COMMON_ARGS=(
  --unsafe-artifact-root "${ARTIFACT_ROOT}"
  --output-root "${PROTOTYPE_OUTPUT_ROOT}"
  --num-prototypes "${NUM_PROTOTYPES}"
  --tokenizer "${TOKENIZER_PATH}"
)
if [[ -n "${PROTOTYPE_MAX_LENGTH}" ]]; then
  COMMON_ARGS+=(--max-length "${PROTOTYPE_MAX_LENGTH}")
fi

if [[ -n "${ARTIFACT_NAMES}" ]]; then
  TARGETS=(${ARTIFACT_NAMES})
else
  TARGETS=()
fi

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "[INFO] No ARTIFACT_NAMES provided; building prototypes for all artifacts in index."
  "${PYTHON_BIN}" -m unsafe_prep.build_unsafe_prototypes "${COMMON_ARGS[@]}"
else
  for artifact in "${TARGETS[@]}"; do
    echo "[INFO] Building prototypes for ${artifact}"
    "${PYTHON_BIN}" -m unsafe_prep.build_unsafe_prototypes "${COMMON_ARGS[@]}" --unsafe-artifact-names "${artifact}"
  done
fi

echo "[INFO] Prototype build complete. Outputs under ${PROTOTYPE_OUTPUT_ROOT}"
