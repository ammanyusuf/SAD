#!/bin/bash
#SBATCH --job-name=sb_semantic_cache
#SBATCH --account=rrg-<your-PI>   # [Compute Canada] replace with your allocation
#SBATCH --time=10:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --mail-user=${USER}@<your-institution.ca>  # [Compute Canada] set to your email
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=/scratch/%u/logs/safe-text-diffusion/semantic_cache_%j.out
#SBATCH --error=/scratch/%u/logs/safe-text-diffusion/semantic_cache_%j.err


 # note, need memory to be 32gb when using LLaDa hf encoder


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT=${REPO_ROOT:-${DEFAULT_REPO_ROOT}}

ARTIFACT_ROOT=${ARTIFACT_ROOT:-${SCRATCH:-$HOME}/safe-text-diffusion/artifacts/unsafe_artifacts}
ARTIFACT_NAMES=${ARTIFACT_NAMES:-beavertails-1500}
PROVIDER=${PROVIDER:-mdlm}                         # hf | mdlm | callable
MDLM_FN=${MDLM_FN:-third_party.mdlm.encoders:embed_from_checkpoint}
EMBED_FN=${EMBED_FN:-}                             # module:function when provider=callable
ENCODER_NAME=${ENCODER_NAME:-}    # fallback or provider=hf
BCKPT_PATH=${CHECKPOINT_PATH:-}                    # checkpoint for provider=mdlm
MDLM_EMBED_ATTR=${MDLM_EMBED_ATTR:-}               # optional backbone embedding attr (e.g., vocab_embed)
MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH:-}           # optional model config path for MDLM embedding
TOKENIZER_OVERRIDE=${TOKENIZER_OVERRIDE:-}
BATCH_SIZE=${BATCH_SIZE:-2}
OUTPUT_PATH=${OUTPUT_PATH:-}                       # optional explicit path; if empty, auto-named per artifact under semantic_refs/
DEVICE=${DEVICE:-cuda}
PYTHON_BIN=${PYTHON_BIN:-python}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="${SLURM_TMPDIR}/logs"
mkdir -p "${LOG_DIR}"

module purge
module load StdEnv/2023 cuda/12.2 python/3.11 gcc arrow/21.0.0 scipy-stack

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
# rsync -a "${SRC_HF_HOME}/" "${HF_HOME}/" || true
# rsync -a "${SRC_HF_DATASETS_CACHE}/" "${HF_DATASETS_CACHE}/" || true
echo "HuggingFace caches ready."

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

# Stage encoder if provided
if [[ -n "${ENCODER_NAME}" ]]; then
  enc_source="$(stage_model_path "${ENCODER_NAME}")"
  if [[ -n "${enc_source}" ]]; then
    base_name="$(basename "${enc_source}")"
    dest="${HF_MODELS_CACHE}/${base_name}"
    echo "[INFO] Staging encoder ${base_name} to ${dest}"
    rsync -a --progress --human-readable "${enc_source}/" "${dest}/" || true
  else
    echo "[WARN] Could not locate encoder '${ENCODER_NAME}' to stage; relying on cache." >&2
  fi
fi

export PYTHONPATH="${TMP_REPO}/src:${TMP_REPO}/src/third_party:${TMP_REPO}/src/third_party/mdlm:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
DEFAULT_MDLM_CONFIG="${TMP_REPO}/src/third_party/mdlm/configs/config.yaml"
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1
export NVIDIA_TF32_OVERRIDE=1
export CUDA_LAUNCH_BLOCKING=0

mkdir -p "${ARTIFACT_ROOT}"

# shellcheck disable=SC2206
artifact_list=(${ARTIFACT_NAMES})

for artifact in "${artifact_list[@]}"; do
  out_path="${OUTPUT_PATH}"
  if [[ -z "${out_path}" ]]; then
    out_path="${ARTIFACT_ROOT}/semantic_refs/semantic_ref_embeddings_${artifact}.pt"
  fi

  resolved_model_config="${MODEL_CONFIG_PATH}"
  if [[ -n "${resolved_model_config}" && ! -f "${resolved_model_config}" ]]; then
    if [[ -f "${DEFAULT_MDLM_CONFIG}" ]]; then
      echo "[WARN] MODEL_CONFIG_PATH='${resolved_model_config}' missing; using staged config at ${DEFAULT_MDLM_CONFIG}" >&2
      resolved_model_config="${DEFAULT_MDLM_CONFIG}"
    fi
  fi
  if [[ -z "${resolved_model_config}" ]]; then
    if [[ -f "${DEFAULT_MDLM_CONFIG}" ]]; then
      resolved_model_config="${DEFAULT_MDLM_CONFIG}"
    fi
  fi

  CMD=(
    "${PYTHON_BIN}"
    -m unsafe_prep.build_semantic_ref_cache
    --artifact-root "${ARTIFACT_ROOT}"
    --artifact-name "${artifact}"
    --provider "${PROVIDER}"
    --output "${out_path}"
    --batch-size "${BATCH_SIZE}"
    --device "${DEVICE}"
  )

  if [[ "${PROVIDER}" == "mdlm" ]]; then
    if [[ -n "${MDLM_FN}" ]]; then
      CMD+=(--mdlm-fn "${MDLM_FN}")
    fi
    if [[ -n "${ENCODER_NAME}" ]]; then
      CMD+=(--encoder "${ENCODER_NAME}")
    fi
    if [[ -n "${BCKPT_PATH}" ]]; then
      CMD+=(--checkpoint "${BCKPT_PATH}")
    fi
    if [[ -n "${resolved_model_config}" ]]; then
      CMD+=(--model-config "${resolved_model_config}")
    fi
    if [[ -n "${MDLM_EMBED_ATTR}" ]]; then
      CMD+=(--mdlm-embed-attr "${MDLM_EMBED_ATTR}")
    fi
    if [[ -n "${TOKENIZER_PATH}" ]]; then
      CMD+=(--tokenizer-path "${TOKENIZER_PATH}")
    fi
  fi
  if [[ -n "${TOKENIZER_OVERRIDE}" ]]; then
    CMD+=(--tokenizer "${TOKENIZER_OVERRIDE}")
  fi
  if [[ "${PROVIDER}" == "callable" && -n "${EMBED_FN}" ]]; then
    CMD+=(--embed-fn "${EMBED_FN}")
  fi
  if [[ "${PROVIDER}" == "hf" && -n "${ENCODER_NAME}" ]]; then
    CMD+=(--encoder "${ENCODER_NAME}")
  fi

  echo "[INFO] Building semantic cache for ${artifact}: $(printf '%q ' "${CMD[@]}")"
  "${CMD[@]}"
  echo "[INFO] Semantic cache saved to ${out_path}"
done
