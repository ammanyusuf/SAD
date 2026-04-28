#!/bin/bash
# Single-node generation script. Saturates all requested GPUs on one node.
# Submit with: sbatch --gpus-per-node=<N> src/slurm/generate_single.sh
# See src/slurm/README.md for the full env var reference.
#
# [Compute Canada] Notes:
#   - --account: set to your allocation (e.g. rrg-<PI> or def-<PI>)
#   - --gpus-per-node: a100:1 on narval/trillium, v100:1 on cedar/graham
#   - Log paths (/scratch/%u/logs/...) require the directory to exist before submission
#   - Flash Attention is only compatible with specific torch builds; see src/requirements-cc.txt

#SBATCH --job-name=sb_generate_mdlm
#SBATCH --account=rrg-<your-PI>   # [Compute Canada] replace with your allocation
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=8GB
#SBATCH --mail-user=${USER}@<your-institution.ca>  # [Compute Canada] set to your email
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=/scratch/%u/logs/safe-text-diffusion/%x_%A_%a.out
#SBATCH --error=/scratch/%u/logs/safe-text-diffusion/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT=${REPO_ROOT:-${DEFAULT_REPO_ROOT}}

TRACK_NAME=${TRACK_NAME:-safety}
RUN_ID=${RUN_ID:-${SLURM_JOB_ID}}
RESULTS_ROOT=${RESULTS_ROOT:-${SCRATCH:-$HOME}/results}
TARGET_VRAM_PCT=${TARGET_VRAM_PCT:-0.9}
AUTO_BATCH_WARMUP_PROMPTS=${AUTO_BATCH_WARMUP_PROMPTS:-64}
if [[ -z "${GEN_CONFIG_NAME:-}" ]]; then
  echo "[ERROR] GEN_CONFIG_NAME must be set to a Hydra config name (e.g., experiments/beavertails_prompts)." >&2
  exit 1
fi
EXPERIMENT_SLUG=${EXPERIMENT_SLUG:-${GEN_CONFIG_NAME##*/}}
SKIP_PIP_UPGRADE=${SKIP_PIP_UPGRADE:-0}
PIP_INSTALL_ARGS=${PIP_INSTALL_ARGS:-}
EXTERNAL_VENV_ACTIVATE=${EXTERNAL_VENV_ACTIVATE:-}
SAFETY_ENABLED=${SAFETY_ENABLED:-0}
SAFETY_SCALE=${SAFETY_SCALE:-}
SAFETY_ETA=${SAFETY_ETA:-${SAFETY_SCALE:-1.0}}
UNSAFE_ARTIFACTS=${UNSAFE_ARTIFACTS:-}
UNSAFE_ARTIFACT_ROOT=${UNSAFE_ARTIFACT_ROOT:-}
UNSAFE_ARTIFACT_NAME=${UNSAFE_ARTIFACT_NAME:-}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
TOKENIZER_PATH=${TOKENIZER_PATH:-}
MODEL_VARIANT=${MODEL_VARIANT:-}
PROMPT_LIMIT=${PROMPT_LIMIT:-}
ADD_BOS=${ADD_BOS:-0}
ADD_EOS=${ADD_EOS:-0}
UNCONDITIONAL_SAMPLES=${UNCONDITIONAL_SAMPLES:-0}
DRY_RUN=${DRY_RUN:-0}

LOG_DIR="${SLURM_TMPDIR}/logs"
mkdir -p "${LOG_DIR}"

echo "[INFO] Job ${SLURM_JOB_ID} starting on $(hostname)"
echo "[INFO] Requested GPUs per node: ${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}"

cleanup_ran=0
stage_status="failed"

stage_out() {
  local status=$1
  local run_dir_name="${RUN_ID:-${SLURM_JOB_ID}}"
  local dest="${RESULTS_ROOT}/${EXPERIMENT_SLUG}/${run_dir_name}"
  mkdir -p "${dest}"
  echo "[INFO] Staging outputs to ${dest}"
  rsync -a "${SLURM_TMPDIR}/outputs/" "${dest}/" || true
  rsync -a "${LOG_DIR}/" "${dest}/logs/" || true
  printf "%s\n" "${status}" > "${dest}/status.txt"
  echo "[INFO] Stage complete (status=${status}); artifacts available at ${dest}"
}

cleanup() {
  if [[ ${cleanup_ran} -eq 1 ]]; then
    return
  fi
  cleanup_ran=1
  stage_out "${stage_status}"
}
trap cleanup EXIT
trap 'stage_status="failed"; exit 1' ERR INT

# Stage repo locally for faster imports, but keep checkpoint/tokenizer in place.
echo "[INFO] Staging inputs to SLURM_TMPDIR: ${SLURM_TMPDIR}"
mkdir -p "${SLURM_TMPDIR}"/{repo,data,outputs}
TMP_REPO="${SLURM_TMPDIR}/repo"
TMP_DATA_DIR="${SLURM_TMPDIR}/data"
TMP_OUTPUT_DIR="${SLURM_TMPDIR}/outputs"
rsync -a --exclude=".git" --exclude=".tmp" --exclude=".env" --exclude=".env-gpu" --exclude=".env-jailbreak" "${REPO_ROOT}/" "${TMP_REPO}/"
echo "[INFO] Repo staged to ${TMP_REPO}"

if [[ -n "${CONFIG_SNAPSHOT_PATH:-}" && -d "${CONFIG_SNAPSHOT_PATH}" ]]; then
  echo "[INFO] Overwriting staged configs with snapshot from ${CONFIG_SNAPSHOT_PATH}/configs/"
  rsync -a "${CONFIG_SNAPSHOT_PATH}/configs/" "${TMP_REPO}/configs/"
fi

# Require explicit HF caches so jobs run offline/deterministic.
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

echo "[INFO] Setting up HuggingFace caches in SLURM_TMPDIR"
export HF_HOME=${SLURM_TMPDIR}/hf_home
export HF_DATASETS_CACHE=${SLURM_TMPDIR}/hf_datasets
export HF_MODELS_CACHE=${SLURM_TMPDIR}/hf_models
export TRANSFORMERS_CACHE=${HF_MODELS_CACHE}
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_MODELS_CACHE}"
rsync -a "${SRC_HF_HOME}/" "${HF_HOME}/" || true
rsync -a "${SRC_HF_DATASETS_CACHE}/" "${HF_DATASETS_CACHE}/" || true
#rsync -a --exclude="*.tar.gz" --exclude="LLaDA-8B-Base/" "${SRC_HF_MODELS_CACHE}/" "${HF_MODELS_CACHE}/" || true
export BEAVERTAILS_DATA_DIR=${BEAVERTAILS_DATA_DIR:-${HF_DATASETS_CACHE}/BeaverTails}
export TOXIGEN_DATA_DIR=${TOXIGEN_DATA_DIR:-${HF_DATASETS_CACHE}/toxigen}
export REALTOXICITY_DATA_DIR=${REALTOXICITY_DATA_DIR:-${HF_DATASETS_CACHE}/real-toxicity-prompts}
echo "[INFO] HuggingFace caches set up at ${HF_HOME}, ${HF_DATASETS_CACHE}, ${HF_MODELS_CACHE}"

echo "[INFO] Setting up Python environment"
module purge
module load StdEnv/2023 cuda/12.2 python/3.11 gcc arrow/21.0.0 scipy-stack
USING_EXTERNAL_ENV=0
if [[ -n "${EXTERNAL_VENV_ACTIVATE}" ]]; then
  if [[ -f "${EXTERNAL_VENV_ACTIVATE}" ]]; then
    echo "[INFO] Activating external virtualenv: ${EXTERNAL_VENV_ACTIVATE}"
    # shellcheck disable=SC1090
    source "${EXTERNAL_VENV_ACTIVATE}"
    USING_EXTERNAL_ENV=1
  else
    echo "[WARN] EXTERNAL_VENV_ACTIVATE='${EXTERNAL_VENV_ACTIVATE}' not found; falling back to Compute Canada wheel install." >&2
  fi
fi
if [[ "${USING_EXTERNAL_ENV}" -ne 1 ]]; then
  echo "[INFO] Creating Compute Canada virtualenv at ${SLURM_TMPDIR}/venv"
  ${PYTHON_BIN:-python} -m venv "${SLURM_TMPDIR}/venv"
  # shellcheck disable=SC1090
  source "${SLURM_TMPDIR}/venv/bin/activate"
  if [[ "${SKIP_PIP_UPGRADE}" != "1" ]]; then
    python -m pip install --upgrade pip
  fi
  if [[ -n "${PIP_INSTALL_ARGS}" ]]; then
    python -m pip install ${PIP_INSTALL_ARGS} -r "${TMP_REPO}/src/requirements-cc.txt"
  else
    python -m pip install -r "${TMP_REPO}/src/requirements-cc.txt"
  fi
  echo "[INFO] Compute Canada virtualenv ready"
else
  echo "[INFO] Using caller-provided virtualenv"
fi
echo "[INFO] ----- pip freeze -----"
python -m pip freeze | tee "${LOG_DIR}/pip_freeze.txt" || true
echo "[INFO] ----------------------"

export PYTHONPATH="${PYTHONPATH}:${TMP_REPO}/src:${TMP_REPO}/src/third_party/:${TMP_REPO}/src/third_party/mdlm/"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1
export NVIDIA_TF32_OVERRIDE=1
export CUDA_LAUNCH_BLOCKING=0

IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ ${#GPU_IDS[@]} -eq 0 || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_COUNT=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}
  GPU_IDS=()
  for ((i = 0; i < GPU_COUNT; ++i)); do
    GPU_IDS+=("${i}")
  done
fi
GPU_COUNT=${#GPU_IDS[@]}
TOTAL_GLOBAL_SHARDS=${GPU_COUNT}

declare -a WORKER_PIDS=()
mkdir -p "${TMP_OUTPUT_DIR}"

launch_worker() {
  echo "[INFO] Launching worker on GPU slot $1 (device ID $2)"
  local gpu_slot=$1
  local device_id=$2
  local shard_id=$gpu_slot
  local worker_root="${TMP_OUTPUT_DIR}/gpu_${gpu_slot}"
  mkdir -p "${worker_root}"
  local log_path="${LOG_DIR}/gpu_${gpu_slot}.log"

  local safety_bool="false"
  if [[ "${SAFETY_ENABLED}" == "1" || "${SAFETY_ENABLED,,}" == "true" ]]; then
    safety_bool="true"
  fi
  local dry_bool="false"
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
    dry_bool="true"
  fi
  local bos_bool="false"
  local eos_bool="false"
  if [[ "${ADD_BOS}" == "1" || "${ADD_BOS,,}" == "true" ]]; then
    bos_bool="true"
  fi
  if [[ "${ADD_EOS}" == "1" || "${ADD_EOS,,}" == "true" ]]; then
    eos_bool="true"
  fi

  local overrides=(
    "io.output_dir=${worker_root}"
    "io.run_id=${RUN_ID}"
    "io.target_vram_pct=${TARGET_VRAM_PCT}"
    "io.auto_batch_warmup_prompts=${AUTO_BATCH_WARMUP_PROMPTS}"
    "sharding.shard_id=${shard_id}"
    "sharding.num_shards=${TOTAL_GLOBAL_SHARDS}"
    "safety.enabled=${safety_bool}"
    "safety.eta=${SAFETY_ETA}"
    "gen.add_bos=${bos_bool}"
    "gen.add_eos=${eos_bool}"
    "gen.unconditional_samples=${UNCONDITIONAL_SAMPLES}"
    "gen.dry_run=${dry_bool}"
  )
  if [[ -n "${EXPERIMENT_SLUG}" ]]; then
    overrides+=("io.experiment_slug=${EXPERIMENT_SLUG}")
  fi
  if [[ -n "${TRACK_NAME}" ]]; then
    overrides+=("io.track_name=${TRACK_NAME}")
  fi
  if [[ -n "${PROMPT_LIMIT}" ]]; then
    overrides+=("data.limit=${PROMPT_LIMIT}")
  fi
  if [[ -n "${DATASET_JSON:-}" ]]; then
    overrides+=("data.dataset_json=${DATASET_JSON}")
  fi
  if [[ -n "${MODEL_VARIANT:-}" ]]; then
    overrides+=("model.variant=${MODEL_VARIANT}")
  fi
  if [[ -n "${UNSAFE_ARTIFACTS}" ]]; then
    overrides+=("safety.unsafe_artifacts=${UNSAFE_ARTIFACTS}")
  fi
  if [[ -n "${UNSAFE_ARTIFACT_ROOT}" ]]; then
    overrides+=("safety.unsafe_artifact_root=${UNSAFE_ARTIFACT_ROOT}")
  fi
  if [[ -n "${UNSAFE_ARTIFACT_NAME}" ]]; then
    overrides+=("safety.unsafe_artifact_name=${UNSAFE_ARTIFACT_NAME}")
  fi
  if [[ -n "${UNSAFE_PROTOTYPES:-}" ]]; then
    overrides+=("safety.unsafe_prototypes=${UNSAFE_PROTOTYPES}")
  fi
  if [[ -n "${SAFETY_SCALE}" ]]; then
    overrides+=("safety.scale=${SAFETY_SCALE}")
  fi
  if [[ -n "${UNSAFE_PROTOTYPE_ROOT:-}" ]]; then
    overrides+=("safety.unsafe_prototype_root=${UNSAFE_PROTOTYPE_ROOT}")
  fi
  if [[ -n "${CRITICAL_STEPS:-}" ]]; then
    overrides+=("safety.critical_steps=${CRITICAL_STEPS}")
  fi
  if [[ -n "${SAFETY_T_START:-}" ]]; then
    overrides+=("safety.t_start=${SAFETY_T_START}")
  fi
  if [[ -n "${SAFETY_T_END:-}" ]]; then
    overrides+=("safety.t_end=${SAFETY_T_END}")
  fi
  mkdir -p "${TMP_REPO}/staged_models"
  if [[ -n "${CHECKPOINT_PATH}" && -f "${CHECKPOINT_PATH}" ]]; then
    echo "[INFO] Staging checkpoint ${CHECKPOINT_PATH} to SLURM_TMPDIR"
    local staged_ckpt="${TMP_REPO}/staged_models/$(basename "${CHECKPOINT_PATH}")"
    rsync -a "${CHECKPOINT_PATH}" "${staged_ckpt}" || true
    overrides+=("model.checkpoint=${staged_ckpt}")
    echo "[INFO] Staged checkpoint to ${staged_ckpt}"
  fi
  if [[ -n "${TOKENIZER_PATH}" && -f "${TOKENIZER_PATH}" ]]; then
    echo "[INFO] Staging tokenizer ${TOKENIZER_PATH} to SLURM_TMPDIR"
    local staged_tok="${TMP_REPO}/staged_models/$(basename "${TOKENIZER_PATH}")"
    rsync -a "${TOKENIZER_PATH}" "${staged_tok}" || true
    overrides+=("model.tokenizer_name=${staged_tok}")
    echo "[INFO] Staged tokenizer to ${staged_tok}"
  fi

  (
    set -x
    CUDA_VISIBLE_DEVICES="${device_id}" python -m tools.generate --config-name "${GEN_CONFIG_NAME}" "${overrides[@]}"
  ) 2>&1 | tee "${log_path}" | sed -e "s/^/[gpu_${gpu_slot}] /" &
  WORKER_PIDS+=($!)
}

for idx in "${!GPU_IDS[@]}"; do
  launch_worker "${idx}" "${GPU_IDS[$idx]}"
done

for pid in "${WORKER_PIDS[@]}"; do
  wait "${pid}"
done
echo "[INFO] All workers completed successfully."

python -m utils.slurm_helpers aggregate \
  --root "${TMP_OUTPUT_DIR}" \
  --output "${TMP_OUTPUT_DIR}/job_run_metadata.json" \
  --job-id "${SLURM_JOB_ID}" \
  --experiment-slug "${EXPERIMENT_SLUG}" \
  --total-prompts 0 \
  --gpu-shards "${GPU_COUNT}" || true

stage_status="success"
echo "[INFO] Job ${SLURM_JOB_ID:-local} completed; results staged under ${RESULTS_ROOT}/${EXPERIMENT_SLUG}/${RUN_ID:-${SLURM_JOB_ID}}"
