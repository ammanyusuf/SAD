#!/bin/bash
# Array-aware generation script. Each array task receives a contiguous prompt slice.
# Submit with: sbatch --array=0-<N-1> --gpus-per-node=<G> src/slurm/generate_array.sh
# See src/slurm/README.md for the full env var reference.
#
# [Compute Canada] Notes:
#   - --account: set to your allocation (e.g. rrg-<PI> or def-<PI>)
#   - --array: the default 0-3 runs 4 tasks; adjust to total shards needed
#   - Log directory (/scratch/%u/logs/safe-text-diffusion/) must exist before submission

#SBATCH --job-name=sb_generate_mdlm
#SBATCH --account=rrg-<your-PI>   # [Compute Canada] replace with your allocation
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32G
#SBATCH --mail-user=${USER}@<your-institution.ca>  # [Compute Canada] set to your email
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --array=0-3
#SBATCH --output=/scratch/%u/logs/safe-text-diffusion/%x_%A_%a.out
#SBATCH --error=/scratch/%u/logs/safe-text-diffusion/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT=${REPO_ROOT:-${DEFAULT_REPO_ROOT}}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[ERROR] SLURM_ARRAY_TASK_ID is unset; run via sbatch --array." >&2
  exit 1
fi

TRACK_NAME=${TRACK_NAME:-safety}
RUN_ID=${RUN_ID:-${SLURM_JOB_ID}}
RESULTS_ROOT=${RESULTS_ROOT:-${SCRATCH:-$HOME}/results}
TARGET_VRAM_PCT=${TARGET_VRAM_PCT:-0.9}
AUTO_BATCH_WARMUP_PROMPTS=${AUTO_BATCH_WARMUP_PROMPTS:-64}
CONFIG_BATCH_SPECS=${CONFIG_BATCH_SPECS:-}
CONFIG_BATCH_FILE=${CONFIG_BATCH_FILE:-}
GEN_CONFIG_NAME=${GEN_CONFIG_NAME:-}
EXPERIMENT_SLUG=${EXPERIMENT_SLUG:-${GEN_CONFIG_NAME##*/}}
SKIP_PIP_UPGRADE=${SKIP_PIP_UPGRADE:-0}
PIP_INSTALL_ARGS=${PIP_INSTALL_ARGS:-}
EXTERNAL_VENV_ACTIVATE=${EXTERNAL_VENV_ACTIVATE:-}
SAFETY_ENABLED=${SAFETY_ENABLED:-0}
SAFETY_SCALE=${SAFETY_SCALE:-}
SAFETY_ETA=${SAFETY_ETA:-${SAFETY_SCALE:-1.0}}
SAFETY_SEMANTIC_WEIGHT=${SAFETY_SEMANTIC_WEIGHT:-}
SAFETY_SEMANTIC_TEMP=${SAFETY_SEMANTIC_TEMP:-}
SAFETY_SEMANTIC_SIGMA=${SAFETY_SEMANTIC_SIGMA:-}
USE_SEMANTIC_GATING=${USE_SEMANTIC_GATING:-}
UNSAFE_ARTIFACTS=${UNSAFE_ARTIFACTS:-}
UNSAFE_ARTIFACT_ROOT=${UNSAFE_ARTIFACT_ROOT:-}
UNSAFE_ARTIFACT_NAME=${UNSAFE_ARTIFACT_NAME:-}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
TOKENIZER_PATH=${TOKENIZER_PATH:-}
PROMPT_LIMIT=${PROMPT_LIMIT:-}
SAMPLING_STEPS=${SAMPLING_STEPS:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-}
PROMPT_VARIANT=${PROMPT_VARIANT:-}
PROMPT_SOURCE_NAME=${PROMPT_SOURCE_NAME:-}
MODEL_FAMILY=${MODEL_FAMILY:-}
MODEL_NAME=${MODEL_NAME:-}
MODEL_VARIANT=${MODEL_VARIANT:-}
LLAMAGUARD_CHECKPOINT_PATH=${LLAMAGUARD_CHECKPOINT_PATH:-}
N_PER_PROMPT=${N_PER_PROMPT:-}
FK_K_PARTICLES=${FK_K_PARTICLES:-}
FK_ROBERTA_CHECKPOINT_PATH=${FK_ROBERTA_CHECKPOINT_PATH:-}
TEMPERATURE=${TEMPERATURE:-}
ADD_BOS=${ADD_BOS:-0}
ADD_EOS=${ADD_EOS:-0}
UNCONDITIONAL_SAMPLES=${UNCONDITIONAL_SAMPLES:-0}
DRY_RUN=${DRY_RUN:-0}
GEN_SEED=${GEN_SEED:-}
RUN_SUBDIR=${RUN_SUBDIR:-}
CURRENT_STAGE_STATUS="failed"

ARRAY_MIN=${SLURM_ARRAY_TASK_MIN:-0}
ARRAY_MAX=${SLURM_ARRAY_TASK_MAX:-${SLURM_ARRAY_TASK_ID}}
ARRAY_STEP=${SLURM_ARRAY_TASK_STEP:-1}
ARRAY_SIZE=$(( (ARRAY_MAX - ARRAY_MIN) / ARRAY_STEP + 1 ))
TASK_ORD=$(( (SLURM_ARRAY_TASK_ID - ARRAY_MIN) / ARRAY_STEP ))

LOG_ROOT="${SLURM_TMPDIR}/logs/task_${SLURM_ARRAY_TASK_ID}"
JOB_LOG_DIR="${LOG_ROOT}/job"
mkdir -p "${JOB_LOG_DIR}"
TMP_OUTPUT_ROOT="${SLURM_TMPDIR}/outputs/task_${SLURM_ARRAY_TASK_ID}"

echo "[INFO] Job ${SLURM_JOB_ID} starting on $(hostname)"
echo "[INFO] Requested GPUs per node: ${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}"

cleanup_ran=0
stage_pending=0
CURRENT_EXPERIMENT_SLUG="${EXPERIMENT_SLUG}"
CURRENT_RUN_ID="${RUN_ID}"
CURRENT_OUTPUT_DIR=""
CURRENT_LOG_DIR=""

stage_out_config() {
  local status="$1"
  local slug="$2"
  local run_id="$3"
  local output_dir="$4"
  local log_dir="$5"
  local task_id="${SLURM_ARRAY_TASK_ID:-0}"
  if [[ -z "${slug}" || -z "${run_id}" || -z "${output_dir}" ]]; then
    echo "[WARN] Missing slug/run_id/output_dir for staging; skipping." >&2
    return
  fi
  local dest="${RESULTS_ROOT}/${slug}/${run_id}"
  if [[ -n "${RUN_SUBDIR}" ]]; then
    dest="${dest}/${RUN_SUBDIR}"
  fi
  dest="${dest}/task_${task_id}"
  mkdir -p "${dest}"
  echo "[INFO] [task_${task_id}] Staging outputs for ${slug}/${run_id} to ${dest}"
  if [[ -d "${output_dir}" ]]; then
    rsync -a "${output_dir}/" "${dest}/" || true
  fi
  if [[ -d "${log_dir:-}" ]]; then
    rsync -a "${log_dir}/" "${dest}/logs/" || true
  fi
  if [[ -d "${JOB_LOG_DIR:-}" ]]; then
    rsync -a "${JOB_LOG_DIR}/" "${dest}/logs/job/" || true
  fi
  printf "%s\n" "${status}" > "${dest}/status.txt"
  echo "[INFO] [task_${task_id}] Stage complete (status=${status}); artifacts available at ${dest}"
  stage_pending=0
}

stage_out_pending() {
  if [[ ${stage_pending} -ne 1 ]]; then
    return
  fi
  stage_out_config "${CURRENT_STAGE_STATUS}" "${CURRENT_EXPERIMENT_SLUG}" "${CURRENT_RUN_ID}" "${CURRENT_OUTPUT_DIR}" "${CURRENT_LOG_DIR}"
}

cleanup() {
  if [[ ${cleanup_ran} -eq 1 ]]; then
    return
  fi
  cleanup_ran=1
  stage_out_pending
}
trap cleanup EXIT
trap 'CURRENT_STAGE_STATUS="failed"; stage_out_pending; exit 1' ERR INT

echo "[INFO] Staging inputs to SLURM_TMPDIR: ${SLURM_TMPDIR}"
mkdir -p "${SLURM_TMPDIR}"/{repo,outputs}
TMP_REPO="${SLURM_TMPDIR}/repo"
rsync -a --exclude=".git" --exclude=".tmp" --exclude=".env" --exclude=".env-gpu" --exclude=".env-jailbreak" "${REPO_ROOT}/" "${TMP_REPO}/"
echo "[INFO] Repo staged to ${TMP_REPO}"

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

echo "[INFO] Setting up HuggingFace caches in SLURM_TMPDIR"
export HF_HOME=${SLURM_TMPDIR}/hf_home
export HF_DATASETS_CACHE=${SLURM_TMPDIR}/hf_datasets
export HF_MODELS_CACHE=${SLURM_TMPDIR}/hf_models
export TRANSFORMERS_CACHE=${HF_MODELS_CACHE}
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_MODELS_CACHE}"
rsync -a "${SRC_HF_HOME}/" "${HF_HOME}/" || true
rsync -a "${SRC_HF_DATASETS_CACHE}/" "${HF_DATASETS_CACHE}/" || true
#rsync -a --exclude="*.tar.gz" --exclude="LLaDA-8B-Base/" -v --stats --progress "${SRC_HF_MODELS_CACHE}/" "${HF_MODELS_CACHE}/" || true
# Redirect common dataset env vars to the staged HF_DATASETS_CACHE so prompt sources read from local disk.
export BEAVERTAILS_DATA_DIR=${BEAVERTAILS_DATA_DIR:-${HF_DATASETS_CACHE}/BeaverTails}
export TOXIGEN_DATA_DIR=${TOXIGEN_DATA_DIR:-${HF_DATASETS_CACHE}/toxigen}
export REALTOXICITY_DATA_DIR=${REALTOXICITY_DATA_DIR:-${HF_DATASETS_CACHE}/real-toxicity-prompts}
echo "[INFO] Refreshing staged RealToxicityPrompts cache at ${REALTOXICITY_DATA_DIR}"
rm -rf "${HF_DATASETS_CACHE}/real-toxicity-prompts/default"
rsync -a --delete "${REALTOXICITY_DATA_DIR}/" "${HF_DATASETS_CACHE}/real-toxicity-prompts/" || true
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
python -m pip freeze | tee "${JOB_LOG_DIR}/pip_freeze.txt" || true
echo "[INFO] ----------------------"

export PYTHONPATH="${PYTHONPATH}:${TMP_REPO}/src:${TMP_REPO}/src/third_party/:${TMP_REPO}/src/third_party/mdlm"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1
export NVIDIA_TF32_OVERRIDE=1
export CUDA_LAUNCH_BLOCKING=0
# Sometimes these warnings are annoying, so you can quiet them down during submission
# export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1


IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ ${#GPU_IDS[@]} -eq 0 || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_COUNT=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}
  GPU_IDS=()
  for ((i = 0; i < GPU_COUNT; ++i)); do
    GPU_IDS+=("${i}")
  done
fi
GPU_COUNT=${#GPU_IDS[@]}
TOTAL_GLOBAL_SHARDS=$((ARRAY_SIZE * GPU_COUNT))

declare -A STAGED_PATH_CACHE=()
STAGED_MODELS_DIR="${TMP_REPO}/staged_models"
STAGED_ARTIFACTS_DIR="${TMP_REPO}/staged_artifacts"
mkdir -p "${STAGED_MODELS_DIR}" "${STAGED_ARTIFACTS_DIR}" "${TMP_OUTPUT_ROOT}"

stage_path_once() {
  echo "[INFO] Staging path once: $1" >&2
  local src="$1"
  local dest_root="$2"
  if [[ -z "${src}" ]]; then
    return
  fi
  if [[ -n "${STAGED_PATH_CACHE["${src}"]:-}" ]]; then
    echo "[INFO] Reusing staged path for ${src} -> ${STAGED_PATH_CACHE["${src}"]}" >&2
    echo "${STAGED_PATH_CACHE["${src}"]}"
    return
  fi
  if [[ ! -e "${src}" ]]; then
    echo "[WARN] Source path '${src}' not found for staging; using original path." >&2
    echo "${src}"
    return
  fi
  mkdir -p "${dest_root}"
  local digest=""
  if command -v sha1sum >/dev/null 2>&1; then
    echo "[INFO] Computing digest for ${src} using sha1sum" >&2
    digest=$(printf "%s" "${src}" | sha1sum | awk '{print $1}')
  elif command -v shasum >/dev/null 2>&1; then
    echo "[INFO] Computing digest for ${src} using shasum" >&2
    digest=$(printf "%s" "${src}" | shasum | awk '{print $1}')
  else
    echo "[INFO] Computing digest for ${src} using python hashlib" >&2
    digest=$(python - "$src" <<'PY'
import hashlib, sys
print(hashlib.sha1(sys.argv[1].encode("utf-8")).hexdigest())
PY
    )
  fi
  echo "[INFO] Digest for ${src}: ${digest}" >&2
  local dest="${dest_root}/${digest}_$(basename "${src}")"
  echo "[INFO] Staging ${src} -> ${dest}" >&2
  if [[ -d "${src}" ]]; then
    rsync -a "${src}/" "${dest}/" || true
  else
    rsync -a "${src}" "${dest}" || true
  fi
  STAGED_PATH_CACHE["${src}"]="${dest}"
  echo "${dest}"
}

resolve_model_path() {
  local candidate="$1"
  if [[ -z "${candidate}" ]]; then
    return
  fi
  if [[ -f "${candidate}" || -d "${candidate}" ]]; then
    echo "${candidate}"
    return
  fi
  if [[ -n "${HF_MODELS_CACHE:-}" && -d "${HF_MODELS_CACHE}/${candidate}" ]]; then
    echo "${HF_MODELS_CACHE}/${candidate}"
    return
  fi
}

echo "[INFO] Preparing global staged paths (checkpoint/tokenizer/artifacts)"
RESOLVED_CHECKPOINT_PATH="$(resolve_model_path "${CHECKPOINT_PATH:-}")"
RESOLVED_TOKENIZER_PATH="$(resolve_model_path "${TOKENIZER_PATH:-}")"
if [[ -n "${CHECKPOINT_PATH:-}" && -z "${RESOLVED_CHECKPOINT_PATH}" ]]; then
  echo "[WARN] CHECKPOINT_PATH not found on node: ${CHECKPOINT_PATH}" >&2
fi
if [[ -n "${TOKENIZER_PATH:-}" && -z "${RESOLVED_TOKENIZER_PATH}" ]]; then
  echo "[WARN] TOKENIZER_PATH not found on node: ${TOKENIZER_PATH}" >&2
fi
STAGED_CHECKPOINT_PATH="$(stage_path_once "${RESOLVED_CHECKPOINT_PATH:-}" "${STAGED_MODELS_DIR}")"
STAGED_TOKENIZER_PATH="$(stage_path_once "${RESOLVED_TOKENIZER_PATH:-}" "${STAGED_MODELS_DIR}")"
STAGED_UNSAFE_PROTOTYPES="$(stage_path_once "${UNSAFE_PROTOTYPES:-}" "${STAGED_ARTIFACTS_DIR}")"
echo "[INFO] Global staged paths ready."

# To add a new Hydra override flowing from submit_sbatch_experiments.py into tools.generate,
# list the env var here, default it above, and append it to the overrides array in run_one_config.
PER_CONFIG_VARS=(
  GEN_CONFIG_NAME
  EXPERIMENT_SLUG
  RUN_ID
  TRACK_NAME
  RESULTS_ROOT
  SAFETY_ENABLED
  SAFETY_ETA
  SAFETY_SCALE
  SAFETY_SEMANTIC_WEIGHT
  SAFETY_SEMANTIC_TEMP
  SAFETY_SEMANTIC_SIGMA
  UNSAFE_ARTIFACTS
  UNSAFE_ARTIFACT_ROOT
  UNSAFE_ARTIFACT_NAME
  UNSAFE_PROTOTYPES
  UNSAFE_PROTOTYPE_ROOT
  CHECKPOINT_PATH
  TOKENIZER_PATH
  PROMPT_LIMIT
  PROMPT_SOURCE_NAME
  MODEL_FAMILY
  MODEL_NAME
  MODEL_VARIANT
  ADD_BOS
  ADD_EOS
  UNCONDITIONAL_SAMPLES
  DRY_RUN
  CRITICAL_STEPS
  SAFETY_T_START
  SAFETY_T_END
  SAMPLING_STEPS
  MAX_NEW_TOKENS
  BLOCK_LENGTH
  TEMPERATURE
  GEN_BATCH_SIZE
  USE_SEMANTIC_GATING
  DATASET_JSON
  PROMPT_VARIANT
  LLAMAGUARD_CHECKPOINT_PATH
  N_PER_PROMPT
  FK_K_PARTICLES
  FK_ROBERTA_CHECKPOINT_PATH
)
for var in "${PER_CONFIG_VARS[@]}"; do
  eval "BASE_${var}=\${${var}:-}"
done
declare -A BASE_PROMPT_SOURCE_PARAMS=()
while IFS= read -r var; do
  BASE_PROMPT_SOURCE_PARAMS["${var}"]="${!var}"
done < <(compgen -v PROMPT_SOURCE_PARAM_)

reset_config_env() {
  for var in "${PER_CONFIG_VARS[@]}"; do
    local base_var="BASE_${var}"
    eval "${var}=\"\${${base_var}:-}\""
  done
  while IFS= read -r var; do
    unset "${var}"
  done < <(compgen -v PROMPT_SOURCE_PARAM_)
  for var in "${!BASE_PROMPT_SOURCE_PARAMS[@]}"; do
    export "${var}"="${BASE_PROMPT_SOURCE_PARAMS[$var]}"
  done
}

prepare_stage_paths() {
  echo "[INFO] Preparing per-config staged paths (checkpoint/tokenizer/artifacts)"
  CONFIG_CHECKPOINT_OVERRIDE="$(stage_path_once "${RESOLVED_CHECKPOINT_PATH:-}" "${STAGED_MODELS_DIR}")"
  CONFIG_TOKENIZER_OVERRIDE="$(stage_path_once "${RESOLVED_TOKENIZER_PATH:-}" "${STAGED_MODELS_DIR}")"
  # Stage LlamaGuard checkpoint (used by posthoc_filter / best_of_n variants)
  CONFIG_LLAMAGUARD_OVERRIDE=""
  if [[ -n "${LLAMAGUARD_CHECKPOINT_PATH:-}" ]]; then
    CONFIG_LLAMAGUARD_OVERRIDE="$(stage_path_once "${LLAMAGUARD_CHECKPOINT_PATH}" "${STAGED_MODELS_DIR}")"
    echo "[INFO] Staged LlamaGuard checkpoint: ${LLAMAGUARD_CHECKPOINT_PATH} -> ${CONFIG_LLAMAGUARD_OVERRIDE}"
  fi
  local artifact_src=""
  CONFIG_SEMANTIC_REF_OVERRIDE=""
  if [[ -n "${UNSAFE_ARTIFACT_ROOT:-}" && -n "${UNSAFE_ARTIFACT_NAME:-}" ]]; then
    echo "[INFO] Staging specific unsafe artifact ${UNSAFE_ARTIFACT_NAME} from root ${UNSAFE_ARTIFACT_ROOT}" >&2
    artifact_src="${UNSAFE_ARTIFACT_ROOT%/}/${UNSAFE_ARTIFACT_NAME}"
    local staged_dest="${STAGED_ARTIFACTS_DIR}/${UNSAFE_ARTIFACT_NAME}"
    mkdir -p "${staged_dest}"
    rsync -a "${artifact_src}/" "${staged_dest}/" || true
    CONFIG_ARTIFACT_ROOT_OVERRIDE="${STAGED_ARTIFACTS_DIR}"
    local src_index="${UNSAFE_ARTIFACT_ROOT%/}/index.json"
    if [[ -f "${src_index}" ]]; then
      rsync -a "${src_index}" "${STAGED_ARTIFACTS_DIR}/" || true
    fi

    local sem_root="${UNSAFE_SEMANTIC_ROOT:-${UNSAFE_ARTIFACT_ROOT%/}/semantic_refs}"
    local sem_file="${sem_root%/}/semantic_ref_embeddings_${UNSAFE_ARTIFACT_NAME}.pt"
    if [[ -f "${sem_file}" ]]; then
      local sem_dest_dir="${STAGED_ARTIFACTS_DIR}/semantic_refs"
      mkdir -p "${sem_dest_dir}"
      rsync -a "${sem_file}" "${sem_dest_dir}/" || true
      CONFIG_SEMANTIC_REF_OVERRIDE="${sem_dest_dir}/$(basename "${sem_file}")"
      echo "[INFO] Staged semantic cache ${sem_file} -> ${CONFIG_SEMANTIC_REF_OVERRIDE}" >&2
    else
      echo "[INFO] No semantic cache found at ${sem_file}; skipping semantic staging." >&2
    fi
  elif [[ -n "${UNSAFE_ARTIFACT_ROOT:-}" ]]; then
    echo "[INFO] Staging ENTIRE unsafe artifact root ${UNSAFE_ARTIFACT_ROOT}" >&2
    CONFIG_ARTIFACT_ROOT_OVERRIDE="$(stage_path_once "${UNSAFE_ARTIFACT_ROOT}" "${STAGED_ARTIFACTS_DIR}")"
  else
    echo "[INFO] No unsafe artifact root/name provided; skipping artifact staging." >&2
    CONFIG_ARTIFACT_ROOT_OVERRIDE=""
  fi
  CONFIG_PROTOTYPES_OVERRIDE=""
  if [[ -n "${UNSAFE_PROTOTYPES:-}" ]]; then
    echo "[INFO] Staging unsafe prototypes from ${UNSAFE_PROTOTYPES}" >&2
    CONFIG_PROTOTYPES_OVERRIDE="$(stage_path_once "${UNSAFE_PROTOTYPES}" "${STAGED_ARTIFACTS_DIR}")"
  elif [[ -n "${UNSAFE_PROTOTYPE_ROOT:-}" && -n "${UNSAFE_ARTIFACT_NAME:-}" ]]; then
    echo "[INFO] Staging unsafe prototype candidate for artifact ${UNSAFE_ARTIFACT_NAME} from root ${UNSAFE_PROTOTYPE_ROOT}" >&2
    local proto_candidate="${UNSAFE_PROTOTYPE_ROOT%/}/${UNSAFE_ARTIFACT_NAME}_k64.pt"
    if [[ -e "${proto_candidate}" ]]; then
      echo "[INFO] Staging unsafe prototype ${proto_candidate}" >&2
      CONFIG_PROTOTYPES_OVERRIDE="$(stage_path_once "${proto_candidate}" "${STAGED_ARTIFACTS_DIR}")"
    fi
  fi
}

declare -a WORKER_PIDS=()
declare -a WORKER_SLOTS=()

launch_worker() {
  echo "[INFO] Launching worker on GPU slot $1 (device_id=$2)"
  local gpu_slot=$1
  local device_id=$2
  local shard_id=$((TASK_ORD * GPU_COUNT + gpu_slot))
  local worker_root="${CURRENT_OUTPUT_DIR}/gpu_${gpu_slot}"
  mkdir -p "${worker_root}"
  local log_path="${CURRENT_LOG_DIR}/gpu_${gpu_slot}.log"

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
    "io.run_id=${CURRENT_RUN_ID}"
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
  if [[ -n "${CURRENT_EXPERIMENT_SLUG}" ]]; then
    overrides+=("io.experiment_slug=${CURRENT_EXPERIMENT_SLUG}")
  fi
  if [[ -n "${TRACK_NAME}" ]]; then
    overrides+=("io.track_name=${TRACK_NAME}")
  fi
  if [[ -n "${PROMPT_LIMIT}" ]]; then
    overrides+=("data.limit=${PROMPT_LIMIT}")
  fi
  if [[ -n "${SAMPLING_STEPS}" ]]; then
    overrides+=("gen.sampling_steps=${SAMPLING_STEPS}")
  fi
  if [[ -n "${MAX_NEW_TOKENS}" ]]; then
    overrides+=("gen.max_new_tokens=${MAX_NEW_TOKENS}")
  fi
  if [[ -n "${BLOCK_LENGTH}" ]]; then
    overrides+=("gen.block_length=${BLOCK_LENGTH}")
  fi
  if [[ -n "${TEMPERATURE}" ]]; then
    overrides+=("gen.temperature=${TEMPERATURE}")
  fi
  if [[ -n "${GEN_BATCH_SIZE}" ]]; then
    overrides+=("gen.batch_size=${GEN_BATCH_SIZE}")
  fi
  if [[ -n "${GEN_SEED}" ]]; then
    overrides+=("gen.seed=${GEN_SEED}")
  fi
  if [[ -n "${DATASET_JSON:-}" ]]; then
    overrides+=("data.dataset_json=${DATASET_JSON}")
  fi
  if [[ -n "${PROMPT_SOURCE_NAME:-}" ]]; then
    overrides+=("++data.prompt_source.name=${PROMPT_SOURCE_NAME}")
    overrides+=("data.dataset_json=null")
  fi
  mapfile -t PROMPT_SOURCE_PARAM_VARS < <(compgen -v PROMPT_SOURCE_PARAM_ | sort)
  for var in "${PROMPT_SOURCE_PARAM_VARS[@]}"; do
    val="${!var}"
    [[ -z "${val:-}" ]] && continue
    key="${var#PROMPT_SOURCE_PARAM_}"
    key="${key,,}"
    overrides+=("++data.prompt_source.params.${key}=${val}")
  done
  if [[ -n "${PROMPT_VARIANT:-}" ]]; then
    overrides+=("data.prompt_variant=${PROMPT_VARIANT}")
  fi
  if [[ -n "${MODEL_FAMILY:-}" ]]; then
    overrides+=("model.family=${MODEL_FAMILY}")
  fi
  if [[ -n "${MODEL_NAME:-}" ]]; then
    overrides+=("model.model_name=${MODEL_NAME}")
  fi
  if [[ -n "${MODEL_VARIANT:-}" ]]; then
    overrides+=("model.variant=${MODEL_VARIANT}")
  fi
  if [[ -n "${UNSAFE_ARTIFACTS}" ]]; then
    overrides+=("safety.unsafe_artifacts=${UNSAFE_ARTIFACTS}")
  fi
  if [[ -n "${CONFIG_ARTIFACT_ROOT_OVERRIDE:-}" ]]; then
    overrides+=("safety.unsafe_artifact_root=${CONFIG_ARTIFACT_ROOT_OVERRIDE}")
  fi
  if [[ -n "${UNSAFE_ARTIFACT_NAME}" ]]; then
    overrides+=("safety.unsafe_artifact_name=${UNSAFE_ARTIFACT_NAME}")
  fi
  if [[ -n "${USE_SEMANTIC_GATING:-}" ]]; then
    overrides+=("safety.use_semantic_gating=${USE_SEMANTIC_GATING}")
  fi
  if [[ -n "${CONFIG_SEMANTIC_REF_OVERRIDE:-}" ]]; then
    overrides+=("safety.semantic_ref_path=${CONFIG_SEMANTIC_REF_OVERRIDE}")
    overrides+=("safety.cache_semantic_ref=true")
    local semantic_flag="${USE_SEMANTIC_GATING:-true}"
    overrides+=("safety.use_semantic_gating=${semantic_flag}")
    if [[ -n "${SAFETY_SEMANTIC_WEIGHT:-}" ]]; then
      overrides+=("safety.semantic_weight=${SAFETY_SEMANTIC_WEIGHT}")
    fi
    if [[ -n "${SAFETY_SEMANTIC_TEMP:-}" ]]; then
      overrides+=("safety.semantic_temp=${SAFETY_SEMANTIC_TEMP}")
    fi
    if [[ -n "${SAFETY_SEMANTIC_SIGMA:-}" ]]; then
      overrides+=("safety.semantic_sigma=${SAFETY_SEMANTIC_SIGMA}")
    fi
  fi
  if [[ -n "${CONFIG_PROTOTYPES_OVERRIDE:-}" ]]; then
    overrides+=("safety.unsafe_prototypes=${CONFIG_PROTOTYPES_OVERRIDE}")
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
  if [[ -n "${CONFIG_CHECKPOINT_OVERRIDE:-}" ]]; then
    overrides+=("model.checkpoint=${CONFIG_CHECKPOINT_OVERRIDE}")
  fi
  if [[ -n "${CONFIG_TOKENIZER_OVERRIDE:-}" ]]; then
    overrides+=("model.tokenizer_name=${CONFIG_TOKENIZER_OVERRIDE}")
  fi
  # Export staged LlamaGuard path so posthoc_filter/best_of_n backends find it.
  if [[ -n "${CONFIG_LLAMAGUARD_OVERRIDE:-}" ]]; then
    export LLAMAGUARD_CHECKPOINT_PATH="${CONFIG_LLAMAGUARD_OVERRIDE}"
  fi
  if [[ -n "${N_PER_PROMPT:-}" ]]; then
    export N_PER_PROMPT="${N_PER_PROMPT}"
  fi
  if [[ -n "${FK_K_PARTICLES:-}" ]]; then
    export FK_K_PARTICLES="${FK_K_PARTICLES}"
  fi
  if [[ -n "${FK_ROBERTA_CHECKPOINT_PATH:-}" ]]; then
    export FK_ROBERTA_CHECKPOINT_PATH="${FK_ROBERTA_CHECKPOINT_PATH}"
  fi

  (
    set -x
    CUDA_VISIBLE_DEVICES="${device_id}" python -m tools.generate --config-name "${GEN_CONFIG_NAME}" "${overrides[@]}"
  ) 2>&1 | tee "${log_path}" | sed -e "s/^/[task_${SLURM_ARRAY_TASK_ID:-0}|gpu_${gpu_slot}] /" &
  WORKER_PIDS+=($!)
  WORKER_SLOTS+=("${gpu_slot}")
  echo "[INFO] Started worker gpu_${gpu_slot} pid=${WORKER_PIDS[-1]} shard_id=${shard_id}"
}

run_one_config() {
  local config_idx="$1"
  if [[ ${CONFIG_BATCH_MODE} -eq 1 ]]; then
    CURRENT_OUTPUT_DIR="${TMP_OUTPUT_ROOT}/config_${config_idx}"
    CURRENT_LOG_DIR="${LOG_ROOT}/config_${config_idx}"
  else
    CURRENT_OUTPUT_DIR="${TMP_OUTPUT_ROOT}"
    CURRENT_LOG_DIR="${LOG_ROOT}"
  fi
  mkdir -p "${CURRENT_OUTPUT_DIR}" "${CURRENT_LOG_DIR}"
  stage_pending=1
  CURRENT_STAGE_STATUS="failed"

  prepare_stage_paths
  WORKER_PIDS=()
  WORKER_SLOTS=()
  echo "[INFO] Launching workers on GPUs: ${GPU_IDS[*]}"
  for idx in "${!GPU_IDS[@]}"; do
    launch_worker "${idx}" "${GPU_IDS[$idx]}"
  done

  for idx in "${!WORKER_PIDS[@]}"; do
    pid="${WORKER_PIDS[$idx]}"
    wait "${pid}"
    echo "[INFO] Worker gpu_${WORKER_SLOTS[$idx]} (pid=${pid}) completed"
  done

  echo "[INFO] All workers for task ${SLURM_ARRAY_TASK_ID} completed for config index ${config_idx}; aggregating telemetry"

  python -m utils.slurm_helpers aggregate \
    --root "${CURRENT_OUTPUT_DIR}" \
    --output "${CURRENT_OUTPUT_DIR}/task_metadata.json" \
    --job-id "${SLURM_JOB_ID}" \
    --experiment-slug "${CURRENT_EXPERIMENT_SLUG}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --total-prompts 0 \
    --gpu-shards "${GPU_COUNT}" || true

  CURRENT_STAGE_STATUS="success"
  stage_out_config "${CURRENT_STAGE_STATUS}" "${CURRENT_EXPERIMENT_SLUG}" "${CURRENT_RUN_ID}" "${CURRENT_OUTPUT_DIR}" "${CURRENT_LOG_DIR}"
}

CONFIG_SPECS=()
CONFIG_BATCH_MODE=0
if [[ -n "${CONFIG_BATCH_FILE:-}" ]]; then
  if [[ ! -f "${CONFIG_BATCH_FILE}" ]]; then
    echo "[ERROR] CONFIG_BATCH_FILE='${CONFIG_BATCH_FILE}' does not exist." >&2
    exit 1
  fi
  CONFIG_BATCH_MODE=1
  while IFS= read -r line; do
    [[ -z "${line// }" ]] && continue
    CONFIG_SPECS+=("${line}")
  done < "${CONFIG_BATCH_FILE}"
elif [[ -n "${CONFIG_BATCH_SPECS:-}" ]]; then
  CONFIG_BATCH_MODE=1
  while IFS= read -r spec; do
    [[ -z "${spec// }" ]] && continue
    CONFIG_SPECS+=("${spec}")
  done <<< "$(echo "${CONFIG_BATCH_SPECS}" | tr ',' '\n')"
fi

if [[ ${CONFIG_BATCH_MODE} -eq 1 && ${#CONFIG_SPECS[@]} -eq 0 ]]; then
  echo "[ERROR] CONFIG_BATCH_* specified but no configs were found to run." >&2
  exit 1
fi

if [[ ${CONFIG_BATCH_MODE} -eq 0 ]]; then
  if [[ -z "${GEN_CONFIG_NAME:-}" ]]; then
    echo "[ERROR] GEN_CONFIG_NAME must be set to a Hydra config name (e.g., experiments/beavertails_prompts)." >&2
    exit 1
  fi
  CONFIG_SPECS+=("")
fi

for idx in "${!CONFIG_SPECS[@]}"; do
  reset_config_env
  spec="${CONFIG_SPECS[$idx]}"
  if [[ -n "${spec}" ]]; then
    echo "[INFO] Applying config spec #${idx}: ${spec}"
    eval "${spec}"
  fi
  echo "[INFO] Refreshing staged RealToxicityPrompts cache between configs"
  rm -rf "${HF_DATASETS_CACHE}/real-toxicity-prompts/default"
  rsync -a --delete "${REALTOXICITY_DATA_DIR}/" "${HF_DATASETS_CACHE}/real-toxicity-prompts/" || true
  if [[ -z "${GEN_CONFIG_NAME:-}" ]]; then
    echo "[ERROR] GEN_CONFIG_NAME must be set for config index ${idx}." >&2
    exit 1
  fi
  if [[ -z "${EXPERIMENT_SLUG:-}" ]]; then
    EXPERIMENT_SLUG="${GEN_CONFIG_NAME##*/}"
  fi
  if [[ -z "${RUN_ID:-}" ]]; then
    if [[ ${CONFIG_BATCH_MODE} -eq 1 ]]; then
      RUN_ID="${SLURM_JOB_ID}_cfg${idx}"
    else
      RUN_ID="${SLURM_JOB_ID}"
    fi
  fi
  CURRENT_EXPERIMENT_SLUG="${EXPERIMENT_SLUG}"
  CURRENT_RUN_ID="${RUN_ID}"
  echo "[INFO] Starting config index ${idx}: slug='${CURRENT_EXPERIMENT_SLUG}', run_id='${CURRENT_RUN_ID}', config='${GEN_CONFIG_NAME}'"
  run_one_config "${idx}"
  echo "[INFO] Finished config index ${idx}: slug='${CURRENT_EXPERIMENT_SLUG}', run_id='${CURRENT_RUN_ID}'"
done

echo "[INFO] Task ${SLURM_ARRAY_TASK_ID} for job ${SLURM_JOB_ID:-local} completed; staged outputs under ${RESULTS_ROOT}"
