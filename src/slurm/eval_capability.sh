#!/bin/bash
#SBATCH --job-name=eval_capability
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32G
#SBATCH --output=/scratch/%u/logs/safe-text-diffusion/%x_%A_%a.out
#SBATCH --error=/scratch/%u/logs/safe-text-diffusion/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT=${REPO_ROOT:-${DEFAULT_REPO_ROOT}}

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

CAP_TASKS=${CAP_TASKS:-}
CAP_MODEL_ARGS=${CAP_MODEL_ARGS:-}
CAP_EVAL_ARGS=${CAP_EVAL_ARGS:-}
CAP_BACKEND=${CAP_BACKEND:-${MODEL_FAMILY:-}}
CAP_BATCH_SIZE=${CAP_BATCH_SIZE:-}
CAP_NUM_FEWSHOT=${CAP_NUM_FEWSHOT:-}
CAP_APPLY_CHAT_TEMPLATE=${CAP_APPLY_CHAT_TEMPLATE:-}
CAP_CONFIRM_RUN_UNSAFE_CODE=${CAP_CONFIRM_RUN_UNSAFE_CODE:-}
CAP_LOG_SAMPLES=${CAP_LOG_SAMPLES:-}
CAP_ALLOW_CODE_EVAL=${CAP_ALLOW_CODE_EVAL:-}
CAP_LM_EVAL_TASK_PATH=${CAP_LM_EVAL_TASK_PATH:-}
CAP_ACCELERATE_PORT=${CAP_ACCELERATE_PORT:-12334}
CAP_ACCELERATE_ARGS=${CAP_ACCELERATE_ARGS:-}
CAP_STAGE_MODELS=${CAP_STAGE_MODELS:-1}
CAP_MONITOR=${CAP_MONITOR:-1}
CAP_MONITOR_INTERVAL=${CAP_MONITOR_INTERVAL:-10}
CAP_DMESG_ON_KILL=${CAP_DMESG_ON_KILL:-0}

MODEL_PATH=${MODEL_PATH:-${CHECKPOINT_PATH:-}}
OUTPUT_DIR=${OUTPUT_DIR:-${RESULTS_ROOT:-${SCRATCH:-$HOME}/results}/capability_eval/${SLURM_JOB_ID:-local}}

SAFETY_ENABLED=${SAFETY_ENABLED:-0}
SAFETY_ETA=${SAFETY_ETA:-}
SAFETY_SCALE=${SAFETY_SCALE:-}
SAFETY_AUTO_BUILD_UNSAFE_ARTIFACTS=${SAFETY_AUTO_BUILD_UNSAFE_ARTIFACTS:-}
UNSAFE_ARTIFACT_ROOT=${UNSAFE_ARTIFACT_ROOT:-}
UNSAFE_ARTIFACT_NAME=${UNSAFE_ARTIFACT_NAME:-}
UNSAFE_ARTIFACTS=${UNSAFE_ARTIFACTS:-}
SAFETY_T_START=${SAFETY_T_START:-}
SAFETY_T_END=${SAFETY_T_END:-}

CONFIG_BATCH_FILE=${CONFIG_BATCH_FILE:-}
CONFIG_BATCH_SPECS=${CONFIG_BATCH_SPECS:-}

BASE_CAP_TASKS="${CAP_TASKS}"
BASE_CAP_MODEL_ARGS="${CAP_MODEL_ARGS}"
BASE_CAP_EVAL_ARGS="${CAP_EVAL_ARGS}"
BASE_CAP_BACKEND="${CAP_BACKEND}"
BASE_CAP_BATCH_SIZE="${CAP_BATCH_SIZE}"
BASE_CAP_NUM_FEWSHOT="${CAP_NUM_FEWSHOT}"
BASE_CAP_APPLY_CHAT_TEMPLATE="${CAP_APPLY_CHAT_TEMPLATE}"
BASE_CAP_CONFIRM_RUN_UNSAFE_CODE="${CAP_CONFIRM_RUN_UNSAFE_CODE}"
BASE_CAP_LOG_SAMPLES="${CAP_LOG_SAMPLES}"
BASE_CAP_ALLOW_CODE_EVAL="${CAP_ALLOW_CODE_EVAL}"
BASE_CAP_LM_EVAL_TASK_PATH="${CAP_LM_EVAL_TASK_PATH}"
BASE_CAP_ACCELERATE_PORT="${CAP_ACCELERATE_PORT}"
BASE_MODEL_PATH="${MODEL_PATH}"
BASE_OUTPUT_DIR="${OUTPUT_DIR}"
BASE_SAFETY_ENABLED="${SAFETY_ENABLED}"
BASE_SAFETY_ETA="${SAFETY_ETA}"
BASE_SAFETY_SCALE="${SAFETY_SCALE}"
BASE_SAFETY_AUTO_BUILD_UNSAFE_ARTIFACTS="${SAFETY_AUTO_BUILD_UNSAFE_ARTIFACTS}"
BASE_UNSAFE_ARTIFACT_ROOT="${UNSAFE_ARTIFACT_ROOT}"
BASE_UNSAFE_ARTIFACT_NAME="${UNSAFE_ARTIFACT_NAME}"
BASE_UNSAFE_ARTIFACTS="${UNSAFE_ARTIFACTS}"
BASE_SAFETY_T_START="${SAFETY_T_START}"
BASE_SAFETY_T_END="${SAFETY_T_END}"

SLURM_TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}/slurm_tmp_${SLURM_JOB_ID:-local}}
mkdir -p "${SLURM_TMPDIR}"

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
    digest=$(printf "%s" "${src}" | sha1sum | awk '{print $1}')
  elif command -v shasum >/dev/null 2>&1; then
    digest=$(printf "%s" "${src}" | shasum | awk '{print $1}')
  else
    digest=$(python - "$src" <<'PY'
import hashlib, sys
print(hashlib.sha1(sys.argv[1].encode("utf-8")).hexdigest())
PY
    )
  fi
  local dest="${dest_root}/${digest}_$(basename "${src}")"
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

reset_config_env() {
  CAP_TASKS="${BASE_CAP_TASKS}"
  CAP_MODEL_ARGS="${BASE_CAP_MODEL_ARGS}"
  CAP_EVAL_ARGS="${BASE_CAP_EVAL_ARGS}"
  CAP_BACKEND="${BASE_CAP_BACKEND}"
  CAP_BATCH_SIZE="${BASE_CAP_BATCH_SIZE}"
  CAP_NUM_FEWSHOT="${BASE_CAP_NUM_FEWSHOT}"
  CAP_APPLY_CHAT_TEMPLATE="${BASE_CAP_APPLY_CHAT_TEMPLATE}"
  CAP_CONFIRM_RUN_UNSAFE_CODE="${BASE_CAP_CONFIRM_RUN_UNSAFE_CODE}"
  CAP_LOG_SAMPLES="${BASE_CAP_LOG_SAMPLES}"
  CAP_ALLOW_CODE_EVAL="${BASE_CAP_ALLOW_CODE_EVAL}"
  CAP_LM_EVAL_TASK_PATH="${BASE_CAP_LM_EVAL_TASK_PATH}"
  CAP_ACCELERATE_PORT="${BASE_CAP_ACCELERATE_PORT}"
  MODEL_PATH="${BASE_MODEL_PATH}"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}"
  SAFETY_ENABLED="${BASE_SAFETY_ENABLED}"
  SAFETY_ETA="${BASE_SAFETY_ETA}"
  SAFETY_SCALE="${BASE_SAFETY_SCALE}"
  SAFETY_AUTO_BUILD_UNSAFE_ARTIFACTS="${BASE_SAFETY_AUTO_BUILD_UNSAFE_ARTIFACTS}"
  UNSAFE_ARTIFACT_ROOT="${BASE_UNSAFE_ARTIFACT_ROOT}"
  UNSAFE_ARTIFACT_NAME="${BASE_UNSAFE_ARTIFACT_NAME}"
  UNSAFE_ARTIFACTS="${BASE_UNSAFE_ARTIFACTS}"
  SAFETY_T_START="${BASE_SAFETY_T_START}"
  SAFETY_T_END="${BASE_SAFETY_T_END}"
}

_append_arg() {
  local arr_name="$1"
  local flag="$2"
  local value="$3"
  if [[ -n "${value}" ]]; then
    eval "${arr_name}+=(\"${flag}\" \"${value}\")"
  fi
}

_append_flag() {
  local arr_name="$1"
  local flag="$2"
  local value="$3"
  if [[ "${value}" == "1" || "${value,,}" == "true" ]]; then
    eval "${arr_name}+=(\"${flag}\")"
  fi
}

_split_extra_args() {
  local raw="$1"
  local -n out_ref=$2
  if [[ -z "${raw}" ]]; then
    return
  fi
  local normalized="${raw//,/ }"
  local -a parts=()
  read -r -a parts <<< "${normalized}"
  for part in "${parts[@]}"; do
    [[ -z "${part// }" ]] && continue
    out_ref+=("${part}")
  done
}

# _monitor_pid() {
#   local pid="$1"
#   local interval="$2"
#   while kill -0 "${pid}" 2>/dev/null; do
#     if [[ -r "/proc/${pid}/status" ]]; then
#       local vmrss vmhwm vmsize vmpeak
#       vmrss=$(awk '/VmRSS:/ {print $2 " " $3}' "/proc/${pid}/status")
#       vmhwm=$(awk '/VmHWM:/ {print $2 " " $3}' "/proc/${pid}/status")
#       vmsize=$(awk '/VmSize:/ {print $2 " " $3}' "/proc/${pid}/status")
#       vmpeak=$(awk '/VmPeak:/ {print $2 " " $3}' "/proc/${pid}/status")
#       echo "[MONITOR] pid=${pid} VmRSS=${vmrss:-?} VmHWM=${vmhwm:-?} VmSize=${vmsize:-?} VmPeak=${vmpeak:-?}" >&2
#     fi
#     if command -v nvidia-smi >/dev/null 2>&1; then
#       nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
#         awk '{print "[MONITOR] gpu_mem_mb used=" $1 " total=" $2}' >&2 || true
#     fi
#     sleep "${interval}"
#   done
# }
_monitor_pid() {
  return 0
}

_run_with_monitor() {
  local -a cmd=("$@")
  if [[ "${CAP_MONITOR}" == "1" || "${CAP_MONITOR,,}" == "true" ]]; then
    "${cmd[@]}" &
    local cmd_pid=$!
    _monitor_pid "${cmd_pid}" "${CAP_MONITOR_INTERVAL}" &
    local mon_pid=$!
    wait "${cmd_pid}"
    local status=$?
    kill "${mon_pid}" 2>/dev/null || true
    if [[ ${status} -eq 137 || ${status} -eq 9 ]]; then
      echo "[MONITOR] process killed (exit=${status})." >&2
      if [[ "${CAP_DMESG_ON_KILL}" == "1" || "${CAP_DMESG_ON_KILL,,}" == "true" ]]; then
        dmesg | tail -n 50 >&2 || true
      fi
    fi
    return ${status}
  fi
  "${cmd[@]}"
}
run_one_config() {
  if [[ -z "${CAP_TASKS}" ]]; then
    echo "[ERROR] CAP_TASKS is required." >&2
    exit 1
  fi
  mkdir -p "${OUTPUT_DIR}"
  echo "[DEBUG] CAP_EVAL_ARGS raw: ${CAP_EVAL_ARGS}"
  echo "[DEBUG] CAP_MODEL_ARGS raw: ${CAP_MODEL_ARGS}"
  echo "[DEBUG] CAP_TASKS: ${CAP_TASKS}"

  local backend="${CAP_BACKEND}"
  backend=$(echo "${backend}" | tr '[:upper:]' '[:lower:]')

  local use_accelerate="${CAP_USE_ACCELERATE:-1}"
  if [[ -z "${backend}" || "${backend}" == "llada" ]]; then
    local lm_eval_task_path="${CAP_LM_EVAL_TASK_PATH}"
    if [[ -z "${lm_eval_task_path}" && "${CAP_TASKS}" == *"mmlu_generative"* ]]; then
      lm_eval_task_path="${REPO_ROOT}/src/third_party/Dream/eval_instruct/lm_eval/tasks"
    fi
    local batch_size_cli="${CAP_BATCH_SIZE}"
    local batch_size_model=""
    if [[ "${CAP_BATCH_SIZE,,}" == "auto" || "${CAP_BATCH_SIZE,,}" == "probe" || "${CAP_BATCH_SIZE,,}" == "oom" ]]; then
      batch_size_cli=""
      batch_size_model="auto"
    fi
    # Strip any batch_size from CAP_MODEL_ARGS to avoid lm_eval double-pass.
    local cleaned_model_args=()
    if [[ -n "${CAP_MODEL_ARGS}" ]]; then
      IFS=',' read -r -a _model_arg_parts <<< "${CAP_MODEL_ARGS}"
      for part in "${_model_arg_parts[@]}"; do
        # Trim leading/trailing whitespace for robust matching.
        part="${part#"${part%%[![:space:]]*}"}"
        part="${part%"${part##*[![:space:]]}"}"
        [[ -z "${part}" ]] && continue
        if [[ "${part}" == batch_size=* ]]; then
          continue
        fi
        cleaned_model_args+=("${part}")
      done
    fi
    local model_args_joined=""
    if [[ ${#cleaned_model_args[@]} -gt 0 ]]; then
      model_args_joined=$(IFS=','; echo "${cleaned_model_args[*]}")
    fi
    local accelerate_args=()
    _split_extra_args "${CAP_ACCELERATE_ARGS}" accelerate_args
    local cmd=()
    cmd=(accelerate launch "${accelerate_args[@]}")
    cmd+=("${REPO_ROOT}/src/third_party/LLaDA/eval_llada.py" \
      --tasks "${CAP_TASKS}" \
      --model llada_dist)
    _append_arg cmd "--batch_size" "${batch_size_cli}"
    _append_arg cmd "--num_fewshot" "${CAP_NUM_FEWSHOT}"
    _append_arg cmd "--output_path" "${OUTPUT_DIR}"
    _append_flag cmd "--confirm_run_unsafe_code" "${CAP_CONFIRM_RUN_UNSAFE_CODE}"
    _append_flag cmd "--apply_chat_template" "${CAP_APPLY_CHAT_TEMPLATE}"

    local model_args="${model_args_joined}"
    if [[ -z "${model_args}" ]]; then
      model_args="model_path=${MODEL_PATH}"
    else
      model_args="model_path=${MODEL_PATH},${model_args}"
    fi
    if [[ -n "${batch_size_model}" ]]; then
      model_args="${model_args},batch_size=${batch_size_model}"
    fi
    cmd+=(--model_args "${model_args}")

    local extra_args=()
    _split_extra_args "${CAP_EVAL_ARGS}" extra_args
    if [[ ${#extra_args[@]} -gt 0 ]]; then
      echo "[DEBUG] LLaDA extra_args: ${extra_args[*]}"
    else
      echo "[DEBUG] LLaDA extra_args: (none)"
    fi
    if [[ ${#extra_args[@]} -gt 0 ]]; then
      cmd+=("${extra_args[@]}")
    fi

    echo "[INFO] Running LLaDA capability eval: ${CAP_TASKS}"
    echo "[DEBUG] LLaDA cmd: ${cmd[*]}"
    PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/src/third_party/LLaDA:${PYTHONPATH:-}" \
      LLADA_GEN_CHECKPOINT_PATH="${OUTPUT_DIR}/llada_gen_cache.jsonl" \
      LM_EVAL_TASK_PATH="${lm_eval_task_path}" \
      HF_ALLOW_CODE_EVAL=${CAP_ALLOW_CODE_EVAL:-0} \
      _run_with_monitor "${cmd[@]}"
    local status=$?
    if [[ ${status} -eq 0 ]]; then
      echo "[INFO] Results saved to: ${OUTPUT_DIR}"
    else
      echo "[WARN] LLaDA eval failed (exit=${status}). Results may be incomplete at: ${OUTPUT_DIR}" >&2
    fi
    return ${status}
  fi

  if [[ "${backend}" == "dream" ]]; then
    local accelerate_args=()
    _split_extra_args "${CAP_ACCELERATE_ARGS}" accelerate_args
    local cmd=()
    cmd=(accelerate launch "${accelerate_args[@]}" --main_process_port "${CAP_ACCELERATE_PORT}" -m lm_eval)
    cmd+=(--model diffllm \
      --tasks "${CAP_TASKS}" \
      --device cuda \
      --output_path "${OUTPUT_DIR}")
    _append_arg cmd "--batch_size" "${CAP_BATCH_SIZE}"
    _append_arg cmd "--num_fewshot" "${CAP_NUM_FEWSHOT}"
    _append_flag cmd "--log_samples" "${CAP_LOG_SAMPLES}"
    _append_flag cmd "--confirm_run_unsafe_code" "${CAP_CONFIRM_RUN_UNSAFE_CODE}"
    _append_flag cmd "--apply_chat_template" "${CAP_APPLY_CHAT_TEMPLATE}"

    local model_args="${CAP_MODEL_ARGS}"
    if [[ -z "${model_args}" ]]; then
      model_args="pretrained=${MODEL_PATH}"
    else
      model_args="pretrained=${MODEL_PATH},${model_args}"
    fi
    cmd+=(--model_args "${model_args}")

    local extra_args=()
    _split_extra_args "${CAP_EVAL_ARGS}" extra_args
    if [[ ${#extra_args[@]} -gt 0 ]]; then
      cmd+=("${extra_args[@]}")
    fi

    echo "[INFO] Running Dream capability eval: ${CAP_TASKS}"
    PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/src/third_party/Dream/eval_instruct:${PYTHONPATH:-}" \
      HF_ALLOW_CODE_EVAL=${CAP_ALLOW_CODE_EVAL:-0} \
      _run_with_monitor "${cmd[@]}"
    local status=$?
    if [[ ${status} -eq 0 ]]; then
      echo "[INFO] Results saved to: ${OUTPUT_DIR}"
    else
      echo "[WARN] Dream eval failed (exit=${status}). Results may be incomplete at: ${OUTPUT_DIR}" >&2
    fi
    return ${status}
  fi

  echo "[ERROR] Unsupported CAP_BACKEND='${CAP_BACKEND}'" >&2
  exit 1
}

CONFIG_SPECS=()
CONFIG_BATCH_MODE=0
if [[ -n "${CONFIG_BATCH_FILE}" ]]; then
  if [[ ! -f "${CONFIG_BATCH_FILE}" ]]; then
    echo "[ERROR] CONFIG_BATCH_FILE='${CONFIG_BATCH_FILE}' does not exist." >&2
    exit 1
  fi
  CONFIG_BATCH_MODE=1
  while IFS= read -r line; do
    [[ -z "${line// }" ]] && continue
    CONFIG_SPECS+=("${line}")
  done < "${CONFIG_BATCH_FILE}"
elif [[ -n "${CONFIG_BATCH_SPECS}" ]]; then
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
  CONFIG_SPECS+=("")
fi

if [[ "${CAP_STAGE_MODELS}" == "1" || "${CAP_STAGE_MODELS,,}" == "true" ]]; then
  echo "[INFO] Staging inputs to SLURM_TMPDIR: ${SLURM_TMPDIR}"
  declare -A STAGED_PATH_CACHE=()
  STAGED_MODELS_DIR="${SLURM_TMPDIR}/staged_models"
  mkdir -p "${STAGED_MODELS_DIR}"

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
  echo "[INFO] HuggingFace caches set up at ${HF_HOME}, ${HF_DATASETS_CACHE}, ${HF_MODELS_CACHE}"
else
  echo "[INFO] Model staging disabled (CAP_STAGE_MODELS=${CAP_STAGE_MODELS})."
  STAGED_MODELS_DIR=""
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

for idx in "${!CONFIG_SPECS[@]}"; do
  reset_config_env
  spec="${CONFIG_SPECS[$idx]}"
  if [[ -n "${spec}" ]]; then
    echo "[INFO] Applying config spec #${idx}: ${spec}"
    eval "${spec}"
  fi
  if [[ -n "${MODEL_PATH}" ]]; then
    RESOLVED_MODEL_PATH="$(resolve_model_path "${MODEL_PATH}")"
    if [[ -z "${RESOLVED_MODEL_PATH}" ]]; then
      echo "[WARN] MODEL_PATH not found on node: ${MODEL_PATH}" >&2
      RESOLVED_MODEL_PATH="${MODEL_PATH}"
    fi
    if [[ "${CAP_STAGE_MODELS}" == "1" || "${CAP_STAGE_MODELS,,}" == "true" ]]; then
      MODEL_PATH="$(stage_path_once "${RESOLVED_MODEL_PATH}" "${STAGED_MODELS_DIR}")"
    else
      MODEL_PATH="${RESOLVED_MODEL_PATH}"
    fi
  fi
  run_one_config
done
