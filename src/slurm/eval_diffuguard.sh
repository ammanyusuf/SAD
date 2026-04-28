#!/bin/bash
#SBATCH --job-name=eval_diffuguard
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

EVAL_CONFIG_NAME=${EVAL_CONFIG_NAME:-config}
MODEL_PATH=${MODEL_PATH:-${CHECKPOINT_PATH:-}}
MODEL_FAMILY=${MODEL_FAMILY:-}
MODEL_VARIANT=${MODEL_VARIANT:-}
MODEL_NAME=${MODEL_NAME:-}
TOKENIZER_PATH=${TOKENIZER_PATH:-}
ATTACK_PROMPT=${ATTACK_PROMPT:-${DATASET_JSON:-}}
OUTPUT_DIR=${OUTPUT_DIR:-${RESULTS_ROOT:-${SCRATCH:-$HOME}/results}/diffuguard_eval/${SLURM_JOB_ID:-local}}
OUTPUT_NAME=${OUTPUT_NAME:-diffuguard_outputs.json}
PROMPT_LIMIT=${PROMPT_LIMIT:-}

SAFETY_ENABLED=${SAFETY_ENABLED:-0}
SAFETY_ETA=${SAFETY_ETA:-}
SAFETY_SCALE=${SAFETY_SCALE:-}
UNSAFE_ARTIFACT_ROOT=${UNSAFE_ARTIFACT_ROOT:-}
UNSAFE_ARTIFACT_NAME=${UNSAFE_ARTIFACT_NAME:-}
UNSAFE_ARTIFACTS=${UNSAFE_ARTIFACTS:-}
SAFETY_T_START=${SAFETY_T_START:-}
SAFETY_T_END=${SAFETY_T_END:-}

JAILBREAK_STEPS=${JAILBREAK_STEPS:-}
JAILBREAK_GEN_LENGTH=${JAILBREAK_GEN_LENGTH:-}
JAILBREAK_BLOCK_LENGTH=${JAILBREAK_BLOCK_LENGTH:-}
JAILBREAK_TEMPERATURE=${JAILBREAK_TEMPERATURE:-}
JAILBREAK_CFG_SCALE=${JAILBREAK_CFG_SCALE:-}
JAILBREAK_REMASKING=${JAILBREAK_REMASKING:-}
JAILBREAK_RANDOM_RATE=${JAILBREAK_RANDOM_RATE:-}
JAILBREAK_INJECTION_STEP=${JAILBREAK_INJECTION_STEP:-}
JAILBREAK_ALPHA0=${JAILBREAK_ALPHA0:-}
JAILBREAK_SP_MODE=${JAILBREAK_SP_MODE:-}
JAILBREAK_SP_THRESHOLD=${JAILBREAK_SP_THRESHOLD:-}
JAILBREAK_REFINEMENT_STEPS=${JAILBREAK_REFINEMENT_STEPS:-}
JAILBREAK_REMASK_RATIO=${JAILBREAK_REMASK_RATIO:-}
JAILBREAK_SUPPRESSION_VALUE=${JAILBREAK_SUPPRESSION_VALUE:-}
JAILBREAK_ATTACK_METHOD=${JAILBREAK_ATTACK_METHOD:-}
JAILBREAK_DEFENSE_METHOD=${JAILBREAK_DEFENSE_METHOD:-}
JAILBREAK_MASK_ID=${JAILBREAK_MASK_ID:-}
JAILBREAK_MASK_COUNTS=${JAILBREAK_MASK_COUNTS:-}
JAILBREAK_FILL_ALL_MASKS=${JAILBREAK_FILL_ALL_MASKS:-0}
JAILBREAK_DEBUG_PRINT=${JAILBREAK_DEBUG_PRINT:-0}
JAILBREAK_CORRECT_ONLY_FIRST_BLOCK=${JAILBREAK_CORRECT_ONLY_FIRST_BLOCK:-}
JAILBREAK_AUTO_PICK_GPU=${JAILBREAK_AUTO_PICK_GPU:-}
GEN_SEED=${GEN_SEED:-}
PPL_GPT2_MODEL=${PPL_GPT2_MODEL:-gpt2}

CONFIG_BATCH_FILE=${CONFIG_BATCH_FILE:-}
CONFIG_BATCH_SPECS=${CONFIG_BATCH_SPECS:-}

BASE_EVAL_CONFIG_NAME="${EVAL_CONFIG_NAME}"
BASE_MODEL_PATH="${MODEL_PATH}"
BASE_MODEL_FAMILY="${MODEL_FAMILY}"
BASE_MODEL_VARIANT="${MODEL_VARIANT}"
BASE_MODEL_NAME="${MODEL_NAME}"
BASE_TOKENIZER_PATH="${TOKENIZER_PATH}"
BASE_ATTACK_PROMPT="${ATTACK_PROMPT}"
BASE_OUTPUT_DIR="${OUTPUT_DIR}"
BASE_OUTPUT_NAME="${OUTPUT_NAME}"
BASE_PROMPT_LIMIT="${PROMPT_LIMIT}"
BASE_SAFETY_ENABLED="${SAFETY_ENABLED}"
BASE_SAFETY_ETA="${SAFETY_ETA}"
BASE_SAFETY_SCALE="${SAFETY_SCALE}"
BASE_UNSAFE_ARTIFACT_ROOT="${UNSAFE_ARTIFACT_ROOT}"
BASE_UNSAFE_ARTIFACT_NAME="${UNSAFE_ARTIFACT_NAME}"
BASE_UNSAFE_ARTIFACTS="${UNSAFE_ARTIFACTS}"
BASE_SAFETY_T_START="${SAFETY_T_START}"
BASE_SAFETY_T_END="${SAFETY_T_END}"
BASE_JAILBREAK_STEPS="${JAILBREAK_STEPS}"
BASE_JAILBREAK_GEN_LENGTH="${JAILBREAK_GEN_LENGTH}"
BASE_JAILBREAK_BLOCK_LENGTH="${JAILBREAK_BLOCK_LENGTH}"
BASE_JAILBREAK_TEMPERATURE="${JAILBREAK_TEMPERATURE}"
BASE_JAILBREAK_CFG_SCALE="${JAILBREAK_CFG_SCALE}"
BASE_JAILBREAK_REMASKING="${JAILBREAK_REMASKING}"
BASE_JAILBREAK_RANDOM_RATE="${JAILBREAK_RANDOM_RATE}"
BASE_JAILBREAK_INJECTION_STEP="${JAILBREAK_INJECTION_STEP}"
BASE_JAILBREAK_ALPHA0="${JAILBREAK_ALPHA0}"
BASE_JAILBREAK_SP_MODE="${JAILBREAK_SP_MODE}"
BASE_JAILBREAK_SP_THRESHOLD="${JAILBREAK_SP_THRESHOLD}"
BASE_JAILBREAK_REFINEMENT_STEPS="${JAILBREAK_REFINEMENT_STEPS}"
BASE_JAILBREAK_REMASK_RATIO="${JAILBREAK_REMASK_RATIO}"
BASE_JAILBREAK_SUPPRESSION_VALUE="${JAILBREAK_SUPPRESSION_VALUE}"
BASE_JAILBREAK_ATTACK_METHOD="${JAILBREAK_ATTACK_METHOD}"
BASE_JAILBREAK_DEFENSE_METHOD="${JAILBREAK_DEFENSE_METHOD}"
BASE_JAILBREAK_MASK_ID="${JAILBREAK_MASK_ID}"
BASE_JAILBREAK_MASK_COUNTS="${JAILBREAK_MASK_COUNTS}"
BASE_JAILBREAK_FILL_ALL_MASKS="${JAILBREAK_FILL_ALL_MASKS}"
BASE_JAILBREAK_DEBUG_PRINT="${JAILBREAK_DEBUG_PRINT}"
BASE_JAILBREAK_CORRECT_ONLY_FIRST_BLOCK="${JAILBREAK_CORRECT_ONLY_FIRST_BLOCK}"
BASE_JAILBREAK_AUTO_PICK_GPU="${JAILBREAK_AUTO_PICK_GPU}"
BASE_GEN_SEED="${GEN_SEED}"
BASE_PPL_GPT2_MODEL="${PPL_GPT2_MODEL}"

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

hydra_quote() {
  # Quote override values so Hydra can parse paths containing special chars (e.g. seed=1).
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "${value}"
}

reset_config_env() {
  EVAL_CONFIG_NAME="${BASE_EVAL_CONFIG_NAME}"
  MODEL_PATH="${BASE_MODEL_PATH}"
  MODEL_FAMILY="${BASE_MODEL_FAMILY}"
  MODEL_VARIANT="${BASE_MODEL_VARIANT}"
  MODEL_NAME="${BASE_MODEL_NAME}"
  TOKENIZER_PATH="${BASE_TOKENIZER_PATH}"
  ATTACK_PROMPT="${BASE_ATTACK_PROMPT}"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}"
  OUTPUT_NAME="${BASE_OUTPUT_NAME}"
  PROMPT_LIMIT="${BASE_PROMPT_LIMIT}"
  SAFETY_ENABLED="${BASE_SAFETY_ENABLED}"
  SAFETY_ETA="${BASE_SAFETY_ETA}"
  SAFETY_SCALE="${BASE_SAFETY_SCALE}"
  UNSAFE_ARTIFACT_ROOT="${BASE_UNSAFE_ARTIFACT_ROOT}"
  UNSAFE_ARTIFACT_NAME="${BASE_UNSAFE_ARTIFACT_NAME}"
  UNSAFE_ARTIFACTS="${BASE_UNSAFE_ARTIFACTS}"
  SAFETY_T_START="${BASE_SAFETY_T_START}"
  SAFETY_T_END="${BASE_SAFETY_T_END}"
  JAILBREAK_STEPS="${BASE_JAILBREAK_STEPS}"
  JAILBREAK_GEN_LENGTH="${BASE_JAILBREAK_GEN_LENGTH}"
  JAILBREAK_BLOCK_LENGTH="${BASE_JAILBREAK_BLOCK_LENGTH}"
  JAILBREAK_TEMPERATURE="${BASE_JAILBREAK_TEMPERATURE}"
  JAILBREAK_CFG_SCALE="${BASE_JAILBREAK_CFG_SCALE}"
  JAILBREAK_REMASKING="${BASE_JAILBREAK_REMASKING}"
  JAILBREAK_RANDOM_RATE="${BASE_JAILBREAK_RANDOM_RATE}"
  JAILBREAK_INJECTION_STEP="${BASE_JAILBREAK_INJECTION_STEP}"
  JAILBREAK_ALPHA0="${BASE_JAILBREAK_ALPHA0}"
  JAILBREAK_SP_MODE="${BASE_JAILBREAK_SP_MODE}"
  JAILBREAK_SP_THRESHOLD="${BASE_JAILBREAK_SP_THRESHOLD}"
  JAILBREAK_REFINEMENT_STEPS="${BASE_JAILBREAK_REFINEMENT_STEPS}"
  JAILBREAK_REMASK_RATIO="${BASE_JAILBREAK_REMASK_RATIO}"
  JAILBREAK_SUPPRESSION_VALUE="${BASE_JAILBREAK_SUPPRESSION_VALUE}"
  JAILBREAK_ATTACK_METHOD="${BASE_JAILBREAK_ATTACK_METHOD}"
  JAILBREAK_DEFENSE_METHOD="${BASE_JAILBREAK_DEFENSE_METHOD}"
  JAILBREAK_MASK_ID="${BASE_JAILBREAK_MASK_ID}"
  JAILBREAK_MASK_COUNTS="${BASE_JAILBREAK_MASK_COUNTS}"
  JAILBREAK_FILL_ALL_MASKS="${BASE_JAILBREAK_FILL_ALL_MASKS}"
  JAILBREAK_DEBUG_PRINT="${BASE_JAILBREAK_DEBUG_PRINT}"
  JAILBREAK_CORRECT_ONLY_FIRST_BLOCK="${BASE_JAILBREAK_CORRECT_ONLY_FIRST_BLOCK}"
  JAILBREAK_AUTO_PICK_GPU="${BASE_JAILBREAK_AUTO_PICK_GPU}"
  GEN_SEED="${BASE_GEN_SEED}"
  PPL_GPT2_MODEL="${BASE_PPL_GPT2_MODEL}"
}

echo "[INFO] Staging inputs to SLURM_TMPDIR: ${SLURM_TMPDIR}"
declare -A STAGED_PATH_CACHE=()
STAGED_MODELS_DIR="${SLURM_TMPDIR}/staged_models"
STAGED_DATA_DIR="${SLURM_TMPDIR}/staged_data"
STAGED_ARTIFACTS_DIR="${SLURM_TMPDIR}/staged_artifacts"
mkdir -p "${STAGED_MODELS_DIR}" "${STAGED_DATA_DIR}" "${STAGED_ARTIFACTS_DIR}"

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

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

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

stage_ppl_model() {
  local requested="${1:-${PPL_GPT2_MODEL:-gpt2}}"
  if [[ -d "${requested}" && "${requested}" == "${HF_MODELS_CACHE}/"* ]]; then
    export PPL_GPT2_MODEL="${requested}"
    echo "[INFO] Reusing staged PPL GPT-2 model at ${PPL_GPT2_MODEL}"
    return
  fi

  local source_path
  source_path="$(stage_model_path "${requested}")"
  if [[ -z "${source_path}" ]]; then
    echo "[WARN] Could not locate PPL GPT-2 model '${requested}' in local cache; PPL defense may fail." >&2
    export PPL_GPT2_MODEL="${requested}"
    return
  fi

  local dest
  if [[ -n "${SRC_HF_MODELS_CACHE:-}" && "${source_path}" == "${SRC_HF_MODELS_CACHE}/"* ]]; then
    local rel_path
    rel_path="${source_path#${SRC_HF_MODELS_CACHE}/}"
    dest="${HF_MODELS_CACHE}/${rel_path}"
  else
    dest="${HF_MODELS_CACHE}/$(basename "${source_path}")"
  fi

  if [[ -d "${source_path}" ]]; then
    mkdir -p "${dest}"
    rsync -a --progress --human-readable "${source_path}/" "${dest}/" || true
  else
    mkdir -p "$(dirname "${dest}")"
    rsync -a --progress --human-readable "${source_path}" "${dest}" || true
  fi
  export PPL_GPT2_MODEL="${dest}"
  echo "[INFO] Staged PPL GPT-2 model to ${PPL_GPT2_MODEL} (requested='${requested}')"
}

# Stage default GPT-2 model for PPL defense once; per-config overrides are restaged in run_one_config.
stage_ppl_model "${PPL_GPT2_MODEL}"

run_one_config() {
  local config_idx="$1"
  if [[ -z "${MODEL_PATH}" ]]; then
    echo "[ERROR] MODEL_PATH is required." >&2
    exit 1
  fi
  if [[ -z "${ATTACK_PROMPT}" ]]; then
    echo "[ERROR] ATTACK_PROMPT is required." >&2
    exit 1
  fi

  RESOLVED_MODEL_PATH="$(resolve_model_path "${MODEL_PATH}")"
  if [[ -n "${MODEL_PATH}" && -z "${RESOLVED_MODEL_PATH}" ]]; then
    echo "[WARN] MODEL_PATH not found on node: ${MODEL_PATH}" >&2
    RESOLVED_MODEL_PATH="${MODEL_PATH}"
  fi
  MODEL_PATH="$(stage_path_once "${RESOLVED_MODEL_PATH}" "${STAGED_MODELS_DIR}")"
  if [[ -f "${ATTACK_PROMPT}" || -d "${ATTACK_PROMPT}" ]]; then
    ATTACK_PROMPT="$(stage_path_once "${ATTACK_PROMPT}" "${STAGED_DATA_DIR}")"
  fi

  if [[ -n "${UNSAFE_ARTIFACTS}" ]]; then
    UNSAFE_ARTIFACTS="$(stage_path_once "${UNSAFE_ARTIFACTS}" "${STAGED_ARTIFACTS_DIR}")"
  elif [[ -n "${UNSAFE_ARTIFACT_ROOT}" && -n "${UNSAFE_ARTIFACT_NAME}" ]]; then
    artifact_src="${UNSAFE_ARTIFACT_ROOT%/}/${UNSAFE_ARTIFACT_NAME}"
    if [[ -d "${artifact_src}" ]]; then
      staged_dest="${STAGED_ARTIFACTS_DIR}/${UNSAFE_ARTIFACT_NAME}"
      mkdir -p "${staged_dest}"
      rsync -a "${artifact_src}/" "${staged_dest}/" || true
      src_index="${UNSAFE_ARTIFACT_ROOT%/}/index.json"
      if [[ -f "${src_index}" ]]; then
        rsync -a "${src_index}" "${STAGED_ARTIFACTS_DIR}/" || true
      fi
      UNSAFE_ARTIFACT_ROOT="${STAGED_ARTIFACTS_DIR}"
    else
      echo "[WARN] Unsafe artifact source not found: ${artifact_src}" >&2
    fi
  elif [[ -n "${UNSAFE_ARTIFACT_ROOT}" ]]; then
    UNSAFE_ARTIFACT_ROOT="$(stage_path_once "${UNSAFE_ARTIFACT_ROOT}" "${STAGED_ARTIFACTS_DIR}")"
  fi

  # CONFIG_BATCH specs may override PPL_GPT2_MODEL after initial staging; normalize it per config.
  stage_ppl_model "${PPL_GPT2_MODEL}"

  safety_bool="false"
  if [[ "${SAFETY_ENABLED}" == "1" || "${SAFETY_ENABLED,,}" == "true" ]]; then
    safety_bool="true"
  fi

  mkdir -p "${OUTPUT_DIR}"

  cd "${REPO_ROOT}"

  export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

  overrides=(
    "io.output_dir=$(hydra_quote "${OUTPUT_DIR}")"
    "model.checkpoint=$(hydra_quote "${MODEL_PATH}")"
    "data.dataset_json=$(hydra_quote "${ATTACK_PROMPT}")"
    "jailbreak.output_name=$(hydra_quote "${OUTPUT_NAME}")"
    "safety.enabled=${safety_bool}"
  )
  [[ -n "${PROMPT_LIMIT}" ]] && overrides+=("data.limit=${PROMPT_LIMIT}")

  [[ -n "${MODEL_FAMILY}" ]] && overrides+=("model.family=$(hydra_quote "${MODEL_FAMILY}")")
  [[ -n "${MODEL_VARIANT}" ]] && overrides+=("model.variant=$(hydra_quote "${MODEL_VARIANT}")")
  [[ -n "${MODEL_NAME}" ]] && overrides+=("model.model_name=$(hydra_quote "${MODEL_NAME}")")
  [[ -n "${TOKENIZER_PATH}" ]] && overrides+=("model.tokenizer_name=$(hydra_quote "${TOKENIZER_PATH}")")

  [[ -n "${SAFETY_ETA}" ]] && overrides+=("safety.eta=${SAFETY_ETA}")
  [[ -n "${SAFETY_SCALE}" ]] && overrides+=("safety.scale=${SAFETY_SCALE}")
  [[ -n "${UNSAFE_ARTIFACTS}" ]] && overrides+=("safety.unsafe_artifacts=$(hydra_quote "${UNSAFE_ARTIFACTS}")")
  [[ -n "${UNSAFE_ARTIFACT_ROOT}" ]] && overrides+=("safety.unsafe_artifact_root=$(hydra_quote "${UNSAFE_ARTIFACT_ROOT}")")
  [[ -n "${UNSAFE_ARTIFACT_NAME}" ]] && overrides+=("safety.unsafe_artifact_name=$(hydra_quote "${UNSAFE_ARTIFACT_NAME}")")
  [[ -n "${SAFETY_T_START}" ]] && overrides+=("safety.t_start=${SAFETY_T_START}")
  [[ -n "${SAFETY_T_END}" ]] && overrides+=("safety.t_end=${SAFETY_T_END}")

  [[ -n "${JAILBREAK_STEPS}" ]] && overrides+=("jailbreak.steps=${JAILBREAK_STEPS}")
  [[ -n "${JAILBREAK_GEN_LENGTH}" ]] && overrides+=("jailbreak.gen_length=${JAILBREAK_GEN_LENGTH}")
  [[ -n "${JAILBREAK_BLOCK_LENGTH}" ]] && overrides+=("jailbreak.block_length=${JAILBREAK_BLOCK_LENGTH}")
  [[ -n "${JAILBREAK_TEMPERATURE}" ]] && overrides+=("jailbreak.temperature=${JAILBREAK_TEMPERATURE}")
  [[ -n "${JAILBREAK_CFG_SCALE}" ]] && overrides+=("jailbreak.cfg_scale=${JAILBREAK_CFG_SCALE}")
  [[ -n "${JAILBREAK_REMASKING}" ]] && overrides+=("jailbreak.remasking=${JAILBREAK_REMASKING}")
  [[ -n "${JAILBREAK_RANDOM_RATE}" ]] && overrides+=("jailbreak.random_rate=${JAILBREAK_RANDOM_RATE}")
  [[ -n "${JAILBREAK_INJECTION_STEP}" ]] && overrides+=("jailbreak.injection_step=${JAILBREAK_INJECTION_STEP}")
  [[ -n "${JAILBREAK_ALPHA0}" ]] && overrides+=("jailbreak.alpha0=${JAILBREAK_ALPHA0}")
  [[ -n "${JAILBREAK_SP_MODE}" ]] && overrides+=("jailbreak.sp_mode=${JAILBREAK_SP_MODE}")
  [[ -n "${JAILBREAK_SP_THRESHOLD}" ]] && overrides+=("jailbreak.sp_threshold=${JAILBREAK_SP_THRESHOLD}")
  [[ -n "${JAILBREAK_REFINEMENT_STEPS}" ]] && overrides+=("jailbreak.refinement_steps=${JAILBREAK_REFINEMENT_STEPS}")
  [[ -n "${JAILBREAK_REMASK_RATIO}" ]] && overrides+=("jailbreak.remask_ratio=${JAILBREAK_REMASK_RATIO}")
  [[ -n "${JAILBREAK_SUPPRESSION_VALUE}" ]] && overrides+=("jailbreak.suppression_value=${JAILBREAK_SUPPRESSION_VALUE}")
  [[ -n "${JAILBREAK_ATTACK_METHOD}" ]] && overrides+=("jailbreak.attack_method=${JAILBREAK_ATTACK_METHOD}")
  [[ -n "${JAILBREAK_DEFENSE_METHOD}" ]] && overrides+=("jailbreak.defense_method=${JAILBREAK_DEFENSE_METHOD}")
  [[ -n "${JAILBREAK_MASK_ID}" ]] && overrides+=("jailbreak.mask_id=${JAILBREAK_MASK_ID}")
  [[ -n "${JAILBREAK_MASK_COUNTS}" ]] && overrides+=("jailbreak.mask_counts=${JAILBREAK_MASK_COUNTS}")
  [[ -n "${JAILBREAK_CORRECT_ONLY_FIRST_BLOCK}" ]] && overrides+=("jailbreak.correct_only_first_block=${JAILBREAK_CORRECT_ONLY_FIRST_BLOCK}")
  [[ -n "${JAILBREAK_AUTO_PICK_GPU}" ]] && overrides+=("jailbreak.auto_pick_gpu=${JAILBREAK_AUTO_PICK_GPU}")
  [[ -n "${GEN_SEED}" ]] && overrides+=("gen.seed=${GEN_SEED}")

  if [[ "${JAILBREAK_FILL_ALL_MASKS}" == "1" || "${JAILBREAK_FILL_ALL_MASKS,,}" == "true" ]]; then
    overrides+=("jailbreak.fill_all_masks=true")
  fi
  if [[ "${JAILBREAK_DEBUG_PRINT}" == "1" || "${JAILBREAK_DEBUG_PRINT,,}" == "true" ]]; then
    overrides+=("jailbreak.debug_print=true")
  fi

  echo "[INFO] Starting config index ${config_idx}: output_dir='${OUTPUT_DIR}' config='${EVAL_CONFIG_NAME}'"
  python -m tools.eval_diffuguard --config-name "${EVAL_CONFIG_NAME}" "${overrides[@]}"
  echo "[INFO] Finished config index ${config_idx}: output_dir='${OUTPUT_DIR}'"
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

for idx in "${!CONFIG_SPECS[@]}"; do
  reset_config_env
  spec="${CONFIG_SPECS[$idx]}"
  if [[ -n "${spec}" ]]; then
    echo "[INFO] Applying config spec #${idx}: ${spec}"
    eval "${spec}"
  fi
  run_one_config "${idx}"
done
