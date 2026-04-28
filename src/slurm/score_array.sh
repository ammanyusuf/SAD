#!/bin/bash
#SBATCH --job-name=sb_score
#SBATCH --account=rrg-<your-PI>   # [Compute Canada] replace with your allocation
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32G
#SBATCH --mail-user=${USER}@<your-institution.ca>  # [Compute Canada] set to your email
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=/scratch/%u/logs/safe-text-diffusion/score_%j.out
#SBATCH --error=/scratch/%u/logs/safe-text-diffusion/score_%j.err
#SBATCH --array=0-0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -gt 0 ]]; then
  REPO_ROOT_PATH="$1"
  shift
elif [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT_PATH="${REPO_ROOT}"
else
  REPO_ROOT_PATH="${DEFAULT_REPO_ROOT}"
fi
REPO_ROOT="$(cd "${REPO_ROOT_PATH}" && pwd)"

RUN_DIR=${RUN_DIR:-}
SCORE_RUN_LIST=${SCORE_RUN_LIST:-}
SCORE_RUN_LIST_FILE=${SCORE_RUN_LIST_FILE:-}
TRACK=${TRACK:-safety}
MODEL=${MODEL:?Set MODEL name}
CLASSIFIER=${CLASSIFIER:-llamaguard}
CLASSIFIER_MODEL=${CLASSIFIER_MODEL:-}
BASELINE_RUN_DIR=${BASELINE_RUN_DIR:-}
BEHAVIORS_CSV=${BEHAVIORS_CSV:-}
INDEXES_DIR=${INDEXES_DIR:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
FORCE=${FORCE:-1}
DRY_RUN=${DRY_RUN:-0}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-16}
SCORE_CONFIG_NAME=${SCORE_CONFIG_NAME:-config}
SCORE_COMPUTE_PERPLEXITY=${SCORE_COMPUTE_PERPLEXITY:-1}
SCORE_PPL_MODEL_NAME=${SCORE_PPL_MODEL_NAME:-}
SCORE_PPL_MODEL_PATH_OVERWRITE=${SCORE_PPL_MODEL_PATH_OVERWRITE:-}
SCORE_PPL_MODEL=${SCORE_PPL_MODEL:-}
SCORE_PPL_BATCH_SIZE=${SCORE_PPL_BATCH_SIZE:-8}
SCORE_PPL_MAX_LENGTH=${SCORE_PPL_MAX_LENGTH:-1024}
SCORE_SKIP_MISSING_GENERATIONS=${SCORE_SKIP_MISSING_GENERATIONS:-0}
SCORE_SKIP_JAILBREAK_EVALS=${SCORE_SKIP_JAILBREAK_EVALS:-0}
SCORE_CONTINUE_ON_ERROR=${SCORE_CONTINUE_ON_ERROR:-0}
SCORE_COMPUTE_HYGIENE_METRICS=${SCORE_COMPUTE_HYGIENE_METRICS:-1}
SCORE_COMPUTE_LEXICAL_METRICS=${SCORE_COMPUTE_LEXICAL_METRICS:-1}
SCORE_OVERLAP_NS=${SCORE_OVERLAP_NS:-1,2,3,4}
SCORE_DISTINCT_NS=${SCORE_DISTINCT_NS:-1,2,3,4}
SCORE_FUZZY_OVERLAP_NGRAM=${SCORE_FUZZY_OVERLAP_NGRAM:-10}
SCORE_FUZZY_MAX_SAMPLES=${SCORE_FUZZY_MAX_SAMPLES:-50}
SCORE_COMPUTE_BERTSCORE=${SCORE_COMPUTE_BERTSCORE:-0}
SCORE_BERTSCORE_MODEL=${SCORE_BERTSCORE_MODEL:-}
SCORE_BERTSCORE_BATCH_SIZE=${SCORE_BERTSCORE_BATCH_SIZE:-8}
SCORE_COMPUTE_MAUVE=${SCORE_COMPUTE_MAUVE:-0}
SCORE_MAUVE_MODEL_NAME=${SCORE_MAUVE_MODEL_NAME:-gpt2}
SCORE_MAUVE_MAX_TEXTS=${SCORE_MAUVE_MAX_TEXTS:-5000}
SCORE_MAUVE_MAX_TEXT_LENGTH=${SCORE_MAUVE_MAX_TEXT_LENGTH:-256}
SCORE_MAUVE_SEED=${SCORE_MAUVE_SEED:-0}
SCORE_COMPUTE_REFUSAL_METRICS=${SCORE_COMPUTE_REFUSAL_METRICS:-1}
SCORE_REFUSAL_MAX_CHARS=${SCORE_REFUSAL_MAX_CHARS:-200}
SCORE_REFUSAL_MAX_TOKENS=${SCORE_REFUSAL_MAX_TOKENS:-40}
SCORE_REFUSAL_CONTENT_RATIO_THRESHOLD=${SCORE_REFUSAL_CONTENT_RATIO_THRESHOLD:-0.2}
SCORE_NON_ANSWER_CONTENT_RATIO_THRESHOLD=${SCORE_NON_ANSWER_CONTENT_RATIO_THRESHOLD:-0.12}
SCORE_COMPUTE_DEGENERATION_METRICS=${SCORE_COMPUTE_DEGENERATION_METRICS:-1}
SCORE_DEGEN_MAX_SPAN_THRESHOLD=${SCORE_DEGEN_MAX_SPAN_THRESHOLD:-50}
SCORE_DEGEN_DISTINCT2_THRESHOLD=${SCORE_DEGEN_DISTINCT2_THRESHOLD:-0.10}
SCORE_DEGEN_REPEAT2_THRESHOLD=${SCORE_DEGEN_REPEAT2_THRESHOLD:-0.30}
SCORE_DEGEN_INCLUDE_EARLY_STOP=${SCORE_DEGEN_INCLUDE_EARLY_STOP:-1}
SCORE_COMPUTE_DISTRIBUTION_MMD=${SCORE_COMPUTE_DISTRIBUTION_MMD:-1}
SCORE_MMD_SPLIT_HALF_TRIALS=${SCORE_MMD_SPLIT_HALF_TRIALS:-5}
MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH:-}
SKIP_PIP_UPGRADE=${SKIP_PIP_UPGRADE:-0}
PIP_INSTALL_ARGS=${PIP_INSTALL_ARGS:-}
EXTERNAL_VENV_ACTIVATE=${EXTERNAL_VENV_ACTIVATE:-}
PYTHON_BIN=${PYTHON_BIN:-python}

LOG_ROOT="${SLURM_TMPDIR}/logs/task_${SLURM_ARRAY_TASK_ID:-0}"
JOB_LOG_DIR="${LOG_ROOT}/job"
mkdir -p "${JOB_LOG_DIR}"

cleanup_ran=0
stage_pending=0
CURRENT_STAGE_STATUS="failed"
CURRENT_RUN_DIR=""
CURRENT_SCORE_DIR=""
CURRENT_LOG_DIR=""

stage_out_score() {
  local status="$1"
  local run_dir="$2"
  local score_dir="$3"
  local log_dir="$4"
  if [[ -z "${run_dir}" || -z "${score_dir}" ]]; then
    echo "[WARN] Missing run_dir/score_dir for staging; skipping." >&2
    return
  fi
  mkdir -p "${score_dir}"
  if [[ -d "${log_dir:-}" ]]; then
    rsync -a "${log_dir}/" "${score_dir}/logs/" || true
  fi
  if [[ -d "${JOB_LOG_DIR:-}" ]]; then
    rsync -a "${JOB_LOG_DIR}/" "${score_dir}/logs/job/" || true
  fi
  printf "%s\n" "${status}" > "${score_dir}/status.txt"
  echo "[INFO] Staged logs for ${run_dir} into ${score_dir}"
  stage_pending=0
}

stage_out_pending() {
  if [[ ${stage_pending} -ne 1 ]]; then
    return
  fi
  stage_out_score "${CURRENT_STAGE_STATUS}" "${CURRENT_RUN_DIR}" "${CURRENT_SCORE_DIR}" "${CURRENT_LOG_DIR}"
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

mkdir -p "${SLURM_TMPDIR}"/{repo,outputs}
TMP_REPO="${SLURM_TMPDIR}/repo"
rsync -a --exclude=".git" --exclude=".tmp" --exclude=".env" --exclude=".env-gpu" --exclude=".env-jailbreak"  "${REPO_ROOT}/" "${TMP_REPO}/"

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

module purge
module load StdEnv/2023 cuda/12.2 python/3.11 gcc arrow/21.0.0 scipy-stack
module load faiss

USING_EXTERNAL_ENV=0
if [[ -n "${EXTERNAL_VENV_ACTIVATE}" ]]; then
  if [[ -f "${EXTERNAL_VENV_ACTIVATE}" ]]; then
    # shellcheck disable=SC1090
    source "${EXTERNAL_VENV_ACTIVATE}"
    USING_EXTERNAL_ENV=1
  else
    echo "[WARN] EXTERNAL_VENV_ACTIVATE='${EXTERNAL_VENV_ACTIVATE}' not found; falling back to Compute Canada wheel install." >&2
  fi
fi
if [[ "${USING_EXTERNAL_ENV}" -ne 1 ]]; then
  ${PYTHON_BIN} -m venv "${SLURM_TMPDIR}/venv"
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
else
  echo "[INFO] Using caller-provided virtualenv"
fi

export HF_HOME=${SLURM_TMPDIR}/hf_home
export HF_DATASETS_CACHE=${SLURM_TMPDIR}/hf_datasets
export HF_MODELS_CACHE=${SLURM_TMPDIR}/hf_models
export TRANSFORMERS_CACHE=${HF_MODELS_CACHE}
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_MODELS_CACHE}"
rsync -a "${SRC_HF_HOME}/" "${HF_HOME}/" || true
echo "[INFO] Staged HF_HOME to ${HF_HOME}"
rsync -a "${SRC_HF_DATASETS_CACHE}/" "${HF_DATASETS_CACHE}/" || true
echo "[INFO] Staged HF_DATASETS_CACHE to ${HF_DATASETS_CACHE}"

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

hydra_quote() {
  # Quote override values so Hydra accepts paths with special chars (e.g. seed=1).
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "${value}"
}

CHECKPOINT_SOURCE="$(stage_model_path "${CHECKPOINT_PATH:-}")"
TOKENIZER_SOURCE="$(stage_model_path "${TOKENIZER_PATH:-}")"

declare -a MODELS_TO_STAGE=()

if [[ -z "${MODEL_CONFIG_PATH}" ]]; then
  if [[ -f "${REPO_ROOT}/src/third_party/mdlm/configs/config.yaml" ]]; then
    MODEL_CONFIG_PATH="${REPO_ROOT}/src/third_party/mdlm/configs/config.yaml"
  elif [[ -f "${REPO_ROOT}/third_party/mdlm/configs/config.yaml" ]]; then
    MODEL_CONFIG_PATH="${REPO_ROOT}/third_party/mdlm/configs/config.yaml"
  fi
fi
if [[ -n "${MODEL_CONFIG_PATH}" ]]; then
  export MODEL_CONFIG_PATH
  echo "[INFO] Using MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH}"
else
  echo "[WARN] MODEL_CONFIG_PATH not set; embedding alignment may be skipped." >&2
fi


ppl_model="${SCORE_PPL_MODEL_PATH_OVERWRITE:-}"
if [[ -z "${ppl_model}" ]]; then
  ppl_model="${SCORE_PPL_MODEL_NAME:-}"
fi
if [[ -z "${ppl_model}" && -n "${SCORE_PPL_MODEL:-}" ]]; then
  ppl_model="${SCORE_PPL_MODEL}"
fi
if [[ -z "${ppl_model}" ]]; then
  ppl_model="gpt2-large"
  SCORE_PPL_MODEL_NAME="gpt2-large"
fi
ppl_source="$(stage_model_path "${ppl_model}")"
# fallback to gpt2-large if needed
if [[ -z "${ppl_source}" ]]; then
  ppl_source="$(stage_model_path "gpt2-large")"
  if [[ -n "${ppl_source}" ]]; then
    SCORE_PPL_MODEL_NAME="gpt2-large"
    SCORE_PPL_MODEL_PATH_OVERWRITE="${ppl_source}"
    SCORE_PPL_MODEL="${ppl_source}"
    echo "[INFO] Falling back to gpt2-large for perplexity model."
  else
    echo "[INFO] Could not stage requested perplexity model; continuing without staging."
  fi
else
  echo "[INFO] Using specified perplexity model for staging: ${ppl_model} with source ${ppl_source}"
fi
if [[ -n "${ppl_source}" ]]; then
  MODELS_TO_STAGE+=("${ppl_source}")
else
  echo "[WARN] Could not locate perplexity model (name='${SCORE_PPL_MODEL_NAME:-gpt2-large}', override='${SCORE_PPL_MODEL_PATH_OVERWRITE:-}') for perplexity; continuing." >&2
fi
echo "[INFO] GPT PPL Model to stage: ${ppl_source}"

classifier_source=""
if [[ -n "${CLASSIFIER_MODEL:-}" ]]; then
  classifier_source="$(stage_model_path "${CLASSIFIER_MODEL}")"
else
  if [[ "${CLASSIFIER}" == "harmbench" ]]; then
    classifier_source="$(stage_model_path "HarmBench-Llama-2-13b-cls")"
  elif [[ "${CLASSIFIER}" == "llamaguard" ]]; then
    classifier_source="$(stage_model_path "Llama-Guard-3-8B")"
  fi
  echo "[INFO] Classifier Model to stage: ${classifier_source}, for CLASSIFIER='${CLASSIFIER}'"
fi
if [[ -n "${CHECKPOINT_SOURCE}" ]]; then
  MODELS_TO_STAGE+=("${CHECKPOINT_SOURCE}")
fi
if [[ -n "${TOKENIZER_SOURCE}" && "${TOKENIZER_SOURCE}" != "${CHECKPOINT_SOURCE}" ]]; then
  MODELS_TO_STAGE+=("${TOKENIZER_SOURCE}")
fi
if [[ -n "${classifier_source}" ]]; then
  MODELS_TO_STAGE+=("${classifier_source}")
else
  echo "[WARN] Could not locate classifier weights for CLASSIFIER='${CLASSIFIER}'; proceeding without staging." >&2
fi

echo "DEBUG compute bertscore flag: ${SCORE_COMPUTE_BERTSCORE} | bertscore model: ${SCORE_BERTSCORE_MODEL}"

bertscore_source=""
if [[ "${SCORE_COMPUTE_BERTSCORE}" == "1" || "${SCORE_COMPUTE_BERTSCORE,,}" == "true" ]]; then
  if [[ -n "${SCORE_BERTSCORE_MODEL}" ]]; then
    bertscore_source="$(stage_model_path "${SCORE_BERTSCORE_MODEL}")"
    if [[ -n "${bertscore_source}" ]]; then
      MODELS_TO_STAGE+=("${bertscore_source}")
      echo "[INFO] BERTScore model to stage: ${bertscore_source}"
    else
      echo "[WARN] Could not locate BERTScore model '${SCORE_BERTSCORE_MODEL}' for staging; will rely on cache/path." >&2
    fi
  else
    echo "[WARN] SCORE_COMPUTE_BERTSCORE enabled but SCORE_BERTSCORE_MODEL not provided." >&2
  fi
fi

echo "DEBUG compute mauve flag: ${SCORE_COMPUTE_MAUVE} | mauve model name:${SCORE_MAUVE_MODEL_NAME}"

mauve_source=""
if [[ "${SCORE_COMPUTE_MAUVE}" == "1" || "${SCORE_COMPUTE_MAUVE,,}" == "true" ]]; then
  if [[ -n "${SCORE_MAUVE_MODEL_NAME}" ]]; then
    mauve_source="$(stage_model_path "${SCORE_MAUVE_MODEL_NAME}")"
    if [[ -n "${mauve_source}" ]]; then
      MODELS_TO_STAGE+=("${mauve_source}")
      echo "[INFO] MAUVE model to stage: ${mauve_source}"
    else
      echo "[WARN] Could not locate MAUVE model '${SCORE_MAUVE_MODEL_NAME}' for staging; will rely on cache/path." >&2
    fi
  fi
fi

echo "[INFO] Staging models ${MODELS_TO_STAGE[@]} into ${HF_MODELS_CACHE}:"
for src in "${MODELS_TO_STAGE[@]}"; do
  if [[ -n "${SRC_HF_MODELS_CACHE:-}" && "${src}" == "${SRC_HF_MODELS_CACHE}/"* ]]; then
    rel_path="${src#${SRC_HF_MODELS_CACHE}/}"
    dest="${HF_MODELS_CACHE}/${rel_path}"
    echo "Staging model: ${rel_path}"
  else
    base_name="$(basename "${src}")"
    dest="${HF_MODELS_CACHE}/${base_name}"
    echo "Staging model: ${base_name}"
  fi
  if [[ -d "${src}" ]]; then
    mkdir -p "${dest}"
    rsync -a --progress --human-readable "${src}/" "${dest}/" || true
  else
    mkdir -p "$(dirname "${dest}")"
    rsync -a --progress --human-readable "${src}" "${dest}" || true
  fi
  echo "  - ${src} -> ${dest}"
done

if [[ -n "${CHECKPOINT_SOURCE}" ]]; then
  CHECKPOINT_PATH="${HF_MODELS_CACHE}/$(basename "${CHECKPOINT_SOURCE}")"
  export CHECKPOINT_PATH
fi
if [[ -n "${TOKENIZER_SOURCE}" ]]; then
  TOKENIZER_PATH="${HF_MODELS_CACHE}/$(basename "${TOKENIZER_SOURCE}")"
  export TOKENIZER_PATH
fi
if [[ -n "${bertscore_source}" ]]; then
  if [[ -n "${SRC_HF_MODELS_CACHE:-}" && "${bertscore_source}" == "${SRC_HF_MODELS_CACHE}/"* ]]; then
    SCORE_BERTSCORE_MODEL="${bertscore_source#${SRC_HF_MODELS_CACHE}/}"
  else
    SCORE_BERTSCORE_MODEL="${HF_MODELS_CACHE}/$(basename "${bertscore_source}")"
  fi
  export SCORE_BERTSCORE_MODEL
fi
if [[ -n "${mauve_source}" ]]; then
  SCORE_MAUVE_MODEL_NAME="${HF_MODELS_CACHE}/$(basename "${mauve_source}")"
  export SCORE_MAUVE_MODEL_NAME
fi

looks_like_jailbreak_run() {
  local run_dir="$1"
  local lowered
  lowered="$(echo "${run_dir}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${lowered}" == *jailbreak* ]]; then
    return 0
  fi
  if [[ ! -d "${run_dir}" ]]; then
    return 1
  fi
  local candidates=()
  if [[ -f "${run_dir}/generations.jsonl" ]]; then
    candidates+=("${run_dir}/generations.jsonl")
  fi
  if [[ -f "${run_dir}/generations.ndjson" ]]; then
    candidates+=("${run_dir}/generations.ndjson")
  fi
  if [[ -d "${run_dir}/generations" ]]; then
    local gen_file
    for gen_file in "${run_dir}"/generations/*.jsonl "${run_dir}"/generations/*.ndjson; do
      [[ -f "${gen_file}" ]] && candidates+=("${gen_file}") && break
    done
  fi
  if [[ ${#candidates[@]} -eq 0 ]]; then
    while IFS= read -r -d '' file; do
      candidates+=("${file}")
      break
    done < <(find "${run_dir}" -maxdepth 2 \( -name "generations.jsonl" -o -name "generations.ndjson" \) -print0 2>/dev/null)
  fi
  if [[ ${#candidates[@]} -eq 0 ]]; then
    return 1
  fi
  local meta_pattern='"vanilla prompt"|"vanilla_prompt"|"refined prompt"|"refined_prompt"|"forbidden_prompt"|"BehaviorID"|"behavior_id"|"jailbreak_variant"|"attack_method"'
  local candidate
  for candidate in "${candidates[@]}"; do
    if grep -m1 -E "${meta_pattern}" "${candidate}" >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

# HF_MODELS_CACHE=~/scratch/hf_models

export PYTHONPATH="${TMP_REPO}/src:${TMP_REPO}/src/third_party:${TMP_REPO}/src/third_party/mdlm:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1
export NVIDIA_TF32_OVERRIDE=1
export CUDA_LAUNCH_BLOCKING=0
# Sometimes these warnings are annoying, so you can quiet them down during submission
# export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

echo "[INFO] ----- pip freeze -----"
python -m pip freeze | tee "${JOB_LOG_DIR}/pip_freeze.txt" || true
echo "[INFO] ----------------------"

FORCE_BOOL="true"
if [[ "${FORCE}" == "0" || "${FORCE,,}" == "false" ]]; then
  FORCE_BOOL="false"
fi
DRY_RUN_BOOL="false"
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN,,}" == "true" ]]; then
  DRY_RUN_BOOL="true"
fi

SCORE_RUNS=()
SCORE_BATCH_MODE=0
if [[ -n "${SCORE_RUN_LIST_FILE:-}" ]]; then
  if [[ ! -f "${SCORE_RUN_LIST_FILE}" ]]; then
    echo "[ERROR] SCORE_RUN_LIST_FILE='${SCORE_RUN_LIST_FILE}' does not exist." >&2
    exit 1
  fi
  SCORE_BATCH_MODE=1
  while IFS= read -r line; do
    [[ -z "${line// }" ]] && continue
    SCORE_RUNS+=("${line}")
  done < "${SCORE_RUN_LIST_FILE}"
elif [[ -n "${SCORE_RUN_LIST:-}" ]]; then
  SCORE_BATCH_MODE=1
  while IFS= read -r run_entry; do
    [[ -z "${run_entry// }" ]] && continue
    SCORE_RUNS+=("${run_entry}")
  done <<< "$(echo "${SCORE_RUN_LIST}" | tr ',' '\n')"
fi

if [[ ${SCORE_BATCH_MODE} -eq 1 && ${#SCORE_RUNS[@]} -eq 0 ]]; then
  echo "[ERROR] SCORE_RUN_LIST* provided but no run directories parsed." >&2
  exit 1
fi

if [[ ${SCORE_BATCH_MODE} -eq 0 ]]; then
  if [[ -z "${RUN_DIR:-}" ]]; then
    echo "[ERROR] Set RUN_DIR to the generation directory to score." >&2
    exit 1
  fi
  SCORE_RUNS+=("${RUN_DIR}")
fi

NEED_JAILBREAK_MODELS=0
for run_dir in "${SCORE_RUNS[@]}"; do
  if looks_like_jailbreak_run "${run_dir}"; then
    NEED_JAILBREAK_MODELS=1
    break
  fi
done

if [[ "${NEED_JAILBREAK_MODELS}" -eq 1 ]]; then
  echo "[INFO] Jailbreak run detected; staging HarmBench and StrongREJECT evaluators."
  declare -a JB_MODELS_TO_STAGE=()
  harmbench_eval_source="$(stage_model_path "HarmBench-Llama-2-13b-cls")"
  strongreject_eval_source="$(stage_model_path "strongreject-15k-v1")"
  if [[ -n "${harmbench_eval_source}" ]]; then
    JB_MODELS_TO_STAGE+=("${harmbench_eval_source}")
  else
    echo "[WARN] Could not locate HarmBench classifier for jailbreak ASR." >&2
  fi
  if [[ -n "${strongreject_eval_source}" ]]; then
    JB_MODELS_TO_STAGE+=("${strongreject_eval_source}")
    strongreject_base_source="$(stage_model_path "gemma-2b")"
    if [[ -n "${strongreject_base_source}" ]]; then
      JB_MODELS_TO_STAGE+=("${strongreject_base_source}")
    else
      echo "[WARN] Could not locate StrongREJECT base model 'gemma-2b' for staging." >&2
    fi
  else
    echo "[WARN] Could not locate StrongREJECT evaluator for jailbreak scoring." >&2
  fi
  for src in "${JB_MODELS_TO_STAGE[@]}"; do
    if [[ -n "${SRC_HF_MODELS_CACHE:-}" && "${src}" == "${SRC_HF_MODELS_CACHE}/"* ]]; then
      rel_path="${src#${SRC_HF_MODELS_CACHE}/}"
      dest="${HF_MODELS_CACHE}/${rel_path}"
      echo "Staging jailbreak model: ${rel_path}"
    else
      base_name="$(basename "${src}")"
      dest="${HF_MODELS_CACHE}/${base_name}"
      echo "Staging jailbreak model: ${base_name}"
    fi
    if [[ -d "${src}" ]]; then
      mkdir -p "${dest}"
      rsync -a --progress --human-readable "${src}/" "${dest}/" || true
    else
      mkdir -p "$(dirname "${dest}")"
      rsync -a --progress --human-readable "${src}" "${dest}" || true
    fi
    echo "  - ${src} -> ${dest}"
  done
  if [[ -n "${harmbench_eval_source}" ]]; then
    if [[ -n "${SRC_HF_MODELS_CACHE:-}" && "${harmbench_eval_source}" == "${SRC_HF_MODELS_CACHE}/"* ]]; then
      export HARM_BENCH_CLASSIFIER="${HF_MODELS_CACHE}/${harmbench_eval_source#${SRC_HF_MODELS_CACHE}/}"
    else
      export HARM_BENCH_CLASSIFIER="${HF_MODELS_CACHE}/$(basename "${harmbench_eval_source}")"
    fi
  fi
  if [[ -n "${strongreject_eval_source}" ]]; then
    if [[ -n "${SRC_HF_MODELS_CACHE:-}" && "${strongreject_eval_source}" == "${SRC_HF_MODELS_CACHE}/"* ]]; then
      export STRONGREJECT_MODEL="${HF_MODELS_CACHE}/${strongreject_eval_source#${SRC_HF_MODELS_CACHE}/}"
    else
      export STRONGREJECT_MODEL="${HF_MODELS_CACHE}/$(basename "${strongreject_eval_source}")"
    fi
  fi
else
  echo "[INFO] No jailbreak runs detected; skipping HarmBench/StrongREJECT model staging."
fi

score_one_run() {
  local run_dir="$1"
  local run_idx="$2"
  if [[ ! -d "${run_dir}" ]]; then
    echo "[ERROR] RUN_DIR='${run_dir}' does not exist." >&2
    exit 1
  fi

  if [[ ${SCORE_BATCH_MODE} -eq 1 ]]; then
    CURRENT_LOG_DIR="${LOG_ROOT}/run_${run_idx}"
    HYDRA_SCORE_DIR="${run_dir}/scores/job_${SLURM_JOB_ID:-local}/task_${SLURM_ARRAY_TASK_ID:-0}/run_${run_idx}"
  else
    CURRENT_LOG_DIR="${LOG_ROOT}"
    HYDRA_SCORE_DIR="${run_dir}/scores/job_${SLURM_JOB_ID:-local}/task_${SLURM_ARRAY_TASK_ID:-0}"
  fi
  mkdir -p "${CURRENT_LOG_DIR}" "${HYDRA_SCORE_DIR}"

  CURRENT_RUN_DIR="${run_dir}"
  CURRENT_SCORE_DIR="${HYDRA_SCORE_DIR}"
  CURRENT_STAGE_STATUS="failed"
  stage_pending=1
  echo "[INFO] Starting score for run_idx=${run_idx}, run_dir=${run_dir}"

  local overrides=(
    "score.track=${TRACK}"
    "score.model=${MODEL}"
    "score.run_dir=$(hydra_quote "${run_dir}")"
    "score.classifier=${CLASSIFIER}"
    "score.max_new_tokens=${MAX_NEW_TOKENS}"
    "score.batch_size=${SCORE_BATCH_SIZE}"
    "score.force=${FORCE_BOOL}"
    "score.dry_run=${DRY_RUN_BOOL}"
    "io.output_dir=$(hydra_quote "${HYDRA_SCORE_DIR}")"
  )
  if [[ "${SCORE_SKIP_MISSING_GENERATIONS}" == "1" || "${SCORE_SKIP_MISSING_GENERATIONS,,}" == "true" ]]; then
    overrides+=("score.skip_missing_generations=true")
  else
    overrides+=("score.skip_missing_generations=false")
  fi
  if [[ "${SCORE_SKIP_JAILBREAK_EVALS}" == "1" || "${SCORE_SKIP_JAILBREAK_EVALS,,}" == "true" ]]; then
    overrides+=("score.skip_jailbreak_evals=true")
  else
    overrides+=("score.skip_jailbreak_evals=false")
  fi
  if [[ -n "${BASELINE_RUN_DIR:-}" ]]; then
    overrides+=("score.baseline_run_dir=$(hydra_quote "${BASELINE_RUN_DIR}")")
  fi

  if [[ "${SCORE_COMPUTE_PERPLEXITY}" == "1" || "${SCORE_COMPUTE_PERPLEXITY,,}" == "true" ]]; then
    local ppl_override="${SCORE_PPL_MODEL_PATH_OVERWRITE}"
    if [[ -z "${ppl_override}" ]]; then
      ppl_override="${SCORE_PPL_MODEL_NAME}"
    fi
    if [[ -z "${ppl_override}" && -n "${SCORE_PPL_MODEL:-}" ]]; then
      ppl_override="${SCORE_PPL_MODEL}"
    fi
    if [[ -n "${ppl_override}" && ! -f "${ppl_override}" && ! -d "${ppl_override}" && -d "${HF_MODELS_CACHE}/${ppl_override}" ]]; then
      ppl_override="${HF_MODELS_CACHE}/${ppl_override}"
    fi
    echo "[INFO] Using perplexity model override: ${ppl_override}"
    overrides+=(
      "score.compute_perplexity=true"
      "score.perplexity_model=$(hydra_quote "${ppl_override}")"
      "score.perplexity_batch_size=${SCORE_PPL_BATCH_SIZE}"
      "score.perplexity_max_length=${SCORE_PPL_MAX_LENGTH}"
    )
  fi

  if [[ -n "${CLASSIFIER_MODEL}" ]]; then
    overrides+=("score.classifier_model=$(hydra_quote "${CLASSIFIER_MODEL}")")
  else
    if [[ "${CLASSIFIER}" == "harmbench" ]]; then
      overrides+=("score.classifier_model=$(hydra_quote "${HF_MODELS_CACHE}/HarmBench-Llama-2-13b-cls")")
    elif [[ "${CLASSIFIER}" == "llamaguard" ]]; then
      overrides+=("score.classifier_model=$(hydra_quote "${HF_MODELS_CACHE}/Llama-Guard-3-8B")")
    fi
  fi
  if [[ -n "${BEHAVIORS_CSV}" ]]; then
    overrides+=("score.behaviors_csv=$(hydra_quote "${BEHAVIORS_CSV}")")
  fi
  if [[ -n "${INDEXES_DIR}" ]]; then
    overrides+=("score.indexes_dir=$(hydra_quote "${INDEXES_DIR}")")
  fi
  if [[ "${SCORE_COMPUTE_HYGIENE_METRICS}" == "1" || "${SCORE_COMPUTE_HYGIENE_METRICS,,}" == "true" ]]; then
    overrides+=("score.compute_hygiene_metrics=true")
  else
    overrides+=("score.compute_hygiene_metrics=false")
  fi
  if [[ "${SCORE_COMPUTE_LEXICAL_METRICS}" == "1" || "${SCORE_COMPUTE_LEXICAL_METRICS,,}" == "true" ]]; then
    overrides+=(
      "score.compute_lexical_metrics=true"
      "score.overlap_ns=[${SCORE_OVERLAP_NS}]"
      "score.distinct_ns=[${SCORE_DISTINCT_NS}]"
      "score.fuzzy_overlap_ngram=${SCORE_FUZZY_OVERLAP_NGRAM}"
      "score.fuzzy_max_samples=${SCORE_FUZZY_MAX_SAMPLES}"
    )
  else
    overrides+=("score.compute_lexical_metrics=false")
  fi
  if [[ "${SCORE_COMPUTE_BERTSCORE}" == "1" || "${SCORE_COMPUTE_BERTSCORE,,}" == "true" ]]; then
    berts_model_override="${SCORE_BERTSCORE_MODEL}"
    if [[ -n "${berts_model_override}" && ! -f "${berts_model_override}" && ! -d "${berts_model_override}" ]]; then
      if [[ -n "${HF_MODELS_CACHE:-}" && -d "${HF_MODELS_CACHE}/${berts_model_override}" ]]; then
        berts_model_override="${HF_MODELS_CACHE}/${berts_model_override}"
      elif [[ -n "${HF_MODELS_CACHE:-}" && -d "${HF_MODELS_CACHE}/$(basename "${berts_model_override}")" ]]; then
        berts_model_override="${HF_MODELS_CACHE}/$(basename "${berts_model_override}")"
      fi
    fi
    overrides+=(
      "score.compute_bertscore=true"
      "score.bertscore_model=$(hydra_quote "${berts_model_override}")"
      "score.bertscore_batch_size=${SCORE_BERTSCORE_BATCH_SIZE}"
    )
  fi
  if [[ "${SCORE_COMPUTE_MAUVE}" == "1" || "${SCORE_COMPUTE_MAUVE,,}" == "true" ]]; then
    mauve_model_override="${SCORE_MAUVE_MODEL_NAME}"
    if [[ -n "${mauve_model_override}" && -d "${HF_MODELS_CACHE}/$(basename "${mauve_model_override}")" ]]; then
      mauve_model_override="${HF_MODELS_CACHE}/$(basename "${mauve_model_override}")"
    fi
    overrides+=(
      "score.compute_mauve=true"
      "score.mauve_model_name=$(hydra_quote "${mauve_model_override}")"
      "score.mauve_max_texts=${SCORE_MAUVE_MAX_TEXTS}"
      "score.mauve_max_text_length=${SCORE_MAUVE_MAX_TEXT_LENGTH}"
      "score.mauve_seed=${SCORE_MAUVE_SEED}"
    )
  fi
  if [[ "${SCORE_COMPUTE_REFUSAL_METRICS}" == "1" || "${SCORE_COMPUTE_REFUSAL_METRICS,,}" == "true" ]]; then
    overrides+=(
      "score.compute_refusal_metrics=true"
      "score.refusal_max_chars=${SCORE_REFUSAL_MAX_CHARS}"
      "score.refusal_max_tokens=${SCORE_REFUSAL_MAX_TOKENS}"
      "score.refusal_content_ratio_threshold=${SCORE_REFUSAL_CONTENT_RATIO_THRESHOLD}"
      "score.non_answer_content_ratio_threshold=${SCORE_NON_ANSWER_CONTENT_RATIO_THRESHOLD}"
    )
  else
    overrides+=("score.compute_refusal_metrics=false")
  fi
  if [[ "${SCORE_COMPUTE_DEGENERATION_METRICS}" == "1" || "${SCORE_COMPUTE_DEGENERATION_METRICS,,}" == "true" ]]; then
    overrides+=(
      "score.compute_degeneration_metrics=true"
      "score.degeneration_max_span_threshold=${SCORE_DEGEN_MAX_SPAN_THRESHOLD}"
      "score.degeneration_distinct2_threshold=${SCORE_DEGEN_DISTINCT2_THRESHOLD}"
      "score.degeneration_repeat2_threshold=${SCORE_DEGEN_REPEAT2_THRESHOLD}"
      "score.degeneration_include_early_stop=${SCORE_DEGEN_INCLUDE_EARLY_STOP}"
    )
  else
    overrides+=("score.compute_degeneration_metrics=false")
  fi
  if [[ "${SCORE_COMPUTE_DISTRIBUTION_MMD}" == "1" || "${SCORE_COMPUTE_DISTRIBUTION_MMD,,}" == "true" ]]; then
    overrides+=(
      "score.compute_distribution_mmd=true"
      "score.mmd_split_half_trials=${SCORE_MMD_SPLIT_HALF_TRIALS}"
    )
  else
    overrides+=("score.compute_distribution_mmd=false")
  fi

  set +e
  (
    set -x
    python -m tools.score --config-name "${SCORE_CONFIG_NAME}" "${overrides[@]}"
  ) 2>&1 | tee "${CURRENT_LOG_DIR}/score.log"
  score_status=${PIPESTATUS[0]}
  set -e

  if [[ ${score_status} -ne 0 ]]; then
    echo "[WARN] Score failed (exit=${score_status}) for run_idx=${run_idx}, run_dir=${run_dir}" >&2
    CURRENT_STAGE_STATUS="failed"
    stage_out_score "${CURRENT_STAGE_STATUS}" "${run_dir}" "${HYDRA_SCORE_DIR}" "${CURRENT_LOG_DIR}"
    if [[ "${SCORE_CONTINUE_ON_ERROR}" == "1" || "${SCORE_CONTINUE_ON_ERROR,,}" == "true" ]]; then
      return 0
    fi
    exit "${score_status}"
  fi

  CURRENT_STAGE_STATUS="success"
  echo "[INFO] Finished score for run_idx=${run_idx}, run_dir=${run_dir}"
  stage_out_score "${CURRENT_STAGE_STATUS}" "${run_dir}" "${HYDRA_SCORE_DIR}" "${CURRENT_LOG_DIR}"
}

for idx in "${!SCORE_RUNS[@]}"; do
  echo "[INFO] Starting scoring for run index ${idx}: ${SCORE_RUNS[$idx]}"
  score_one_run "${SCORE_RUNS[$idx]}" "${idx}"
done

echo "[INFO] All scoring tasks complete for job ${SLURM_JOB_ID:-local}"
