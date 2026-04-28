#!/usr/bin/env bash
#
# Source this file to set up environment variables for Slurm jobs.
# Usage examples (must be sourced):
#   source scripts/env_profile.sh mdlm
#   source scripts/env_profile.sh llada
#   source scripts/env_profile.sh llada --llada-1.5
#   source scripts/env_profile.sh dream
#   source scripts/env_profile.sh llada-instruct
#   source scripts/env_profile.sh mmada
#   source scripts/env_profile.sh mdlm --debug   # enables SAFE_* debug flags
#   source scripts/env_profile.sh mdlm --debug-dist   # enable distribution logging
#   source scripts/env_profile.sh llada --jailbreak --diffuguard   # export ATTACK_PROMPT for DiffuGuard
#   source scripts/env_profile.sh llada --jailbreak --dija         # export ATTACK_PROMPT for DIJA
#   source scripts/env_profile.sh dream --jailbreak --diffuguard   # Dream DiffuGuard run
#   source scripts/env_profile.sh dream --dream-instruct           # Dream-v0-Instruct-7B
#   source scripts/env_profile.sh dream --dream-base               # Dream-v0-Base-7B
#   source scripts/env_profile.sh dream --dream-on                 # DreamOn-v0-7B
#   source scripts/env_profile.sh dream --dream-coder-instruct     # Dream-Coder-v0-Instruct-7B
#
# Memorization experiments:
#   source scripts/env_profile.sh dlm-memorization               # Wikipedia fine-tuned MDLM (702-1250000.ckpt)
#   source scripts/env_profile.sh dlm-memorization --slimpajama  # (future) DLM-1.1B on SlimPajama
#
# If available, module loads and venv activation are applied; otherwise they are skipped.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "[ERROR] Please source this script: 'source scripts/env_profile.sh <mdlm|llada|llada-instruct|dream|mmada> [--debug]'." >&2
  exit 1
fi

profile="$1"
debug_flag="$2"
REQUESTED_MODULES="StdEnv/2023 cuda/12.2 python/3.11 gcc arrow/21.0.0 scipy-stack faiss rust opencv"
MODULE_STATUS="not_attempted"
ACTIVATED_VENV=""

if [[ -z "${profile}" ]]; then
  echo "[ERROR] Missing profile. Use 'mdlm', 'llada', 'llada-instruct', 'dream', or 'mmada'." >&2
  return 1
fi

dist_debug=false
jailbreak_prompts=false
jailbreak_mode=""
jailbreak_dataset=""
dream_variant=""
llada_variant=""
mem_variant=""
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug) debug_flag="--debug" ;;
    --debug-dist) dist_debug=true ;;
    --jailbreak) jailbreak_prompts=true ;;
    --diffuguard) jailbreak_mode="diffuguard" ;;
    --dija) jailbreak_mode="dija" ;;
    --harmbench) jailbreak_dataset="harmbench" ;;
    --strongreject) jailbreak_dataset="strongreject" ;;
    --jailbreakbench) jailbreak_dataset="jailbreakbench" ;;
    --llada-1.5) llada_variant="llada-1.5" ;;
    --dream-instruct) dream_variant="dream-instruct" ;;
    --dream-base) dream_variant="dream-base" ;;
    --dream-on|--dreamon) dream_variant="dream-on" ;;
    --dream-coder-instruct) dream_variant="dream-coder-instruct" ;;
    --slimpajama) mem_variant="slimpajama" ;;
    *) ;;
  esac
  shift
done

DEFAULT_VENV_PATH="$HOME/repos/safe-text-diffusion/.env-gpu/bin/activate"
if [[ "${jailbreak_prompts}" == true ]]; then
  DEFAULT_VENV_PATH="$HOME/repos/safe-text-diffusion/.env-jailbreak/bin/activate"
fi

# Unset prior values to avoid cross-contamination.
unset REPO_ROOT HF_HOME HF_DATASETS_CACHE HF_MODELS_CACHE
unset MEMORIZATION_DATA_DIR MEMORIZATION_RESULTS_DIR MDLM_CONFIG_OVERRIDES
unset JAILBREAK_DATA_ROOT JBB_BEHAVIORS_CSV JBB_DIJA_JSON JBB_DIFFUGUARD_JSON HARM_BENCH_CSV HARM_BENCH_JSON
unset HARMBENCH_DIFFUGUARD_JSON HARMBENCH_PROMPTS_DIJA HARMBENCH_PROMPTS_DIFFUGUARD HARMBENCH_PROMPTS
unset STRONGREJECT_RAW_JSON STRONGREJECT_JSON STRONGREJECT_DIFFUGUARD_JSON STRONGREJECT_PROMPTS_DIJA STRONGREJECT_PROMPTS_DIFFUGUARD STRONGREJECT_PROMPTS
unset ATTACK_PROMPT ATTACK_PROMPT_JBB_DIJA ATTACK_PROMPT_JBB_DIFFUGUARD ATTACK_PROMPT_HARMBENCH
unset CHECKPOINT_PATH TOKENIZER_PATH MODEL_CONFIG_PATH
unset CAP_APPLY_CHAT_TEMPLATE
unset SKIP_PIP_UPGRADE PIP_INSTALL_ARGS EXTERNAL_VENV_ACTIVATE
unset SAFE_REPELLENCY_DEBUG SAFE_KERNEL_MODE SAFE_GUIDANCE_MODE SAFE_BETA_MODE SAFE_REPELLENCY_VALIDATE
unset SAFE_LLADA_DEBUG
unset LLAMAGUARD_CHECKPOINT_PATH FK_ROBERTA_CHECKPOINT_PATH N_PER_PROMPT FK_K_PARTICLES FK_RESAMPLE_FREQ FK_NUM_X0_SAMPLES FK_LAMBDA FK_REWARD_TRIM_LEN
unset SAFE_DIST_LOG_ENABLED SAFE_DIST_LOG_DIR SAFE_DIST_LOG_PATH SAFE_DIST_LOG_TIMESTEPS
unset SAFE_DIST_LOG_TOPK SAFE_DIST_LOG_MAX_POS SAFE_DIST_LOG_POSITIONS SAFE_DIST_LOG_POSITION_MODE
unset SAFE_DIST_LOG_POSITION_SAMPLE_SEED SAFE_DIST_LOG_DUMP_TOKENS SAFE_DIST_LOG_DUMP_PROMPT
unset SAFE_DIST_LOG_FULL_VOCAB SAFE_DIST_LOG_DTYPE SAFE_DIST_LOG_AUTO_TOP_N SAFE_DIST_LOG_RUN_ID

_maybe_module_load() {
  if command -v module >/dev/null 2>&1; then
    module load ${REQUESTED_MODULES} && MODULE_STATUS="loaded:${REQUESTED_MODULES}" || MODULE_STATUS="load_failed:${REQUESTED_MODULES}"
  else
    MODULE_STATUS="module_cmd_unavailable"
  fi
}

_activate_venv() {
  local venv_path="$1"
  if [[ -n "${venv_path}" && -f "${venv_path}" ]]; then
    # shellcheck disable=SC1090
    source "${venv_path}"
    ACTIVATED_VENV="${venv_path}"
  fi
}

_base_common() {
  export REPO_ROOT="${REPO_ROOT:-$HOME/repos/safe-text-diffusion}"
  export HF_HOME="${HF_HOME:-$HOME/scratch/hf_home}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/scratch/hf_datasets}"
  export HF_MODELS_CACHE="${HF_MODELS_CACHE:-$HOME/scratch/hf_models}"
  export JAILBREAK_DATA_ROOT="${JAILBREAK_DATA_ROOT:-$HOME/scratch/jailbreak_datasets}"
  export SKIP_PIP_UPGRADE=1
  export PIP_INSTALL_ARGS="--no-index"
  export EXTERNAL_VENV_ACTIVATE="${EXTERNAL_VENV_ACTIVATE:-$DEFAULT_VENV_PATH}"
  # Filtering baselines: LlamaGuard and RoBERTa toxicity classifier checkpoints (shared across all profiles)
  export LLAMAGUARD_CHECKPOINT_PATH="${LLAMAGUARD_CHECKPOINT_PATH:-$HOME/scratch/hf_models/Llama-Guard-3-8B}"
  _fk_roberta_snapshot=$(ls -d "$HOME/scratch/hf_home/hub/models--SkolkovoInstitute--roberta_toxicity_classifier/snapshots/"* 2>/dev/null | head -1)
  export FK_ROBERTA_CHECKPOINT_PATH="${FK_ROBERTA_CHECKPOINT_PATH:-${_fk_roberta_snapshot}}"
  unset _fk_roberta_snapshot
  if [[ "${jailbreak_prompts}" == true && "${EXTERNAL_VENV_ACTIVATE}" == *".env-gpu"* ]]; then
    export EXTERNAL_VENV_ACTIVATE="${DEFAULT_VENV_PATH}"
  fi

  local base_py="${REPO_ROOT}:${REPO_ROOT}/src:${REPO_ROOT}/src/mdlm"
  local third_party="${REPO_ROOT}/src/third_party/mdlm"
  if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${base_py}:${third_party}:${PYTHONPATH}"
  else
    export PYTHONPATH="${base_py}:${third_party}"
  fi
}

case "${profile}" in
  mdlm)
    _base_common
    _maybe_module_load
    _activate_venv "${EXTERNAL_VENV_ACTIVATE}"
    export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$HOME/scratch/models/text-diffusion/mdlm.ckpt}"
    export TOKENIZER_PATH="${TOKENIZER_PATH:-$HOME/scratch/hf_models/gpt2-large}"
    export MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-$HOME/repos/safe-text-diffusion/src/third_party/mdlm/configs/config.yaml}"
    ;;
  llada)
    _base_common
    _maybe_module_load
    _activate_venv "${EXTERNAL_VENV_ACTIVATE}"
    if [[ "${llada_variant}" == "llada-1.5" ]]; then
      export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$HOME/scratch/hf_models/LLaDA-1.5}"
      export TOKENIZER_PATH="${TOKENIZER_PATH:-$HOME/scratch/hf_models/LLaDA-1.5}"
    else
      export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$HOME/scratch/hf_models/LLaDA-8B-Base}"
      export TOKENIZER_PATH="${TOKENIZER_PATH:-$HOME/scratch/hf_models/LLaDA-8B-Base}"
    fi
    export MODEL_CONFIG_PATH=""
    ;;
  llada-instruct)
    _base_common
    _maybe_module_load
    _activate_venv "${EXTERNAL_VENV_ACTIVATE}"
    export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$HOME/scratch/hf_models/LLaDA-8B-Instruct}"
    export TOKENIZER_PATH="${TOKENIZER_PATH:-$HOME/scratch/hf_models/LLaDA-8B-Instruct}"
    export MODEL_CONFIG_PATH=""
    ;;
  dream)
    _base_common
    _maybe_module_load
    _activate_venv "${EXTERNAL_VENV_ACTIVATE}"
    case "${dream_variant}" in
      dream-base)
        export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$HOME/scratch/hf_models/Dream-v0-Base-7B}"
        export TOKENIZER_PATH="${TOKENIZER_PATH:-$HOME/scratch/hf_models/Dream-v0-Base-7B}"
        ;;
      dream-on)
        export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$HOME/scratch/hf_models/DreamOn-v0-7B}"
        export TOKENIZER_PATH="${TOKENIZER_PATH:-$HOME/scratch/hf_models/DreamOn-v0-7B}"
        ;;
      dream-coder-instruct)
        export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$HOME/scratch/hf_models/Dream-Coder-v0-Instruct-7B}"
        export TOKENIZER_PATH="${TOKENIZER_PATH:-$HOME/scratch/hf_models/Dream-Coder-v0-Instruct-7B}"
        ;;
      *)
        export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$HOME/scratch/hf_models/Dream-v0-Instruct-7B}"
        export TOKENIZER_PATH="${TOKENIZER_PATH:-$HOME/scratch/hf_models/Dream-v0-Instruct-7B}"
        ;;
    esac
    export MODEL_CONFIG_PATH=""
    ;;
  mmada)
    _base_common
    _maybe_module_load
    _activate_venv "${EXTERNAL_VENV_ACTIVATE}"
    export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$HOME/scratch/hf_models/MMaDA-8B-MixCoT}"
    export TOKENIZER_PATH="${TOKENIZER_PATH:-$HOME/scratch/hf_models/MMaDA-8B-MixCoT}"
    export MODEL_CONFIG_PATH=""
    ;;
  dlm-memorization)
    # Memorization experiments: MDLM Lightning checkpoint (Wikipedia fine-tune or SlimPajama)
    _base_common
    _maybe_module_load
    _activate_venv "${EXTERNAL_VENV_ACTIVATE}"
    if [[ "${mem_variant}" == "slimpajama" ]]; then
      # Future: DLM-1.1B pretrained on SlimPajama (not yet available locally)
      export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$HOME/scratch/models/text-diffusion/dlm-1.1b-slimpajama.ckpt}"
      export TOKENIZER_PATH="${TOKENIZER_PATH:-$HOME/scratch/hf_models/gpt2-large}"
    else
      # Default: MDLM fine-tuned on Wikipedia (702-1250000.ckpt)
      export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$HOME/scratch/models/text-diffusion/702-1250000.ckpt}"
      export TOKENIZER_PATH="${TOKENIZER_PATH:-$HOME/scratch/hf_models/gpt2-large}"
    fi
    export MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-$HOME/repos/safe-text-diffusion/src/third_party/mdlm/configs/config.yaml}"
    # Comma-separated Hydra overrides passed straight through to the memorization runner
    # e.g. "data=wikitext,sampling.steps=128"
    export MDLM_CONFIG_OVERRIDES="${MDLM_CONFIG_OVERRIDES:-}"
    export MEMORIZATION_DATA_DIR="${MEMORIZATION_DATA_DIR:-$HOME/scratch/data/memorization}"
    export MEMORIZATION_RESULTS_DIR="${MEMORIZATION_RESULTS_DIR:-$HOME/scratch/results/memorization}"
    ;;
  *)
    echo "[ERROR] Unknown profile '${profile}'. Use 'mdlm', 'llada', 'llada-instruct', 'dream', 'mmada', or 'dlm-memorization'." >&2
    return 1
    ;;
esac

# Enable chat template by default for LLaDA instruct/chat checkpoints,
# but keep explicit user-provided values untouched.
if [[ -z "${CAP_APPLY_CHAT_TEMPLATE+x}" ]]; then
  model_id="${CHECKPOINT_PATH:-},${TOKENIZER_PATH:-}"
  model_id="${model_id,,}"
  if [[ "${profile}" == "llada-instruct" ]] || [[ "${model_id}" == *"llada"* && ( "${model_id}" == *"instruct"* || "${model_id}" == *"chat"* ) ]]; then
    export CAP_APPLY_CHAT_TEMPLATE=1
  fi
fi

if [[ "${debug_flag}" == "--debug" ]]; then
  export SAFE_REPELLENCY_DEBUG=1
  export SAFE_KERNEL_MODE=both
  export SAFE_GUIDANCE_MODE=both
  export SAFE_BETA_MODE=both
  export SAFE_REPELLENCY_VALIDATE=1
  export SAFE_LLADA_DEBUG=1
  export SAFE_REPELLENCY_CSV_LOG="${RESULTS_ROOT:-${SCRATCH:-$HOME}/results}/repellency_stats.csv"
fi

if [[ "${dist_debug}" == true ]]; then
  export SAFE_DIST_LOG_ENABLED=1
  export SAFE_DIST_LOG_DIR="${SAFE_DIST_LOG_DIR:-${REPO_ROOT}/results/diagnostics/mdlm/dist_logs}"
  export SAFE_DIST_LOG_PATH="${SAFE_DIST_LOG_PATH:-${SAFE_DIST_LOG_DIR}/dist_logs.jsonl}"
  export SAFE_DIST_LOG_TIMESTEPS="${SAFE_DIST_LOG_TIMESTEPS:-all}"
  export SAFE_DIST_LOG_TOPK="${SAFE_DIST_LOG_TOPK:-50}"
  export SAFE_DIST_LOG_MAX_POS="${SAFE_DIST_LOG_MAX_POS:-16}"
  export SAFE_DIST_LOG_POSITIONS="${SAFE_DIST_LOG_POSITIONS:-all}"
  export SAFE_DIST_LOG_POSITION_SAMPLE_SEED="${SAFE_DIST_LOG_POSITION_SAMPLE_SEED:-0}"
  export SAFE_DIST_LOG_DUMP_TOKENS="${SAFE_DIST_LOG_DUMP_TOKENS:-1}"
  export SAFE_DIST_LOG_DUMP_PROMPT="${SAFE_DIST_LOG_DUMP_PROMPT:-1}"
  export SAFE_DIST_LOG_FULL_VOCAB="${SAFE_DIST_LOG_FULL_VOCAB:-0}"
  export SAFE_DIST_LOG_DTYPE="${SAFE_DIST_LOG_DTYPE:-float16}"
  export SAFE_DIST_LOG_AUTO_TOP_N="${SAFE_DIST_LOG_AUTO_TOP_N:-3}"
  export SAFE_DIST_LOG_RUN_ID="${SAFE_DIST_LOG_RUN_ID:-${RUN_ID:-${SLURM_JOB_ID:-dist}}}"
fi

if [[ "${jailbreak_prompts}" == true ]]; then
  export JBB_BEHAVIORS_CSV="${JAILBREAK_DATA_ROOT}/jbb_behaviors_harmful.csv"
  export JBB_DIJA_JSON="${JAILBREAK_DATA_ROOT}/jbb_behaviors_harmful_dija.json"
  export JBB_DIFFUGUARD_JSON="${JAILBREAK_DATA_ROOT}/jbb_behaviors_harmful_diffuguard.json"
  export HARM_BENCH_CSV="${JAILBREAK_DATA_ROOT}/harmbench_behaviors_text_all.csv"
  export HARM_BENCH_JSON="${JAILBREAK_DATA_ROOT}/harmbench_behaviors_text_all.json"
  export HARMBENCH_DIFFUGUARD_JSON="${JAILBREAK_DATA_ROOT}/harmbench_prompts_diffuguard.json"
  export STRONGREJECT_RAW_JSON="${JAILBREAK_DATA_ROOT}/strongreject_raw.json"
  export STRONGREJECT_JSON="${JAILBREAK_DATA_ROOT}/strongreject_prompts.json"
  export STRONGREJECT_DIFFUGUARD_JSON="${JAILBREAK_DATA_ROOT}/strongreject_prompts_diffuguard.json"
  export ATTACK_PROMPT_JBB_DIJA="${JBB_DIJA_JSON}"
  export ATTACK_PROMPT_JBB_DIFFUGUARD="${JBB_DIFFUGUARD_JSON}"
  export ATTACK_PROMPT_HARMBENCH="${HARM_BENCH_JSON}"
  export HARMBENCH_PROMPTS_DIJA="${HARM_BENCH_JSON}"
  export HARMBENCH_PROMPTS_DIFFUGUARD="${HARMBENCH_DIFFUGUARD_JSON}"
  export STRONGREJECT_PROMPTS_DIJA="${STRONGREJECT_JSON}"
  export STRONGREJECT_PROMPTS_DIFFUGUARD="${STRONGREJECT_DIFFUGUARD_JSON}"
  # Example: choose JBB prompts for DiffuGuard/DIJA
  # export ATTACK_PROMPT="${ATTACK_PROMPT_JBB_DIFFUGUARD}"   # for DiffuGuard
  # export ATTACK_PROMPT="${ATTACK_PROMPT_JBB_DIJA}"         # for DIJA
  if [[ -z "${ATTACK_PROMPT:-}" ]]; then
    if [[ "${jailbreak_dataset}" == "harmbench" ]]; then
      export ATTACK_PROMPT="${ATTACK_PROMPT_HARMBENCH}"
    else 
      if [[ "${jailbreak_mode}" == "dija" ]]; then
        export ATTACK_PROMPT="${ATTACK_PROMPT_JBB_DIJA}"
      else
        export ATTACK_PROMPT="${ATTACK_PROMPT_JBB_DIFFUGUARD}"
      fi
    fi
  fi
  if [[ -z "${HARMBENCH_PROMPTS:-}" && "${jailbreak_dataset}" == "harmbench" ]]; then
    if [[ "${jailbreak_mode}" == "diffuguard" ]]; then
      export HARMBENCH_PROMPTS="${HARMBENCH_PROMPTS_DIFFUGUARD}"
    else
      export HARMBENCH_PROMPTS="${HARMBENCH_PROMPTS_DIJA}"
    fi
  fi
  if [[ -z "${STRONGREJECT_PROMPTS:-}" && "${jailbreak_dataset}" == "strongreject" ]]; then
    if [[ "${jailbreak_mode}" == "diffuguard" ]]; then
      export STRONGREJECT_PROMPTS="${STRONGREJECT_PROMPTS_DIFFUGUARD}"
    else
      export STRONGREJECT_PROMPTS="${STRONGREJECT_PROMPTS_DIJA}"
    fi
  fi
fi

echo "[INFO] Loaded profile '${profile}'."
echo "[INFO] Module status: ${MODULE_STATUS}"
if [[ -n "${ACTIVATED_VENV}" ]]; then
  echo "[INFO] Activated venv: ${ACTIVATED_VENV}"
else
  echo "[INFO] Activated venv: <none>"
fi
echo "[INFO] Exported environment variables:"

for key in SAFE_REPELLENCY_CSV_LOG REPO_ROOT HF_HOME HF_DATASETS_CACHE HF_MODELS_CACHE MEMORIZATION_DATA_DIR MEMORIZATION_RESULTS_DIR MDLM_CONFIG_OVERRIDES JAILBREAK_DATA_ROOT JBB_BEHAVIORS_CSV JBB_DIJA_JSON JBB_DIFFUGUARD_JSON HARM_BENCH_CSV HARM_BENCH_JSON HARMBENCH_DIFFUGUARD_JSON HARMBENCH_PROMPTS_DIJA HARMBENCH_PROMPTS_DIFFUGUARD HARMBENCH_PROMPTS STRONGREJECT_RAW_JSON STRONGREJECT_JSON STRONGREJECT_DIFFUGUARD_JSON STRONGREJECT_PROMPTS_DIJA STRONGREJECT_PROMPTS_DIFFUGUARD STRONGREJECT_PROMPTS ATTACK_PROMPT ATTACK_PROMPT_JBB_DIJA ATTACK_PROMPT_JBB_DIFFUGUARD ATTACK_PROMPT_HARMBENCH CHECKPOINT_PATH TOKENIZER_PATH MODEL_CONFIG_PATH LLAMAGUARD_CHECKPOINT_PATH FK_ROBERTA_CHECKPOINT_PATH CAP_APPLY_CHAT_TEMPLATE SKIP_PIP_UPGRADE PIP_INSTALL_ARGS EXTERNAL_VENV_ACTIVATE PYTHONPATH SAFE_REPELLENCY_DEBUG SAFE_KERNEL_MODE SAFE_GUIDANCE_MODE SAFE_BETA_MODE SAFE_REPELLENCY_VALIDATE SAFE_LLADA_DEBUG SAFE_DIST_LOG_ENABLED SAFE_DIST_LOG_DIR SAFE_DIST_LOG_PATH SAFE_DIST_LOG_TIMESTEPS SAFE_DIST_LOG_TOPK SAFE_DIST_LOG_MAX_POS SAFE_DIST_LOG_POSITIONS SAFE_DIST_LOG_POSITION_MODE SAFE_DIST_LOG_POSITION_SAMPLE_SEED SAFE_DIST_LOG_DUMP_TOKENS SAFE_DIST_LOG_DUMP_PROMPT SAFE_DIST_LOG_FULL_VOCAB SAFE_DIST_LOG_DTYPE SAFE_DIST_LOG_AUTO_TOP_N SAFE_DIST_LOG_RUN_ID; do
  if [[ -n "${!key+x}" ]]; then
    echo "  ${key}=${!key}"
  fi
done
