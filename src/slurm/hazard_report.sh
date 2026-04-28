#!/bin/bash
#SBATCH --job-name=sb_hazard_report
#SBATCH --account=rrg-<your-PI>   # [Compute Canada] replace with your allocation
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --mail-user=${USER}@<your-institution.ca>  # [Compute Canada] set to your email
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=/scratch/%u/logs/safe-text-diffusion/hazard_report_%j.out
#SBATCH --error=/scratch/%u/logs/safe-text-diffusion/hazard_report_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT=${REPO_ROOT:-${DEFAULT_REPO_ROOT}}

# logic to handle RUN_DIRS or RUN_DIRS_FILE
RUN_DIR_ARRAY=()
if [[ -n "${RUN_DIRS_FILE:-}" ]] && [[ -f "${RUN_DIRS_FILE}" ]]; then
  while IFS= read -r line; do
    [[ -n "$line" ]] && RUN_DIR_ARRAY+=("$line")
  done < "${RUN_DIRS_FILE}"
elif [[ -n "${RUN_DIRS:-}" ]]; then
  while IFS= read -r run_dir; do
    [[ -z "${run_dir// }" ]] && continue
    RUN_DIR_ARRAY+=("${run_dir}")
  done <<< "$(echo "${RUN_DIRS}" | tr ',' '\n')"
else
  echo "Error: Neither RUN_DIRS_FILE nor RUN_DIRS provided." >&2
  exit 1
fi

TS=${HAZARD_TIMESTAMP:-$(date +"%Y%m%d%H%M%S")}
# default behavious:  the first run dir to set default output dir if not provided
FIRST_RUN_DIR="${RUN_DIR_ARRAY[0]}"
OUTPUT_DIR=${OUTPUT_DIR:-"${FIRST_RUN_DIR}/../hazard_report_${TS}"}
EXAMPLES_PER_HAZARD=${EXAMPLES_PER_HAZARD:-10}
PYTHON_BIN=${PYTHON_BIN:-python}
EXTERNAL_VENV_ACTIVATE=${EXTERNAL_VENV_ACTIVATE:-}
PIP_INSTALL_ARGS=${PIP_INSTALL_ARGS:-}
SKIP_PIP_UPGRADE=${SKIP_PIP_UPGRADE:-0}

module purge
module load StdEnv/2023 python/3.11 gcc arrow/21.0.0 scipy-stack

TMP_REPO="${SLURM_TMPDIR}/repo"
mkdir -p "${TMP_REPO}"
rsync -a --exclude=".git" --exclude=".env" --exclude=".env-gpu" --exclude=".env-jailbreak" "${REPO_ROOT}/" "${TMP_REPO}/"

if [[ -n "${CONFIG_SNAPSHOT_PATH:-}" && -d "${CONFIG_SNAPSHOT_PATH}" ]]; then
  echo "[INFO] Overwriting staged configs with snapshot from ${CONFIG_SNAPSHOT_PATH}/configs/"
  rsync -a "${CONFIG_SNAPSHOT_PATH}/configs/" "${TMP_REPO}/configs/"
fi

USING_EXTERNAL_ENV=0
if [[ -n "${EXTERNAL_VENV_ACTIVATE}" ]]; then
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
  if [[ "${SKIP_PIP_UPGRADE}" != "1" ]]; then
    python -m pip install --upgrade pip
  fi
  if [[ -n "${PIP_INSTALL_ARGS}" ]]; then
    python -m pip install ${PIP_INSTALL_ARGS} -r "${TMP_REPO}/src/requirements-cc.txt"
  else
    python -m pip install -r "${TMP_REPO}/src/requirements-cc.txt"
  fi
fi

export PYTHONPATH="${TMP_REPO}/src:${TMP_REPO}/src/third_party:${PYTHONPATH:-}"

expanded=()
for rd in "${RUN_DIR_ARRAY[@]}"; do
  matches=( $rd )
  if [[ ${#matches[@]} -gt 1 ]] || [[ "${matches[0]}" != "${rd}" ]]; then
    expanded+=("${matches[@]}")
  else
    expanded+=("${rd}")
  fi
done

expanded_with_seeds=()
for rd in "${expanded[@]}"; do
  seed_dirs=( "${rd}"/seed=* )
  if [[ -d "${seed_dirs[0]}" ]]; then
    expanded_with_seeds+=("${seed_dirs[@]}")
  else
    expanded_with_seeds+=("${rd}")
  fi
done

CMD=(
  "${PYTHON_BIN}"
  src/tools/hazard_report.py
  --output-dir "${OUTPUT_DIR}"
  --examples-per-hazard "${EXAMPLES_PER_HAZARD}"
  --examples-per-metric "${EXAMPLES_PER_METRIC:-${EXAMPLES_PER_HAZARD:-5}}"
  --examples-per-transition "${EXAMPLES_PER_TRANSITION:-${EXAMPLES_PER_HAZARD:-5}}"
)
if [[ "${HAZARD_JAILBREAK_SPLIT:-0}" == "1" ]]; then
  CMD+=(--jailbreak-split)
fi
for rd in "${expanded_with_seeds[@]}"; do
  CMD+=(--run-dirs "${rd}")
done

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"

TAR_OUTPUT=${TAR_OUTPUT:-"${OUTPUT_DIR}.tar.gz"}
if [[ -n "${TAR_OUTPUT}" ]]; then
  tar -czvf "${TAR_OUTPUT}" -C "$(dirname "${OUTPUT_DIR}")" "$(basename "${OUTPUT_DIR}")"
  echo "[INFO] Hazard report archive written to ${TAR_OUTPUT}"
fi

echo "[INFO] Hazard report complete. Outputs at ${OUTPUT_DIR}"
