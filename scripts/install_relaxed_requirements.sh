#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/install_relaxed_requirements.sh [options]

Options:
  --requirements PATH   requirements.txt path (repeatable)
  --constraints PATH    constraints file passed to pip install (optional)
  --venv PATH           create/use venv at PATH (optional)
  --no-install          only write relaxed files, do not pip install
  --skip-nvidia         drop nvidia* packages (default: on)
  --skip-prefix PREFIX  drop packages with this prefix (repeatable)
  --pip-args "..."      extra args passed to pip install

Defaults:
  --requirements src/third_party/DIJA/requirements.txt
  --requirements src/third_party/DiffuGuard/requirements.txt
EOF
}

REQ_FILES=()
CONSTRAINTS_FILE=""
VENV_PATH=""
NO_INSTALL=0
PIP_ARGS=""
SKIP_GOOGLE=0
SKIP_NVIDIA=1
SKIP_PREFIXES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --requirements)
      REQ_FILES+=("$2")
      shift 2
      ;;
    --constraints)
      CONSTRAINTS_FILE="$2"
      shift 2
      ;;
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --no-install)
      NO_INSTALL=1
      shift
      ;;
    --skip-nvidia)
      SKIP_NVIDIA=1
      shift
      ;;
    --skip-prefix)
      SKIP_PREFIXES+=("$2")
      shift 2
      ;;
    --pip-args)
      PIP_ARGS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ${#REQ_FILES[@]} -eq 0 ]]; then
  REQ_FILES+=("src/third_party/DIJA/requirements.txt")
  REQ_FILES+=("src/third_party/DiffuGuard/requirements.txt")
fi

if [[ -n "${VENV_PATH}" ]]; then
  python - <<PY
import venv
venv.create("${VENV_PATH}", with_pip=True, clear=False)
PY
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
  python -m pip install --upgrade pip
fi

RELAXED_FILES=()
for req in "${REQ_FILES[@]}"; do
  if [[ ! -f "${req}" ]]; then
    echo "[WARN] Missing requirements file: ${req}" >&2
    continue
  fi
  out="./$(basename "${req}").relaxed.txt"
  if [[ "${#SKIP_PREFIXES[@]}" -gt 0 ]]; then
    SKIP_PREFIXES_JSON=$(python - "${SKIP_PREFIXES[@]}" <<'PY'
import json, sys
print(json.dumps(sys.argv[1:]))
PY
    )
  else
    SKIP_PREFIXES_JSON="[]"
  fi
  SKIP_PREFIXES_JSON="${SKIP_PREFIXES_JSON}" python - <<PY
from pathlib import Path
import json
import os

src = Path("${req}")
dst = Path("${out}")
skip_google = ${SKIP_GOOGLE}
skip_nvidia = ${SKIP_NVIDIA}
skip_prefixes = json.loads(os.environ.get("SKIP_PREFIXES_JSON", "[]"))
extra_prefixes = []
extra_prefixes += [
    "numpy", "scipy", "matplotlib", "pandas", "pyarrow", "opencv-",
    "scikit-", "sklearn", "numexpr", "pybind11"
]
if skip_google:
    extra_prefixes += ["google-", "google_", "google"]
if skip_nvidia:
    extra_prefixes += ["nvidia-"]
if skip_prefixes:
    extra_prefixes += list(skip_prefixes)
out_lines = []
for line in src.read_text().splitlines():
    raw = line.strip()
    if not raw or raw.startswith("#"):
        out_lines.append(line)
        continue
    if " @ " in raw or raw.startswith("git+"):
        out_lines.append(line)
        continue
    name = raw.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip()
    low = name.lower()
    if any(low == p.rstrip("-") or low.startswith(p) for p in extra_prefixes):
        continue
    if "==" in raw:
        out_lines.append(name)
        continue
    out_lines.append(line)
dst.write_text("\n".join(out_lines) + "\n")
print(dst)
PY
  RELAXED_FILES+=("${out}")
done

if [[ "${NO_INSTALL}" -eq 1 ]]; then
  echo "[INFO] Relaxed files written:"
  printf "  %s\n" "${RELAXED_FILES[@]}"
  exit 0
fi

DEFAULT_PIP_ARGS="--only-binary=:all: --prefer-binary --no-build-isolation"
for relaxed in "${RELAXED_FILES[@]}"; do
  echo "[INFO] Installing ${relaxed}"
  if [[ -n "${PIP_ARGS}" ]]; then
    python -m pip install  ${DEFAULT_PIP_ARGS}  ${PIP_ARGS} -r "${relaxed}" -c "${CONSTRAINTS_FILE}" --upgrade-strategy only-if-needed
  else
    python -m pip install  ${DEFAULT_PIP_ARGS}  -r "${relaxed}" -c "${CONSTRAINTS_FILE}" --upgrade-strategy only-if-needed
  fi
done

echo "[INFO] Done."
