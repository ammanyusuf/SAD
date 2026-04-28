#!/bin/bash
#SBATCH --job-name=sb_index_corpus
#SBATCH --account=rrg-<your-PI>   # [Compute Canada] replace with your allocation
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=${USER}@<your-institution.ca>  # [Compute Canada] set to your email
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --output=/scratch/%u/logs/safe-text-diffusion/index_%j.out
#SBATCH --error=/scratch/%u/logs/safe-text-diffusion/index_%j.err
#SBATCH --array=0-0

set -euo pipefail

if [[ -z ${1:-} ]]; then
  echo "Usage: $0 /path/to/repo" >&2
  exit 1
fi
REPO_ROOT=$1
shift || true

module load cuda python/3.11 gcc arrow/21.0.0 scipy-stack
source "${REPO_ROOT}/.env/bin/activate"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$SCRATCH/hf_cache}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$SCRATCH/hf_datasets}
export PYTHONPATH=${REPO_ROOT}/src:${REPO_ROOT}/src/third_party:${PYTHONPATH:-}

cd "${REPO_ROOT}"

export PYTHONHASHSEED=0

CORPUS=${CORPUS:?Set CORPUS to target corpus (enwiki8|enwiki9|pile_uncopyrighted)}
TOKENIZER=${TOKENIZER:?Set TOKENIZER to tokenizer name/path}
OUTDIR=${OUTDIR:-$SCRATCH/safe-text-diffusion/results/indexes}
DEDUP=${DEDUP:-none}
MIN_GRAM=${MIN_GRAM:-11}
MAX_GRAM=${MAX_GRAM:-13}

mkdir -p "${OUTDIR}"

python -m tools.index_corpus \
  --corpus "${CORPUS}" \
  --tokenizer "${TOKENIZER}" \
  --dedupe "${DEDUP}" \
  --min_gram "${MIN_GRAM}" \
  --max_gram "${MAX_GRAM}" \
  --outdir "${OUTDIR}" \
  --shard_id "${SLURM_ARRAY_TASK_ID}" \
  --num_shards "${SLURM_ARRAY_TASK_COUNT}"
