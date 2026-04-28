#!/usr/bin/env bash
set -euo pipefail


START_LINE=160
END_LINE=200
END_INDEX=$((END_LINE + 1))   

TS="$(date +'%Y%m%d_%H%M%S')"

OUT_DIR="smdm/workspace/icml/results/new_free_gene"
LOG_DIR="smdm/workspace/icml/logs/new_free_gene"

mkdir -p "$OUT_DIR" "$LOG_DIR"

OUT_PATH="${OUT_DIR}/1bpretrain100000_${TS}_s${START_LINE}_e${END_LINE}.jsonl"
LOG_PATH="${LOG_DIR}/1bpretrain100000_${TS}_s${START_LINE}_e${END_LINE}.log"

python smdm/n_eval_freegeneration.py \
  --input_path smdm/data/generation_target/1b_200_samples.jsonl \
  --start_index "${START_LINE}" \
  --end_index "${END_INDEX}" \
  --lit_model_name 1028 \
  --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --ckpt_path smdm/workdir/scaling_debug/mdm-1028M-100.0/final-ckpt.pth \
  --diff_mask_id 32000 \
  --tail_len 100 \
  --total_traj 100000 \
  --traj_batch_size 256 \
  --gumbel_temperature 1.0 \
  --alg origin \
  --cfg_scale 0.0 \
  --device cuda \
  --dtype bfloat16 \
  --default_mask_ratio 0.2 \
  --use_sample_mask_ratio \
  --output_path "${OUT_PATH}" \
  --log_path "${LOG_PATH}"
