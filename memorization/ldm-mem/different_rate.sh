#!/usr/bin/env bash
set -euo pipefail

python smdm/n_evaluation_different_ratio.py \
  --samples_path smdm/dataset/raw_data/trec_win100.jsonl \
  --ckpt_path smdm/workdir/scaling_debug/mdm-170M-0.0/enron.pth\
  --lit_model_name 170 \
  --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --device cuda \
  --max_samples 10000 \
  --diff_mc_samples 512 \
  --diff_mc_batch_size 512 \
  --mask_rates "0.25" \
  --diff_mask_id 32000 \
  --tail_len 100 \
  --only_gen1 \
  --start_line 1 \
  --end_line 10000 \
  --output_path ./output/10000-different_trec170final.jsonl \
  --log_path ./results/10000-different_trec170-final.log
