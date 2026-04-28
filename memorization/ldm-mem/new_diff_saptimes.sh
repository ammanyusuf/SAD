#!/usr/bin/env bash
set -euo pipefail


python smdm/n_eval_multi_samplingtimes.py \
  --ckpt_path smdm/workdir/scaling_debug/mdm-1028M-100.0/final.pth \
  --input_path smdm/data/generation_target/1b_200_samples.jsonl  \
  --device cuda \
  --lit_model_name 1028 \
  --dtype bfloat16 \
  --cfg_scale 0.0 \
  --gumbel_temperature 1.0 \
  --tokenizer_name smdm/dataset/TinyLlama/checkpoints \
  --diff_mask_id 32000 \
  --traj_list "128" \
  --traj_batch_size 1024 \
  --alpha_clip 100.0 \
  --alg origin \
  --output_path ./results/new_free_gene/clip10000.jsonl \
  --log_path ./logs/new_free_gene/clip10000.log