set -euo pipefail

python smdm/evaluation_ar_diff_ratio.py \
  --samples_path smdm/data/1b_pretrain_ar.jsonl \
  --ckpt_path smdm/arm-1028M-100.0/iter-164972-ckpt.pth \
  --lit_model_name 1028 \
  --tail_len 100 \
  --mask_rates "0.25" \
  --gen_temperature 1.0 \
  --gen_top_k 40 \
  --p_targets "0.1,0.5,0.9,0.99" \
  --n_targets "1,10,100" \
  --device cuda \
  --max_samples 10000 \
  --batch_size 128 \
  --output_path .smdm/results/pretrain_train_memo_results_mask_rates_ar.jsonl \
  --log_path .smdm/output/pretrain_train_memo_run_mask_rates_ar.log
