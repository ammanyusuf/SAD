
python smdm/ar_distribution.py \
  --samples_path smdm/ar_gene.jsonl \
  --ckpt_path smdm/workdir/scaling_debug/arm-1028M-0.0/final.pth \
  --lit_model_name 1028 \
  --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --gen_temperature 1.0 \
  --gen_top_k 40 \
  --batch_size 128 \
  --n_trials 10000 \
  --prompt_len 80 \
  --gen_len 20 \
  --truncate_prompt_to_fit \
  --output_path smdm/workspace/icml/results/pii_result/ARGENERATION.jsonl
