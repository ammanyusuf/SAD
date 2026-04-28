 \
python smdm/n_eval_pii_ar.py \
  --samples_path smdm/enron_pii.phone.pre100.jsonl \
  --ckpt_path smdm/workdir/scaling_debug/arm-1028M-0.0/enron-finetune-final.pth \
  --lit_model_name 1028 \
  --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --gen_temperature 1.0 \
  --gen_top_k 40 \
  --batch_size 32 \
  --output_path smdm/workspace/icml/results/pii_result/phoneAR.jsonl \
  --log_path smdm/workspace/icml/logs/ar.log \
  --debug_print_first_k 2
