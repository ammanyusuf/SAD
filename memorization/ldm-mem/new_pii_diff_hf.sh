base="smdm/workspace/icml/results/pii_result/PREPOST2K_8BFINE_enron_pii.phone.pre100.jsonl"
lit=8096
steps=1
out="${base%.*}_lit${lit}_steps${steps}.${base##*.}"

 \
python smdm/n_eval_pii_diff.py \
  --input_path smdm/enron_pii.phone.post2k.jsonl \
  --output_path "$out" \
  --log_path smdm/workspace/icml/logs/justfordoublecheckdiffpii1beachstep.log \
  --ckpt_path smdm/workdir/scaling_debug/hf-GSAI-ML_LLaDA-8B-Base-lr1e-05-gb192/enronfinetune-ckpt.pth \
  --hf_model_name GSAI-ML/LLaDA-8B-Base \
  --mask_id 126336 \
  --device cuda \
  --dtype bf16 \
  --mc_samples 64 \
  --mc_batch_size 64 \
  --recover_steps "$steps" \
  --recover_alg origin \
  --gen_temperature 1.0 \
  --recover_eps 1e-3 \
  --recover_cfg_scale 0.0 \
  --trust_remote_code \
  --recover_each_token \
  --save_log_sums