base="smdm/workspace/icml/results/pii_result/1.1Bemailgreedy.jsonl"
lit=1028
steps=1
out="${base%.*}_lit${lit}_steps${steps}.${base##*.}"


python smdm/n_eval_pii_diff.py \
  --input_path smdm/enron_aligntest/realenron_aligntestemail_contexts.jsonl \
  --output_path "$out" \
  --log_path smdm/workspace/icml/logs/justfordoublecheckdiffpii1beachstep.log \
  --lit_model_name "$lit" \
  --ckpt_path smdm/workdir/scaling_debug/mdm-1028M-0.0/enron.pth \
  --tokenizer_path smdm/dataset/TinyLlama/checkpoints \
  --mask_id 32000 \
  --device cuda \
  --dtype bf16 \
  --mc_samples 32 \
  --mc_batch_size 32 \
  --recover_steps "$steps" \
  --recover_alg greddy \
  --gen_temperature 0.0 \
  --recover_eps 1e-3 \
  --recover_cfg_scale 0.0 \
  --recover_each_token \
  --save_log_sums


