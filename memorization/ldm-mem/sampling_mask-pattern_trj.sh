
 python smdm/newpara_mask.py \
  --input_path smdm/data/generation_target/1b_200_samples.jsonl \
  --lit_model_name 1028 \
  --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --ckpt_path smdm/workdir/scaling_debug/mdm-1028M-0.0/final.pth \
  --mask_id 32000 \
  --output_path smdm/sampled_trj/1b_final_001.jsonl \
  --tail_len 100 \
  --default_mask_ratio 0.2 \
  --num_traj 128 \
  --traj_batch_size 64 \
  --gen_temperature 1.0 \
  --gen_top_k 40 \
  --device cuda \
  --dtype bfloat16
