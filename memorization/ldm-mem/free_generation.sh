OUT_DIR="smdm/workspace/icml/results/new_free_gene"
LOG_DIR="smdm/workspace/icml/logs/new_free_gene"
START_LINE=96
END_LINE=128
mkdir -p "$OUT_DIR" "$LOG_DIR"

OUT_PATH="${OUT_DIR}/8benron_freegene_${TS}_s${START_LINE}_e${END_LINE}.jsonl"
LOG_PATH="${LOG_DIR}/8benron_freegene_${TS}_s${START_LINE}_e${END_LINE}.log"

 \
python smdm/evaluation_free_generation_hf.py \
  --input_path smdm/data/generation_target/8b_newtarget.jsonl \
  --start_index "${START_LINE}" \
  --end_index "${END_LINE}" \
  --hf_model_name GSAI-ML/LLaDA-8B-Base \
  --ckpt_path smdm/workdir/scaling_debug/hf-GSAI-ML_LLaDA-8B-Base-lr1e-05-gb192/enronfinetune-ckpt.pth \
  --diff_mask_id 126336 \
  --tail_len 100 \
  --total_traj 10000 \
  --traj_batch_size 64 \
  --gumbel_temperature 1.0 \
  --device cuda \
  --dtype bfloat16 \
  --hf_trust_remote_code \
  --output_path "${OUT_PATH}" \
  --log_path "${LOG_PATH}"
