set -euo pipefail

TRAJ_START=175
TRAJ_END=200

 \
python smdm/new_freegene_trj.py \
  --input_path  smdm/sampled_trj/1b_final_001_goodtraj_20260120_092432_s0_eend.jsonl \
  --output_path smdm/gene_fix_trj_result/10000masked_001_step_all_hits_s${TRAJ_START}_e${TRAJ_END}.jsonl \
  --lit_model_name 1028 \
  --ckpt_path smdm/workdir/scaling_debug/mdm-1028M-0.0/final.pth \
  --runs 10000 \
  --eval_batch_size 512 \
  --steps_2 2 \
  --steps_5 5 \
  --steps_10 10 \
  --target_trajs 200 \
  --enable_per_token_steps \
  --enable_multistep_eval \
  --gen_temperature 1.0 \
  --cfg_scale 0.0 \
  --device cuda \
  --max_traj_per_sample 1 \
  --traj_start ${TRAJ_START} \
  --traj_end ${TRAJ_END} \
  --dtype bf16
