set -euo pipefail

TRAJ_START=0
TRAJ_END=1


python smdm/new_free_gene_hf.py \
  --input_path  smdm/enron_aligntest/newonlyone.jsonl \
  --output_path smdm/gene_fix_trj_result/find_first_page${TRAJ_START}_e${TRAJ_END}.jsonl \
  --use_hf \
  --hf_model_name GSAI-ML/LLaDA-8B-Base \
  --ckpt_path smdm/workdir/scaling_debug/hf-GSAI-ML_LLaDA-8B-Base-lr1e-05-gb192/enronfinetune-ckpt.pth \
  --enable_gen_dump \
  --gen_dump_dir smdm/gene_fix_trj_result/firstpageresult.jsonl \
  --dump_full_sequence \
  --runs 100 \
  --eval_batch_size 64 \
  --steps_5 5 \
  --target_trajs 1 \
  --enable_per_token_steps \
  --enable_multistep_eval \
  --gen_temperature 1.0 \
  --cfg_scale 0.0 \
  --device cuda \
  --max_traj_per_sample 1 \
  --traj_start ${TRAJ_START} \
  --traj_end ${TRAJ_END} \
  --dtype bf16
