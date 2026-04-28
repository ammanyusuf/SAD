# Jailbreak Evaluation

Evaluates the safe denoiser against jailbreak attacks (PAD, DIJA) using DiffuGuard and DIJA harnesses, with LlamaGuard + HarmBench ASR scoring. Three instruct models are evaluated: LLaDA-8B-Instruct, LLaDA-1.5, and Dream-v0-Instruct-7B.

## Dataset preparation

Convert benchmark datasets to DiffuGuard/DIJA JSON format before running evals:

```bash
python scripts/setup_jailbreak_assets.py --out $JAILBREAK_DATA_ROOT
python scripts/convert_jbb_to_diffuguard_json.py --out $JAILBREAK_DATA_ROOT
python scripts/convert_jbb_to_dija_json.py --out $JAILBREAK_DATA_ROOT
```

Dataset catalogs are in `configs/data/catalog_jailbreak.yaml`. Set `JAILBREAK_DATA_ROOT` to the directory containing the converted JSON files.

## Ad-hoc generation (no Slurm)

```bash
# Upstream LLaDA (no jailbreak defense)
python -m tools.generate model.family=llada model.variant=upstream model.checkpoint=/path/to/llada

# With DiffuGuard defense
python -m tools.generate model.family=llada model.variant=diffuguard model.checkpoint=/path/to/llada

# With DIJA attack prompts
python -m tools.generate model.family=llada model.variant=dija model.checkpoint=/path/to/llada

# With safe denoiser enabled
python -m tools.generate \
  model.family=llada model.variant=upstream model.checkpoint=/path/to/llada \
  safety.enabled=true safety.eta=1.5 safety.t_start=0 safety.t_end=18
```

## DiffuGuard jailbreak eval

Runs jailbreak attacks (zero-shot, PAD, DIJA) with and without the safe denoiser, across multiple datasets (JBB, HarmBench, AdvBench, StrongREJECT, WildJailbreak).

```bash
# Ad-hoc
python -m tools.eval_diffuguard \
  model.checkpoint=/path/to/llada \
  jailbreak.attack_prompt=$JAILBREAK_DATA_ROOT/jbb_behaviors_harmful_diffuguard.json \
  io.output_dir=/path/to/results \
  safety.enabled=true safety.eta=1.5 safety.t_start=0 safety.t_end=18

# Slurm (LLaDA-8B-Instruct)
python src/slurm/submit_sbatch_jailbreak.py \
  --config configs/slurm/sbatch_eval_diffguard_jailbreak_llada_instruct.yaml \
  --repo-root "$REPO_ROOT"

# Slurm (LLaDA-1.5)
python src/slurm/submit_sbatch_jailbreak.py \
  --config configs/slurm/sbatch_eval_diffguard_jailbreak_llada_1.5.yaml \
  --repo-root "$REPO_ROOT"

# Slurm (Dream-v0-Instruct)
python src/slurm/submit_sbatch_jailbreak.py \
  --config configs/slurm/sbatch_eval_diffguard_jailbreak_dream_instruct.yaml \
  --repo-root "$REPO_ROOT"
```

DiffuGuard defense variants (set via `jailbreak.defense_method`):
`none`, `self-reminder`, `ppl`, `diffuguard` (hidden-state audit).

## DIJA jailbreak eval

```bash
# Ad-hoc
python -m tools.eval_dija \
  model.checkpoint=/path/to/llada \
  jailbreak.attack_prompt=$JAILBREAK_DATA_ROOT/jbb_behaviors_harmful_dija.json \
  io.output_dir=/path/to/results \
  safety.enabled=true safety.eta=1.5 safety.t_start=0 safety.t_end=18

# Slurm
python src/slurm/submit_sbatch_jailbreak.py \
  --config configs/slurm/sbatch_eval_dija_jailbreak.yaml \
  --repo-root "$REPO_ROOT"
```

## Multi-seed runs and aggregation

```bash
# Run seeds 1–5 (outputs land in run_dir/seed=<n>/)
python src/slurm/submit_sbatch_jailbreak.py \
  --config configs/slurm/sbatch_eval_diffguard_jailbreak_llada_instruct.yaml \
  --repo-root "$REPO_ROOT" \
  --seeds 1 2 3 4 5

# Aggregate metrics (mean ± std) after scoring finishes
python src/slurm/submit_sbatch_jailbreak.py \
  --aggregate-only \
  --aggregate-run-dir /path/to/results/<slug>/<run_id> \
  --seeds 1 2 3 4 5

# Or chain aggregation automatically
python src/slurm/submit_sbatch_jailbreak.py \
  --config configs/slurm/sbatch_eval_diffguard_jailbreak_llada_instruct.yaml \
  --repo-root "$REPO_ROOT" \
  --seeds 1 2 3 --auto-aggregate
```

## Mask token reference

| Model | Mask token | HuggingFace ID |
|---|---|---|
| LLaDA-8B-Base/Instruct | `<\|mdm_mask\|>` (id=126336) | `GSAI-ML/LLaDA-8B-Base` |
| LLaDA-1.5 | `<\|mask\|>` | `inclusionAI/LLaDA2.0-mini` |
| Dream-v0-7B | `<\|mask\|>` (id=151666) | `Dream-org/Dream-v0-Base-7B` |
