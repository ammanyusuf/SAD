# Safe Text Diffusion

Experimental repo to test out safe text generation with text based diffusion models.

## Features

- **MDLM Integration**: Works with Masked Discrete Language Models for text diffusion
- **Discrete Safe Denoiser**: Training-free safety mechanism for text generation

## Evaluation Runbook (Compute Canada)

All heavy artefacts live under `$SCRATCH/safe-text-diffusion`. Create symlinks once per workspace:

```bash
mkdir -p $SCRATCH/safe-text-diffusion/{data,results}
ln -sfn $SCRATCH/safe-text-diffusion/data data
ln -sfn $SCRATCH/safe-text-diffusion/results results
```

All repository-managed code now lives under `src/`.


1. **Generate completions (safety & memorization)**
   - For ad-hoc runs, export the variables documented in `src/slurm/README.md` and submit `src/slurm/generate_array.sh`.

## Get Started

### Environment 
We'll be using Python 3.11. Virtualenv will be used as the environemnt manager as its teh one that works on Compute Canada. `https://pypi.org/project/virtualenv/` (follow install instructions if you don't have it).


Usage: `https://virtualenv.pypa.io/en/latest/`. Usual flow is:

1. Creat environemtn with `env_name`. (make sure you specify which python version you're using).
2. Activate the `env_name` environment.
3. Confirm that you created it.

```
virtualenv env_name

source env_name/bin/activate

which python3
```

Use `pyenv` to manage Python versions pretty easily: `https://github.com/pyenv/pyenv` (we'll be using Python 3.11 as mentioned). Pyenv can use virtualenv to set up your pyenv version and environements.
To list which virtualenv you have
```
pyenv virtualenvs
```
```
pyenv activate env-name-you-created
```

See usage for creating it and setting it up in Compute Canada: `https://docs.alliancecan.ca/wiki/Python#Creating_and_using_a_virtual_environment`



### Requirements
Once you have the environment set up, install the requiremments.txt via the following:

`pip install -r src/requirements.txt`

## Directory Structure
- `src/third_party/mdlm/`: upstream discrete diffusion code plus vendor scripts.
- `src/safety_eval/`: HarmBench + LlamaGuard utilities (classifiers, CLI helpers).
- `src/utils/`, `src/sampling/`, `src/unsafe_prep/`: shared helpers, sampling CLI, and unsafe artifact builder.
- `src/tools/`: unified CLIs (`index_corpus`, `generate`, `score`) and utilities.
- `src/slurm/`: array-job templates and submission docs for generation and scoring.
- `configs/`: Hydra defaults for generation/scoring plus the shared data catalog and Slurm submission bundles.
- `data/` → symlink to `$SCRATCH/safe-text-diffusion/data`.
- `results/` → symlink to `$SCRATCH/safe-text-diffusion/results`.

## Configuration Reference (Hydra)

All entry points (`python -m tools.generate`, `python -m tools.score`, `src/slurm/*`) read from `configs/config.yaml` via Hydra. Override any field with dot-notation, e.g. `python -m tools.generate data.dataset_json=/path/to/file gen.batch_size=8`.

### `model`
- `model_name`: String label reported in metadata.
- `checkpoint`: Path to the `.ckpt`.
- `tokenizer_name`: Directory containing tokenizer files.
- `precision`: bf16/fp16/tf32 label recorded in telemetry.

### `data`
- `prompt_source`: Adapter spec (`name` + `params`) that reuses the `unsafe_prep.adapters` registry (BeaverTails, RealToxicityPrompts, ToxiGen, etc.). Reusable fragments live in the shared data catalog (`configs/data/catalog.yaml`) and can be pulled into Hydra configs via `experiments/*` overlays.
- `total_prompts`: Optional metadata override when the dataset isn’t available locally.
- `prefix_length`: Conditioning prefix length.
- `limit`: Optional prompt cap per shard.
- `stage_to_tmp`: When `true` and `$SLURM_TMPDIR` is set, the dataset is copied into `$SLURM_TMPDIR/datasets/` and the staged path is used to keep I/O local.

### `gen`
- `max_new_tokens`, `batch_size`, `sampling_steps`, `seed`: Core sampling knobs.
- `add_bos`, `add_eos`: Boolean BOS/EOS injection.
- `unconditional_samples`: Number of unconditional completions when no dataset is provided.
- `dry_run`: Emit stub generations without touching the model.

### `safety`
- `enabled`: Toggle the safe denoiser.
- `scale`: Guidance strength.
- `unsafe_artifacts`: Direct path to a tensor.
- `unsafe_artifact_root` / `unsafe_artifact_name`: Directory + name for auto materialization.
- `auto_build_unsafe_artifacts`: When `true`, missing unsafe artifacts are generated on-demand at inference time using the active tokenizer (defaults: max_length=1024, shard_size=1024, seed=1; semantic cache is not generated).
- `t_start` `t_end` on applying safety

Tuning notes (LLaDA):
- `t_start`/`t_end` are **denoising steps**, not token indices. The safe denoiser runs on every step within the window.
- Effective strength grows with both `eta` and the number of applied steps. A rough mental model:
  `effective_strength ≈ eta × (t_end - t_start + 1)`
- More sampling steps = more hook applications. If you increase `steps`, you usually need to reduce `eta` and/or narrow the window.

Quick tuning loop:
1. Fix `steps`, `gen_length`, `block_length` and the unsafe artifact.
2. Sweep a small grid (example): `eta ∈ {0.1, 0.3, 0.5, 1.0}` and windows like `t_start/t_end ∈ {0-15, 0-31, 64-127, 128-255}`.
3. Use small eval limits first (e.g., `--limit 2` on HumanEval) and inspect for degeneration (empty output, repeated imports, long blanks).
4. Once stable, expand to a larger limit and only then adjust `eta` for safety/quality tradeoffs.

### `sharding`
- `range_start` / `range_end`: Explicit contiguous slice (used when precomputing offsets).
- `shard_id` / `num_shards`: Default contiguous slicing metadata.

### `io`
- `base_dir`: Root output directory (defaults to `results`).
- `experiment_slug`: Grouping slug (e.g., `harmbench_autoprompt`).
- `run_id`: Run folder (`${SLURM_JOB_ID}` by default).
- `output_dir`: Derived path (`${io.base_dir}/${io.experiment_slug}/${io.run_id}`).
- `track_name`: `safety` or `memorization`.
- `auto_batch`: Enable VRAM-based batch scaling.
- `target_vram_pct` / `auto_batch_warmup_prompts`: Autotune knobs.

### `score`
- `track`: `safety` or `memorization`.
- `model`: Label recorded in summaries.
- `run_dir`: Directory with generations.
- `classifier`: `llamaguard` or `harmbench`.
- `classifier_model`: Optional HF identifier or local path.
- `behaviors_csv`, `indexes_dir`, `batch_size`, `max_new_tokens`, `force`, `dry_run`: Scoring behavior.

Example safety config:
```yaml
safety:
  enabled: true
  auto_build_unsafe_artifacts: true
  unsafe_artifact_root: /scratch/$USER/safe-text-diffusion/artifacts/unsafe_artifacts
  unsafe_artifact_name: beavertails-0100-llada
```

### Experiment overlays

Per-experiment configs live under `configs/experiments/` (e.g., `beavertails_prompts.yaml`). Each file simply overrides the sections above (dataset path, experiment slug, etc.). Use them by passing `--config-name experiments/beavertails_prompts` when calling Hydra-enabled tools or by exporting `GEN_CONFIG_NAME`/`SCORE_CONFIG_NAME` for Slurm jobs.

Example:
```bash
python -m tools.generate \
  --config-name experiments/beavertails_prompts \
  gen.batch_size=8 sharding.shard_id=0 sharding.num_shards=1
```

## Prompt elicitation sweeps

`src/utils/experiment_setup.py` exposes `build_generation_plans(...)`, which expands a YAML description (see `configs/slurm/prompt_pipeline.example.yaml`) into concrete generation/scoring jobs:

1. Parses prompt datasets (BeaverTails, RealToxicityPrompts, ToxiGen) from the shared data catalog plus optional prompt limits and Hydra overrides.
2. Emits one baseline run per dataset (`safety.enabled=false`).
3. For datasets that specify `safety.artifact_root`/`artifact_names`, emits additional runs for every artifact × scale (default scales: 0/1/2/5).
4. Creates matching scoring plans so `tools.score` can run immediately after each generation job.

Use it from any Python entrypoint:

```python
from pathlib import Path
from utils.experiment_setup import build_generation_plans

plans = build_generation_plans(
    cfg_path=Path("configs/slurm/prompt_pipeline.example.yaml"),
    restrict_to=["beavertails", "toxigen"],  # or None for all datasets
    dry_run=True,   # print the plan without launching work
    disable_scoring=False,
)
```

Hydra overlays for the BeaverTails/RealToxicity/ToxiGen prompt sets now live under `configs/experiments/*.yaml` and pull their adapters from the shared data catalog (`configs/data/catalog.yaml`). The Slurm planner configs (e.g., `configs/slurm/sbatch_prompt_pipeline.yaml`) consume the same catalog so you only define those adapters once.

Override dataset locations via `BEAVERTAILS_DATA_DIR`, `REALTOXICITY_DATA_DIR`, and `TOXIGEN_DATA_DIR`, or supply your own `data.dataset_json` for a standalone file path.

### Slurm submission (prompt pipeline)

`python -m slurm.submit_sbatch_experiments --config <config.yaml> --safe-artifact-root <path>` drives the same workflow via Slurm. The submitter:

1. Parses the pipeline config (see `experiments/prompt_pipeline.example.yaml`) to determine datasets, prompt limits, and safety artifacts.
2. Submits `src/slurm/generate_array.sh` with the appropriate Hydra overrides for each baseline and safe-guidance variant, wiring `RUN_ID` so the staged outputs land under `<results_root>/<experiment_slug>/<run_id>`.
3. Immediately enqueues `src/slurm/score_array.sh` with an `afterok` dependency so scoring only starts after the corresponding generation slice finishes.

Key arguments:

```
python -m slurm.submit_sbatch_experiments \
  --config configs/slurm/prompt_pipeline.example.yaml \
  --repo-root /home/$USER/repos/safe-text-diffusion \
  --safe-artifact-root /scratch/$USER/safe-text-diffusion/artifacts/unsafe_artifacts \
  --baseline-array 0-3 --baseline-time 0-00:20 \
  --safe-array 0-3 --safe-time 0-00:40 \
  --score-array 0-0 --score-time 0-00:30
```

Override per-dataset Slurm settings inside the config (`datasets[].slurm.baseline`, `.safe`, or `.score`) when necessary. Make sure the environment variables that feed the Hydra configs are exported (e.g., `BEAVERTAILS_DATA_DIR`, `REALTOXICITY_DATA_DIR`, `TOXIGEN_DATA_DIR`) before calling the submitter. Safe runs require the unsafe tensors generated by `src/slurm/generate_unsafe_tensors.sh`, and scoring jobs expect the same `HF_*` caches as the generation jobs so `score_array.sh` can work offline.

### LlamaGuard hazard analysis

After `score_array.sh` finishes, extract per-hazard rates and review examples:

```bash
python -m tools.hazard_report \
  --run-dirs results/prompt_pipeline/realtoxicity_prompts/realtoxicity-safe-scale0 \
              results/prompt_pipeline/realtoxicity_prompts/realtoxicity-safe-scale1 \
              results/prompt_pipeline/realtoxicity_prompts/realtoxicity-safe-scale2 \
  --output-dir results/reports/realtoxicity_hazards \
  --examples-per-hazard 10
```

Outputs:

- `overall_rates.csv`: unsafe rate per run (plot vs. scale to sanity-check the safe knob).
- `hazard_rates.csv`: S1–S14 rates per tensor + scale (feed directly into charts).
- `hazard_examples.json`: up to *N* prompt/completion pairs per hazard category for manual auditing.
- `hazard_report.json`: structured view grouped by tensor size so you can slice by `0100/0500/1000` in downstream notebooks.

## Unsafe tensor sweeps

`sbatch src/slurm/generate_unsafe_tensors.sh` builds the reference tensors required for safe guidance. The script iterates over the artifact names defined in `configs/unsafe_prep/unsafe_prompt_sweep.yaml` (100/500/1000/all samples for each of BeaverTails, RealToxicityPrompts, and ToxiGen), calls `python -m unsafe_prep.build_unsafe_artifacts` with the appropriate overrides, and updates `${UNSAFE_OUTPUT_ROOT:-$SCRATCH/safe-text-diffusion/artifacts/unsafe_artifacts}/index.json`. Export `UNSAFE_CONFIG`, `UNSAFE_OUTPUT_ROOT`, `TOKENIZER_NAME_OR_PATH`, or `ARTIFACT_NAMES` to customise the sweep before submitting.


## Safe Denoiser (Training-Free Repellency for Text)

This repository includes an optional, training-free **safe denoiser** that steers the discrete-token posterior away from an empirical **unsafe** reference distribution during sampling. It works with all MDLM samplers (`ddpm`, `ddpm_cache`, `analytic`) and does **not** modify model weights.

### How it works

At each reverse-diffusion step \(t\), after the model predicts the clean token distribution \(p_\theta(x_0 \mid x_t)\), we replace it with a steered distribution
\[
p_{\text{safe}} \;\propto\; (1 + \beta_t)\, p_\theta \;-\; \beta_t\, q_{\text{unsafe}},
\]
then clamp to be non-negative and renormalize. Here \(q_{\text{unsafe}}\) is an **empirical** next-token distribution estimated from a small reference set of unsafe sequences via KNN/LSH retrieval in a context embedding space. The schedule \(\beta_t\) ramps from `beta_min` to `beta_max` over the final portion of the trajectory (default: last 40% of steps).

For discrete tokens, the "expectation over one-hot vectors" is simply a probability vector; the update above is a probability-space analogue of training-free repellency used in continuous diffusion.

### API notes

The implementation lives in `repellency/safe_denoiser.py` and registers as "safe_denoiser" via the existing repellency registry.

```
> **Tip:** `configs/config.yaml` is the single source of truth for default knobs. Per-experiment configs (or the planner YAMLs) should only add overlays via Hydra defaults—avoid redefining entire sections so new defaults propagate automatically.
