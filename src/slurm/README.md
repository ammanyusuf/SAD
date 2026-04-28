# Slurm Job Templates

These helpers maximise GPU utilisation on Compute Canada by staging all inputs to `$SLURM_TMPDIR`, autotuning the batch size, and writing structured telemetry (seq/s, peak VRAM, GPU utilisation, `(B,L,T)`, commit hash) alongside the generations.

## Shared requirements

Export the following before calling `sbatch`:

- `REPO_ROOT` – repository checkout containing `src/`.
- `CHECKPOINT_PATH` – absolute path to the `.ckpt`.
- `TOKENIZER_NAME` – tokenizer directory.
- `DATASET_JSON` – prompt file to slice.
- `EXPERIMENT_SLUG` – output namespace.

Optional knobs: `MODEL_NAME`, `TRACK_NAME`, `RUN_ID`, `MAX_NEW_TOKENS`, `PREFIX_LENGTH`, `SAMPLING_STEPS`, `SEED`, `SAFETY_*`, `UNSAFE_ARTIFACT_*`, `PROMPT_LIMIT`, `UNCONDITIONAL_SAMPLES`, `TARGET_VRAM_PCT` (default 0.9), `PRECISION` (default `bf16`).

Both scripts:

1. `rsync` the repo, checkpoint, tokenizer, and dataset into `$SLURM_TMPDIR/{repo,models,tokenizer,data}`.
2. Create a fresh venv under `$SLURM_TMPDIR/venv`, install the pre-built Compute Canada stack via `src/requirements-cc.txt`.
3. Point HF caches at `$SLURM_TMPDIR`, enable TF32/bf16, disable tokenizer multithreading.
4. Count prompts via `tools.generate._load_dataset`, then slice contiguously (no modulo).
5. On each visible GPU, run a warmup loop that grows `batch_size` until ~`TARGET_VRAM_PCT` VRAM is used (or the first OOM), then lock that `batch_size` for the rest of the run.
6. Write results to `$SLURM_TMPDIR/outputs/...`, collect `run_metadata.json` per shard plus a top-level summary (`job_run_metadata.json` or `task_metadata.json`).
7. Stage outputs and logs back to `$RESULTS_ROOT/${EXPERIMENT_SLUG}/${RUN_ID}/...` even on failure (`RUN_ID` defaults to `SLURM_JOB_ID`, but you can override it to match Hydra's `io.run_id`).

### Environment variables

| Variable | Purpose |
| --- | --- |
| `REPO_ROOT` | Repository checkout that holds the `src/` tree. |
| `CHECKPOINT_PATH` | Absolute path to the checkpoint to sample. |
| `TOKENIZER_NAME` | Directory containing tokenizer files. |
| `DATASET_JSON` | Prompt file that will be sliced contiguously (optional when your Hydra config uses `data.prompt_source`). |
| `EXPERIMENT_SLUG` | Namespace inside `$SCRATCH/results/` for this run. |
| `RUN_ID` | Folder name under `${RESULTS_ROOT}/${EXPERIMENT_SLUG}`. Defaults to `${SLURM_JOB_ID}` so each job writes to its own subdirectory. |
| `MODEL_NAME` | Label recorded in metadata (`mdlm-0p5b` default). |
| `TRACK_NAME` | `safety` or `memorization` (defaults to `safety`). |
| `TARGET_VRAM_PCT` | Auto-batch utilization target (default `0.9`). |
| `AUTO_BATCH_WARMUP_PROMPTS` | Samples used during the warmup probe. |
| `RESULTS_ROOT` | Destination for staged outputs (defaults to `$SCRATCH/results`). |
| `HF_HOME`, `HF_DATASETS_CACHE`, `HF_MODELS_CACHE`/`TRANSFORMERS_CACHE` | Local Hugging Face caches (tokenizers, datasets, models). Provide `HF_MODELS_CACHE`; if missing, the scripts accept `TRANSFORMERS_CACHE`. All three caches are copied into `$SLURM_TMPDIR` before sampling. |
| `SAFETY_ENABLED`, `SAFETY_ETA`, `SAFETY_SCALE`, `UNSAFE_ARTIFACT_ROOT`, `UNSAFE_ARTIFACT_NAME`, `UNSAFE_ARTIFACTS` | Optional knobs for the safe denoiser and artifact selection (`SAFETY_SCALE` is deprecated; use `SAFETY_ETA`). |
| `SKIP_PIP_UPGRADE` | Set to `1` to skip `pip install --upgrade pip` (useful on offline nodes). |
| `PIP_INSTALL_ARGS` | Extra flags forwarded to `pip install -r src/requirements-cc.txt` (e.g., `--no-index --find-links file:///cvmfs/soft.computecanada.ca/custom/python/wheelhouse/avx2`). |
| `EXTERNAL_VENV_ACTIVATE` | Absolute path to an existing `activate` script. If set and found, the scripts source it instead of creating a fresh Compute Canada venv. |

Keep a small shell snippet (e.g., `.experiment_env`) that exports the values above and `source` it before launching jobs. All defaults now live under `configs/config.yaml` and are consumed through Hydra, so overrides take the form `section.key=value`:

```bash
python -m tools.generate model.model_name=mdlm-0p5b data.dataset_json=/path/to/test_cases.json
```

### `generate_single.sh`

Single-node launcher that saturates however many GPUs you request (default 1). Each GPU receives a contiguous prompt slice, stages outputs under `$RESULTS_ROOT/${EXPERIMENT_SLUG}/${RUN_ID}/gpu_<r>/…`, and emits telemetry in `job_run_metadata.json`.

Internally the script shells out to `python -m tools.generate` with Hydra overrides (e.g., `model.checkpoint=...`, `io.output_dir=...`), so you can mirror the same syntax for ad-hoc runs.

Example:

```bash
export REPO_ROOT=/home/$USER/repos/safe-text-diffusion
export CHECKPOINT_PATH=/home/$USER/scratch/models/text-diffusion/mdlm.ckpt
export TOKENIZER_NAME=/home/$USER/scratch/hf_models/gpt2-large
export DATASET_JSON=/home/$USER/scratch/harmbench/.../test_cases.json
export GEN_CONFIG_NAME=experiments/beavertails_prompts
export EXPERIMENT_SLUG=harmbench_safe_autobatch
export SAFETY_ENABLED=1
export UNSAFE_ARTIFACT_ROOT=/scratch/$USER/safe-text-diffusion/artifacts/unsafe_artifacts

sbatch --gpus-per-node=4 --cpus-per-task=1 --mem=8G src/slurm/generate_single.sh
```

Set `GEN_CONFIG_NAME` to the Hydra config you want to launch (e.g., `experiments/beavertails_prompts`) before submission; the script does not pick a default on your behalf.

### `generate_array.sh`

Array-aware launcher for large sweeps. Set `--array=0-(S-1)`; each task receives its contiguous outer slice, then the script further subdivides that range across the GPUs on the node. Outputs land under `$RESULTS_ROOT/${EXPERIMENT_SLUG}/${RUN_ID}/task_<array_id>/gpu_<r>/…`. Pass the same env vars as above—no positional arguments are needed—so the staged results align with Hydra's `io.run_id`.

Each GPU worker receives its own Hydra override bundle, so you can replicate the launch locally with the same `python -m tools.generate key=value …` pattern.

Example (20 shards × 2 GPUs per task):

```bash
export REPO_ROOT=...
export CHECKPOINT_PATH=...
export TOKENIZER_NAME=...
export DATASET_JSON=...
export EXPERIMENT_SLUG=harmbench_full
export GEN_CONFIG_NAME=experiments/beavertails_prompts
sbatch --array=0-19 --gpus-per-node=2 --cpus-per-task=1 --mem=8G src/slurm/generate_array.sh
```

Like the single-job script, you must set `GEN_CONFIG_NAME` to the Hydra config path you want (e.g., `experiments/beavertails_prompts`) so all workers inherit the correct defaults.

### `generate_template.sh`

Annotated template showing the full staging workflow inside a single script (env validation, venv creation, requirements-cc installation, contiguous slicing, telemetry aggregation, and staging back to `$SCRATCH`). Use it as a reference when crafting custom launchers.

### `generate_unsafe_tensors.sh`

Builds a grid of unsafe reference tensors (BeaverTails, RealToxicityPrompts, ToxiGen) at the sample sizes described in `configs/unsafe_prep/unsafe_prompt_sweep.yaml`. The script iterates over the desired artifact names (default: 100/500/1000/all for each dataset), launches `python -m unsafe_prep.build_unsafe_artifacts` with the appropriate overrides, and updates the shared `index.json` under `${UNSAFE_OUTPUT_ROOT}` (defaults to `$SCRATCH/safe-text-diffusion/artifacts/unsafe_artifacts`). It reuses the same staging logic as the generation scripts (HF caches copied into `$SLURM_TMPDIR`, offline flags set) so the build remains deterministic. Export `UNSAFE_CONFIG`, `UNSAFE_OUTPUT_ROOT`, `TOKENIZER_NAME_OR_PATH`, or `ARTIFACT_NAMES` to customise the sweep, then submit via `sbatch src/slurm/generate_unsafe_tensors.sh`.

### `submit_sbatch_unsafe_prep.py`

Splits the unsafe sweep (e.g., `configs/unsafe_prep/unsafe_prompt_sweep.yaml`) across a requested number of jobs and submits one `generate_unsafe_tensors.sh` per chunk. When `semantic_cache.enabled` is true in the config (see `configs/slurm/unsafe_prep_submit_test.yaml`), the script chains a `generate_semantic_cache.sh` submission with an `afterok` dependency on the tensor job. Dry-run/integration example:

```bash
python src/slurm/submit_sbatch_unsafe_prep.py \
  --config configs/slurm/unsafe_prep_submit_test.yaml \
  --repo-root "$REPO_ROOT" \
  --integration-test
```

### `generate_semantic_cache.sh`

Builds and caches semantic embeddings for unsafe references so sampling can load `semantic_ref_embeddings.pt` without re-encoding. It stages the repo and HF caches into `$SLURM_TMPDIR`, builds a venv (or reuses `EXTERNAL_VENV_ACTIVATE`), and runs:

```bash
python -m unsafe_prep.build_semantic_ref_cache \
  --artifact-root "${ARTIFACT_ROOT}" \
  --artifact-name "${ARTIFACT_NAME}" \
  --provider "${PROVIDER}" \
  --encoder "${ENCODER_NAME}" \
  --output "${OUTPUT_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  [--mdlm-fn "${MDLM_FN}"] [--embed-fn "${EMBED_FN}"]
```

Defaults: `ARTIFACT_ROOT=$SCRATCH/safe-text-diffusion/artifacts/unsafe_artifacts`, `ARTIFACT_NAMES="beavertails-1500"`, `PROVIDER=mdlm`, `MDLM_FN=third_party.mdlm.encoders:embed_from_checkpoint`, `ENCODER_NAME` unset (no fallback), `OUTPUT_PATH` unset (auto `${ARTIFACT_ROOT}/semantic_ref_embeddings_<artifact>.pt`), `BATCH_SIZE=64`, `DEVICE=cuda`. Submit with:

```bash
sbatch src/slurm/generate_semantic_cache.sh
```

Example with multiple artifacts:

```bash
export ARTIFACT_ROOT=${SCRATCH}/safe-text-diffusion/artifacts/unsafe_artifacts
export ARTIFACT_NAMES="beavertails-0100 real-toxicity-prompts-prompt-and-continuations-1000 toxigen-0500"
export PROVIDER=mdlm
export MDLM_FN=third_party.mdlm.encoders:embed_from_checkpoint
# Optionally set CHECKPOINT_PATH and MDLM_EMBED_ATTR to control which embedding layer is used.
sbatch src/slurm/generate_semantic_cache.sh
```

### `score_array.sh`

Scores generation directories using the same Hydra config while mirroring the bootstrap pipeline from `generate_single.sh`. The script stages the repo into `$SLURM_TMPDIR`, syncs the Hugging Face caches, creates (or reuses via `EXTERNAL_VENV_ACTIVATE`) a virtualenv, and logs all activity under `<run_dir>/scores/logs/job_<job>/<array>`. Provide `RUN_DIR`, `TRACK`, `MODEL`, and optional classifier assets via environment variables; it translates them into overrides such as `score.track=safety score.run_dir=/path score.classifier=llamaguard`. Remember to export the same `HF_HOME` / `HF_DATASETS_CACHE` / `HF_MODELS_CACHE` directories before submission so scoring runs offline just like generation.

### `submit_sbatch_experiments.py`

Python launcher that ties everything together. It calls `utils.experiment_setup.build_generation_plans` to expand a prompt-sweep config, then submits `generate_array.sh` (plus `score_array.sh` dependencies) with the appropriate environment variables. Example:

```bash
python -m slurm.submit_sbatch_experiments \
  --config configs/slurm/prompt_pipeline.example.yaml \
  --repo-root /home/$USER/repos/safe-text-diffusion \
  --safe-artifact-root /scratch/$USER/safe-text-diffusion/artifacts/unsafe_artifacts
```

Use `--only`, `--baseline-array`, `--safe-array`, `--score-array`, and their corresponding `--*-time` knobs to control shard counts per variant. Pair it with `configs/slurm/sbatch_prompt_pipeline.yaml` for a ready-to-run config that points at the mirrored BeaverTails/RealToxicity/ToxiGen prompt dumps under `${SCRATCH}/hf_datasets`.

### `index_corpus_array.sh`

Stub that writes placeholder manifests for future memorisation indexes. Variables: `CORPUS`, `TOKENIZER`, `OUTDIR` (default `$SCRATCH/safe-text-diffusion/results/indexes`).

## Typical workflow

1. Build unsafe reference tensors once per tokenizer (`python -m unsafe_prep.build_unsafe_artifacts ...`).
2. Launch either `generate_single.sh` (for dense multi-GPU nodes) or `generate_array.sh` (for sweeps).
3. Score via `src/slurm/score_array.sh`, aggregate with `python -m tools.aggregate_scores`, then build reports under `reports/`.
