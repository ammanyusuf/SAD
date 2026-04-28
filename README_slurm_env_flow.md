# Slurm pipeline parameter flow

How settings in `configs/slurm/sbatch_*.yaml` become environment variables and Hydra overrides inside the job scripts.

## The path from config to sbatch

```
configs/slurm/sbatch_prompt_pipeline.yaml
  └─ run.* (model, gen.*, score.*, safety_etas, t_start/t_end)
  └─ datasets: [use: beavertails, ...]
       └─ configs/data/catalog.yaml (per-dataset prompt_variants, safety, slurm blocks)
            │
            ▼
    src/utils/experiment_setup.py
      ExperimentPlanBuilder.build()   → List[GenerationPlan]
        each plan has .overrides      → Hydra key=value list
        and .score                    → ScorePlan with score overrides
            │
            ▼
    src/slurm/submit_sbatch_experiments.py
      submits generate_array.sh with env vars baked in (--export)
      chains score_array.sh (afterok) + hazard_report.sh (afterok)
            │
            ▼
    src/slurm/generate_array.sh
      reads env vars → builds Hydra override list → python -m tools.generate key=value ...
```

For jailbreak runs, the parallel path is:
```
configs/slurm/sbatch_eval_diffguard_jailbreak_*.yaml
  └─ src/utils/jailbreak_experiment_setup.py  →  List[JailbreakPlan]
  └─ src/slurm/submit_sbatch_jailbreak.py
  └─ src/slurm/eval_diffuguard.sh  /  eval_dija.sh
```

---

## `generate_array.sh` — env var reference

All variables default to empty/disabled unless set by the submitter.

### Infrastructure

| Env var | Default | Purpose |
|---|---|---|
| `REPO_ROOT` | auto-detected | Root of the repo; used for rsync staging |
| `RESULTS_ROOT` | `$SCRATCH/results` | Output root for run dirs |
| `RUN_ID` | `$SLURM_JOB_ID` | Sub-directory under experiment slug |
| `TRACK_NAME` | `safety` | `io.track_name` Hydra override |
| `EXPERIMENT_SLUG` | from config name | `io.experiment_slug` |
| `EXTERNAL_VENV_ACTIVATE` | — | Path to pre-built venv `activate` script (skips venv creation) |
| `PIP_INSTALL_ARGS` | — | Extra args passed to `pip install` (e.g. `--no-index` for CC) |
| `SKIP_PIP_UPGRADE` | `0` | Set to `1` to skip `pip install --upgrade pip` |
| `DRY_RUN` | `0` | Set to `1` to skip the actual `python -m tools.generate` call |
| `CONFIG_BATCH_FILE` | — | Path to a file of per-job Hydra config specs (batch mode) |
| `CONFIG_BATCH_SPECS` | — | Inline JSON batch spec string |
| `GEN_CONFIG_NAME` | — | Hydra config name override (default: `config`) |

### Model

| Env var | Hydra override | Notes |
|---|---|---|
| `CHECKPOINT_PATH` | `model.checkpoint` | Staged to `$SLURM_TMPDIR/models/` before generation |
| `TOKENIZER_PATH` | `model.tokenizer_name` | Staged to `$SLURM_TMPDIR/tokenizer/` |
| `MODEL_FAMILY` | `model.family` | `llada`, `mdlm`, `dream`, etc. |
| `MODEL_NAME` | `model.model_name` | Logical model name (e.g. `llada-8b-instruct`) |
| `MODEL_VARIANT` | `model.variant` | `upstream`, `diffuguard`, `dija`, `local`, etc. |

### Generation

| Env var | Hydra override | Notes |
|---|---|---|
| `SAMPLING_STEPS` | `gen.sampling_steps` | |
| `MAX_NEW_TOKENS` | `gen.max_new_tokens` | |
| `TEMPERATURE` | `gen.temperature` | |
| `BLOCK_LENGTH` | `gen.block_length` | Semi-AR block size; leave unset for pure diffusion |
| `PROMPT_LIMIT` | `data.limit` | Truncate prompt set for testing |
| `PROMPT_VARIANT` | `data.prompt_variant` | Which prompt variant to use from the catalog |
| `PROMPT_SOURCE_NAME` | `data.prompt_source.name` | Override prompt source |
| `GEN_SEED` | `gen.seed` | Random seed |
| `ADD_BOS` | `gen.add_bos` | |
| `ADD_EOS` | `gen.add_eos` | |
| `UNCONDITIONAL_SAMPLES` | `gen.unconditional_samples` | |
| `TARGET_VRAM_PCT` | `io.target_vram_pct` | Auto-batch VRAM target fraction |
| `AUTO_BATCH_WARMUP_PROMPTS` | `io.auto_batch_warmup_prompts` | |
| `RUN_SUBDIR` | `io.run_id` suffix | Appended to run_id for sub-experiment labeling |

### Safety (safe denoiser)

| Env var | Hydra override | Notes |
|---|---|---|
| `SAFETY_ENABLED` | `safety.enabled` | `0`/`1` |
| `SAFETY_ETA` | `safety.eta` | Main guidance strength knob |
| `SAFETY_SCALE` | `safety.scale` | Alternative to eta (deprecated) |
| `SAFETY_T_START` | `safety.t_start` | First denoising step to apply guidance |
| `SAFETY_T_END` | `safety.t_end` | Last step. For LLaDA-64 steps, `18` ≈ final 28% |
| `UNSAFE_ARTIFACT_ROOT` | `safety.unsafe_artifact_root` | Root dir for pre-built reference tensors |
| `UNSAFE_ARTIFACT_NAME` | `safety.unsafe_artifact_name` | Selects a specific tensor under the root |
| `UNSAFE_ARTIFACTS` | `safety.unsafe_artifacts` | Full path override (bypasses root+name resolution) |
| `USE_SEMANTIC_GATING` | `safety.use_semantic_gating` | Enable RBF-kernel reference reweighting |
| `SAFETY_SEMANTIC_WEIGHT` | `safety.semantic_weight` | Interpolation: 0=uniform, 1=fully semantic |
| `SAFETY_SEMANTIC_TEMP` | `safety.semantic_temp` | Softmax temperature for semantic weights |
| `SAFETY_SEMANTIC_SIGMA` | `safety.semantic_sigma` | RBF kernel bandwidth |
| `UNSAFE_SEMANTIC_ROOT` | — | Root for staged semantic cache files (internal) |

### Baselines (FK Diffusion Steering / Best-of-N)

| Env var | Hydra override | Notes |
|---|---|---|
| `N_PER_PROMPT` | — | Number of samples per prompt (best-of-N backend) |
| `FK_K_PARTICLES` | — | Number of FK particles |
| `FK_ROBERTA_CHECKPOINT_PATH` | — | RoBERTa toxicity model path for FK steering |
| `LLAMAGUARD_CHECKPOINT_PATH` | `safety.classifier_model` | LlamaGuard path, forwarded to the scorer too |

### HuggingFace caching (set automatically)

| Env var | Value | Purpose |
|---|---|---|
| `HF_HOME` | `$SLURM_TMPDIR/hf_home` | Redirected to node-local storage |
| `HF_DATASETS_CACHE` | `$SLURM_TMPDIR/hf_datasets` | |
| `HF_MODELS_CACHE` | `$SLURM_TMPDIR/hf_models` | |
| `HF_HUB_OFFLINE` | `1` | Forces offline mode after staging |
| `BEAVERTAILS_DATA_DIR` | `$HF_DATASETS_CACHE/BeaverTails` | Staged dataset path |
| `TOXIGEN_DATA_DIR` | `$HF_DATASETS_CACHE/toxigen` | |
| `REALTOXICITY_DATA_DIR` | `$HF_DATASETS_CACHE/real-toxicity-prompts` | |

---

## `score_array.sh` — env var reference

| Env var | Hydra override | Notes |
|---|---|---|
| `RUN_DIR` | `score.run_dir` | Path to the generation run to score |
| `SCORE_RUN_LIST` / `SCORE_RUN_LIST_FILE` | — | Score multiple runs in one job |
| `MODEL` | `score.model` | Model tag (for output labeling) |
| `CLASSIFIER` | `score.classifier` | `llamaguard`, `toxigen`, `harmbench` |
| `CLASSIFIER_MODEL` | `score.classifier_model` | Path to classifier checkpoint |
| `SCORE_BATCH_SIZE` | `score.batch_size` | |
| `MAX_NEW_TOKENS` | `score.max_new_tokens` | Classifier max decode length |
| `FORCE` | `score.force` | Re-score even if results exist |
| `DRY_RUN` | `score.dry_run` | |
| `BASELINE_RUN_DIR` | `score.baseline_run_dir` | For computing delta vs. baseline |
| `BEHAVIORS_CSV` | `score.behaviors_csv` | HarmBench behaviors CSV |
| `INDEXES_DIR` | `score.indexes_dir` | HarmBench indexes |
| `SCORE_COMPUTE_PERPLEXITY` | `score.compute_perplexity` | `0`/`1` |
| `SCORE_PPL_MODEL_NAME` | `score.perplexity_model_name` | e.g. `gpt2-large` |
| `SCORE_PPL_MODEL_PATH_OVERWRITE` | `score.perplexity_model_path_overwrite` | Override PPL model path |
| `SCORE_PPL_BATCH_SIZE` | `score.perplexity_batch_size` | |
| `SCORE_PPL_MAX_LENGTH` | `score.perplexity_max_length` | |
| `SCORE_COMPUTE_BERTSCORE` | `score.compute_bertscore` | `0`/`1` (slow) |
| `SCORE_BERTSCORE_MODEL` | `score.bertscore_model` | |
| `SCORE_COMPUTE_MAUVE` | `score.compute_mauve` | `0`/`1` (slow) |
| `SCORE_MAUVE_MODEL_NAME` | `score.mauve_model_name` | |
| `SCORE_COMPUTE_HYGIENE_METRICS` | `score.compute_hygiene_metrics` | Refusal / degeneration / lexical |
| `SCORE_COMPUTE_LEXICAL_METRICS` | `score.compute_lexical_metrics` | n-gram overlap / distinct |
| `SCORE_COMPUTE_DEGENERATION_METRICS` | `score.compute_degeneration_metrics` | |
| `SCORE_COMPUTE_REFUSAL_METRICS` | `score.compute_refusal_metrics` | |
| `SCORE_COMPUTE_DISTRIBUTION_MMD` | `score.compute_distribution_mmd` | |
| `SCORE_SKIP_MISSING_GENERATIONS` | `score.skip_missing_generations` | |
| `SCORE_SKIP_JAILBREAK_EVALS` | `score.skip_jailbreak_evals` | |
| `SCORE_CONTINUE_ON_ERROR` | — | Continue scoring if one run fails |
| `HARM_BENCH_CLASSIFIER` | — | Staged HarmBench classifier path (set internally) |
| `STRONGREJECT_MODEL` | — | Staged StrongREJECT model path (set internally) |

---

## `hazard_report.sh` — env var reference

| Env var | Default | Purpose |
|---|---|---|
| `RUN_DIRS` | required | Comma-separated list of scored run dirs |
| `RUN_DIRS_FILE` | — | Alternative: newline-separated file of run dirs |
| `OUTPUT_DIR` | `<first_run>/../hazard_report_<ts>` | Where to write the report |
| `EXAMPLES_PER_HAZARD` | `10` | Number of example generations per hazard category |
| `EXAMPLES_PER_METRIC` | `$EXAMPLES_PER_HAZARD` | |
| `EXAMPLES_PER_TRANSITION` | `$EXAMPLES_PER_HAZARD` | |
| `HAZARD_JAILBREAK_SPLIT` | `0` | Set to `1` to split report by jailbreak method |
| `HAZARD_TIMESTAMP` | auto | Timestamp suffix on output dir |
| `TAR_OUTPUT` | `<output_dir>.tar.gz` | Archive path; set to empty to skip archiving |

---

## `eval_diffuguard.sh` / `eval_dija.sh` — env var reference

These scripts are used by the jailbreak evaluation pipeline (`submit_sbatch_jailbreak.py`).

### Model + paths (shared with generate_array.sh)

| Env var | Purpose |
|---|---|
| `MODEL_PATH` / `CHECKPOINT_PATH` | Model checkpoint |
| `TOKENIZER_PATH` | Tokenizer path |
| `MODEL_FAMILY` / `MODEL_VARIANT` / `MODEL_NAME` | Model selection |
| `ATTACK_PROMPT` / `DATASET_JSON` | Path to jailbreak attack prompts JSON |
| `OUTPUT_DIR` | Where to write eval results |
| `OUTPUT_NAME` | Output filename |
| `PROMPT_LIMIT` | Truncate dataset for testing |

### Safety (same as generate_array.sh)

`SAFETY_ENABLED`, `SAFETY_ETA`, `SAFETY_SCALE`, `SAFETY_T_START`, `SAFETY_T_END`, `UNSAFE_ARTIFACT_ROOT`, `UNSAFE_ARTIFACT_NAME`, `UNSAFE_ARTIFACTS`

### Jailbreak-specific

| Env var | Hydra override | Notes |
|---|---|---|
| `JAILBREAK_STEPS` | `jailbreak.steps` | Denoising steps for jailbreak generation |
| `JAILBREAK_GEN_LENGTH` | `jailbreak.gen_length` | Generation length |
| `JAILBREAK_BLOCK_LENGTH` | `jailbreak.block_length` | (DiffuGuard only) |
| `JAILBREAK_TEMPERATURE` | `jailbreak.temperature` | |
| `JAILBREAK_CFG_SCALE` | `jailbreak.cfg_scale` | Classifier-free guidance scale |
| `JAILBREAK_REMASKING` | `jailbreak.remasking` | `low_confidence` or `random` |
| `JAILBREAK_RANDOM_RATE` | `jailbreak.random_rate` | |
| `JAILBREAK_INJECTION_STEP` | `jailbreak.injection_step` | (DiffuGuard only) Step at which suffix is injected |
| `JAILBREAK_SP_MODE` | `jailbreak.sp_mode` | Suppression mode: `off`, `soft`, `hard` |
| `JAILBREAK_SP_THRESHOLD` | `jailbreak.sp_threshold` | Suppression threshold |
| `JAILBREAK_REFINEMENT_STEPS` | `jailbreak.refinement_steps` | |
| `JAILBREAK_REMASK_RATIO` | `jailbreak.remask_ratio` | |
| `JAILBREAK_SUPPRESSION_VALUE` | `jailbreak.suppression_value` | |
| `JAILBREAK_ATTACK_METHOD` | `jailbreak.attack_method` | `zeroshot`, `pad`, `dija` |
| `JAILBREAK_DEFENSE_METHOD` | `jailbreak.defense_method` | `diffuguard`, `repellency`, etc. |
| `JAILBREAK_MASK_ID` | `jailbreak.mask_id` | Override mask token id |
| `JAILBREAK_MASK_COUNTS` | `jailbreak.mask_counts` | |
| `JAILBREAK_FILL_ALL_MASKS` | `jailbreak.fill_all_masks` | |
| `JAILBREAK_CORRECT_ONLY_FIRST_BLOCK` | `jailbreak.correct_only_first_block` | |
| `JAILBREAK_AUTO_PICK_GPU` | `jailbreak.auto_pick_gpu` | |
| `JAILBREAK_DEBUG_PRINT` | `jailbreak.debug_print` | |
| `GEN_SEED` | `gen.seed` | |

---

## Cluster-specific switches

| Flag / var | Effect |
|---|---|
| `--trillium` (submit script) | Skips `--mem` in sbatch (Trillium enforces per-node memory) |
| `--nodes N` | Request N nodes per job |
| `--account` | Slurm allocation to charge (e.g. `rrg-<your-PI>`) |
| `--seeds 1 2 3` | Run each plan with multiple seeds; each gets its own `seed=<n>/` subdir |
| `--aggregate-only` | Skip generation/scoring; just aggregate existing seed results |
| `--dry-run` | Print sbatch commands without submitting; logs `[generate] hydra config = …` |

## `--integration-test` mode

`--integration-test` runs each `.sh` script directly as `bash <script>.sh` instead of submitting it with `sbatch`. This lets you test the full pipeline on Compute Canada without requesting a Slurm allocation — just use an interactive session or login node for small runs.

What it does differently from a normal submit:
- Strips the `sbatch` wrapper and calls `bash generate_array.sh` (or `score_array.sh`, `hazard_report.sh`) directly
- Injects stub Slurm env vars so the scripts don't fail: `SLURM_JOB_ID=local`, `SLURM_ARRAY_TASK_ID=0`, etc.
- Creates `.slurm_tmp_local/` in the repo root as a stand-in for `$SLURM_TMPDIR`
- Runs jobs sequentially (no job chaining — score runs immediately after generate finishes)
- Prints `[integration-test] with env vars: …` before each run so you can see exactly what the script receives

Usage:
```bash
# Test the full safety eval pipeline on a single dataset without submitting a job
python src/slurm/submit_sbatch_experiments.py \
  --config configs/slurm/sbatch_submit_test.yaml \
  --repo-root "$REPO_ROOT" \
  --integration-test

# Same for jailbreak
python src/slurm/submit_sbatch_jailbreak.py \
  --config configs/slurm/sbatch_eval_diffguard_jailbreak_llada_instruct.yaml \
  --repo-root "$REPO_ROOT" \
  --integration-test
```

The jailbreak submitter also prompts for confirmation before each job (`Proceed with integration-test run? [y/N]`) — pass responses interactively or pipe `y` for automation.

> **Note**: `--integration-test` is not a substitute for a full cluster run — it uses the same staging logic (rsync to `$SLURM_TMPDIR`, pip install, etc.) so it can be slow on a login node with a large model. Use `configs/slurm/sbatch_submit_test.yaml` (short prompt limit, 1 dataset) to keep it fast.

## Debugging

If a Hydra override is missing from the generated output, either `ExperimentPlanBuilder` didn't put it in `plan.overrides`, or `generate_array.sh` doesn't forward it. Check what `generate_array.sh` echoes before the `python -m tools.generate` line.

Dry-run mode prints the full Hydra override list for each job:
```bash
python src/slurm/submit_sbatch_experiments.py \
  --config configs/slurm/sbatch_prompt_pipeline.yaml \
  --repo-root "$REPO_ROOT" --dry-run
```
