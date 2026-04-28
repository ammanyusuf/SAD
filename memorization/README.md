# Memorization Experiments

Replicates Table 1 of "Characterizing Memorization in Diffusion Language Models"
(Luo et al., arXiv:2603.02333): PII extraction on Enron emails, DLM vs ARM baseline.

---

## Quick Reference

### 0. Set up assets (login node — has internet)

Before going to a compute node, run this once to download models and datasets:

```bash
source scripts/env_profile.sh dlm-memorization
python memorization/setup_assets.py
```

To just check what's present without downloading:

```bash
python memorization/setup_assets.py --check-only
```

### 1. Get an interactive GPU node

```bash
salloc --account=<your-account> --gres=gpu:1 --mem=32G --time=1:00:00 --ntasks=1 --cpus-per-task=4
```

### 2. Set up environment

```bash
cd $HOME/repos/safe-text-diffusion
source scripts/env_profile.sh dlm-memorization
```

Verify the key variables are set:

```bash
echo $CHECKPOINT_PATH        # ~/scratch/models/text-diffusion/702-1250000.ckpt
echo $TOKENIZER_PATH         # ~/scratch/hf_models/gpt2-large
echo $MODEL_CONFIG_PATH      # .../src/third_party/mdlm/configs/config.yaml
echo $HF_MODELS_CACHE        # ~/scratch/hf_models
echo $MEMORIZATION_DATA_DIR  # ~/scratch/data/memorization
echo $MEMORIZATION_RESULTS_DIR
```

### 3. Debug run — GPT-2 Large (fast, verify pipeline)

Always test the ARM baseline first: no checkpoint loading, fast, catches data/pipeline issues.

```bash
python experiments/replicate_table1.py \
  --config memorization/configs/table1.yaml \
  --debug \
  --model-name gpt2-large \
  --pii-type email
```

Debug mode uses 10 examples and R=32 (instead of 3000 examples and R=512).

### 4. Debug run — MDLM Wikipedia checkpoint

```bash
python experiments/replicate_table1.py \
  --config memorization/configs/table1.yaml \
  --debug \
  --model-name mdlm-wiki \
  --pii-type email
```

### 5. Full run — all models, all PII types

```bash
python experiments/replicate_table1.py \
  --config memorization/configs/table1.yaml
```

Results are written to `$MEMORIZATION_RESULTS_DIR/table1/`:
- `<model>_<pii_type>_raw.json` — per-example p_hat_z and metrics
- `table1_summary.csv` — aggregated Table 1 numbers

---

## Models

| Config name   | Family | Checkpoint                                          |
|---------------|--------|-----------------------------------------------------|
| `mdlm-wiki`   | mdlm   | `~/scratch/models/text-diffusion/702-1250000.ckpt`  |
| `gpt2-large`  | arm    | `~/scratch/hf_models/gpt2-large`                    |

To add LLaDA-8B-Base, uncomment the `llada-8b-base` entry in `memorization/configs/table1.yaml`
and run `source scripts/env_profile.sh llada` first.

---

## Key Hyperparameters (`memorization/configs/table1.yaml`)

| Parameter          | Default     | Description                                      |
|--------------------|-------------|--------------------------------------------------|
| `extraction.R`     | 512         | MC sampling trials per example                   |
| `extraction.N_values` | [1,2,5,10] | Denoising steps to sweep (|M| added at runtime) |
| `extraction.alg`   | `"origin"`  | Unmasking algorithm: `origin` or `greddy`        |
| `extraction.temperature` | 1.0   | Gumbel temperature                               |
| `extraction.trial_batch_size` | 16 | Parallel trials per forward pass (reduce if OOM) |
| `data.n_samples`   | 3000        | Examples per PII type                            |
| `data.prefix_max_tokens` | 100   | Context window length                            |

---

## Troubleshooting

**PYTHONPATH error on import**
```bash
export PYTHONPATH=$(pwd):$(pwd)/src:$(pwd)/src/third_party/mdlm:$PYTHONPATH
```

**OOM on GPU**
Reduce `trial_batch_size` in `table1.yaml` (try 4 or 8), or pass a smaller `--R` override.

**Enron data not found**
Both datasets must be downloaded on a login node (internet access) before going offline.
Use the setup script (step 0 above) — it handles both automatically.

Two sources are used:
- `corbt/enron-emails` (~517k emails, real PII) — required for Table 1
- `Yale-LILY/aeslc` (~18k emails, no PII) — for fast pipeline testing only

**MDLM checkpoint not loading**
- Confirm `$CHECKPOINT_PATH` points to `702-1250000.ckpt`
- Confirm `$MODEL_CONFIG_PATH` points to the MDLM Hydra config yaml
- The tokenizer must be `gpt2-large` (set via `$TOKENIZER_PATH`)

---

## Experiment 2: Verbatim Wikipedia Memorization

Measures how often MDLM (trained on WikiText-103) can reconstruct verbatim
training sequences given a prefix — no PII required.

The "secret" is a plain suffix of `suffix_tokens` tokens that follows a
`prefix_tokens`-token context in the training data.  Same p_hat_z estimator,
different dataset.

### Setup (login node)

```bash
source scripts/env_profile.sh dlm-memorization
python memorization/setup_assets.py   # also caches wikitext-103
```

### Debug run

```bash
python experiments/measure_wiki_memorization.py \
  --config memorization/configs/wiki_memorization.yaml \
  --debug \
  --model-name mdlm-wiki
```

### Full run

```bash
python experiments/measure_wiki_memorization.py \
  --config memorization/configs/wiki_memorization.yaml
```

### High-frequency sequences (more likely memorized)

```bash
python experiments/measure_wiki_memorization.py \
  --config memorization/configs/wiki_memorization.yaml \
  --strategy high_freq
```

Results in `$MEMORIZATION_RESULTS_DIR/wiki_memorization/`:
- `<model>_<strategy>_raw.json`
- `wiki_memorization_summary.csv`

### Interpreting results

| Observation | Meaning |
|---|---|
| mdlm-wiki p=50% memorized >> 0% | Model has verbatim memorized some training text |
| mdlm-wiki ≈ gpt2-large | Both models memorize at similar rates |
| high_freq > random | Repetition in training drives memorization (expected) |
| Larger `suffix_tokens` → lower p_hat_z | Longer suffixes are harder to reconstruct exactly |

**Expected for the Wikipedia-pretrained MDLM** (before any fine-tuning on Enron):
- Some verbatim memorization of frequently-repeated Wikipedia phrases/sentences
- Lower memorization than a model trained for many more epochs

### Key hyperparameters (`wiki_memorization.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `data.prefix_tokens` | 100 | Context window length |
| `data.suffix_tokens` | 50 | Tokens to reconstruct |
| `data.strategy` | `random` | `random` or `high_freq` |
| `data.min_freq` | 2 | Min repetitions for `high_freq` |
| `extraction.R` | 512 | MC trials |
| `extraction.epsilon_values` | [0,1,2] | Hamming relaxation |

---

## Running Tests

```bash
python -m pytest memorization/tests/test_pz_estimator.py -v
```

All 16 unit tests cover: mask sampling, extraction metrics, DLM p_z estimator
(uniform + perfect toy models), ARM estimator, and relaxed (epsilon) estimator.
