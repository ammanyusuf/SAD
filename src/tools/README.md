# src/tools — CLI entry points

All tools are run as `python -m tools.<name>` from the repo root with `PYTHONPATH` set.
They are Hydra-based: overrides are passed as `key=value` on the command line.

---

## `generate.py` — text generation

The main generation entry point. Loads a model, slices prompts, auto-sizes batches, runs the sampling loop (optionally with the safe denoiser), and writes JSONL completions.

```bash
python -m tools.generate \
  model.checkpoint=$CHECKPOINT_PATH \
  model.tokenizer_name=$TOKENIZER_PATH \
  io.experiment_slug=my_run \
  gen.max_new_tokens=256 \
  safety.enabled=true safety.eta=1.0 safety.t_start=0 safety.t_end=18
```

Key config sections (see `configs/config.yaml`):
- `model`: `family` (llada/mdlm/dream), `variant`, `checkpoint`, `tokenizer_name`, `precision`
- `gen`: `max_new_tokens`, `sampling_steps`, `batch_size`, `temperature`, `block_length`
- `safety`: `enabled`, `eta`, `t_start`, `t_end`, `unsafe_artifact_name`, `use_semantic_gating`
- `io`: `experiment_slug`, `run_id`, `output_dir`, `auto_batch`
- `data`: `prompt_source`, `prompt_limit`, `dataset_json`

Dry-run (validates config without generating):
```bash
python -m tools.generate --config-name experiments/beavertails_prompts gen.dry_run=true
```

---

## `score.py` — safety scoring

Runs safety classifiers over the JSONL completions produced by `generate.py`. Supports LlamaGuard, ToxiGen, and HarmBench ASR. Optionally computes perplexity, BERTScore, and degeneration metrics.

```bash
python -m tools.score \
  score.run_dir=/path/to/run \
  score.classifier=llamaguard \
  score.dry_run=true   # remove to score for real
```

Key options:
- `score.classifier`: `llamaguard`, `toxigen`, `harmbench`
- `score.compute_perplexity`: true/false (uses GPT-2-large by default)
- `score.force`: re-score even if results already exist

---

## `aggregate_scores.py` — result aggregation

Aggregates per-shard score files (produced by `score.py`) into a single CSV and summary JSON for a run directory. Used internally by the Slurm pipeline after scoring finishes.

```bash
python -m tools.aggregate_scores --run-dir /path/to/run
```

---

## `hazard_report.py` — hazard breakdown

Generates a per-category hazard report from scored completions. Outputs per-hazard unsafe rates, example generations, and an `overall_rates.csv` used by the plotting scripts.

```bash
python -m tools.hazard_report \
  io.run_dirs=[/path/to/run1,/path/to/run2] \
  io.output_dir=/path/to/reports
```

The `overall_rates.csv` output is the input to `scripts/plot_ablation_tradeoff.py` and `scripts/plot_ablations.py`.

---

## `eval_diffuguard.py` — DiffuGuard jailbreak evaluation

Runs jailbreak attacks (zero-shot, PAD, DIJA) through the DiffuGuard harness. Supports optional safe denoiser guidance during the evaluation loop.

```bash
python -m tools.eval_diffuguard \
  model.checkpoint=/path/to/llada \
  jailbreak.attack_prompt=/path/to/jbb_behaviors_harmful_diffuguard.json \
  io.output_dir=/path/to/results \
  safety.enabled=true safety.eta=1.5 safety.t_start=0 safety.t_end=18
```

See `README_jailbreak.md` for the full evaluation setup.

---

## `eval_dija.py` — DIJA jailbreak evaluation

Runs the DIJA jailbreak attack harness. Functionally parallel to `eval_diffuguard.py` but using DIJA-formatted attack prompts and the DIJA adversarial suffix injection pipeline.

```bash
python -m tools.eval_dija \
  model.checkpoint=/path/to/llada \
  jailbreak.attack_prompt=/path/to/jbb_behaviors_harmful_dija.json \
  io.output_dir=/path/to/results \
  safety.enabled=true safety.eta=1.5
```

---

## `index_corpus.py` — corpus indexing (not implemented)

Placeholder for a corpus indexing tool intended for the memorization experiments. Not used in the main safety evaluation pipeline.
