# Unsafe Prep Guide

The `unsafe_prep` package builds “unsafe-answer” tensor artifacts that plug straight into MDLM’s repellency samplers. Each artifact is a directory that contains fixed-size shards (`shard-00000.pt`, …) plus statistics files and an `index.json` manifest.

Canonical dataset metadata (splits, fields, configurable filters, and typed record models) lives in `unsafe_prep/schemas.py` and `unsafe_prep/constants.py`. Every adapter instantiates those dataclasses, so downstream code references well-defined attributes instead of raw dict keys.

```
artifacts/unsafe_artifacts/
  beavertails-violence-1500/
    shard-00000.pt
    shard-00000.stats.json
    …
  real-toxicity-prompts-all/
  toxigen-train-alice-0.8/
  index.json
```

## Dataset Schemas

| Source | Key fields we consume | Unsafe selection knobs |
|--------|-----------------------|-------------------------|
| **BeaverTails**<br>`prompt`, `response`, `category`, `is_safe` | `response` is the model answer; `is_safe=false` marks unsafe rows.<br>`category` is a dict of 14 boolean flags, one for each risk bucket:<br>· animal_abuse<br>· child_abuse<br>· controversial_topics,politics<br>· discrimination,stereotype,injustice<br>· drug_abuse,weapons,banned_substance<br>· financial_crime,property_crime,theft<br>· hate_speech,offensive_language<br>· misinformation_regarding_ethics,laws_and_safety<br>· non_violent_unethical_behavior<br>· privacy_violation<br>· self_harm<br>· sexually_explicit,adult_content<br>· terrorism,organized_crime<br>· violence,aiding_and_abetting,incitement.<br>Splits: 330k_train/test (≈301k/33k rows), 30k_train/test (≈27k/3k). | Optional `categories` list keeps any record whose flagged categories intersect the list. Leave empty to include all unsafe answers. |
| **RealToxicityPrompts**<br>`filename`, `begin`, `end`, `challenging`, `prompt`, `continuation` | Each `prompt`/`continuation` dict includes text plus Perspective scores (floats in [0,1]):<br>· toxicity<br>· severe_toxicity<br>· threat<br>· identity_attack<br>· insult<br>· profanity<br>· sexually_explicit<br>· flirtation.<br>Additional columns: `filename`, character offsets (`begin`, `end`), `challenging` flag.<br>Split: train (≈99.4k rows). | `toxicity_threshold` compares `continuation.toxicity`. Optional `toxicity_filters` lets you set floors using any of the metrics above (e.g., `threat ≥ 0.2`). |
| **Toxigen**<br>configs: `train`, `annotated`, `annotations` | **train** (`split=train`): `prompt`, `generation`, `generation_method ∈ {ALICE, TopK}`, `group ∈ {asian, black, chinese, jewish, latino, lgbtq, mental_dis, middle_east, mexican, muslim, native_american, physical_dis, women}`, `prompt_label ∈ {0,1}`, `roberta_prediction ∈ [0,1]`.<br>**annotated** (`split=train`, `test`): `text`, `target_group` (same 13 groups), categorical tags (`factual?`, `ingroup_effect`, `lewd`, `framing`, `stereotyping`), numeric scores `intent`, `toxicity_ai`, `toxicity_human` ∈ [1,5], `predicted_author ∈ {ai, human}`, `actual_method ∈ {topk, cbs}`.<br>**annotations** exposes per-annotator rows. | `config_name` selects the subset. `roberta_threshold`, `toxicity_ai_threshold`, `toxicity_human_threshold` filter scores; `generation_methods` (`["alice"]`, `["topk"]`, …) and `groups` restrict methods/targets. |

Each shard contains a torch dictionary:

```
{
  "input_ids": LongTensor [num_records, max_length],
  "lengths":   LongTensor [num_records],
  "meta":      List[Dict[str, Any]]  # source, category, toxicity scores, provenance
}
```

## Sample Configuration (`configs/unsafe_prep/unsafe_prep.yaml`)

```yaml
tokenizer_name_or_path: /scratch/${USER}/hf_models/gpt2-large
max_length: 1024
shard_size: 1024
seed: 17
output_dir: artifacts/unsafe_artifacts

datasets:
  # BeaverTails: sample 1.5k unsafe responses. Categories optional; list here for reference.
  - source: beavertails
    split: 330k_train               # options: 330k_train/test, 30k_train/test
    sample_size: 1500
    # categories:
    #   ["animal_abuse", "child_abuse", "controversial_topics,politics",
    #    "discrimination,stereotype,injustice", "drug_abuse,weapons,banned_substance",
    #    "financial_crime,property_crime,theft", "hate_speech,offensive_language",
    #    "misinformation_regarding_ethics,laws_and_safety",
    #    "non_violent_unethical_behavior", "privacy_violation", "self_harm",
    #    "sexually_explicit,adult_content", "terrorism,organized_crime",
    #    "violence,aiding_and_abetting,incitement"]
    output_name: beavertails-1500

  # RealToxicityPrompts: optional field-specific filters use Perspective scores.
  - source: real-toxicity-prompts
    split: train
    toxicity_threshold: 0.5
    # toxicity_filters:
    #   threat: 0.2
    #   identity_attack: 0.2

  # Toxigen: highlight configurable knobs (config, roberta_threshold, methods, groups).
  - source: toxigen
    config_name: train              # choices: train, annotated, annotations
    split: train
    sample_size: 5000
    roberta_threshold: 0.8
    generation_methods: ["alice"]   # optional; case-insensitive
    groups: []                      # optional target groups
    # toxicity_ai_threshold: 3.0    # annotated config only
    # toxicity_human_threshold: 3.0
```

**Granularity tips**

- Beavertails: combine multiple selections if you want separate artifacts (e.g., one per category cluster). The builder names the artifact `source-[categories]-[count|all]` unless `output_name` is set.
- RealToxicityPrompts: any Perspective field in `continuation` can go into `toxicity_filters`. The adapter copies the filters into metadata for traceability.
- Toxigen: use `config_name=train` for scale (roberta-based filtering), `config_name=annotated` for high-precision human labels. You can run two entries—one with annotated seeds, another bulked-up train set—and point each at its own `output_name`.

## CLI Run Examples

```bash
python -m unsafe_prep.build_unsafe_artifacts \
  --config configs/unsafe_prep/unsafe_prep.yaml \
  --out /scratch/$USER/unsafe_artifacts \
  --force \
  --set datasets[0].split=330k_test \
  --set datasets[1].toxicity_filters.threat=0.25 \
  --set datasets[2].roberta_threshold=0.85
```

Use `--set KEY=VALUE` overrides to tweak any field without editing the YAML. Accepted values are parsed via `ast.literal_eval`, so lists can be written as `--set datasets[2].groups='["black","lgbtq"]'`.

## Offline Usage

- Download the datasets ahead of time into `$HF_DATASETS_CACHE` (e.g., `/scratch/$USER/hf_datasets/<dataset>`). The pipeline first checks `SAFE_TEXT_DIFFUSION_DATASETS`, `UNSAFE_ARTIFACTS_DATA_ROOT`, `UNSAFE_PREP_DATA_ROOT`, then `HF_DATASETS_CACHE` for a matching folder name.
- Point `tokenizer_name_or_path` at a local tokenizer directory to avoid hub lookups.
- Override splits explicitly for local mirrors (e.g., BeaverTails’ `evaluation` alias does not exist offline; use `330k_train` or similar).

## Semantic cache for repellency

Cache semantic embeddings for unsafe refs offline so sampling doesn’t re-encode them:

```bash
python -m unsafe_prep.build_semantic_ref_cache \
  --artifact-root artifacts/unsafe_artifacts \
  --artifact-name beavertails-1500 \
  --provider mdlm \
  --mdlm-fn third_party.mdlm.encoders:embed_last_hidden \  # uses diffusion encoder callable
  --encoder bert-base-uncased \  # fallback if mdlm callable import fails
  --output artifacts/unsafe_artifacts/semantic_ref_embeddings.pt \
  --batch-size 64 --device cuda
```

Providers:
- `hf` (default): use `--encoder` or `SEMANTIC_REF_ENCODER`.
- `mdlm`: import callable via `--mdlm-fn` or `MDLM_EMBED_FN`; falls back to `--encoder` if import fails.
- `callable`: use `--embed-fn` or `UNSAFE_PREP_EMBED_FN` (module:function).

Then point `MaskKernelRepellency` at that file with `cache_semantic_ref=True` and `semantic_ref_path=/path/to/semantic_ref_embeddings.pt`.

Pass the embedding source explicitly via CLI flags (e.g., `--encoder` for HF or `--embed-fn module:function` for a callable). Environment-variable fallbacks are no longer used.
