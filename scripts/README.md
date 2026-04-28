# Scripts overview

Utility scripts for quick experiments and debugging. Source `env_profile.sh` first to set caches/checkpoints and PYTHONPATH.

## Environment profiles
- `env_profile.sh`: source with `mdlm` or `llada` (optionally `--debug` for SAFE_* flags). Prints modules requested, venv activated, and exported env vars.
  - Examples:
    - `source scripts/env_profile.sh mdlm`
    - `source scripts/env_profile.sh llada --debug`
  - [Compute Canada] This script sets `PYTHONPATH`, activates the venv, and exports model/artifact paths. Edit the path variables at the top to match your `$SCRATCH` layout.

## Jailbreak asset preparation

Before running jailbreak evaluations, convert the benchmark datasets to DiffuGuard/DIJA format:

```bash
# Convert JailbreakBench to DiffuGuard and DIJA formats
python scripts/convert_jbb_to_diffuguard_json.py --out $JAILBREAK_DATA_ROOT
python scripts/convert_jbb_to_dija_json.py --out $JAILBREAK_DATA_ROOT

# Download / convert remaining benchmarks (AdvBench, HarmBench, StrongREJECT, WildJailbreak)
python scripts/setup_jailbreak_assets.py --out $JAILBREAK_DATA_ROOT
```

## Debug / experiment scripts
- `debug_llada_generation.py`: quick LLaDA generation sanity check.
  - Example: `python scripts/debug_llada_generation.py --checkpoint-path $CHECKPOINT_PATH --model-name llada-8b-base --batch-size 1 --max-new-tokens 128`
- `debug_llada_parity.py`: compares repo LLaDA against official implementation on sample prompts.
  - Example: `python scripts/debug_llada_parity.py --checkpoint $CHECKPOINT_PATH --unsafe-artifacts /path/to/artifacts --eta 1.0`
- `debug_dream_diffuguard_generation.py`: Dream + DiffuGuard generation sanity check.
  - Example: `python scripts/debug_dream_diffuguard_generation.py --checkpoint-path $CHECKPOINT_PATH --model-name dream-v0-instruct-7b`
- `show_safe_token_push.py`: searches prompt/seed pairs and prints baseline vs safe comparisons ranked by token-level divergence (first divergence step/token + final token diff count).
  - Example: `python scripts/show_safe_token_push.py --checkpoint-path $CHECKPOINT_PATH --model-name llada-8b-base --unsafe-artifacts /path/to/unsafe_reference.pt --eta 1.5 --num-seeds 20 --num-show 3`
  - Interleaved masked prompts (true mask infill + continuation by default): `python scripts/show_safe_token_push.py --checkpoint-path $CHECKPOINT_PATH --model-name llada-8b-base --unsafe-artifacts /path/to/unsafe_reference.pt --eta 1.5 --interleaved-masked-prompts --num-seeds 20`
  - Prompt-only infill (no continuation): add `--infill-only`
  - Dense interleaved masks (more editable slots): `python scripts/show_safe_token_push.py --checkpoint-path $CHECKPOINT_PATH --model-name llada-8b-base --unsafe-artifacts /path/to/unsafe_reference.pt --eta 1.5 --interleaved-masked-prompts --interleaved-prompt-style dense --num-seeds 20`
  - Step replay + GIF: `python scripts/show_safe_token_push.py --checkpoint-path $CHECKPOINT_PATH --model-name llada-8b-base --unsafe-artifacts /path/to/unsafe_reference.pt --eta 1.5 --num-seeds 10 --animate --animate-target both --animate-delay 0.12 --animate-gif outputs/replay.gif`
- `repellency_diagnostic_dashboard.py`: per-step guidance magnitude visualizations.
  - Example: `python scripts/repellency_diagnostic_dashboard.py --run-dir /path/to/generation/run`

## Plotting + tables

Use these for paper plots and summary tables. Input CSVs come from `tools/hazard_report.py` output directories.
Replace `/path/to/...` with your local report locations (outputs of `python -m tools.hazard_report`).

### Figure 2: Safety/utility tradeoff (MDLM on RealToxicityPrompts)

`--overall-csv` is the `overall_rates.csv` file inside a `hazard_report_combined_*/` directory.

```bash
# BERTScore axis (Figure 2, left)
python scripts/plot_ablation_tradeoff.py \
  --overall-csv /path/to/rtp_mdlm_hazard_report_combined/overall_rates.csv \
  --output-dir paper/section5/5.1/bertscore \
  --x-metric bertscore \
  --include-baseline \
  --delta-plot \
  --show-pareto \
  --label-top-k 4 \
  --baseline-lines \
  --color-by time_window \
  --x-label "Δ BERTScore" \
  --y-label-left "Δ Unsafe Rate" \
  --title "Tradeoff: Safety vs Utility" \
  --interactive-legend \
  --legend-name "Time Window"

python scripts/plot_speed_refsize_paper.py \
  --summary-csv /path/to/speed_by_refsize_timewindow.csv \
  --output-prefix /path/to/speed_by_refsize_timewindow \
  --time-window median \
  --interactive-rename \
  --x-label "Negation set size" \
  --y-label-seq "Throughput (seq/s)" \
  --y-label-sec "Latency (s/sample)" \
  --title "Sample Time vs Unsafe Reference Size" \
  --legend-title "Active Steps" \
  --bucket-step 10

python scripts/summarize_speed.py \
  --speed-csv /path/to/speed_stats_llada_trillium.csv \
  --output-csv /path/to/speed_llada_by_ref.csv

python scripts/plot_ablations.py \
  --overall-csv /path/to/rtp_llada_hazard_report/overall_rates.csv \
  --output-dir paper/section5/5.2 \
  --ablation eta \
  --primary-metric unsafe_rate \
  --artifact real-toxicity-prompts-0100-llada \
  --include-baseline \
  --time-window-steps 256 \
  --x-label "η" \
  --y-label-left "Unsafe Rate" \
  --title "η Sensitivity" \
  --interactive-legend

python scripts/plot_ablation_tradeoff.py \
  --overall-csv /path/to/rtp_mdlm_hazard_report_combined/overall_rates.csv \
  --output-dir paper/section5/5.1 \
  --x-metric perplexity \
  --include-baseline \
  --delta-plot \
  --show-pareto \
  --label-top-k 4 \
  --baseline-lines \
  --color-by time_window \
  --x-label "Δ Perplexity" \
  --y-label-left "Δ Unsafe Rate" \
  --title "Tradeoff: Safety vs Utility" \
  --interactive-legend \
  --legend-name "Time Window"
```

### Figure 3: η sensitivity ablation (LLaDA on RealToxicityPrompts)

`--overall-csv` is the `overall_rates.csv` from the LLaDA hyperparam search hazard report.

```bash
python scripts/plot_ablations.py \
  --overall-csv /path/to/rtp-llada-hyperparam-search-hazard_report_combined/overall_rates.csv \
  --output-dir paper/section5/5.2 \
  --ablation eta \
  --primary-metric unsafe_rate \
  --artifact real-toxicity-prompts-0100-llada \
  --include-baseline \
  --time-window-steps 256 \
  --x-label "η" \
  --y-label-left "Unsafe Rate" \
  --title "η Sensitivity" \
  --interactive-legend
```

### Figure 4: Sampling speed vs reference size

`--summary-csv` is the output of `scripts/summarize_speed.py` over per-shard speed CSVs.

```bash
python scripts/plot_speed_refsize_paper.py \
  --summary-csv /path/to/speed_by_refsize_timewindow.csv \
  --output-prefix /path/to/speed_by_refsize_timewindow \
  --time-window median \
  --interactive-rename \
  --x-label "Negation set size" \
  --y-label-seq "Throughput (seq/s)" \
  --y-label-sec "Latency (s/sample)" \
  --title "Sample Time vs Unsafe Reference Size" \
  --legend-title "Active Steps" \
  --bucket-step 10

# Summarize raw per-shard speed CSVs first:
python scripts/summarize_speed.py \
  --speed-csv /path/to/speed_stats_llada.csv \
  --output-csv /path/to/speed_by_ref.csv
```

### Ablation plots (Appendix)

```bash
# Tradeoff curve (generic, for any dataset/model)
python scripts/plot_ablation_tradeoff.py \
  --overall-csv results/reports/realtoxicity_hazards/overall_rates.csv \
  --output-dir results/reports \
  --x-metric bertscores_f1_mean

# Section 5.5.1: Sensitivity to eta (fix artifact + time window, vary eta)
python scripts/plot_ablations.py \
  --overall-csv /path/to/overall_rates.csv \
  --output-dir /path/to/plots \
  --ablation eta \
  --artifact unsafe_refs_k4096.pt \
  --t-start 0 --t-end 16 \
  --primary-metric unsafe_rate \
  --add-perplexity --add-bertscore

# Section 5.5.2: Sensitivity to unsafe reference size (fix eta + time window, vary size)
python scripts/plot_ablations.py \
  --overall-csv /path/to/overall_rates.csv \
  --output-dir /path/to/plots \
  --ablation ref_size \
  --eta 0.2 \
  --t-start 0 --t-end 16 \
  --primary-metric unsafe_rate_delta \
  --add-mmd

# Section 5.5.3: Sensitivity to time window (fix artifact + eta, vary t_start/t_end)
python scripts/plot_ablations.py \
  --overall-csv /path/to/overall_rates.csv \
  --output-dir /path/to/plots \
  --ablation time_window \
  --artifact unsafe_refs_k4096.pt \
  --eta 0.2 \
  --primary-metric unsafe_rate \
  --add-perplexity
```

### Regenerate jailbreak tables (Table 2)

`--model-csv` takes `"ModelName=/path/to/model/overall_rates.csv"` pairs.
Each path is the `overall_rates.csv` from that model's jailbreak hazard report.

```bash
python scripts/generate_jailbreak_table.py \
  --model-csv "LLaDA-Instruct=/path/to/llada_instruct/overall_rates.csv" \
  --model-csv "LLaDA-1.5=/path/to/llada_1.5/overall_rates.csv" \
  --model-csv "Dream-Instruct=/path/to/dream_instruct/overall_rates.csv" \
  --metrics unsafe_rate harmbench_asr \
  --dataset jailbreakbench \
  --dataset-label JailbreakBench \
  --output-dir paper/tables \
  --output-prefix jailbreak_jbb

python scripts/generate_jailbreak_table.py \
  --model-csv "LLaDA-Instruct=/path/to/llada_instruct/overall_rates.csv" \
  --model-csv "LLaDA-1.5=/path/to/llada_1.5/overall_rates.csv" \
  --model-csv "Dream-Instruct=/path/to/dream_instruct/overall_rates.csv" \
  --metrics unsafe_rate harmbench_asr \
  --dataset jailbreakbench \
  --dataset-label JailbreakBench \
  --output-dir paper/tables/updated \
  --output-prefix jailbreak_jbb

python scripts/generate_jailbreak_table.py \
  --model-csv "LLaDA-Instruct=/path/to/llada_instruct/overall_rates.csv" \
  --model-csv "LLaDA-1.5=/path/to/llada_1.5/overall_rates.csv" \
  --model-csv "Dream-Instruct=/path/to/dream_instruct/overall_rates.csv" \
  --metrics unsafe_rate harmbench_asr \
  --dataset harmbench \
  --dataset-label HarmBench \
  --output-dir paper/tables \
  --output-prefix jailbreak_harmbench

python scripts/generate_jailbreak_table.py \
  --model-csv "LLaDA-Instruct=/path/to/llada_instruct/overall_rates.csv" \
  --model-csv "LLaDA-1.5=/path/to/llada_1.5/overall_rates.csv" \
  --model-csv "Dream-Instruct=/path/to/dream_instruct/overall_rates.csv" \
  --metrics unsafe_rate harmbench_asr \
  --dataset wildjailbreak \
  --dataset-label WildJailbreak \
  --output-dir paper/tables \
  --output-prefix jailbreak_wildjailbreak

python scripts/generate_jailbreak_table.py \
  --model-csv "LLaDA-Instruct=/path/to/llada_instruct/overall_rates.csv" \
  --model-csv "LLaDA-1.5=/path/to/llada_1.5/overall_rates.csv" \
  --model-csv "Dream-Instruct=/path/to/dream_instruct/overall_rates.csv" \
  --metrics unsafe_rate harmbench_asr \
  --dataset advbench \
  --dataset-label AdvBench \
  --output-dir paper/tables \
  --output-prefix jailbreak_advbench

python scripts/generate_jailbreak_table.py \
  --model-csv "LLaDA-Instruct=/path/to/llada_instruct/overall_rates.csv" \
  --model-csv "LLaDA-1.5=/path/to/llada_1.5/overall_rates.csv" \
  --model-csv "Dream-Instruct=/path/to/dream_instruct/overall_rates.csv" \
  --metrics unsafe_rate harmbench_asr \
  --dataset strongreject \
  --dataset-label StrongREJECT \
  --output-dir paper/tables \
  --output-prefix jailbreak_strongreject
```

### Main safety results table (Table 1)

```bash
python scripts/generate_hazard_table.py \
  --report-dir /path/to/hazard_report_combined \
  --output-dir paper/tables
```

## Data conversion utilities

```bash
# Convert HarmBench CSV to JSON for use as an unsafe artifact source
python scripts/convert_harmbench_csv_to_json.py --input harmbench.csv --out harmbench.json

# Flatten HarmBench result directories into a single CSV
python scripts/flatten_harmbench_results.py --results-dir /path/to/harmbench_results

# Collect per-shard generation speed into a summary CSV
python scripts/collect_generation_speed_csv.py --run-dir /path/to/run
```

## Distributed environment debugging

See `scripts/README_dist_env_debugs.md` for tips on diagnosing NCCL/CUDA issues
on multi-node Compute Canada allocations.

## Installation notes

`scripts/install_relaxed_requirements.sh` installs dependencies without strict version pins,
useful for environments where the exact pinned wheels are unavailable.

[Compute Canada] Use `src/requirements-cc.txt` (or `requirements-cc-h100.txt` for H100 nodes)
instead — these reference pre-built wheels from the CC software stack and are significantly
faster to install.
