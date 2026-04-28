# Repellency Debug Env Vars (LLaDA + MDLM)

This repo uses environment variables to control repellency debug logging and distribution diagnostics. These apply to LLaDA generation **when safety/repellency is enabled** (i.e., `safety.enabled=true` and unsafe artifacts are provided). LLaDA routes through `MaskKernelRepellency` via `sampling/safe_hooks.py`, so these vars apply to LLaDA runs too.

## Quick usage (LLaDA)

```bash
source scripts/env_profile.sh llada --debug-dist
# optionally add --debug to enable extra per-step CSV logging
```

That script sets sensible defaults for the distribution logger and prints the effective environment values.

## Core debug toggles

- `SAFE_REPELLENCY_DEBUG` (bool): Enables extra per-step logging and diagnostics for the safe denoiser.
- `SAFE_REPELLENCY_VALIDATE` (bool): Enables additional runtime validation checks.
- `SAFE_KERNEL_MODE` (string): Repellency kernel mode override. Typical values: `both`, `prob`, `logit`.
- `SAFE_GUIDANCE_MODE` (string): Guidance mode override. Typical values: `both`, `prob`, `logit`.
- `SAFE_BETA_MODE` (string): Beta mode override. Typical values: `both`, `raw`, `len`, `mean`.
- `SAFE_LLADA_DEBUG` (bool): LLaDA-specific debug prints.
- `SAFE_REPELLENCY_CSV_LOG` (path): Path to per-step CSV logging (e.g., `results/repellency_stats.csv`). The CSV is truncated automatically at sampler start. Logs are written under a subfolder that encodes run hyperparams (eta/t_start/t_end/schedule/unsafe tensor).

Note: Exact allowed values for kernel/beta/guidance modes live in `src/third_party/mdlm/repellency/README.md` and are validated at runtime.

## Distribution logging (JSONL top-k)

These control the optional distribution-level debug logger that writes JSONL files under the configured directory.

- `SAFE_DIST_LOG_ENABLED` (bool): Master switch for dist logging. The dist-log directory is cleared automatically at sampler start and written under a tagged subfolder for the active hyperparams.
- `SAFE_DIST_LOG_DIR` (path): Base directory to store logs (default: `results/diagnostics/dist_logs`).
- `SAFE_DIST_LOG_PATH` (path): Explicit JSONL output path. If relative, it’s resolved under `SAFE_DIST_LOG_DIR`.
- `SAFE_DIST_LOG_TIMESTEPS` (list, `auto`, or `all`): Comma-separated steps to log (e.g., `0,32,64`), `auto`, or `all` for every step.
- `SAFE_DIST_LOG_TOPK` (int): Top-k tokens to record per position (default: `50`).
- `SAFE_DIST_LOG_MAX_POS` (int): Max positions logged per step (default: `16`).
- `SAFE_DIST_LOG_POSITIONS` (legacy string): `masked`, `all`, `subset`.
- `SAFE_DIST_LOG_POSITION_MODE` (string): Preferred selector. One of:
  - `masked_only` (default)
  - `unmasked_only`
  - `all`
  - `sampled`
- `SAFE_DIST_LOG_POSITION_SAMPLE_SEED` (int): Seed for deterministic sampling (only used with `sampled`).
- `SAFE_DIST_LOG_DTYPE` (string): `float16` or `float32` for stored probabilities.
- `SAFE_DIST_LOG_FULL_VOCAB` (bool): If set, logs full vocab distributions (very large output).
- `SAFE_DIST_LOG_AUTO_TOP_N` (int): When `SAFE_DIST_LOG_TIMESTEPS=auto`, also log the top-N steps by effective strength.
- `SAFE_DIST_LOG_RUN_ID` (string): Run identifier stored in JSONL records.
- `SAFE_DIST_LOG_DUMP_TOKENS` (bool): If set, decode token IDs in the per-step samples.
- `SAFE_DIST_LOG_DUMP_PROMPT` (bool): Parsed, but currently **not emitted** because the logger doesn’t receive prompt text.

## Plotting distribution diagnostics

Use the existing script to visualize the JSONL + CSV diagnostics:

```bash
python scripts/plot_dist_debug.py \
  --csv /path/to/repellency_stats.csv \
  --json /path/to/dist_logs.jsonl \
  --outdir diagnostics/plots
```

Note: token labels in the distribution plots are currently token IDs (not decoded text).

## LLaDA-specific notes

- Distribution logging happens **only when repellency is active**, which requires `safety.enabled=true` and valid unsafe artifacts.
- The safe denoiser for LLaDA is initialized lazily in the logits hook (`sampling/safe_hooks.py`).
- If you see no dist logs, check that `SAFE_DIST_LOG_ENABLED=1` and that safety/repellency is actually enabled.
