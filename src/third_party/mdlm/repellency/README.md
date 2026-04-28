# Discrete Safe Denoiser (Repellency)

This directory contains the main contribution of the paper: a training-free discrete safe denoiser for Masked Discrete Language Models (MDLMs). It steers token predictions away from unsafe content during sampling by computing a per-position unsafe posterior from a set of reference sequences, then repelling the model's predicted distribution away from that posterior.

## Files

```
repellency/
  safe_denoiser.py            # MaskKernelRepellency — main class
  repellency_methods_fast.py  # RepellencyMethod base class + registration system
  alignment.py                # Prompt alignment strategies (left/right/none)
  discrete_utils.py           # Utility functions (masking, token matching)
  tests/                      # Unit tests for the safe denoiser
```

---

## How it works

At each denoising step `t`, the model produces a predicted clean-token distribution `p_x0[b, i, v]` (batch × position × vocab). The safe denoiser modifies this distribution before the ancestral sampling step.

**Step 1 — Compute unsafe likelihood weights.**

For each unsafe reference sequence `U_n` (a tokenized example of unsafe text), compute the forward-process likelihood of the current noisy state `x_t` given `U_n`:

```
log q_t(x_t | U_n)  =  Σ_i  log q_t(x_t[i] | U_n[i])
```

where `q_t(x_t[i] | U_n[i])` is the masking probability at timestep `t`. This is computed with the **mask kernel** (Eq. 3 in the paper):

- `relaxed` mode (default): mismatch between `x_t[i]` and `U_n[i]` contributes a soft penalty.
- `strict` mode: any mismatch with a non-masked token makes `log q_t = -∞` for that reference.

**Step 2 — Build unsafe posterior.**

Normalize the weights across references:

```
w_n  ∝  exp(log q_t(x_t | U_n))
p_unsafe[i, v]  =  Σ_n  w_n * 1{U_n[i] == v}
```

This gives a per-position categorical distribution over tokens that appear in unsafe contexts.

**Step 3 — Compute guidance strength β̂.**

```
β̂(x_t)  =  Σ_n  w_n  (normalized mean likelihood over references)
strength  =  η × β̂(x_t) × g(t)
```

where `η` (eta) is the user-set guidance scale and `g(t)` is a timestep schedule (`hard_window` or `cosine_ramp`).

**Step 4 — Repel.**

Apply a logit-ratio update:

```
p_safe  =  softmax(log p_x0  +  strength × (log p_x0 − log p_unsafe_smooth))
```

This reduces the probability mass on tokens that appear disproportionately in unsafe references, without touching the rest of the distribution.

**Step 5 — Optional semantic gating.**

If `use_semantic_gating=True`, each reference is additionally weighted by a semantic similarity score between the current generation and the reference (computed via RBF kernel over model embeddings). References that are semantically close to the current output are upweighted; distant references are downweighted. This prevents guidance from firing on references that are irrelevant to the current prompt context.

---

## Integration

### MDLM

The hook is at `src/third_party/mdlm/diffusion.py:_ddpm_caching_update()` (~line 630). `MaskKernelRepellency.conditioning()` is called on `p_x0` before the ancestral sampling step. Requires `sampling.predictor=ddpm_cache`.

### LLaDA and Dream

The hook is in `src/sampling/safe_hooks.py`. `build_llada_repellency_hook()` and `build_dream_repellency_hook()` return logits hook closures that are passed to the upstream generation functions. See `src/sampling/README.md` for details.

---

## Configuration

Set via `safety.*` Hydra keys (see `configs/config.yaml`) or environment variables.

| Parameter | Hydra key | Default | Description |
|---|---|---|---|
| Guidance scale | `safety.eta` | 1.0 | Main tuning knob. Higher = stronger repellency. Paper range: 0.1–8. |
| Window start | `safety.t_start` | 0 | First diffusion step to apply guidance (0 = last denoising step). |
| Window end | `safety.t_end` | — | Last step to apply guidance. For LLaDA-256 steps, `t_end=18` ≈ final 7%. |
| Schedule | `safety.schedule_mode` | `hard_window` | `hard_window`: uniform within [t_start, t_end]. `cosine_ramp`: tapers at edges. |
| Kernel mode | `SAFE_KERNEL_MODE` env | `relaxed` | `relaxed`: soft mismatch penalty. `strict`: exact match only. |
| Beta mode | `SAFE_BETA_MODE` env | `len` | `len`: length-normalized beta. `raw`: unnormalized. |
| Guidance mode | `SAFE_GUIDANCE_MODE` env | `logit` | `logit`: logit-ratio update (default). `prob`: additive probability update. |
| Semantic gating | `safety.use_semantic_gating` | false | Weight references by semantic similarity to current output. |
| Semantic weight | `safety.semantic_weight` | 1.0 | Interpolation between uniform (0) and fully semantic (1) reference weighting. |

---

## Recommended starting points

```bash
# Baseline (paper default for LLaDA)
safety.enabled=true safety.eta=4.0 safety.t_start=0 safety.t_end=18

# Stronger guidance (trades more utility for safety)
safety.enabled=true safety.eta=8.0 safety.t_start=0 safety.t_end=64

# With semantic gating (more selective, less utility cost)
safety.enabled=true safety.eta=4.0 safety.t_start=0 safety.t_end=18 \
  safety.use_semantic_gating=true safety.semantic_weight=1.0
```

---

## Environment variable toggles (debug / ablation)

```bash
# Verbose per-step logging
SAFE_REPELLENCY_DEBUG=1

# Validate fast scatter_add_ against slow reference implementation
SAFE_REPELLENCY_VALIDATE=1

# Kernel mode: relaxed (default) / strict / both
SAFE_KERNEL_MODE=relaxed

# Beta normalization: len (default) / raw / both
SAFE_BETA_MODE=len

# Guidance update rule: logit (default) / prob / both
SAFE_GUIDANCE_MODE=logit
```

Setting `*=both` computes both variants and logs them for comparison, but applies only the primary mode.

---

## Tests

```bash
python -m pytest -q src/third_party/mdlm/repellency/tests/test_safe_denoiser_alignment.py
python -m pytest -q src/unsafe_prep/tests
```
