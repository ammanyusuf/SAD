# src/sampling — generation layer

Model-agnostic generation layer. Sits between `tools/generate.py` and the per-model backends.

```
sampling/
  sample_text.py        # Dataclasses + MDLM generation loop (run_generation)
  safe_hooks.py         # Safe denoiser hooks for LLaDA and Dream
  llada_engine.py       # Experimental LLaDA generation engine (local backend only)
  transfer_schedule.py  # Diffusion transfer schedule utilities
  backends/             # Per-model backend implementations
```

---

## `sample_text.py`

Contains the shared dataclasses (`GenerationSettings`, `ModelSettings`, `SafetySettings`, `PromptRecord`, `GenerationResult`, `GenerationRun`) used by all backends, plus the MDLM generation loop.

**`run_generation()`** is the MDLM-specific entry point. It:
1. Loads the MDLM checkpoint via Lightning + Hydra.
2. Optionally initializes the MDLM `MaskKernelRepellency` safe denoiser via the config in `src/third_party/mdlm/diffusion.py`.
3. Runs the DDPM caching sampler (`sampling.predictor=ddpm_cache`) over the prompt batch.
4. Returns a `GenerationRun` with completions, timing, and metadata.

> **Note**: `run_generation()` is only called by `MDLMBackend`. LLaDA and Dream use their own backends that do not go through this function — they use the upstream `generate()` functions directly with a `logits_hook`.

**Key helpers:**
- `_prepare_unsafe_artifacts()` — resolves unsafe artifact path from `SafetySettings`, auto-building if needed.
- `resolve_eta_config()` — maps `safety.eta` / `safety.scale` to a float guidance strength.
- `_resolve_stop_tokens()` — extracts EOS/EOT/PAD ids from the tokenizer for guidance masking.
- `StopTokens` — dataclass wrapping stop token ids so they can be passed through the hook.

---

## `safe_hooks.py`

Implements the safe denoiser as a **logits hook** for LLaDA and Dream. This is the integration point for models that do not use the MDLM sampling loop.

**`build_llada_repellency_hook(tokenizer, safety, device)`**

Returns a `_logits_hook(logits, *, x, t, mask_index, prompt_index, attention_mask, extra)` closure that is passed as `logits_hook=` to the upstream LLaDA `generate()` function. On the first call it lazy-initializes a `MaskKernelRepellency` instance (loads the unsafe tensor, resolves eta, sets up the guidance window). On subsequent calls it applies repellency conditioning if `t_start <= t <= t_end`.

The hook:
1. Detects prompt width from `extra["prompt_width"]` or from the position of the first mask token.
2. Builds `prompt_mask` (True for non-mask prompt tokens — these are frozen and excluded from guidance).
3. Calls `_compute_move_proxy()` to get the linear diffusion schedule alignment.
4. Passes `softmax(logits)` through `MaskKernelRepellency.conditioning()`.
5. Returns `log(p_safe)` as the new logits.

**`build_dream_repellency_hook(tokenizer, safety, device, ...)`**

Same concept but for Dream's `step_callback(step, x, logits)` interface. Dream uses a different mask token id (151666 vs LLaDA's 126336) and does not support `low_confidence` remasking.

---

## `llada_engine.py`

An experimental local LLaDA generation engine used only by `LLaDALocalBackend`. It re-implements the LLaDA sampling loop natively in this repo (rather than calling the upstream `generate()`) to expose finer-grained control over the denoising trajectory.

**Status:** Used only when `model.variant=local` or `model.variant=native`. In all paper experiments, `model.variant=upstream` (i.e., `LLaDAUpstreamBackend`) was used. The local engine is not validated to produce identical output to the upstream implementation — use `scripts/debug_llada_parity.py` to check.

---

## `transfer_schedule.py`

Utilities for computing the discrete diffusion transfer schedule (move grid). Used by `safe_hooks.py` to align the guidance strength proxy `move(t)` with the sampler's timestep discretization.

---

## `backends/`

Each backend wraps a specific model family + generation strategy and exposes a uniform interface:

```python
class SomeBackend:
    name: str          # used in logging
    family: str        # model family tag
    supports_logits_hook: bool

    def load(self, model_settings: ModelSettings, device: str) -> None: ...
    def generate_batch(self, prompts, generation, safety, shard_metadata) -> GenerationRun: ...
```

### Backend inventory

| File | Class | When used | Notes |
|---|---|---|---|
| `mdlm_backend.py` | `MDLMBackend` | `model.family=mdlm` (default) | Delegates to `sample_text.run_generation()`. Uses MDLM-native safe denoiser via Lightning/Hydra. |
| `llada_upstream_backend.py` | `LLaDAUpstreamBackend` | `model.family=llada` (default variant) | Calls upstream `third_party/LLaDA/generate.py` with `logits_hook=`. **Used in all paper experiments for LLaDA.** |
| `dream_backend.py` | `DreamBackend` | `model.family=dream` | Calls upstream Dream generation with `safe_hooks.build_dream_repellency_hook`. **Used in paper experiments for Dream.** |
| `llada_diffuguard_backend.py` | `LLaDADiffuGuardBackend` | `model.family=llada model.variant=diffuguard` | Wraps LLaDA generation with DiffuGuard's hidden-state audit defense. Used in jailbreak eval. |
| `dream_diffuguard_backend.py` | `DreamDiffuGuardBackend` | `model.family=dream model.variant=diffuguard` | Same as above but for Dream. Note: Dream does not support `low_confidence` remasking. |
| `llada_dija_backend.py` | `LLaDADIJABackend` | `model.family=llada model.variant=dija` | Wraps LLaDA with DIJA adversarial suffix injection. Used in jailbreak eval. |
| `llada_local_backend.py` | `LLaDALocalBackend` | `model.family=llada model.variant=local` | Uses `llada_engine.py` (the local reimplementation). **Experimental — not used in paper.** |
| `mdlm_fk_backend.py` | `MDLMFKBackend` | `model.family=mdlm model.variant=fk_steering` | FK Diffusion Steering baseline (RoBERTa toxicity reward). **Experimental — comparison method.** |
| `posthoc_filter_backend.py` | `PosthocFilterBackend` / `BestOfNBackend` | `model.variant=posthoc_filter` or `best_of_n` | Post-hoc LlamaGuard filtering baselines. Wraps any inner backend and runs N samples per prompt. |
| `mmada_backend.py` | `MMADABackend` | `model.family=mmada` | **Not implemented** — raises `NotImplementedError`. Stub for future MMaDA support. |
| `base.py` | `TextGenerationBackend` | — | Abstract base class. |
| `registry.py` | `get_backend()` | `tools/generate.py` | Dispatches `(family, variant)` → backend instance. |

### Safe denoiser wiring per backend

| Backend | Where hook attaches |
|---|---|
| `MDLMBackend` | `src/third_party/mdlm/diffusion.py:_ddpm_caching_update()` — modifies `p_x0` directly |
| `LLaDAUpstreamBackend` | `logits_hook=` argument to `third_party/LLaDA/generate.py:generate()` |
| `DreamBackend` | `step_callback=` passed into Dream's generation loop |
| `LLaDADiffuGuardBackend` | `logits_hook=` on the DiffuGuard-wrapped LLaDA generator |
| `DreamDiffuGuardBackend` | `step_callback=` on the DiffuGuard-wrapped Dream generator |
| `LLaDADIJABackend` | `logits_hook=` on the DIJA-wrapped LLaDA generator |
