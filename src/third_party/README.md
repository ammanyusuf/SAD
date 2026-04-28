# src/third_party — vendored dependencies

All directories here are vendored upstream repos. **Do not modify them directly** unless noted below.
If you need to override behaviour, prefer monkey-patching or subclassing in `src/sampling/` or `src/safety_eval/`.

---

## mdlm

**Upstream**: [kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm)

**What it is**: The core Masked Discrete Language Model (MDLM). Implements training, sampling
(DDPM-Cache ancestral sampler), and perplexity evaluation for absorbing-state diffusion on text.

**What we use from it**: `diffusion.py` — the `Diffusion` Lightning module and its
`_ddpm_caching_update()` method, which is our main integration point for the safe denoiser.

**What we added**: `repellency/` is entirely new code — it was not in the upstream repo.
It contains `MaskKernelRepellency` (the safe denoiser), alignment utilities, and tests.
See `repellency/README.md` for full documentation.

---

## LLaDA

**Upstream**: [GSAI-ML/LLaDA](https://github.com/GSAI-ML/LLaDA)

**What it is**: LLaDA-8B, a large masked diffusion language model. Two checkpoints are used:
`LLaDA-8B-Base` (pretrained) and `LLaDA-8B-Instruct` (instruction-tuned).

**What we use from it**: `generate.py` — specifically the `generate()` function, which exposes
a `logits_hook=` callback argument. We pass our safe denoiser hook through this argument;
no source changes to LLaDA are needed.

---

## Dream

**Upstream**: [Dream-org/Dream](https://github.com/Dream-org/Dream)

**What it is**: Dream-v0, another large masked diffusion LM from the Dream-org team.

**What we use from it**: The generation loop's `step_callback=` interface, which gives us
per-step access to the token state and logits. We pass `build_dream_repellency_hook()` as
the callback.

**Note**: Dream uses mask token id 151666 (vs LLaDA's 126336) and does not support
`low_confidence` remasking. The Dream backend handles these differences.

---

## HarmBench

**Upstream**: [centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench)

**What it is**: An adversarial robustness benchmark for LLMs. Provides attack prompts, a
standard set of behaviors, and a Llama-2-13B classifier for evaluating attack success rate (ASR).

**What we use from it**: The Llama-2-13B classifier checkpoint (`HarmBench-Llama-2-13b-cls`),
loaded via `src/safety_eval/classifiers/harmbench.py`. We do not run HarmBench attacks
directly, we re-use only the ASR classifier and the generated harmful examples from their exposed dataset.

---

## DiffuGuard

**Upstream**: [DiffuGuard paper codebase](https://github.com/fywang/DiffuGuard)

**What it is**: A defense method for masked diffusion models that audits hidden states during
generation. If an intermediate hidden state is flagged as unsafe, the model is redirected.

**What we use from it**: `defense_utils.py` — the hidden-state audit logic. Wrapped by
`src/sampling/backends/llada_diffuguard_backend.py` and `dream_diffuguard_backend.py`.
Used as a **comparison baseline** in the jailbreak evaluation.

---

## DIJA

**Upstream**: DIJA adversarial suffix attack (authors' shared codebase)

**What it is**: A gradient-based jailbreak attack method that injects adversarial suffixes into
a prompt to bypass safety filters.

**What we use from it**: The adversarial suffix injection pipeline. Wrapped by
`src/sampling/backends/llada_dija_backend.py`. Used as an **attack method**.

---

## Fk-Diffusion-Steering

**Upstream**: [Fk-Diffusion-Steering](https://github.com/fk-diffusion-steering/Fk-Diffusion-Steering)

**What it is**: FK Diffusion Steering, a reward-guided sampling approach that uses a
RoBERTa toxicity classifier as a reward signal to steer diffusion sampling away from unsafe
content.

**What we use from it**: The `discrete_diffusion/` steering utilities, wrapped by
`src/sampling/backends/mdlm_fk_backend.py`. Used as a **comparison baseline** only.
