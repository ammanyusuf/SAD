"""
dlm_wrapper.py -- Wrapper around a masked diffusion LM (LLaDA / Dream / MDLM)
that exposes per-step logits for the p_z estimator (Equation 8 of
"Characterizing Memorization in Diffusion Language Models", Luo et al. 2025).

Given a masked sequence x (with mask tokens at positions M), one forward pass
returns the full logit tensor [B, L, vocab_size] so that the caller can read off
  Pr(z_hat_pi = z_pi | x) = softmax(logits)[pi, z_pi]
for every masked position pi.

No denoising loop is run here -- that lives in pz_estimator.py.

Supported backends
------------------
  "llada"  : AutoModel with model(x).logits   (LLaDA-8B-Base/Instruct)
  "dream"  : same interface as llada
  "mdlm"   : diffusion.Diffusion.forward(x, sigma_t) returns log-probs;
             sigma_t is set to match the mask ratio of the input
             (sigma = -log(1 - mask_ratio), so sigma~0.7 for 50% masking).

MDLM checkpoint loading
-----------------------
Use DLMWrapper.from_mdlm_checkpoint() to load a Lightning .ckpt file:

  wrapper, tokenizer = DLMWrapper.from_mdlm_checkpoint(
      ckpt_path="/scratch/models/text-diffusion/702-1250000.ckpt",
      config_path=".../src/third_party/mdlm/configs/config.yaml",
      tokenizer_name="gpt2-large",
      device="cuda",
  )
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


class DLMWrapper:
  """Wraps a masked diffusion LM to expose per-step logit tensors.

  Parameters
  ----------
  model:
      The loaded transformer model.  For LLaDA/Dream this is an AutoModel
      with a ``.logits`` output.  For MDLM this is the ``Diffusion``
      lightning module (its .forward() returns log p_x0).
  mask_id:
      Token id used for [MASK] tokens (126336 for LLaDA; vocab_size for MDLM
      with GPT-2 tokenizer).
  backend:
      One of "llada", "dream", "mdlm".
  device:
      Torch device string.
  """

  def __init__(
    self,
    model,
    mask_id: int,
    backend: str = "llada",
    device: Optional[str] = None,
  ) -> None:
    self.model = model
    self.mask_id = mask_id
    self.backend = backend
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if backend not in ("llada", "dream", "mdlm"):
      raise ValueError(f"Unknown backend {backend!r}; expected llada, dream, or mdlm.")

  # ------------------------------------------------------------------
  # MDLM checkpoint loader (class method)
  # ------------------------------------------------------------------

  @classmethod
  def from_mdlm_checkpoint(
    cls,
    ckpt_path: str,
    config_path: str,
    tokenizer_name: str = "gpt2-large",
    device: Optional[str] = None,
    config_overrides: Optional[str] = None,
  ) -> Tuple["DLMWrapper", object]:
    """Load an MDLM Lightning checkpoint and return (wrapper, tokenizer).

    Parameters
    ----------
    ckpt_path:
        Absolute path to a .ckpt file, e.g.
        "~/scratch/models/text-diffusion/702-1250000.ckpt".
    config_path:
        Path to the Hydra config YAML, e.g.
        ".../src/third_party/mdlm/configs/config.yaml".
    tokenizer_name:
        HF model name or local path for the tokenizer (default: gpt2-large).
    device:
        Torch device (default: cuda if available).
    config_overrides:
        Optional comma-separated Hydra-style overrides, e.g.
        "data=wikitext,sampling.steps=128".  Mirrors $MDLM_CONFIG_OVERRIDES.
    """
    import numpy as np
    import omegaconf
    from pathlib import Path as _Path
    from transformers import AutoTokenizer
    # MDLM lives under src/third_party/mdlm/ and is on PYTHONPATH
    from src.third_party.mdlm import diffusion as mdlm_diffusion

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("Loading MDLM config from %s", config_path)

    # OmegaConf.load() does NOT resolve Hydra defaults (model, data, etc. are
    # separate YAML files listed under `defaults:`).  Use Hydra's compose API
    # so all defaults are merged into a single DictConfig.
    config_dir = str(_Path(config_path).parent.resolve())
    overrides_list = ["eval.checkpoint_path=" + ckpt_path]
    if config_overrides:
      for ov in config_overrides.split(","):
        ov = ov.strip()
        if ov:
          overrides_list.append(ov)

    try:
      from hydra import compose, initialize_config_dir
      from hydra.core.global_hydra import GlobalHydra
      GlobalHydra.instance().clear()
      with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=_Path(config_path).stem, overrides=overrides_list)
      LOGGER.info("Hydra compose succeeded (model.hidden_size=%s)", cfg.model.hidden_size)
    except Exception as hydra_err:
      LOGGER.warning("Hydra compose failed (%s); falling back to OmegaConf.load + manual merge", hydra_err)
      cfg = omegaconf.OmegaConf.load(config_path)
      # Manually merge the model/small defaults so the checkpoint can load
      model_cfg_path = _Path(config_path).parent / "model" / "small.yaml"
      if model_cfg_path.exists():
        model_cfg = omegaconf.OmegaConf.load(model_cfg_path)
        cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.create({"model": model_cfg}))
      omegaconf.OmegaConf.update(cfg, "eval.checkpoint_path", ckpt_path, merge=True)
      if config_overrides:
        for override in config_overrides.split(","):
          override = override.strip()
          if "=" in override:
            key, val = override.split("=", 1)
            try:
              omegaconf.OmegaConf.update(cfg, key.strip(), val.strip(), merge=True)
            except Exception as e:
              LOGGER.warning("Could not apply config override '%s': %s", override, e)

    LOGGER.info("Loading tokenizer: %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading MDLM checkpoint: %s -> %s", ckpt_path, device)
    safe_dtypes = [np.dtype]
    if hasattr(np, "dtypes") and hasattr(np.dtypes, "Float64DType"):
      safe_dtypes.append(np.dtypes.Float64DType)
    with torch.serialization.safe_globals(safe_dtypes):
      model = mdlm_diffusion.Diffusion.load_from_checkpoint(
        ckpt_path,
        tokenizer=tokenizer,
        config=cfg,
        map_location=device,
        weights_only=False,
      )
    model = model.to(device).eval()

    # MDLM mask token is vocab_size (one beyond the GPT-2 vocabulary)
    mask_id = model.mask_index
    LOGGER.info(
      "MDLM loaded. mask_index=%d  vocab_size=%d  device=%s",
      mask_id, model.vocab_size, device,
    )
    wrapper = cls(model=model, mask_id=mask_id, backend="mdlm", device=device)
    return wrapper, tokenizer

  # ------------------------------------------------------------------
  # Single-forward-pass logit extraction
  # ------------------------------------------------------------------

  @torch.no_grad()
  def get_logits(
    self,
    x: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sigma_t: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Run one forward pass and return logits [B, L, vocab_size].

    Parameters
    ----------
    x:
        Token-id tensor of shape (B, L).  Masked positions contain
        self.mask_id; observed positions contain ground-truth token ids.
    attention_mask:
        Optional (B, L) int/bool mask (1=real token, 0=padding).
        Only used by llada/dream backends.
    sigma_t:
        MDLM noise level tensor (B,).  If None and backend=="mdlm", a
        sigma_t is auto-computed from the mask fraction in x.
    """
    x = x.to(self.device)
    if attention_mask is not None:
      attention_mask = attention_mask.to(self.device)

    if self.backend in ("llada", "dream"):
      if attention_mask is not None:
        out = self.model(x, attention_mask=attention_mask)
      else:
        out = self.model(x)
      return out.logits  # [B, L, V]

    elif self.backend == "mdlm":
      if sigma_t is None:
        sigma_t = self._sigma_from_mask_frac(x)
      sigma_t = sigma_t.to(self.device)
      # MDLM forward() returns log p_x0 [B, L, V] (log-probabilities over vocab).
      # These are directly usable as logits for log_softmax.
      log_p_x0 = self.model.forward(x, sigma_t)  # [B, L, V]
      return log_p_x0

    else:
      raise ValueError(f"Unsupported backend: {self.backend}")

  def _sigma_from_mask_frac(self, x: torch.Tensor) -> torch.Tensor:
    """Infer sigma_t from the fraction of masked tokens in x.

    Under the loglinear noise schedule:  move_chance = 1 - exp(-sigma)
    so sigma = -log(1 - move_chance).

    We compute the per-sequence mask fraction and convert.
    Clipped to [0.01, 5.0] for numerical stability.
    """
    B = x.shape[0]
    mask_frac = (x == self.mask_id).float().mean(dim=1).clamp(0.01, 0.99)  # (B,)
    sigma = -torch.log(1.0 - mask_frac)  # (B,)
    return sigma.clamp(max=5.0)

  # ------------------------------------------------------------------
  # Convenience: conditional log-prob of ground-truth tokens at mask positions
  # ------------------------------------------------------------------

  @torch.no_grad()
  def log_prob_at_mask_positions(
    self,
    x: torch.Tensor,
    z_true: torch.Tensor,
    mask_positions: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sigma_t: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Return log Pr(z_pi | x) for each masked position pi.

    Parameters
    ----------
    x:
        Noisy sequence (B, L) with mask tokens at mask_positions.
    z_true:
        Ground-truth token ids (B, L).
    mask_positions:
        Boolean tensor (B, L); True at positions that are currently masked.
    attention_mask, sigma_t:
        See get_logits().

    Returns
    -------
    Tensor of shape (total_masked_positions,) -- log-prob of the ground-truth
    token at each masked position, flattened row-major across batch & sequence.
    """
    logits = self.get_logits(x, attention_mask=attention_mask, sigma_t=sigma_t)
    log_probs_all = F.log_softmax(logits, dim=-1)  # [B, L, V]
    z_true = z_true.to(self.device)
    mask_positions = mask_positions.to(self.device)
    gathered = log_probs_all.gather(-1, z_true.unsqueeze(-1)).squeeze(-1)  # [B, L]
    return gathered[mask_positions]  # (total_masked,)
