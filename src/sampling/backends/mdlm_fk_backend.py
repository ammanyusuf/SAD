"""FK Steering backend for MDLM.

Wraps FKDiffusion from src/third_party/Fk-Diffusion-Steering with a RoBERTa
toxicity classifier (label='negative'), steering generation *away* from toxicity.

Configuration via env vars:
  FK_K_PARTICLES      number of parallel particles (default 8)
  FK_RESAMPLE_FREQ    resampling frequency in timesteps (default 20, i.e. 50 resamples over 1000 steps)
  FK_NUM_X0_SAMPLES   x0 samples per particle per step (default 16, "many r_phi" in paper)
  FK_LAMBDA           potential temperature (default 10.0)
  FK_REWARD_TRIM_LEN  token length for reward truncation (default 50)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from sampling.sample_text import (
  GenerationResult,
  GenerationRun,
  GenerationSettings,
  ModelSettings,
  PromptRecord,
  SafetySettings,
  _compose_sampling_config,
  _strip_completion_tokens,
  _resolve_stop_tokens,
  _chunk,
)
from sampling.backends.base import TextGenerationBackend

LOGGER = logging.getLogger(__name__)

_FK_THIRD_PARTY = str(
  Path(__file__).resolve().parents[2]
  / "third_party"
  / "Fk-Diffusion-Steering"
  / "discrete_diffusion"
)


def _ensure_fk_on_path() -> None:
  if _FK_THIRD_PARTY not in sys.path:
    sys.path.insert(0, _FK_THIRD_PARTY)


def _int_env(key: str, default: int) -> int:
  val = os.getenv(key)
  if val:
    try:
      return int(val)
    except ValueError:
      pass
  return default


def _float_env(key: str, default: float) -> float:
  val = os.getenv(key)
  if val:
    try:
      return float(val)
    except ValueError:
      pass
  return default


class MDLMFKBackend(TextGenerationBackend):
  """FK Steering (toxicity, label=negative) wrapper around FKDiffusion."""

  name = "fk_steering"
  family = "mdlm"
  supports_logits_hook = False

  def __init__(self) -> None:
    self.model_settings: Optional[ModelSettings] = None

  def load(self, model_settings: ModelSettings, device: Optional[str] = None) -> None:
    self.model_settings = model_settings

  def generate_batch(
    self,
    prompts: Optional[Sequence[PromptRecord]],
    generation: GenerationSettings,
    safety: SafetySettings,
    shard_metadata: Dict[str, Any],
  ) -> GenerationRun:
    if self.model_settings is None:
      raise RuntimeError("MDLMFKBackend requires load() before generate_batch().")

    start_time = time.perf_counter()
    _ensure_fk_on_path()

    from fk_diffusion import FKDiffusion  # type: ignore[import]
    from third_party.mdlm import dataloader

    # FK requires batch_size=1
    if generation.batch_size != 1:
      LOGGER.warning(
        "FK Steering requires batch_size=1; overriding (was %d).", generation.batch_size
      )

    # Build base MDLM config and augment with fk_steering section
    base_cfg = _compose_sampling_config(self.model_settings, generation, safety)
    OmegaConf.set_struct(base_cfg, False)
    base_cfg.loader.eval_batch_size = 1
    base_cfg.loader.batch_size = 1

    k_particles = _int_env("FK_K_PARTICLES", 8)
    resample_freq = _int_env("FK_RESAMPLE_FREQ", 20)
    num_x0_samples = _int_env("FK_NUM_X0_SAMPLES", 16)
    lmbda = _float_env("FK_LAMBDA", 10.0)
    reward_trim_len = _int_env("FK_REWARD_TRIM_LEN", 50)

    fk_cfg = OmegaConf.create({
      "fk_steering": {
        "reward_fn": "toxicity",
        "reward_label": "negative",  # steer AWAY from toxicity
        "k_particles": k_particles,
        "num_x0_samples": num_x0_samples,
        "lmbda": lmbda,
        "resample_frequency": resample_freq,
        "reward_trim_length": reward_trim_len,
        "potential_type": "diff",  # differential potential at each step
      }
    })
    cfg = OmegaConf.merge(base_cfg, fk_cfg)
    OmegaConf.set_struct(cfg, True)

    LOGGER.info(
      "FK Steering config: k=%d, resample_freq=%d, num_x0=%d, lambda=%.2f",
      k_particles, resample_freq, num_x0_samples, lmbda,
    )

    load_start = time.perf_counter()
    tokenizer = dataloader.get_tokenizer(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_dtypes = [np.dtype]
    if hasattr(np, "dtypes") and hasattr(np.dtypes, "Float64DType"):
      safe_dtypes.append(np.dtypes.Float64DType)
    with torch.serialization.safe_globals(safe_dtypes):
      model: FKDiffusion = FKDiffusion.load_from_checkpoint(
        cfg.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=cfg,
        map_location=device,
        weights_only=False,
      )
    model = model.to(device)
    model.eval()
    load_seconds = time.perf_counter() - load_start

    tokenizer = model.tokenizer
    stop_tokens = _resolve_stop_tokens(tokenizer)
    stop_ids = stop_tokens.stop_ids if stop_tokens else set()
    mask_index = getattr(model, "mask_index", None)

    # Pre-load toxicity classifier from a local path if provided, to avoid
    # network access on offline clusters (e.g. Compute Canada).
    fk_roberta_path = os.getenv("FK_ROBERTA_CHECKPOINT_PATH", "")
    if fk_roberta_path:
      import reward_functions as _rw  # type: ignore[import]
      from transformers import RobertaTokenizer, RobertaForSequenceClassification
      LOGGER.info("Pre-loading RoBERTa toxicity classifier from %s", fk_roberta_path)
      _tok = RobertaTokenizer.from_pretrained(fk_roberta_path)
      _clf = RobertaForSequenceClassification.from_pretrained(fk_roberta_path)
      _clf.eval()
      _clf.to(device)
      _rw.MODELS["toxicity"] = {"tokenizer": _tok, "model": _clf}

    generation_start = time.perf_counter()
    results: List[GenerationResult] = []

    if prompts:
      for record in prompts:
        prompt_text: Optional[str] = record.prompt if record.prompt else None
        try:
          output = model.restore_model_and_sample(
            num_steps=generation.sampling_steps,
            prompt_text=prompt_text,
          )
        except Exception as exc:
          LOGGER.error("FK Steering failed for prompt %s: %s", record.prompt_id, exc)
          raise

        best_z = output["best"]  # shape [1, seq_len]
        tokens = best_z[0].tolist()

        # Determine prompt length by encoding
        if prompt_text:
          enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
          prompt_len = enc["input_ids"].shape[1]
        else:
          prompt_len = 0

        completion_tokens, _, _ = _strip_completion_tokens(
          tokens,
          prompt_len,
          stop_ids,
          mask_index,
          stop_sequences=stop_tokens.stop_sequences if stop_tokens else (),
        )
        completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()

        metadata = {
          **record.metadata,
          **shard_metadata,
          "safe_sampling_enabled": False,
          "fk_steering": True,
          "fk_k_particles": k_particles,
          "fk_best_reward": float(output["best_r"]),
        }
        results.append(GenerationResult(
          prompt_id=record.prompt_id,
          prompt=record.prompt,
          completion=completion_text,
          full_text=completion_text,
          token_ids=tokens,
          prompt_length=prompt_len,
          prompt_mask=[1] * prompt_len + [0] * (len(tokens) - prompt_len),
          metadata=metadata,
        ))

    generation_seconds = time.perf_counter() - generation_start
    total_seconds = time.perf_counter() - start_time
    peak_vram = 0
    if torch.cuda.is_available():
      try:
        peak_vram = torch.cuda.max_memory_allocated()
      except RuntimeError:
        pass

    timings = {
      "load_seconds": load_seconds,
      "generation_seconds": generation_seconds,
      "total_seconds": total_seconds,
      "peak_vram_bytes": peak_vram,
      "repellency": {},
    }
    from dataclasses import asdict
    resolved_config = {
      "model": asdict(self.model_settings),
      "generation": asdict(generation),
      "safety": asdict(safety),
      "fk_steering": OmegaConf.to_container(fk_cfg["fk_steering"], resolve=True),
    }
    return GenerationRun(results=results, timings=timings, resolved_config=resolved_config)
