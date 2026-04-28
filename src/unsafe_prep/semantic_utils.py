"""Shared utilities for semantic embedding pooling and provider resolution."""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from functools import lru_cache
from abc import ABC, abstractmethod
from importlib import import_module
from typing import Callable, Optional, Tuple
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

LOGGER = logging.getLogger(__name__)


def masked_mean_pool(
    hidden: torch.Tensor,
    tokens: torch.Tensor,
    pad_id: Optional[int] = None,
    mask_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Mean-pool per-token hidden states over non-pad/non-mask positions.

    Args:
        hidden: [B, L, D] or [B, D]
        tokens: [B, L] LongTensor
        pad_id: token id to exclude
        mask_id: token id to exclude
    Returns:
        pooled: [B, D]
    """
    if hidden.dim() == 2:
        return hidden
    if hidden.dim() != 3:
        raise ValueError(f"hidden must be [B, L, D] or [B, D], got {hidden.shape}")
    mask = torch.ones_like(tokens, dtype=torch.float32, device=hidden.device)
    if pad_id is not None:
        mask = mask * (tokens != pad_id).float()
    if mask_id is not None:
        mask = mask * (tokens != mask_id).float()
    weights = mask.unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1e-12)
    return (hidden * weights).sum(dim=1) / denom


class EmbeddingProvider(ABC):
    @abstractmethod
    def resolve(self) -> Tuple[Optional[Callable[[torch.Tensor], torch.Tensor]], Optional[torch.nn.Module], str]:
        """Return (embed_fn, model, label). embed_fn takes tokens -> embeddings."""
        raise NotImplementedError


class HFEmbeddingProvider(EmbeddingProvider):
    def __init__(self, encoder_name: str, device: torch.device):
        if not encoder_name:
            raise ValueError("HFEmbeddingProvider requires an encoder name/path.")
        self.encoder_name = encoder_name
        self.device = device

    def resolve(self):
        model = AutoModel.from_pretrained(
            self.encoder_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        model.eval()
        return None, model, self.encoder_name


class CallableEmbeddingProvider(EmbeddingProvider):
    def __init__(self, fn_path: str, label: str = "callable"):
        if ":" not in fn_path:
            raise ValueError(f"embed function path '{fn_path}' must be module:function")
        self.fn_path = fn_path
        self.label = label

    def resolve(self):
        mod_name, fn_name = self.fn_path.split(":", 1)
        fn = getattr(import_module(mod_name), fn_name)
        return fn, None, self.label


@lru_cache(maxsize=1)
def _load_mdlm_checkpoint(checkpoint_path: str, model_config_path: str, tokenizer_path: str, device: str = "cuda"):
    from third_party.mdlm.main import _load_from_checkpoint
    import numpy as _np
    import torch.serialization as _serialization
    LOGGER.info("Loading MDLM checkpoint from %s with config %s...", checkpoint_path, model_config_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    config_path = Path(model_config_path)
    config_dir = str(config_path.parent)
    config_name = config_path.stem
    LOGGER.info("Loading MDLM checkpoint from %s with config %s...", checkpoint_path, model_config_path)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir, job_name="mdlm_semantic_cache"):
        cfg = compose(config_name=config_name, overrides=[])
    cfg.eval.checkpoint_path = checkpoint_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    safe_items = [_np.core.multiarray.scalar, _np.dtype]
    if hasattr(_np, "dtypes") and hasattr(_np.dtypes, "Float64DType"):
        safe_items.append(_np.dtypes.Float64DType)
    try:
        ctx = _serialization.safe_globals(safe_items)
    except AttributeError:
        _serialization.add_safe_globals(safe_items)
        ctx = nullcontext()
    with ctx:
        model = _load_from_checkpoint(config=cfg, tokenizer=tokenizer)
    model = model.to(device)
    LOGGER.info("MDLM checkpoint loaded.")
    return model


def _extract_mdlm_embeddings(backbone, input_ids: torch.Tensor, embed_attr: Optional[str]):
    LOGGER.info("Extracting MDLM embeddings using attr '%s'...", embed_attr or "<default>")
    if embed_attr:
        emb_layer = getattr(backbone, embed_attr, None)
        if emb_layer is None:
            raise RuntimeError(f"Backbone missing embed_attr '{embed_attr}'")
    else:
        emb_layer = None
        if hasattr(backbone, "get_input_embeddings"):
            emb_layer = backbone.get_input_embeddings()
        elif hasattr(backbone, "vocab_embed"):
            emb_layer = backbone.vocab_embed
        elif hasattr(backbone, "embedding"):
            emb_layer = backbone.embedding
        elif hasattr(backbone, "word_embeddings"):
            emb_layer = backbone.word_embeddings
    if emb_layer is None:
        raise RuntimeError("Backbone does not expose an embedding layer.")
    LOGGER.info("Using embedding layer: %s", emb_layer)
    return emb_layer(input_ids)


def make_mdlm_embed_fn(checkpoint_path: str, model_config_path: str, tokenizer_path: str, embed_attr: Optional[str] = None) -> Callable[[torch.Tensor], torch.Tensor]:
    @torch.no_grad()
    def _fn(input_ids: torch.Tensor) -> torch.Tensor:
        model = _load_mdlm_checkpoint(checkpoint_path, model_config_path, tokenizer_path, device=str(input_ids.device))
        hidden = _extract_mdlm_embeddings(model.backbone, input_ids, embed_attr)
        pad_id = getattr(model.tokenizer, "pad_token_id", None)
        mask_id = getattr(model, "mask_index", None)
        mask = torch.ones_like(input_ids, dtype=torch.float32, device=input_ids.device)
        if pad_id is not None:
            mask = mask * (input_ids != pad_id).float()
        if mask_id is not None:
            mask = mask * (input_ids != mask_id).float()
        weights = mask.unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1e-12)
        return (hidden * weights).sum(dim=1) / denom

    return _fn


class MDLMEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        fn_path: Optional[str],
        fallback_encoder: Optional[str],
        device: torch.device,
        checkpoint: Optional[str] = None,
        embed_attr: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        model_config_path: Optional[str] = None,
    ):
        self.fn_path = fn_path or os.getenv("MDLM_EMBED_FN")
        self.fallback_encoder = fallback_encoder
        self.device = device
        self.checkpoint = checkpoint or os.getenv("CHECKPOINT_PATH")
        self.embed_attr = embed_attr
        self.tokenizer_path = tokenizer_path or os.getenv("TOKENIZER_PATH")
        self.model_config_path = model_config_path or os.getenv("MODEL_CONFIG_PATH")

    def resolve(self):
        if self.checkpoint:
            if not self.tokenizer_path:
                raise ValueError("MDLM embedding requires tokenizer_path (set TOKENIZER_PATH or pass --tokenizer-path).")
            if not self.model_config_path:
                raise ValueError("MDLM embedding requires model_config_path (set MODEL_CONFIG_PATH or pass --model-config).")
            return make_mdlm_embed_fn(
                self.checkpoint,
                self.model_config_path,
                self.tokenizer_path,
                embed_attr=self.embed_attr,
            ), None, f"mdlm_ckpt:{self.checkpoint}"
        if self.fn_path:
            try:
                mod_name, fn_name = self.fn_path.split(":", 1)
                fn = getattr(import_module(mod_name), fn_name)
                return fn, None, f"mdlm:{self.fn_path}"
            except Exception as exc:
                LOGGER.warning("Failed to import MDLM embed_fn '%s': %s", self.fn_path, exc)
        if self.fallback_encoder:
            model = AutoModel.from_pretrained(self.fallback_encoder).to(self.device)
            model.eval()
            return None, model, self.fallback_encoder
        raise ValueError("MDLM provider requires a valid --mdlm-fn/MDLM_EMBED_FN or explicit fallback encoder.")
