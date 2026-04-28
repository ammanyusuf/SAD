"""Lightweight helpers to extract sequence embeddings from MDLM checkpoints."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from third_party.mdlm.main import _load_from_checkpoint


@lru_cache(maxsize=1)
def _load_diffusion(checkpoint_path: str, model_config_path: str, tokenizer_path: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    cfg = OmegaConf.load(model_config_path)
    cfg.eval.checkpoint_path = checkpoint_path
    model = _load_from_checkpoint(config=cfg, tokenizer=tokenizer)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def embed_from_checkpoint(
    input_ids: torch.Tensor,
    checkpoint_path: Optional[str] = None,
    model_config_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    pad_id: Optional[int] = None,
    mask_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Produce pooled sequence embeddings using the MDLM model's input embedding table.

    Args:
        input_ids: LongTensor [B, L]
        checkpoint_path: path to the MDLM Lightning checkpoint (or env CHECKPOINT_PATH)
        model_config_path: path to the MDLM Hydra config (or env MODEL_CONFIG_PATH)
        tokenizer_path: path/name for tokenizer (or env TOKENIZER_PATH)
        pad_id: optional pad token id to exclude from pooling
        mask_id: optional mask token id to exclude from pooling
    Returns:
        pooled: FloatTensor [B, D]
    """
    checkpoint_path = checkpoint_path or os.getenv("CHECKPOINT_PATH")
    model_config_path = model_config_path or os.getenv("MODEL_CONFIG_PATH")
    tokenizer_path = tokenizer_path or os.getenv("TOKENIZER_PATH")
    if not checkpoint_path or not model_config_path or not tokenizer_path:
        raise ValueError("embed_from_checkpoint requires checkpoint_path, model_config_path, and tokenizer_path (or corresponding env vars).")

    model = _load_diffusion(checkpoint_path, model_config_path, tokenizer_path, device=str(input_ids.device))
    # Try to use the backbone's input embeddings; fall back to vocab_embed if needed.
    emb_layer = getattr(model.backbone, "get_input_embeddings", None)
    if callable(emb_layer):
        emb_layer = emb_layer()
    else:
        emb_layer = getattr(model.backbone, "vocab_embed", None)
    if emb_layer is None:
        raise RuntimeError("Backbone does not expose input embeddings (get_input_embeddings or vocab_embed).")
    hidden = emb_layer(input_ids)  # [B, L, D]

    mask = torch.ones_like(input_ids, dtype=torch.float32, device=input_ids.device)
    if pad_id is None:
        pad_id = getattr(model.tokenizer, "pad_token_id", None)
    if mask_id is None:
        mask_id = getattr(model, "mask_index", None)
    if pad_id is not None:
        mask = mask * (input_ids != pad_id).float()
    if mask_id is not None:
        mask = mask * (input_ids != mask_id).float()
    weights = mask.unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1e-12)
    pooled = (hidden * weights).sum(dim=1) / denom
    return pooled
