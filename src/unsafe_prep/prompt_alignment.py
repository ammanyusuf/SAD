from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from .schemas import RawUnsafeRecord
from .registry import get_adapter

LOGGER = logging.getLogger(__name__)


def _device() -> torch.device:
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mean_pool(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
  mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
  summed = torch.sum(token_embeddings * mask, dim=1)
  counts = torch.clamp(mask.sum(dim=1), min=1e-9)
  return summed / counts


def _normalize(vecs: Tensor) -> Tensor:
  return torch.nn.functional.normalize(vecs, p=2, dim=-1)


def _encode_texts(texts: Sequence[str], model_name: str, batch_size: int = 16) -> Tensor:
  device = _device()
  resolved_name = _resolve_encoder_path(model_name)
  tokenizer = AutoTokenizer.from_pretrained(resolved_name, local_files_only=True)
  # Ensure padding works for models without a pad token (e.g., GPT2-family)
  if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
      tokenizer.pad_token = tokenizer.eos_token
    else:
      tokenizer.add_special_tokens({"pad_token": tokenizer.unk_token or "<pad>"})
  model = AutoModel.from_pretrained(resolved_name, local_files_only=True).to(device)
  model.eval()
  outputs: List[Tensor] = []
  with torch.no_grad():
    for start in range(0, len(texts), batch_size):
      batch_texts = texts[start : start + batch_size]
      encoded = tokenizer(
          batch_texts,
          padding=True,
          truncation=True,
          return_tensors="pt",
      ).to(device)
      hidden = model(**encoded).last_hidden_state
      pooled = _mean_pool(hidden, encoded.attention_mask)
      outputs.append(pooled.cpu())
  if not outputs:
    return torch.empty(0, 0)
  return _normalize(torch.cat(outputs, dim=0))


def _cosine_topk(queries: Tensor, keys: Tensor, k: int) -> Tuple[Tensor, Tensor]:
  if queries.numel() == 0 or keys.numel() == 0:
    return torch.empty(0, dtype=torch.long), torch.empty(0)
  sims = torch.matmul(queries, keys.T)
  values, indices = torch.topk(sims, k=min(k, keys.size(0)), dim=-1)
  return indices, values


def _get_prompt_text(meta: Optional[Dict]) -> Optional[str]:
  if not meta:
    return None
  prompt = meta.get("prompt_text") or meta.get("prompt")
  if isinstance(prompt, dict):
    prompt = prompt.get("text") or prompt.get("prompt") or prompt.get("content")
  if isinstance(prompt, str) and prompt.strip():
    return prompt.strip()
  return None


def _resolve_encoder_path(model_name: str) -> str:
  LOGGER.debug("Resolving encoder model path for '%s'", model_name)
  candidate = os.environ.get("ALIGN_ENCODER_PATH")
  if candidate and os.path.exists(candidate):
    return candidate
  if os.path.isabs(model_name) and os.path.exists(model_name):
    return model_name
  cache_root = os.environ.get("HF_MODELS_CACHE") or os.environ.get("TRANSFORMERS_CACHE")
  if cache_root:
    cache_candidate = os.path.join(cache_root, model_name)
    if os.path.exists(cache_candidate):
      return cache_candidate
  return model_name


@dataclass
class PromptAligner:
  encoder_name: str = "gpt2-large"
  prompt_source: str = "real-toxicity-prompts"
  prompt_limit: Optional[int] = None

  def __post_init__(self) -> None:
    self._prompt_ids: List[str] = []
    self._prompt_texts: List[str] = []
    self._prompt_embs: Optional[Tensor] = None

  def _load_prompt_pool(self) -> None:
    adapter = get_adapter(self.prompt_source)
    collected: List[Tuple[str, str]] = []
    for idx, rec in enumerate(adapter.iter_unsafe_answers()):  # type: ignore[attr-defined]
      prompt_text = _get_prompt_text(rec.meta)
      if not prompt_text:
        continue
      pid = rec.meta.get("prompt_id") if rec.meta else str(idx)
      collected.append((str(pid), prompt_text))
      if self.prompt_limit is not None and len(collected) >= self.prompt_limit:
        break
    if not collected:
      raise RuntimeError(f"No prompts found for source '{self.prompt_source}'")
    self._prompt_ids = [p[0] for p in collected]
    self._prompt_texts = [p[1] for p in collected]
    LOGGER.info("Loaded %d prompts from %s", len(self._prompt_texts), self.prompt_source)

  def _ensure_embs(self) -> None:
    if self._prompt_embs is not None:
      return
    self._load_prompt_pool()
    LOGGER.info("Encoding %d prompts with %s", len(self._prompt_texts), self.encoder_name)
    self._prompt_embs = _encode_texts(self._prompt_texts, self.encoder_name)
    if self._prompt_embs.numel() == 0:
      raise RuntimeError("Failed to encode prompt pool; empty embeddings.")

  def align_records(self, records: Sequence[RawUnsafeRecord], k: int = 1, source_prefix: str = "") -> List[RawUnsafeRecord]:
    if not records:
      return []
    self._ensure_embs()
    query_texts = []
    for rec in records:
      query_texts.append(_get_prompt_text(rec.meta) or rec.answer_text[:512])
    query_embs = _encode_texts(query_texts, self.encoder_name)
    idxs, _ = _cosine_topk(query_embs, self._prompt_embs, k)
    aligned: List[RawUnsafeRecord] = []
    for i, rec in enumerate(records):
      if idxs.numel() == 0:
        continue
      top_idx = idxs[i, 0].item()
      knn_prompt = self._prompt_texts[top_idx]
      knn_id = self._prompt_ids[top_idx]
      meta = dict(rec.meta or {})
      original_prompt = _get_prompt_text(rec.meta)
      meta["original_prompt"] = original_prompt
      meta["knn_owt_prompt"] = knn_prompt
      meta["knn_owt_id"] = knn_id
      # Replace prompt fields so downstream consumers (append_prompt, etc.) see the aligned prompt.
      meta["prompt_text"] = knn_prompt
      meta["prompt"] = knn_prompt
      aligned.append(
          RawUnsafeRecord(
              source=f"{source_prefix or rec.source}-knn-owt",
              category=rec.category,
              answer_text=rec.answer_text,
              toxicity_score=rec.toxicity_score,
              meta=meta,
          )
      )
    return aligned
