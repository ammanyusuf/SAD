# -*- coding: utf-8 -*-


import argparse
import json
import logging
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import load_file


# ============================================================
# Logger
# ============================================================
def setup_logger(log_path: Optional[Path]):
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def _flush_logs():
    logger = logging.getLogger()
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass


# ============================================================
# JSONL Reader
# ============================================================
def iter_jsonl(path: Path):
    """
    Read jsonl line-by-line and attach _line_no to help locate errors.
    """
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_line_no"] = ln
            yield obj


# ============================================================
# Utils
# ============================================================
def _safe_seed(x: int) -> int:
    return int(x) % (2**63 - 1)


def parse_dtype(s: str) -> torch.dtype:
    """
    Parse command-line string into a torch dtype.
    """
    s = s.lower().strip()
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if s in ["fp16", "float16"]:
        return torch.float16
    if s in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unknown dtype: {s} (use bf16/fp16/fp32)")


def pick_suffix_text(obj: Dict[str, Any], fields: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the first usable suffix text from a prioritized list of fields.

    Supports str / int / float; skips other types.
    Returns (field_name, text). If none found, returns (None, None).
    """
    for k in fields:
        if k not in obj:
            continue
        v = obj.get(k, None)
        if v is None:
            continue

        # phone_number may be int/float
        if isinstance(v, (int, float)):
            text = str(v)
        elif isinstance(v, str):
            text = v
        else:
            continue

        text = text.strip()
        if text:
            return k, text

    return None, None


# ============================================================
# Unified checkpoint parsing utilities (compatible with many .pth/.pt formats)
# ============================================================
def extract_state_dict_from_ckpt(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """
    Support multiple checkpoint layouts:
    - {"model": {...}}
    - {"state_dict": {...}}
    - {"model_state_dict": {...}}
    - Or the object itself is a state_dict (key->tensor)
    """
    if isinstance(ckpt_obj, dict):
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        for k in ("model_state_dict", "model_state"):
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj
    raise TypeError(f"Unrecognized ckpt format: {type(ckpt_obj)}")


def load_state_dict_any(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """
    Unified weight loading:
    - .safetensors: safetensors.torch.load_file
    - Others (.pth/.pt): torch.load + extract_state_dict_from_ckpt
    """
    p = Path(ckpt_path)
    if p.suffix == ".safetensors":
        sd = load_file(str(p))
        # safetensors already returns a state_dict-like dict
        return dict(sd)

    ckpt = torch.load(str(p), map_location="cpu")
    sd = extract_state_dict_from_ckpt(ckpt)
    return sd


# ============================================================
# HF checkpoint loading (core recover logic unchanged; only loading is handled here)
# ============================================================
class HFLogitsWrapper(torch.nn.Module):
    """
    Make forward(input_ids) -> logits Tensor with a unified interface:
    recover code only depends on model(x) returning [B, L, V].
    """

    def __init__(self, hf_model: torch.nn.Module):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, input_ids: torch.Tensor):
        out = self.hf_model(input_ids=input_ids)
        if hasattr(out, "logits") and out.logits is not None:
            return out.logits
        if torch.is_tensor(out):
            return out
        if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
            return out[0]
        raise RuntimeError("HF model forward() did not produce logits.")


def clean_state_dict_keys_hf(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    HF path: strip common prefixes (consistent with the provided reference):
    - hf_model.
    - _forward_module.
    - module.
    """
    keys = list(sd.keys())
    has_hf_model = any(k.startswith("hf_model.") for k in keys)
    has_forward_module = any(k.startswith("_forward_module.") for k in keys)
    has_module = any(k.startswith("module.") for k in keys)

    def strip(k: str) -> str:
        if has_forward_module and k.startswith("_forward_module."):
            k = k[len("_forward_module.") :]
        if has_module and k.startswith("module."):
            k = k[len("module.") :]
        if has_hf_model and k.startswith("hf_model."):
            k = k[len("hf_model.") :]
        return k

    return {strip(k): v for k, v in sd.items()}


def _get_input_embedding_param_name(hf_model: torch.nn.Module) -> Optional[str]:
    """
    Find the state_dict key name of hf_model's input embedding weight.
    This is used to resize embeddings before load_state_dict to avoid shape mismatch.
    """
    try:
        emb_w = hf_model.get_input_embeddings().weight
    except Exception:
        return None

    for name, param in hf_model.named_parameters():
        # Compare parameter object identity
        if param is emb_w:
            return name

    # Some models expose embedding weight as a buffer or hide it; fall back to None.
    return None


def load_hf_model_with_pth_weights(
    hf_model_name: str,
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype,
    mask_id: int,
    trust_remote_code: bool = True,
) -> Tuple[HFLogitsWrapper, Any, int]:
    """
    Load model *structure* from HuggingFace and then load local pth/safetensors weights.

    Returns:
      wrapped_model: HFLogitsWrapper (forward -> logits)
      tokenizer: AutoTokenizer instance
      vocab_size: embedding_rows (final vocab is determined by embedding row count)
    """
    # Only import transformers on HF path (avoid hard dependency when running local lit-gpt only)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # type: ignore

    logging.info("Loading HF config: %s", hf_model_name)
    cfg = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=trust_remote_code)
    hf_vocab_size = int(getattr(cfg, "vocab_size", 0))

    logging.info("Instantiating HF model from_pretrained: %s", hf_model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # Load checkpoint (CPU) and clean keys
    logging.info("Loading checkpoint: %s", ckpt_path)
    sd = load_state_dict_any(ckpt_path)
    sd = clean_state_dict_keys_hf(sd)

    # Key step: resize embeddings before load_state_dict to avoid shape mismatch
    emb_key = _get_input_embedding_param_name(hf_model)
    if emb_key is not None and emb_key in sd and torch.is_tensor(sd[emb_key]):
        ckpt_rows = int(sd[emb_key].shape[0])
        cur_rows = int(hf_model.get_input_embeddings().weight.shape[0])
        if ckpt_rows != cur_rows:
            hf_model.resize_token_embeddings(ckpt_rows)
            logging.info("Resized token embeddings to match ckpt: %d -> %d", cur_rows, ckpt_rows)

    # Ensure mask_id is valid (resize to mask_id+1)
    emb_rows = int(hf_model.get_input_embeddings().weight.shape[0])
    if emb_rows <= int(mask_id):
        hf_model.resize_token_embeddings(int(mask_id) + 1)
        logging.info("Resized token embeddings for mask_id: %d -> %d", emb_rows, int(mask_id) + 1)

    missing, unexpected = hf_model.load_state_dict(sd, strict=False)
    logging.info("load_state_dict(strict=False): missing=%d unexpected=%d", len(missing), len(unexpected))
    if unexpected:
        logging.warning("Unexpected keys head: %s", unexpected[:20])
    if missing:
        logging.warning("Missing keys head: %s", missing[:20])

    hf_model.eval()
    hf_model.to(device=device, dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=trust_remote_code, use_fast=True)

    wrapped = HFLogitsWrapper(hf_model).to(device=device, dtype=dtype)

    vocab_size = int(wrapped.hf_model.get_input_embeddings().weight.shape[0])
    if hf_vocab_size and vocab_size != hf_vocab_size:
        logging.info("HF vocab_size(cfg)=%d, embedding_rows=%d (using embedding_rows)", hf_vocab_size, vocab_size)

    return wrapped, tokenizer, vocab_size


# ============================================================
# Local lit-gpt diffusion model loading 
# ============================================================
def clean_state_dict_keys_local(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Local lit-gpt diffusion path: strip common prefixes
    - _forward_module.
    - module.
    - model.
    - diff_model.
    """
    keys = list(sd.keys())
    has_module = any(k.startswith("module.") for k in keys)
    has_forward_module = any(k.startswith("_forward_module.") for k in keys)
    has_model = any(k.startswith("model.") for k in keys)
    has_diff_model = any(k.startswith("diff_model.") for k in keys)

    def strip(k: str) -> str:
        if has_forward_module and k.startswith("_forward_module."):
            k = k[len("_forward_module.") :]
        if has_module and k.startswith("module."):
            k = k[len("module.") :]
        if has_model and k.startswith("model."):
            k = k[len("model.") :]
        if has_diff_model and k.startswith("diff_model."):
            k = k[len("diff_model.") :]
        return k

    return {strip(k): v for k, v in sd.items()}


def infer_vocab_size_local(model: torch.nn.Module, config: Any) -> Optional[int]:
    """
    Best-effort vocab_size inference (for logging only; does not affect recover logic).
    """
    for attr in ("vocab_size", "padded_vocab_size"):
        v = getattr(config, attr, None)
        if isinstance(v, int) and v > 0:
            return int(v)

    v = getattr(model, "vocab_size", None)
    if isinstance(v, int) and v > 0:
        return int(v)

    for path in (
        ("transformer", "wte"),
        ("model", "embed_tokens"),
        ("tok_embeddings",),
        ("embedding",),
        ("emb",),
    ):
        cur = model
        ok = True
        for p in path:
            if not hasattr(cur, p):
                ok = False
                break
            cur = getattr(cur, p)
        if ok and hasattr(cur, "weight") and torch.is_tensor(cur.weight):
            return int(cur.weight.shape[0])

    return None


def load_local_diff_model(
    lit_model_name: str,
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[nn.Module, Any, Optional[int]]:
    """
    Keep the original local model loading style:
    - Config.from_name("Diff_LLaMA_{lit_model_name}M")
    - TransEncoder(config)
    - strict=False load_state_dict (after cleaning keys)
    """
    # Only import lit_gpt on local path (avoid hard dependency when running HF only)
    from lit_gpt.diffmodel import TransEncoder, Config  # type: ignore

    model_name = f"Diff_LLaMA_{lit_model_name}M"
    config = Config.from_name(model_name)

    model = TransEncoder(config).to(device)

    sd_raw = load_state_dict_any(ckpt_path)
    sd = clean_state_dict_keys_local(sd_raw)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    logging.info("load_state_dict(strict=False): missing=%d unexpected=%d", len(missing), len(unexpected))
    if unexpected:
        logging.warning("Unexpected keys head: %s", unexpected[:20])
    if missing:
        logging.warning("Missing keys head: %s", missing[:20])

    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Align buffer dtypes
    param_dtype = next(model.parameters()).dtype
    for name, buf in model.named_buffers():
        if torch.is_floating_point(buf) and buf.dtype != param_dtype:
            model._buffers[name] = buf.to(dtype=param_dtype)

    vocab_size = infer_vocab_size_local(model, config)
    return model, config, vocab_size


# ============================================================
# Tokenizer wrapper (unified encode interface; supports:
# - lit-gpt tokenizer (tokenizer_path is a lit-gpt tokenizer dir)
# - transformers tokenizer (tokenizer_path or hf_tokenizer instance)
# ============================================================
class TokenizerWrapper:
    """
    Unified encode(text)->List[int], ensuring no automatic special tokens.
    """

    def __init__(self, tokenizer_path: Optional[str] = None, hf_tokenizer: Any = None):
        """
        Choose one:
        - tokenizer_path: load tokenizer from a directory/name (lit-gpt or HF)
        - hf_tokenizer: pass an already-constructed AutoTokenizer instance
        """
        if hf_tokenizer is not None:
            self._impl = hf_tokenizer
            self.name = "transformers.AutoTokenizer(instance)"
            return

        if tokenizer_path is None:
            raise ValueError("TokenizerWrapper requires tokenizer_path or hf_tokenizer.")

        self._impl = None
        self.name = ""

        # 1) Prefer lit-gpt Tokenizer
        try:
            from lit_gpt.tokenizer import Tokenizer as LitTokenizer  # type: ignore

            self._impl = LitTokenizer(tokenizer_path)
            self.name = "lit_gpt.tokenizer.Tokenizer"
            logging.info("Loaded tokenizer via %s from %s", self.name, tokenizer_path)
            return
        except Exception as e:
            logging.warning("lit_gpt Tokenizer load failed, fallback to transformers. err=%s", str(e))

        # 2) Fallback transformers
        try:
            from transformers import AutoTokenizer  # type: ignore

            self._impl = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
            self.name = "transformers.AutoTokenizer"
            logging.info("Loaded tokenizer via %s from %s", self.name, tokenizer_path)
            return
        except Exception as e:
            raise RuntimeError(
                "Failed to load tokenizer. Please provide a valid tokenizer_path for lit-gpt or HF. "
                f"Last err={e}"
            )

    def encode(self, text: str) -> List[int]:
        """
        Core requirement: do not add special tokens (BOS/EOS, etc.).
        """
        if self.name.startswith("lit_gpt"):
            # lit-gpt encode signature differs across versions; handle both
            try:
                return list(self._impl.encode(text, bos=False, eos=False))
            except TypeError:
                return list(self._impl.encode(text))

        # transformers: explicitly disable special tokens
        return list(self._impl.encode(text, add_special_tokens=False))


# ============================================================
# Recover logic (kept exactly as provided)
# ============================================================
def _rand(shape, *, device, dtype, generator: Optional[torch.Generator] = None):
    return torch.rand(shape, device=device, dtype=dtype, generator=generator)


def add_gumbel_noise(logits: torch.Tensor, temperature: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    logits = logits.to(torch.float64)
    noise = _rand(logits.shape, device=logits.device, dtype=torch.float64, generator=generator)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def gt_logprob_under_gumbel_sampling(logits_fp64: torch.Tensor, gt_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Gumbel-max corresponds to sampling distribution softmax(logits/temperature); return GT token logprob.
    """
    if temperature < 0:
        raise ValueError("gumbel_temperature must be >= 0")
    scaled = logits_fp64 / float(temperature)
    log_probs = F.log_softmax(scaled, dim=-1)
    return log_probs.gather(1, gt_ids.unsqueeze(1)).squeeze(1)


def recover_logprob_from_masked_batch(
    model: torch.nn.Module,
    x: torch.Tensor,            # [B, L] masked input (mask_id indicates masked tokens)
    gt: torch.Tensor,           # [B, L] ground truth ids
    mask_id: int,
    steps: int,
    alg: str,                   # "origin" | "greddy"
    temperature: float,
    eps: float,
    cfg_scale: float = 0.0,
    prompt_len: int = 0,
    torch_gen: Optional[torch.Generator] = None,
    debug_log_rank: bool = False,
    debug_log_rank_max_positions: int = 5,
    log_steps: bool = False,
    log_steps_max_samples: int = 1,
    log_steps_label: str = "",
):
    """

    Returns:
      x_out: [B, L] (teacher forcing will restore to gt)
      total_log: [B] float64
      invalid_mask: [B] bool
    """
    if steps <= 0:
        invalid = torch.zeros((x.shape[0],), device=x.device, dtype=torch.bool)
        return x, torch.zeros((x.shape[0],), device=x.device, dtype=torch.float64), invalid

    B, L = x.shape
    device = x.device

    with torch.no_grad():
        num_mask_row = (x == mask_id).sum(dim=1).to(torch.int64)  # [B]
        mask_count_float = num_mask_row.to(torch.float64).mean().item()
        p0 = float(mask_count_float) / float(L) if L > 0 else 0.0
        t0 = (p0 - eps) / (1.0 - eps) if (1.0 - eps) != 0 else 1.0
        if not math.isfinite(t0):
            t0 = 1.0
        t0 = max(min(t0, 1.0), eps)

    timesteps = torch.linspace(t0, eps, steps + 1, device=device)
    total_log = torch.zeros((B,), device=device, dtype=torch.float64)
    invalid_mask = torch.zeros((B,), device=device, dtype=torch.bool)


    use_amp = (device.type == "cuda")
    amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_amp else nullcontext()

    with torch.no_grad():
        for i in range(steps):
            mask_index = (x == mask_id)  # [B, L]
            if int(mask_index.sum().item()) == 0:
                break

            # forward
            with amp_ctx:
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    if prompt_len > 0:
                        un_x[:, :prompt_len] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)  # [2B, L]
                    logits_all = model(x_)            # [2B, L, V]
                    logits_c, logits_u = torch.chunk(logits_all, 2, dim=0)
                    logits_full = logits_u + (cfg_scale + 1.0) * (logits_c - logits_u)  # [B, L, V]
                else:
                    logits_full = model(x)            # [B, L, V]

            t = timesteps[i]
            s = timesteps[i + 1]

            if alg == "origin":
                p_transfer = 1.0 - (s / t).item() if i < steps - 1 else 1.0
                r = _rand((B, L), device=device, dtype=torch.float32, generator=torch_gen)
                transfer_pos_mask = (r < p_transfer) & mask_index  # [B, L]

                b_idx, pos_idx = transfer_pos_mask.nonzero(as_tuple=True)
                if b_idx.numel() > 0:
                    sel_logits = logits_full[b_idx, pos_idx, :].to(torch.float64)  # [N, V]
                    sel_gt = gt[b_idx, pos_idx]                                     # [N]
                    sel_lp = gt_logprob_under_gumbel_sampling(sel_logits, sel_gt, temperature=temperature)

                    finite = torch.isfinite(sel_lp)
                    if not finite.all():
                        bad_b = b_idx[~finite]
                        invalid_mask[bad_b] = True
                        sel_lp = torch.where(finite, sel_lp, torch.zeros_like(sel_lp))

                    total_log.index_add_(0, b_idx, sel_lp)

                    good = finite
                    if good.any():
                        x[b_idx[good], pos_idx[good]] = sel_gt[good]

                    if log_steps:
                        for j in range(min(B, log_steps_max_samples)):
                            if j == 0:
                                sel_pos_j = pos_idx[b_idx == j].detach().cpu().tolist()
                                sel_tok_j = gt[j, sel_pos_j].detach().cpu().tolist() if sel_pos_j else []
                                logging.info(
                                    "%s step=%d origin transfer_count=%d pos=%s token_ids=%s",
                                    log_steps_label, i, len(sel_pos_j), sel_pos_j[:50], sel_tok_j[:50],
                                )

                    if debug_log_rank:
                        for j in range(min(B, log_steps_max_samples)):
                            sel_pos_j = pos_idx[b_idx == j]
                            if sel_pos_j.numel() == 0:
                                continue
                            sel_pos_j = sel_pos_j[: max(0, debug_log_rank_max_positions)]
                            ranks = []
                            for p_ in sel_pos_j.tolist():
                                tid = int(gt[j, p_].item())
                                gt_logit = logits_full[j, p_, tid]
                                rank = int((logits_full[j, p_] > gt_logit).sum().item()) + 1
                                ranks.append(rank)
                            logging.info(
                                "%s step=%d origin ranks(sample=%d) pos=%s ranks=%s",
                                log_steps_label, i, j, sel_pos_j.detach().cpu().tolist(), ranks,
                            )

            elif alg == "greddy":
                logits_masked = logits_full[mask_index]  # [Nmask_total, V]
                logits_with_noise = add_gumbel_noise(logits_masked, temperature=temperature, generator=torch_gen)
                x0_masked = torch.argmax(logits_with_noise, dim=-1)  # [Nmask_total]

                logits_masked_fp64 = logits_masked.to(torch.float64)
                p = F.softmax(logits_masked_fp64, dim=-1)
                confidence_masked = torch.gather(p, dim=-1, index=x0_masked.unsqueeze(-1)).squeeze(-1)

                confidence_full = torch.full((B, L), float("-inf"), device=device, dtype=torch.float64)
                confidence_full[mask_index] = confidence_masked

                num_mask = mask_index.sum(dim=1).to(torch.int64)
                if i < steps - 1:
                    frac = (1.0 - (s / t).item())
                    k = torch.floor(num_mask.to(torch.float64) * float(frac)).to(torch.int64)
                else:
                    k = num_mask

                max_k = int(k.max().item())
                if max_k <= 0:
                    continue

                _, top_pos = torch.topk(confidence_full, k=max_k, dim=1)

                ar = torch.arange(max_k, device=device).unsqueeze(0)
                take = ar < k.unsqueeze(1)

                b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_k)
                sel_b = b_idx[take]
                sel_pos = top_pos[take]

                if sel_b.numel() == 0:
                    continue

                sel_logits = logits_full[sel_b, sel_pos, :].to(torch.float64)
                sel_gt = gt[sel_b, sel_pos]
                sel_lp = gt_logprob_under_gumbel_sampling(sel_logits, sel_gt, temperature=temperature)

                finite = torch.isfinite(sel_lp)
                if not finite.all():
                    bad_b = sel_b[~finite]
                    invalid_mask[bad_b] = True
                    sel_lp = torch.where(finite, sel_lp, torch.zeros_like(sel_lp))

                total_log.index_add_(0, sel_b, sel_lp)

                good = finite
                if good.any():
                    x[sel_b[good], sel_pos[good]] = sel_gt[good]

                if log_steps:
                    for j in range(min(B, log_steps_max_samples)):
                        pos_j = sel_pos[sel_b == j].detach().cpu().tolist()
                        tok_j = gt[j, pos_j].detach().cpu().tolist() if pos_j else []
                        logging.info(
                            "%s step=%d greddy transfer_count=%d pos=%s token_ids=%s",
                            log_steps_label, i, len(pos_j), pos_j[:50], tok_j[:50],
                        )

                if debug_log_rank:
                    for j in range(min(B, log_steps_max_samples)):
                        pos_j = sel_pos[sel_b == j]
                        if pos_j.numel() == 0:
                            continue
                        pos_j = pos_j[: max(0, debug_log_rank_max_positions)]
                        ranks = []
                        for p_ in pos_j.tolist():
                            tid = int(gt[j, p_].item())
                            gt_logit = logits_full[j, p_, tid]
                            rank = int((logits_full[j, p_] > gt_logit).sum().item()) + 1
                            ranks.append(rank)
                        logging.info(
                            "%s step=%d greddy ranks(sample=%d) pos=%s ranks=%s",
                            log_steps_label, i, j, pos_j.detach().cpu().tolist(), ranks,
                        )

            else:
                raise NotImplementedError(f"Unknown alg={alg}")

    total_log = torch.where(invalid_mask, torch.full_like(total_log, float("nan")), total_log)
    return x, total_log, invalid_mask


# ============================================================
# Stats: log_sums -> descriptive stats + p_hat
# ============================================================
def summarize_finite(xs: List[float]) -> Dict[str, float]:
    """
    Compute stats over a list that may include nan/inf: mean/stdev/min/max/count.
    """
    finite = [float(v) for v in xs if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if len(finite) == 0:
        return {"count": 0, "mean": float("nan"), "stdev": float("nan"), "min": float("nan"), "max": float("nan")}
    m = sum(finite) / float(len(finite))
    if len(finite) >= 2:
        var = sum((v - m) ** 2 for v in finite) / float(len(finite) - 1)
        sd = math.sqrt(var)
    else:
        sd = 0.0
    return {"count": int(len(finite)), "mean": float(m), "stdev": float(sd), "min": float(min(finite)), "max": float(max(finite))}


def estimate_p_hat_from_log_sums(log_sums: List[float]) -> Dict[str, float]:
    """
    Monte Carlo estimate p_hat = E[exp(total_log)] over finite samples:
      log_p_hat = logsumexp(log_i) - log(N)
      p_hat = exp(log_p_hat)
    """
    finite = [float(v) for v in log_sums if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if len(finite) == 0:
        return {"p_hat": float("nan"), "log_p_hat": float("nan"), "count": 0}

    t = torch.tensor(finite, dtype=torch.float64)
    log_p_hat = torch.logsumexp(t, dim=0) - math.log(float(t.numel()))
    p_hat = torch.exp(log_p_hat)
    return {"p_hat": float(p_hat.item()), "log_p_hat": float(log_p_hat.item()), "count": int(t.numel())}


# ============================================================
# Core: evaluate one record with fixed prefix+suffix masked recover (teacher forcing)
# ============================================================
def eval_one_record_prefix_email(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    context_text: str,
    email_text: str,  # kept for backward compatibility: actually suffix_text (email/name/phone_number)
    mask_id: int,
    mc_samples: int,
    mc_batch_size: int,
    seed: int,
    recover_steps: int,
    recover_each_token: bool,
    recover_alg: str,
    gen_temperature: float,
    recover_eps: float,
    recover_cfg_scale: float,
    max_seq_len: Optional[int],
    line_no: int,
    save_log_sums: bool,
    debug_log_rank: bool,
    debug_log_rank_max_positions: int,
    log_steps: bool,
    log_steps_max_samples: int,
    suffix_field: str = "email",
) -> Dict[str, Any]:
    """
    Pipeline:
    1) tokenize context_text/suffix_text
    2) concatenate to gt_ids
    3) mask the suffix span -> x0 (fixed mask positions)
    4) choose steps_used (if recover_each_token=True, steps_used=suffix_len)
    5) run MC recover repeatedly and collect total_log
    6) summarize and return
    """
    device = next(model.parameters()).device

    # -------- 1) tokenize: text -> token ids (no special tokens) --------
    ctx_ids = tokenizer.encode(context_text)
    suffix_ids = tokenizer.encode(email_text)

    prompt_len = len(ctx_ids)              # prefix length
    suffix_len = len(suffix_ids)           # suffix token count
    gt_ids = ctx_ids + suffix_ids
    seq_len = len(gt_ids)

    if seq_len <= 0 or suffix_len <= 0:
        raise ValueError(f"tokenize produced empty sequence: seq_len={seq_len}, suffix_len={suffix_len}")

    # -------- 2) Overlength handling: keep full suffix; truncate left side of context if needed --------
    if max_seq_len is not None and seq_len > int(max_seq_len):
        overflow = seq_len - int(max_seq_len)
        if overflow >= prompt_len:
            raise ValueError(
                f"Sequence too long even after dropping all context. seq_len={seq_len}, "
                f"prompt_len={prompt_len}, suffix_len={suffix_len}, max_seq_len={max_seq_len}"
            )
        logging.warning(
            "Line %d: seq_len=%d > max_seq_len=%d, truncate context by %d tokens (keep full suffix).",
            line_no, seq_len, int(max_seq_len), overflow,
        )
        ctx_ids = ctx_ids[overflow:]
        prompt_len = len(ctx_ids)
        gt_ids = ctx_ids + suffix_ids
        seq_len = len(gt_ids)

    # -------- 3) Build tensors: gt + masked_input (mask suffix span only) --------
    gt = torch.tensor(gt_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]
    x0 = gt.clone()
    # suffix span = [prompt_len, seq_len)
    x0[:, prompt_len:] = int(mask_id)

    # -------- 4) Choose steps_used (recover every token if enabled) --------
    if recover_each_token:
        steps_used = max(1, int(suffix_len))
    else:
        steps_used = max(1, int(recover_steps))

    # -------- 5) MC recover runs (same sample, multiple stochastic trajectories) --------
    log_sums: List[float] = []
    invalid_flags: List[bool] = []

    done = 0
    base_seed = _safe_seed(int(seed) + int(line_no) * 1_000_000)

    t0_time = time.time()
    while done < int(mc_samples):
        cur_bs = min(int(mc_batch_size), int(mc_samples - done))

        # Batch semantics: cur_bs independent MC replicates for the same record
        x = x0.repeat(cur_bs, 1)   # [B, L]
        gtb = gt.repeat(cur_bs, 1) # [B, L]

        g = torch.Generator(device=device)
        g.manual_seed(_safe_seed(int(base_seed) + int(done)))

        _, total_log, invalid = recover_logprob_from_masked_batch(
            model=model,
            x=x,
            gt=gtb,
            mask_id=int(mask_id),
            steps=int(steps_used),
            alg=str(recover_alg),
            temperature=float(gen_temperature),
            eps=float(recover_eps),
            cfg_scale=float(recover_cfg_scale),
            prompt_len=int(prompt_len),
            torch_gen=g,
            debug_log_rank=bool(debug_log_rank),
            debug_log_rank_max_positions=int(debug_log_rank_max_positions),
            log_steps=bool(log_steps),
            log_steps_max_samples=int(log_steps_max_samples),
            log_steps_label=f"Line{line_no}/Recover",
        )

        log_sums.extend(total_log.detach().cpu().tolist())
        invalid_flags.extend(invalid.detach().cpu().tolist())
        done += cur_bs

    elapsed = time.time() - t0_time

    # -------- 6) Summaries --------
    stats = summarize_finite(log_sums)
    pstats = estimate_p_hat_from_log_sums(log_sums)

    per_token_mean = float("nan")
    if math.isfinite(stats["mean"]) and suffix_len > 0:
        per_token_mean = float(stats["mean"] / float(suffix_len))

    out: Dict[str, Any] = {
        "line_no": int(line_no),

        # Compatibility + clarity: suffix_field indicates which field provided the suffix
        "prefix_suffix_type": f"prefix(context_text) + suffix({suffix_field})",
        "suffix_field": str(suffix_field),

        "context_text": context_text,
        "email_text": email_text,  # backward-compatible field name: stores suffix text

        "prompt_len": int(prompt_len),
        "email_len": int(suffix_len),  # backward-compatible field name: suffix_len
        "seq_len": int(seq_len),
        "mask_id": int(mask_id),

        "mc_samples": int(mc_samples),
        "mc_batch_size": int(mc_batch_size),

        "recover_steps_arg": int(recover_steps),
        "recover_each_token": bool(recover_each_token),
        "recover_steps_used": int(steps_used),

        "recover_alg": str(recover_alg),
        "gen_temperature": float(gen_temperature),
        "recover_eps": float(recover_eps),
        "recover_cfg_scale": float(recover_cfg_scale),

        "recover_log_sum_stats": stats,
        "recover_log_sum_per_token_mean": per_token_mean,

        "p_hat": pstats["p_hat"],
        "log_p_hat": pstats["log_p_hat"],
        "p_hat_count": int(pstats["count"]),

        "invalid_count": int(sum(1 for v in invalid_flags if bool(v))),
        "elapsed_sec": float(elapsed),
    }

    # Optional: store all MC log_sums (can be large)
    if save_log_sums:
        out["recover_log_sums"] = log_sums
        out["invalid_mask"] = invalid_flags

    # Token/debug info (helps verify mask span correctness)
    out["tokens"] = {
        "context_token_ids": ctx_ids,
        "email_token_ids": suffix_ids,  # backward-compatible field name: suffix token ids
        "gt_token_ids": gt_ids,
        "masked_input_token_ids": (x0.squeeze(0).detach().cpu().tolist()),
        "email_span": [int(prompt_len), int(seq_len)],  # backward-compatible field name: suffix span
        "suffix_field": str(suffix_field),
    }

    return out


# ============================================================
# CLI main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    # ---------- I/O ----------
    parser.add_argument("--input_path", type=Path, required=True, help="jsonl: must contain context_text and suffix field(s)")
    parser.add_argument("--output_path", type=Path, required=True, help="write results jsonl here")
    parser.add_argument("--log_path", type=Path, default=None)

    # ---------- Suffix field compatibility ----------
    parser.add_argument(
        "--suffix_fields",
        type=str,
        default="email,name,phone_number",
        help="Which fields to search for suffix (comma-separated, in priority order). Default: email,name,phone_number",
    )

    # ---------- Model source: HF or local ----------
    parser.add_argument(
        "--hf_model_name",
        type=str,
        default="",
        help="If provided, use HF: load structure from hf_model_name, then load --ckpt_path weights",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to set trust_remote_code for HF models (default False; enable if your model needs custom code)",
    )

    # local lit-gpt diffusion (used when --hf_model_name is not provided)
    parser.add_argument("--lit_model_name", type=str, default="", help="local: e.g. 1028 -> Diff_LLaMA_1028M")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="Required for local path: lit-gpt tokenizer dir or HF tokenizer dir",
    )

    # ---------- Checkpoint path ----------
    parser.add_argument("--ckpt_path", type=str, required=True, help="local pth/pt/safetensors checkpoint")

    # ---------- Mask id ----------
    parser.add_argument("--mask_id", type=int, required=True)

    # ---------- MC ----------
    parser.add_argument("--mc_samples", type=int, default=256, help="how many MC runs per record")
    parser.add_argument("--mc_batch_size", type=int, default=32)

    # ---------- Recover ----------
    parser.add_argument("--recover_steps", type=int, default=10, help="base steps if not using --recover_each_token")
    parser.add_argument(
        "--recover_each_token",
        action="store_true",
        help="Set steps_used = suffix_len per record (recover every token), overriding --recover_steps",
    )
    parser.add_argument("--recover_alg", type=str, default="origin", choices=["origin", "greddy"])
    parser.add_argument("--gen_temperature", type=float, default=1.0)
    parser.add_argument("--recover_eps", type=float, default=1e-3)
    parser.add_argument("--recover_cfg_scale", type=float, default=0.0)

    # ---------- Runtime ----------
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16/fp16/fp32")

    # ---------- Reproducibility ----------
    parser.add_argument("--seed", type=int, default=12345)

    # ---------- Length protection ----------
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=0,
        help=">0: truncate context to fit max_seq_len (keep suffix)",
    )

    # ---------- Output controls ----------
    parser.add_argument("--save_log_sums", action="store_true", help="store full recover_log_sums into output jsonl")

    # ---------- Debug ----------
    parser.add_argument("--debug_log_rank", action="store_true")
    parser.add_argument("--debug_log_rank_max_positions", type=int, default=5)
    parser.add_argument("--log_steps", action="store_true")
    parser.add_argument("--log_steps_max_samples", type=int, default=1)

    args = parser.parse_args()

    setup_logger(args.log_path)
    logging.info("Args: %s", vars(args))

    if not args.input_path.exists():
        raise FileNotFoundError(f"input_path not found: {args.input_path}")

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)

    # ============================================================
    # 1) Load model + tokenizer (based on whether hf_model_name is provided)
    # ============================================================
    hf_model_name = str(args.hf_model_name).strip()
    if hf_model_name:
        # -------- HF path: load structure from HF + load local weights --------
        model, hf_tokenizer, vocab_size = load_hf_model_with_pth_weights(
            hf_model_name=hf_model_name,
            ckpt_path=args.ckpt_path,
            device=device,
            dtype=dtype,
            mask_id=int(args.mask_id),
            trust_remote_code=bool(args.trust_remote_code),
        )
        tokenizer = TokenizerWrapper(hf_tokenizer=hf_tokenizer)
        logging.info("HF model loaded. vocab_size(embedding_rows)=%d", int(vocab_size))
    else:
        # -------- Local path: lit-gpt diffusion TransEncoder --------
        if not args.lit_model_name:
            raise ValueError("When --hf_model_name is empty, you must provide --lit_model_name for local model.")
        if not args.tokenizer_path:
            raise ValueError("When using local model, you must provide --tokenizer_path.")

        model, _config, vocab_size = load_local_diff_model(
            lit_model_name=str(args.lit_model_name),
            ckpt_path=args.ckpt_path,
            device=device,
            dtype=dtype,
        )
        tokenizer = TokenizerWrapper(tokenizer_path=str(args.tokenizer_path))
        logging.info("Local model loaded. vocab_size(inferred)=%s", str(vocab_size))

    # ============================================================
    # 2) Open output file
    # ============================================================
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.output_path, "w", encoding="utf-8")

    max_seq_len = int(args.max_seq_len) if int(args.max_seq_len) > 0 else None

    # Suffix fields priority
    suffix_fields = [s.strip() for s in str(args.suffix_fields).split(",") if s.strip()]
    logging.info("Suffix fields priority: %s", suffix_fields)

    # ============================================================
    # 3) Iterate jsonl and evaluate record-by-record
    # ============================================================
    t_all = time.time()
    written = 0

    for obj in iter_jsonl(args.input_path):
        line_no = int(obj.get("_line_no", -1))

        context_text = obj.get("context_text", None)
        if not isinstance(context_text, str):
            logging.warning("Skip line %d: missing context_text", line_no)
            continue

        suffix_field, suffix_text = pick_suffix_text(obj, suffix_fields)
        if not isinstance(suffix_text, str) or not suffix_text:
            logging.warning("Skip line %d: missing suffix in fields=%s", line_no, suffix_fields)
            continue

        # Backward-compatible variable naming: email_text carries the suffix text
        email_text = suffix_text

        try:
            res = eval_one_record_prefix_email(
                model=model,
                tokenizer=tokenizer,
                context_text=context_text,
                email_text=email_text,
                mask_id=int(args.mask_id),
                mc_samples=int(args.mc_samples),
                mc_batch_size=int(args.mc_batch_size),
                seed=int(args.seed),
                recover_steps=int(args.recover_steps),
                recover_each_token=bool(args.recover_each_token),
                recover_alg=str(args.recover_alg),
                gen_temperature=float(args.gen_temperature),
                recover_eps=float(args.recover_eps),
                recover_cfg_scale=float(args.recover_cfg_scale),
                max_seq_len=max_seq_len,
                line_no=line_no,
                save_log_sums=bool(args.save_log_sums),
                debug_log_rank=bool(args.debug_log_rank),
                debug_log_rank_max_positions=int(args.debug_log_rank_max_positions),
                log_steps=bool(args.log_steps),
                log_steps_max_samples=int(args.log_steps_max_samples),
                suffix_field=str(suffix_field or "unknown"),
            )
        except Exception as e:
            logging.exception("Line %d failed: %s", line_no, str(e))
            # Write a failure record too, to help locate issues
            err_obj = {
                "line_no": line_no,
                "error": str(e),
                "context_text": context_text,
                "email_text": email_text,   # backward-compatible field name: actually suffix_text
                "suffix_field": str(suffix_field or "unknown"),
            }
            out_f.write(json.dumps(err_obj, ensure_ascii=False) + "\n")
            out_f.flush()
            continue

        out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
        out_f.flush()
        written += 1

        if written % 10 == 0:
            logging.info("Progress: written=%d elapsed=%.1fs", written, time.time() - t_all)
            _flush_logs()

    out_f.close()
    logging.info(
        "Done. written=%d elapsed=%.1fs output=%s",
        written,
        time.time() - t_all,
        str(args.output_path),
    )


if __name__ == "__main__":
    main()
