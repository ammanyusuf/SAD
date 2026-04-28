import argparse
import json
import logging
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from lit_gpt.diffmodel import TransEncoder, Config
from safetensors.torch import load_file


# =========================
# Logging
# =========================
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


# =========================
# JSONL Reader
# =========================
def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_line_no"] = ln
            yield obj


# =========================
# Checkpoint I/O
# =========================
def extract_state_dict_from_ckpt(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
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
    Load a checkpoint and return a raw state_dict (key normalization is done by the caller).
    """
    p = Path(ckpt_path)
    if p.suffix == ".safetensors":
        return load_file(str(p))
    ckpt = torch.load(str(p), map_location="cpu")
    return extract_state_dict_from_ckpt(ckpt)


# =========================
# Local lit-gpt checkpoint normalization
# =========================
def clean_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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


def load_state_dict_local(ckpt_path: str) -> Dict[str, torch.Tensor]:
    sd = load_state_dict_any(ckpt_path)
    return clean_state_dict_keys(sd)


def infer_vocab_size(model: torch.nn.Module, config: Any) -> Optional[int]:
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
    model_name = f"Diff_LLaMA_{lit_model_name}M"
    config = Config.from_name(model_name)

    model = TransEncoder(config).to(device)
    sd = load_state_dict_local(ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    logging.info("load_state_dict(strict=False): missing=%d unexpected=%d", len(missing), len(unexpected))
    if unexpected:
        logging.warning("Unexpected keys head: %s", unexpected[:20])
    if missing:
        logging.warning("Missing keys head: %s", missing[:20])

    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Align buffer dtypes to parameter dtype.
    param_dtype = next(model.parameters()).dtype
    for name, buf in model.named_buffers():
        if torch.is_floating_point(buf) and buf.dtype != param_dtype:
            model._buffers[name] = buf.to(dtype=param_dtype)

    vocab_size = infer_vocab_size(model, config)
    return model, config, vocab_size


# =========================
# HF checkpoint loading
# =========================
class HFLogitsWrapper(torch.nn.Module):
    """
    Adapter that normalizes HF forward(input_ids) to return logits [B, L, V].
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
    Strip common HF-related prefixes:
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
    Return the state_dict key for the input embedding weight, if discoverable.
    """
    try:
        emb_w = hf_model.get_input_embeddings().weight
    except Exception:
        return None

    for name, param in hf_model.named_parameters():
        if param is emb_w:
            return name
    return None


def load_hf_model_with_pth_weights(
    hf_model_name: str,
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype,
    mask_id: int,
    trust_remote_code: bool = True,
) -> Tuple[HFLogitsWrapper, Any, int]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # type: ignore

    logging.info("Loading HF config: %s", hf_model_name)
    cfg = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=trust_remote_code)
    hf_vocab_size = int(getattr(cfg, "vocab_size", 0))

    logging.info("Instantiating HF model: %s", hf_model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    logging.info("Loading checkpoint: %s", ckpt_path)
    sd = load_state_dict_any(ckpt_path)
    sd = clean_state_dict_keys_hf(sd)

    # Resize embeddings before loading if checkpoint and model vocab sizes differ.
    emb_key = _get_input_embedding_param_name(hf_model)
    if emb_key is not None and emb_key in sd and torch.is_tensor(sd[emb_key]):
        ckpt_rows = int(sd[emb_key].shape[0])
        cur_rows = int(hf_model.get_input_embeddings().weight.shape[0])
        if ckpt_rows != cur_rows:
            hf_model.resize_token_embeddings(ckpt_rows)
            logging.info("Resized token embeddings to match ckpt: %d -> %d", cur_rows, ckpt_rows)

    # Ensure mask_id is in range.
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


# =========================
# Misc utilities
# =========================
def _safe_seed(x: int) -> int:
    return int(x) % (2**63 - 1)


def parse_dtype(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if s in ["fp16", "float16"]:
        return torch.float16
    if s in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unknown dtype: {s} (use bf16/fp16/fp32)")


def build_linear_k_schedule(mask_count: int, steps: int) -> List[int]:
    """
    Distribute remaining masked tokens across steps approximately uniformly:
      k_t = ceil(remaining / steps_left)
    """
    if mask_count <= 0:
        return []
    steps_eff = max(1, min(int(steps), int(mask_count)))
    rem = int(mask_count)
    out: List[int] = []
    for t in range(steps_eff):
        left = steps_eff - t
        k = int(math.ceil(rem / float(left)))
        k = max(0, min(k, rem))
        out.append(k)
        rem -= k
        if rem <= 0:
            break
    while len(out) < steps_eff:
        out.append(0)
    return out


def _sanitize_name(s: str) -> str:
    s = str(s)
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:200]


def _token_from_id(tokenizer: Optional[Any], idx: int) -> str:
    if tokenizer is None:
        return f"<id:{int(idx)}>"
    try:
        tok = tokenizer.convert_ids_to_tokens(int(idx))
        if tok is None:
            return f"<id:{int(idx)}>"
        return str(tok)
    except Exception:
        return f"<id:{int(idx)}>"


def _tokens_from_ids(tokenizer: Optional[Any], ids: Union[List[int], torch.Tensor]) -> List[str]:
    if torch.is_tensor(ids):
        ids_list = ids.detach().to("cpu").tolist()
    else:
        ids_list = list(ids)
    return [_token_from_id(tokenizer, int(i)) for i in ids_list]


# =========================
# Gumbel-max sampling
# =========================
def sample_gumbel_argmax(
    logits: torch.Tensor,  # [..., V]
    temperature: float,
    torch_gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample via Gumbel-max:
      g ~ -log(-log(U))
      argmax(logits + temperature * g)
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    if torch_gen is None:
        u = torch.rand_like(logits, dtype=torch.float32)
    else:
        u = torch.rand(
            logits.shape,
            generator=torch_gen,
            device=logits.device,
            dtype=torch.float32,
        )
    u = u.clamp_(1e-12, 1.0 - 1e-12)

    g = -torch.log(-torch.log(u))
    scores = logits.to(torch.float32) + float(temperature) * g
    return torch.argmax(scores, dim=-1).to(torch.long)


# ============================================================
# One-step generation on fixed mask positions (hit counting)
# ============================================================
@torch.inference_mode()
def step1_generate_fixedmask_hits(
    model: nn.Module,
    x_masked: torch.Tensor,   # [B, L] already masked
    gt: torch.Tensor,         # [B, L]
    mask_id: int,
    torch_gen: torch.Generator,
    temperature: float,
    cfg_scale: float,
    prompt_len_for_cfg: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One-step generation for all masked positions in x_masked.

    Returns:
      hit_count_per_row: [B]
      exact_flags:       [B]
    """
    device = x_masked.device
    B, L = x_masked.shape

    mask_index = (x_masked == int(mask_id))
    mask_count_per_row = mask_index.sum(dim=1)
    max_mask = int(mask_count_per_row.max().item())
    if max_mask <= 0:
        hit = torch.zeros((B,), dtype=torch.long, device=device)
        exact = torch.ones((B,), dtype=torch.bool, device=device)
        return hit, exact

    idx = torch.zeros((B, max_mask), dtype=torch.long, device=device)
    for b in range(B):
        pos = torch.nonzero(mask_index[b], as_tuple=False).squeeze(-1)
        n = int(pos.numel())
        if n > 0:
            idx[b, :n] = pos

    model_dtype = next(model.parameters()).dtype
    use_amp = (device.type == "cuda") and (model_dtype in (torch.float16, torch.bfloat16))
    amp_ctx = torch.cuda.amp.autocast(dtype=model_dtype) if use_amp else nullcontext()

    with amp_ctx:
        if cfg_scale is not None and float(cfg_scale) > 0.0:
            un_x = x_masked.clone()
            if prompt_len_for_cfg > 0:
                un_x[:, : int(prompt_len_for_cfg)] = int(mask_id)
            x_ = torch.cat([x_masked, un_x], dim=0)
            logits_all = model(x_)
            logits_c, logits_u = torch.chunk(logits_all, 2, dim=0)
            logits_full = logits_u + (float(cfg_scale) + 1.0) * (logits_c - logits_u)
        else:
            logits_full = model(x_masked)

    b_ar = torch.arange(B, device=device).unsqueeze(1).expand(B, max_mask)
    logits_sel = logits_full[b_ar, idx, :]
    sampled = sample_gumbel_argmax(logits_sel, temperature=float(temperature), torch_gen=torch_gen)
    gt_sel = gt.gather(1, idx)

    use_mask = torch.arange(max_mask, device=device).unsqueeze(0) < mask_count_per_row.unsqueeze(1)
    correct = ((sampled == gt_sel) & use_mask).to(torch.long)
    hit_count = correct.sum(dim=1)
    exact = (hit_count == mask_count_per_row)
    return hit_count, exact


# ============================================================
# One-step generation on fixed mask positions (return filled sequences)
# ============================================================
@torch.inference_mode()
def step1_generate_fixedmask_full(
    model: nn.Module,
    x_masked: torch.Tensor,   # [B, L] already masked
    mask_id: int,
    torch_gen: torch.Generator,
    temperature: float,
    cfg_scale: float,
    prompt_len_for_cfg: int,
) -> torch.Tensor:
    """
    Same sampling as step1_generate_fixedmask_hits, but returns the completed sequence.
    """
    device = x_masked.device
    B, L = x_masked.shape

    mask_index = (x_masked == int(mask_id))
    mask_count_per_row = mask_index.sum(dim=1)
    max_mask = int(mask_count_per_row.max().item())
    if max_mask <= 0:
        return x_masked

    idx = torch.zeros((B, max_mask), dtype=torch.long, device=device)
    for b in range(B):
        pos = torch.nonzero(mask_index[b], as_tuple=False).squeeze(-1)
        n = int(pos.numel())
        if n > 0:
            idx[b, :n] = pos

    model_dtype = next(model.parameters()).dtype
    use_amp = (device.type == "cuda") and (model_dtype in (torch.float16, torch.bfloat16))
    amp_ctx = torch.cuda.amp.autocast(dtype=model_dtype) if use_amp else nullcontext()

    with amp_ctx:
        if cfg_scale is not None and float(cfg_scale) > 0.0:
            un_x = x_masked.clone()
            if prompt_len_for_cfg > 0:
                un_x[:, : int(prompt_len_for_cfg)] = int(mask_id)
            x_ = torch.cat([x_masked, un_x], dim=0)
            logits_all = model(x_)
            logits_c, logits_u = torch.chunk(logits_all, 2, dim=0)
            logits_full = logits_u + (float(cfg_scale) + 1.0) * (logits_c - logits_u)
        else:
            logits_full = model(x_masked)

    b_ar = torch.arange(B, device=device).unsqueeze(1).expand(B, max_mask)
    logits_sel = logits_full[b_ar, idx, :]
    sampled = sample_gumbel_argmax(logits_sel, temperature=float(temperature), torch_gen=torch_gen)

    use_mask = torch.arange(max_mask, device=device).unsqueeze(0) < mask_count_per_row.unsqueeze(1)
    bb = b_ar[use_mask]
    pp = idx[use_mask]
    tt = sampled[use_mask]

    x_out = x_masked.clone()
    x_out[bb, pp] = tt
    return x_out


# ============================================================
# Multi-step generation on fixed mask positions (linear schedule)
# ============================================================
@torch.inference_mode()
def free_generate_fixedmask_multistep_linear(
    model: nn.Module,
    x: torch.Tensor,  # [B, L] (already masked)
    mask_id: int,
    steps: int,
    torch_gen: torch.Generator,
    temperature: float,
    cfg_scale: float,
    prompt_len_for_cfg: int,
) -> torch.Tensor:
    """
    Multi-step generation with fixed mask positions:
      - At each internal step, select random positions among remaining masks.
      - Number of filled positions follows a linear schedule.
      - Sampling uses Gumbel-max without top-k.
    """
    if steps <= 0:
        return x

    device = x.device
    B, L = x.shape

    mask_index0 = (x == int(mask_id))
    mask_count = int(mask_index0.sum(dim=1).max().item())
    if mask_count <= 0:
        return x

    steps_eff = max(1, min(int(steps), int(mask_count)))
    k_schedule = build_linear_k_schedule(mask_count=mask_count, steps=steps_eff)

    model_dtype = next(model.parameters()).dtype
    use_amp = (device.type == "cuda") and (model_dtype in (torch.float16, torch.bfloat16))
    amp_ctx = torch.cuda.amp.autocast(dtype=model_dtype) if use_amp else nullcontext()

    for k_step in k_schedule:
        if k_step <= 0:
            continue

        mask_index = (x == int(mask_id))
        remaining = mask_index.sum(dim=1)
        max_rem = int(remaining.max().item())
        if max_rem <= 0:
            break

        k_sel = min(int(k_step), int(max_rem))

        rand_scores = torch.rand((B, L), generator=torch_gen, device=device, dtype=torch.float32)
        rand_scores = torch.where(mask_index, rand_scores, torch.full_like(rand_scores, -torch.inf))
        _, select_pos = torch.topk(rand_scores, k=k_sel, dim=1, largest=True, sorted=False)

        with amp_ctx:
            if cfg_scale is not None and float(cfg_scale) > 0.0:
                un_x = x.clone()
                if prompt_len_for_cfg > 0:
                    un_x[:, : int(prompt_len_for_cfg)] = int(mask_id)
                x_ = torch.cat([x, un_x], dim=0)
                logits_all = model(x_)
                logits_c, logits_u = torch.chunk(logits_all, 2, dim=0)
                logits_full = logits_u + (float(cfg_scale) + 1.0) * (logits_c - logits_u)
            else:
                logits_full = model(x)

        b_ar = torch.arange(B, device=device).unsqueeze(1).expand(B, k_sel)
        logits_sel = logits_full[b_ar, select_pos, :]
        sampled = sample_gumbel_argmax(logits_sel, temperature=float(temperature), torch_gen=torch_gen)

        use_mask = torch.arange(k_sel, device=device).unsqueeze(0) < remaining.unsqueeze(1)
        if use_mask.any():
            bb = b_ar[use_mask]
            pp = select_pos[use_mask]
            tt = sampled[use_mask]
            x[bb, pp] = tt

    return x


# ============================================================
# Multi-step generation with intermediate capture (for dumps)
# ============================================================
@torch.inference_mode()
def free_generate_fixedmask_multistep_linear_capture(
    model: nn.Module,
    x: torch.Tensor,  # [B, L]
    mask_id: int,
    steps: int,
    torch_gen: torch.Generator,
    temperature: float,
    cfg_scale: float,
    prompt_len_for_cfg: int,
    mask_positions: torch.Tensor,  # [M]
    capture_full_sequence: bool = False,
) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """
    Same as free_generate_fixedmask_multistep_linear, but captures intermediate outputs:
      - inter_mask_ids_steps: list of [B, M] tensors after each internal step
      - inter_full_steps: optional list of [B, L] tensors after each internal step
    """
    if steps <= 0:
        return x, [], [] if capture_full_sequence else None

    device = x.device
    B, L = x.shape

    mask_index0 = (x == int(mask_id))
    mask_count = int(mask_index0.sum(dim=1).max().item())
    if mask_count <= 0:
        return x, [], [] if capture_full_sequence else None

    steps_eff = max(1, min(int(steps), int(mask_count)))
    k_schedule = build_linear_k_schedule(mask_count=mask_count, steps=steps_eff)

    model_dtype = next(model.parameters()).dtype
    use_amp = (device.type == "cuda") and (model_dtype in (torch.float16, torch.bfloat16))
    amp_ctx = torch.cuda.amp.autocast(dtype=model_dtype) if use_amp else nullcontext()

    inter_mask_ids_steps: List[torch.Tensor] = []
    inter_full_steps: Optional[List[torch.Tensor]] = [] if capture_full_sequence else None

    for k_step in k_schedule:
        if k_step <= 0:
            continue

        mask_index = (x == int(mask_id))
        remaining = mask_index.sum(dim=1)
        max_rem = int(remaining.max().item())
        if max_rem <= 0:
            break

        k_sel = min(int(k_step), int(max_rem))

        rand_scores = torch.rand((B, L), generator=torch_gen, device=device, dtype=torch.float32)
        rand_scores = torch.where(mask_index, rand_scores, torch.full_like(rand_scores, -torch.inf))
        _, select_pos = torch.topk(rand_scores, k=k_sel, dim=1, largest=True, sorted=False)

        with amp_ctx:
            if cfg_scale is not None and float(cfg_scale) > 0.0:
                un_x = x.clone()
                if prompt_len_for_cfg > 0:
                    un_x[:, : int(prompt_len_for_cfg)] = int(mask_id)
                x_ = torch.cat([x, un_x], dim=0)
                logits_all = model(x_)
                logits_c, logits_u = torch.chunk(logits_all, 2, dim=0)
                logits_full = logits_u + (float(cfg_scale) + 1.0) * (logits_c - logits_u)
            else:
                logits_full = model(x)

        b_ar = torch.arange(B, device=device).unsqueeze(1).expand(B, k_sel)
        logits_sel = logits_full[b_ar, select_pos, :]
        sampled = sample_gumbel_argmax(logits_sel, temperature=float(temperature), torch_gen=torch_gen)

        use_mask = torch.arange(k_sel, device=device).unsqueeze(0) < remaining.unsqueeze(1)
        if use_mask.any():
            bb = b_ar[use_mask]
            pp = select_pos[use_mask]
            tt = sampled[use_mask]
            x[bb, pp] = tt

        inter_mask_ids_steps.append(x.index_select(1, mask_positions).detach().clone())
        if inter_full_steps is not None:
            inter_full_steps.append(x.detach().clone())

    return x, inter_mask_ids_steps, inter_full_steps


# ============================================================
# Generation dump writer
# ============================================================
class GenDumpWriter:
    def __init__(
        self,
        root_dir: Path,
        tokenizer: Optional[Any],
        dump_full_sequence: bool,
    ):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.dump_full_sequence = bool(dump_full_sequence)

    def make_traj_dir(
        self,
        line_no: Any,
        index: Any,
        traj_id: Any,
        step_key: str,
        steps_eff: int,
    ) -> Path:
        name = f"line{_sanitize_name(line_no)}_idx{_sanitize_name(index)}_traj{_sanitize_name(traj_id)}"
        d = self.root_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_meta_once(self, traj_dir: Path, meta: Dict[str, Any]):
        meta_path = traj_dir / "meta.json"
        if meta_path.exists():
            return
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def append_step_runs(
        self,
        traj_dir: Path,
        step_key: str,
        steps_eff: int,
        records: List[Dict[str, Any]],
    ):
        fn = f"gen_step_{_sanitize_name(step_key)}_eff{int(steps_eff)}.jsonl"
        outp = traj_dir / fn
        with open(outp, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================
# Per-step evaluation (with optional per-run dumps)
# ============================================================
def estimate_hits_for_step(
    model: nn.Module,
    gt_ids: List[int],
    masked_input_ids: List[int],
    mask_id: int,
    step_key: str,
    steps_eff: int,
    runs: int,
    batch_size: int,
    base_seed: int,
    temperature: float,
    cfg_scale: float,
    dump_writer: Optional[GenDumpWriter] = None,
    traj_dir: Optional[Path] = None,
) -> Tuple[List[int], float, float]:
    """
    Run repeated generations from a fixed masked_input_ids and report:
      - hit_list: per-run number of correctly recovered masked tokens
      - exact_rate: fraction of runs with all masked tokens correct
      - mean_hit: average hit count

    If dump_writer is provided, per-run predictions are written under traj_dir.
    """
    assert len(gt_ids) == len(masked_input_ids)
    L = len(gt_ids)

    device = next(model.parameters()).device
    gt_1 = torch.tensor(gt_ids, dtype=torch.long, device=device).unsqueeze(0)
    x0_1 = torch.tensor(masked_input_ids, dtype=torch.long, device=device).unsqueeze(0)

    mask_positions = torch.nonzero((x0_1[0] == int(mask_id)), as_tuple=False).squeeze(-1)
    mask_count = int(mask_positions.numel())
    if mask_count == 0:
        return [0 for _ in range(runs)], 1.0, 0.0

    prompt_len_for_cfg = int(L - mask_count)

    hit_list: List[int] = []
    exact_hits = 0
    done = 0

    while done < runs:
        cur_bs = min(int(batch_size), int(runs - done))

        x = x0_1.repeat(cur_bs, 1)
        gt = gt_1.repeat(cur_bs, 1)

        g = torch.Generator(device=device)
        batch_seed = _safe_seed(int(base_seed) + int(done))
        g.manual_seed(batch_seed)

        if int(steps_eff) == 1:
            if dump_writer is None:
                hit_count, exact_flags = step1_generate_fixedmask_hits(
                    model=model,
                    x_masked=x,
                    gt=gt,
                    mask_id=int(mask_id),
                    torch_gen=g,
                    temperature=float(temperature),
                    cfg_scale=float(cfg_scale),
                    prompt_len_for_cfg=int(prompt_len_for_cfg),
                )
                x_gen = None
                inter_mask_ids_steps = None
                inter_full_steps = None
            else:
                x_gen = step1_generate_fixedmask_full(
                    model=model,
                    x_masked=x,
                    mask_id=int(mask_id),
                    torch_gen=g,
                    temperature=float(temperature),
                    cfg_scale=float(cfg_scale),
                    prompt_len_for_cfg=int(prompt_len_for_cfg),
                )
                hit_count = (
                    (x_gen.index_select(1, mask_positions) == gt.index_select(1, mask_positions))
                    .to(torch.long)
                    .sum(dim=1)
                )
                exact_flags = (hit_count == int(mask_count))
                inter_mask_ids_steps = [x_gen.index_select(1, mask_positions).detach().clone()]
                inter_full_steps = [x_gen.detach().clone()] if dump_writer.dump_full_sequence else None
        else:
            if dump_writer is None:
                x_gen = free_generate_fixedmask_multistep_linear(
                    model=model,
                    x=x,
                    mask_id=int(mask_id),
                    steps=int(steps_eff),
                    torch_gen=g,
                    temperature=float(temperature),
                    cfg_scale=float(cfg_scale),
                    prompt_len_for_cfg=int(prompt_len_for_cfg),
                )
                hit_count = (
                    (x_gen.index_select(1, mask_positions) == gt.index_select(1, mask_positions))
                    .to(torch.long)
                    .sum(dim=1)
                )
                exact_flags = (hit_count == int(mask_count))
                inter_mask_ids_steps = None
                inter_full_steps = None
            else:
                x_gen, inter_mask_ids_steps, inter_full_steps = free_generate_fixedmask_multistep_linear_capture(
                    model=model,
                    x=x,
                    mask_id=int(mask_id),
                    steps=int(steps_eff),
                    torch_gen=g,
                    temperature=float(temperature),
                    cfg_scale=float(cfg_scale),
                    prompt_len_for_cfg=int(prompt_len_for_cfg),
                    mask_positions=mask_positions,
                    capture_full_sequence=bool(dump_writer.dump_full_sequence),
                )
                hit_count = (
                    (x_gen.index_select(1, mask_positions) == gt.index_select(1, mask_positions))
                    .to(torch.long)
                    .sum(dim=1)
                )
                exact_flags = (hit_count == int(mask_count))

        hit_cpu = hit_count.detach().to("cpu").tolist()
        hit_list.extend([int(v) for v in hit_cpu])
        exact_hits += int(exact_flags.sum().item())

        if dump_writer is not None and traj_dir is not None:
            assert x_gen is not None
            tok = dump_writer.tokenizer
            records: List[Dict[str, Any]] = []

            final_mask_ids = x_gen.index_select(1, mask_positions)  # [B, M]
            final_mask_tokens_batch = [_tokens_from_ids(tok, final_mask_ids[i]) for i in range(cur_bs)]

            inter_mask_tokens_batch: Optional[List[List[List[str]]]] = None
            if inter_mask_ids_steps is not None:
                inter_mask_tokens_batch = []
                for i in range(cur_bs):
                    per_run_steps: List[List[str]] = []
                    for t_ids in inter_mask_ids_steps:
                        per_run_steps.append(_tokens_from_ids(tok, t_ids[i]))
                    inter_mask_tokens_batch.append(per_run_steps)

            final_full_tokens_batch: Optional[List[List[str]]] = None
            inter_full_tokens_batch: Optional[List[List[List[str]]]] = None
            if dump_writer.dump_full_sequence:
                final_full_tokens_batch = [_tokens_from_ids(tok, x_gen[i]) for i in range(cur_bs)]
                if inter_full_steps is not None:
                    inter_full_tokens_batch = []
                    for i in range(cur_bs):
                        per_run_steps_full: List[List[str]] = []
                        for t_full in inter_full_steps:
                            per_run_steps_full.append(_tokens_from_ids(tok, t_full[i]))
                        inter_full_tokens_batch.append(per_run_steps_full)

            for i in range(cur_bs):
                run_id = int(done + i)
                is_hit = bool(exact_flags[i].item())
                rec: Dict[str, Any] = {
                    "run_id": run_id,
                    "step_key": str(step_key),
                    "steps_eff": int(steps_eff),
                    "batch_seed": int(batch_seed),
                    "temperature": float(temperature),
                    "cfg_scale": float(cfg_scale),
                    "hit_count": int(hit_cpu[i]),
                    "is_hit": bool(is_hit),
                    "pred_mask_ids": final_mask_tokens_batch[i],
                    "intermediate_pred_mask_ids": inter_mask_tokens_batch[i] if inter_mask_tokens_batch is not None else None,
                }

                if dump_writer.dump_full_sequence:
                    rec["pred_full_sequence"] = final_full_tokens_batch[i] if final_full_tokens_batch is not None else None
                    rec["intermediate_full_sequence"] = inter_full_tokens_batch[i] if inter_full_tokens_batch is not None else None

                records.append(rec)

            dump_writer.append_step_runs(traj_dir=traj_dir, step_key=str(step_key), steps_eff=int(steps_eff), records=records)

        done += cur_bs

    exact_rate = exact_hits / float(runs) if runs > 0 else 0.0
    mean_hit = (sum(hit_list) / float(runs)) if runs > 0 else 0.0
    return hit_list, float(exact_rate), float(mean_hit)


# =========================
# Trajectory sampling: multi-round without duplicates per sample
# =========================
def select_trajectories_multi_round(
    samples: List[Dict[str, Any]],
    target_trajs: int,
    per_round_per_sample: Optional[int],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    if target_trajs is None or int(target_trajs) <= 0:
        target_trajs = 10**18
    target_trajs = int(target_trajs)

    filtered: List[Dict[str, Any]] = []
    for s in samples:
        vt = s.get("_valid_trajs", [])
        if isinstance(vt, list) and len(vt) > 0:
            filtered.append(s)

    ptr: Dict[int, int] = {id(s): 0 for s in filtered}
    selected: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    round_idx = 0

    while len(selected) < target_trajs:
        round_idx += 1
        added_this_round = 0

        for s in filtered:
            if len(selected) >= target_trajs:
                break

            trajs = s["_valid_trajs"]
            i = ptr[id(s)]
            if i >= len(trajs):
                continue

            take = max(0, int(per_round_per_sample)) if per_round_per_sample is not None else (len(trajs) - i)
            if take <= 0:
                continue

            end = min(len(trajs), i + take)
            for j in range(i, end):
                if len(selected) >= target_trajs:
                    break
                selected.append((s, trajs[j]))
                added_this_round += 1

            ptr[id(s)] = end

        logging.info(
            "Sampling round %d done: added=%d total_selected=%d target=%s",
            round_idx,
            added_this_round,
            len(selected),
            str(target_trajs if target_trajs < 10**18 else "ALL"),
        )
        _flush_logs()

        if added_this_round == 0:
            break

    if len(selected) < target_trajs and target_trajs < 10**18:
        logging.warning(
            "Could not reach target_trajs=%d. Selected=%d (exhausted all available trajectories).",
            target_trajs,
            len(selected),
        )

    return selected


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=Path, required=True, help="goodtraj*.jsonl from the previous stage")
    parser.add_argument("--output_path", type=Path, required=True, help="write evaluation results jsonl here")
    parser.add_argument("--log_path", type=Path, default=None)

    parser.add_argument("--lit_model_name", type=str, default=None, help="e.g. 1028 -> Diff_LLaMA_1028M (local lit-gpt)")

    parser.add_argument("--use_hf", action="store_true", help="Load HF structure and then load local weights.")
    parser.add_argument("--hf_model_name", type=str, default=None, help="HuggingFace model name/path for structure")
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="HF only: trust_remote_code (default: True). Use --no-trust_remote_code to disable.",
    )

    parser.add_argument("--ckpt_path", type=str, required=True, help="Local checkpoint path (.pth/.pt/.safetensors)")

    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--gen_temperature", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)

    parser.add_argument("--steps_2", type=int, default=2)
    parser.add_argument("--steps_5", type=int, default=5)
    parser.add_argument("--steps_10", type=int, default=10)
    parser.add_argument("--enable_per_token_steps", action="store_true")

    parser.add_argument(
        "--enable_multistep_eval",
        action="store_true",
        help="If set, evaluate steps_2/5/10 (+per_token if enabled). Otherwise, only step=1 is evaluated.",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16/fp16/fp32")

    parser.add_argument("--traj_start", type=int, default=0)
    parser.add_argument("--traj_end", type=int, default=-1)

    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--target_trajs", type=int, default=200)
    parser.add_argument("--max_traj_per_sample", type=int, default=1)

    parser.add_argument("--enable_gen_dump", action="store_true", help="Dump per-run generations into a folder.")
    parser.add_argument(
        "--gen_dump_dir",
        type=Path,
        default=None,
        help="Dump root directory. Default: output_path.parent/<stem>_gens/",
    )
    parser.add_argument(
        "--dump_full_sequence",
        action="store_true",
        help="Also dump full sequence tokens for each internal step (very large).",
    )

    args = parser.parse_args()

    setup_logger(args.log_path)
    logging.info("Args: %s", vars(args))

    if not args.input_path.exists():
        raise FileNotFoundError(f"input_path not found: {args.input_path}")

    if args.use_hf:
        if not args.hf_model_name:
            raise ValueError("--use_hf is set but --hf_model_name is empty.")
    else:
        if not args.lit_model_name:
            raise ValueError("Local mode requires --lit_model_name (or set --use_hf with --hf_model_name).")

    per_round = int(args.max_traj_per_sample) if args.max_traj_per_sample is not None else None
    if per_round is not None and per_round <= 0:
        raise ValueError("--max_traj_per_sample must be >= 1 (per-round).")

    raw_samples: List[Dict[str, Any]] = list(iter_jsonl(args.input_path))
    logging.info("Loaded %d lines from input jsonl.", len(raw_samples))

    prepped_samples: List[Dict[str, Any]] = []
    skipped_samples = 0
    skipped_trajs = 0
    total_valid_trajs = 0

    for sample in raw_samples:
        gt_ids = sample.get("token_ids", None)
        if not isinstance(gt_ids, list) or not gt_ids or not all(isinstance(x, int) for x in gt_ids):
            skipped_samples += 1
            continue

        sample_mask_id = sample.get("mask_id", None)
        if not isinstance(sample_mask_id, int):
            skipped_samples += 1
            continue

        trajs = sample.get("good_trajectories", [])
        if not isinstance(trajs, list) or not trajs:
            skipped_samples += 1
            continue

        valid_trajs: List[Dict[str, Any]] = []
        for traj in trajs:
            masked_input_ids = traj.get("masked_input_ids", None)
            if not isinstance(masked_input_ids, list) or len(masked_input_ids) != len(gt_ids):
                skipped_trajs += 1
                continue
            valid_trajs.append(traj)

        if not valid_trajs:
            skipped_samples += 1
            continue

        sample["_valid_trajs"] = valid_trajs
        total_valid_trajs += len(valid_trajs)
        prepped_samples.append(sample)

    logging.info(
        "Precheck: samples_kept=%d samples_skipped=%d valid_trajs=%d trajs_skipped=%d",
        len(prepped_samples),
        skipped_samples,
        total_valid_trajs,
        skipped_trajs,
    )

    selected_pairs = select_trajectories_multi_round(
        samples=prepped_samples,
        target_trajs=int(args.target_trajs),
        per_round_per_sample=per_round,
    )
    logging.info("Final selected trajectories: %d", len(selected_pairs))

    start = max(0, int(args.traj_start))
    end = int(args.traj_end)

    if end is None or end < 0:
        end = len(selected_pairs)
    else:
        end = min(end, len(selected_pairs))

    if start > end:
        raise ValueError(f"--traj_start ({start}) must be <= --traj_end ({end})")

    selected_pairs = selected_pairs[start:end]
    logging.info("After slicing: traj_start=%d traj_end=%d kept=%d", start, end, len(selected_pairs))

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)

    vocab_size: Optional[int] = None
    tokenizer: Optional[Any] = None

    if args.use_hf:
        max_mask_id = 0
        if prepped_samples:
            max_mask_id = max(int(s.get("mask_id", 0)) for s in prepped_samples)
        logging.info("HF mode: using max_mask_id=%d for embedding resize guard.", int(max_mask_id))

        model, tokenizer, vocab_size_hf = load_hf_model_with_pth_weights(
            hf_model_name=str(args.hf_model_name),
            ckpt_path=str(args.ckpt_path),
            device=device,
            dtype=dtype,
            mask_id=int(max_mask_id),
            trust_remote_code=bool(args.trust_remote_code),
        )
        vocab_size = int(vocab_size_hf)
        logging.info("HF model loaded. vocab_size(embedding_rows)=%s", str(vocab_size))
    else:
        model, _config, vocab_size_local = load_local_diff_model(
            lit_model_name=str(args.lit_model_name),
            ckpt_path=str(args.ckpt_path),
            device=device,
            dtype=dtype,
        )
        vocab_size = vocab_size_local
        tokenizer = None
        logging.info("Local lit-gpt model loaded. vocab_size(inferred)=%s", str(vocab_size))

    dump_writer: Optional[GenDumpWriter] = None
    dump_root: Optional[Path] = None
    if bool(args.enable_gen_dump):
        dump_root = (args.output_path.parent / f"{args.output_path.stem}_gens") if args.gen_dump_dir is None else Path(args.gen_dump_dir)
        dump_root.mkdir(parents=True, exist_ok=True)
        dump_writer = GenDumpWriter(root_dir=dump_root, tokenizer=tokenizer, dump_full_sequence=bool(args.dump_full_sequence))
        logging.info("Generation dump root: %s", str(dump_root))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.output_path, "w", encoding="utf-8")

    t_all = time.time()
    total_written = 0

    for sample, traj in selected_pairs:
        line_no = sample.get("_line_no")
        index = sample.get("index", None)
        set_name = sample.get("set_name", None)
        mask_ratio = sample.get("mask_ratio", None)
        tail_len = sample.get("tail_len", None)

        gt_ids = sample.get("token_ids")
        sample_mask_id = int(sample.get("mask_id"))

        traj_id = traj.get("traj_id", None)
        traj_seed = traj.get("seed", None)
        traj_logp = traj.get("logp", None)
        traj_p_percent = traj.get("p_percent", None)

        masked_input_ids = traj.get("masked_input_ids", None)
        if not isinstance(masked_input_ids, list) or len(masked_input_ids) != len(gt_ids):
            continue

        mask_positions = traj.get("mask_positions", None)
        if not (isinstance(mask_positions, list) and all(isinstance(x, int) for x in mask_positions)):
            mask_positions = [i for i, v in enumerate(masked_input_ids) if v == int(sample_mask_id)]
        mask_count = int(len(mask_positions))
        prompt_len = int(len(gt_ids) - mask_count)

        logging.info(
            "Eval traj: line=%s index=%s set=%s traj_id=%s mask_count=%d multistep=%s dump=%s",
            str(line_no),
            str(index),
            str(set_name),
            str(traj_id),
            int(mask_count),
            "ON" if args.enable_multistep_eval else "OFF",
            "ON" if dump_writer is not None else "OFF",
        )
        _flush_logs()

        base_seed = _safe_seed(
            int(args.seed)
            + int(index if index is not None else line_no) * 1_000_000
            + int(traj_id if traj_id is not None else 0) * 10_000
        )

        step_items: List[Tuple[str, int]] = [("1", 1)]
        if args.enable_multistep_eval:
            step_items.extend([
                ("2", int(args.steps_2)),
                ("5", int(args.steps_5)),
                ("10", int(args.steps_10)),
            ])
            if args.enable_per_token_steps:
                step_items.append(("per_token", max(1, int(mask_count))))

        hit_by_step: Dict[str, List[int]] = {}
        mean_hit_by_step: Dict[str, float] = {}
        exact_rate_by_step: Dict[str, float] = {}

        traj_dir: Optional[Path] = None
        if dump_writer is not None and dump_root is not None:
            traj_dir = dump_writer.make_traj_dir(line_no=line_no, index=index, traj_id=traj_id, step_key="NA", steps_eff=0)
            dump_writer.write_meta_once(
                traj_dir,
                meta={
                    "line_no": line_no,
                    "index": index,
                    "set_name": set_name,
                    "mask_ratio": mask_ratio,
                    "tail_len": tail_len,
                    "mask_id": int(sample_mask_id),
                    "traj_id": traj_id,
                    "traj_seed": traj_seed,
                    "traj_logp": traj_logp,
                    "traj_p_percent": traj_p_percent,
                    "mask_positions": mask_positions,
                    "mask_count": mask_count,
                    "prompt_len": prompt_len,
                    "runs": int(args.runs),
                    "eval_batch_size": int(args.eval_batch_size),
                    "temperature": float(args.gen_temperature),
                    "cfg_scale": float(args.cfg_scale),
                    "dump_full_sequence": bool(args.dump_full_sequence),
                    "model_load_mode": "hf" if bool(args.use_hf) else "local",
                    "hf_model_name": str(args.hf_model_name) if bool(args.use_hf) else None,
                    "lit_model_name": str(args.lit_model_name) if not bool(args.use_hf) else None,
                },
            )

        t0_all = time.time()
        for key, s in step_items:
            steps_eff = max(1, min(int(s), int(mask_count))) if mask_count > 0 else 1
            step_seed = _safe_seed(int(base_seed) + (hash(key) % 10_000) * 101 + int(steps_eff) * 100)

            t0 = time.time()
            hit_list, exact_rate, mean_hit = estimate_hits_for_step(
                model=model,
                gt_ids=gt_ids,
                masked_input_ids=masked_input_ids,
                mask_id=int(sample_mask_id),
                step_key=str(key),
                steps_eff=int(steps_eff),
                runs=int(args.runs),
                batch_size=int(args.eval_batch_size),
                base_seed=int(step_seed),
                temperature=float(args.gen_temperature),
                cfg_scale=float(args.cfg_scale),
                dump_writer=dump_writer,
                traj_dir=traj_dir,
            )
            hit_by_step[str(key)] = hit_list
            exact_rate_by_step[str(key)] = float(exact_rate)
            mean_hit_by_step[str(key)] = float(mean_hit)

            logging.info(
                "  step=%s(eff=%d) | exact_rate=%.6f | mean_hit=%.3f | runs=%d | time=%.2fs",
                str(key),
                int(steps_eff),
                float(exact_rate),
                float(mean_hit),
                int(args.runs),
                time.time() - t0,
            )
            _flush_logs()

        logging.info("  traj_id=%s | total_eval_time=%.2fs", str(traj_id), time.time() - t0_all)
        _flush_logs()

        out_obj = {
            "index": index,
            "set_name": set_name,
            "line_no": line_no,
            "mask_ratio": mask_ratio,
            "tail_len": tail_len,
            "mask_id": int(sample_mask_id),
            "traj": {
                "traj_id": traj_id,
                "seed": traj_seed,
                "logp": traj_logp,
                "p_percent": traj_p_percent,
                "mask_positions": mask_positions,
                "mask_count": mask_count,
                "prompt_len": prompt_len,
                "masked_input_ids": masked_input_ids,
            },
            "eval": {
                "runs": int(args.runs),
                "eval_batch_size": int(args.eval_batch_size),
                "temperature": float(args.gen_temperature),
                "cfg_scale": float(args.cfg_scale),
                "hit_token": hit_by_step,
                "mean_hit_token": mean_hit_by_step,
                "success_rate_all_correct": exact_rate_by_step,
                "enable_multistep_eval": bool(args.enable_multistep_eval),
                "steps_2": int(args.steps_2),
                "steps_5": int(args.steps_5),
                "steps_10": int(args.steps_10),
                "enable_per_token_steps": bool(args.enable_per_token_steps),
                "target_trajs": int(args.target_trajs),
                "max_traj_per_sample_per_round": int(per_round) if per_round is not None else None,
                "model_load_mode": "hf" if bool(args.use_hf) else "local",
                "hf_model_name": str(args.hf_model_name) if bool(args.use_hf) else None,
                "lit_model_name": str(args.lit_model_name) if not bool(args.use_hf) else None,
                "enable_gen_dump": bool(args.enable_gen_dump),
                "gen_dump_dir": str(dump_root) if dump_root is not None else None,
                "dump_full_sequence": bool(args.dump_full_sequence),
            },
            "token_ids": gt_ids,
        }

        out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        total_written += 1
        out_f.flush()

    out_f.close()
    logging.info(
        "Done. written_trajs=%d elapsed=%.1fs output=%s",
        total_written,
        time.time() - t_all,
        str(args.output_path),
    )


if __name__ == "__main__":
    main()
