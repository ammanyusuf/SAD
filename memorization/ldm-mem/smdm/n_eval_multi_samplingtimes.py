import argparse
import json
import logging
import math
import random
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from lit_gpt.diffmodel import TransEncoder, Config
from safetensors.torch import load_file  


# =========================
# Logging utils
# =========================
def add_timestamp_to_path(path: Path, ts: Optional[str] = None) -> Path:
    if ts is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{ts}{path.suffix}")


def add_run_tags_to_path(path: Path, ts: str, tag: str) -> Path:
    return path.with_name(f"{path.stem}_{tag}_{ts}{path.suffix}")


def add_range_to_path(path: Optional[Path], start: int, end: Optional[int]) -> Optional[Path]:
    if path is None:
        return None
    s = f"s{start}"
    e = f"e{end}" if end is not None else "eend"
    return path.with_name(f"{path.stem}_{s}_{e}{path.suffix}")


def setup_logger(log_path: Optional[Path]):
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


# =========================
# Sample loading
# =========================
def load_jsonl(samples_path: Path, max_samples: int, shuffle: bool, seed: int) -> List[dict]:
    samples = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)
    return samples[:max_samples]


# =========================
# Parse helpers
# =========================
def parse_int_list(value: str) -> List[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(x) for x in value.split(",") if x.strip()]


# =========================
# Stats helpers
# =========================
def summarize_finite(values: List[float]) -> Dict[str, float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {"mean": float("nan"), "stdev": 0.0, "min": float("nan"), "max": float("nan"), "count": 0}
    return {
        "mean": statistics.fmean(finite),
        "stdev": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
        "count": len(finite),
    }


def compute_np_stats(log_sums: List[float], p_targets: List[float], n_targets: List[int]):

    if not log_sums:
        return {"p_hat": float("nan"), "processable_count": 0, "n_for_p": {}, "p_for_n": {}}

    total_count = len(log_sums)
    probs = []
    processable_count = 0
    for v in log_sums:
        if math.isfinite(v):
            probs.append(math.exp(v) if v > -745 else 0.0)
            processable_count += 1
        else:
            probs.append(0.0)

    p_hat = sum(probs) / total_count

    n_for_p = {}
    for p in p_targets:
        if not math.isfinite(p_hat):
            n_for_p[str(p)] = None
            continue
        denom = math.log1p(-p_hat)
        if denom == 0:
            n_for_p[str(p)] = None
        else:
            n_for_p[str(p)] = int(math.ceil(math.log(1 - p) / denom))

    p_for_n = {}
    for n in n_targets:
        if not math.isfinite(p_hat):
            p_for_n[str(n)] = float("nan")
            continue
        p_for_n[str(n)] = -math.expm1(n * math.log1p(-p_hat))

    return {"p_hat": p_hat, "processable_count": processable_count, "n_for_p": n_for_p, "p_for_n": p_for_n}


# =========================
# Local ckpt loading 
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
    p = Path(ckpt_path)
    if p.suffix == ".safetensors":
        sd = load_file(str(p))
        return clean_state_dict_keys(sd)

    ckpt = torch.load(str(p), map_location="cpu")
    sd = extract_state_dict_from_ckpt(ckpt)
    sd = clean_state_dict_keys(sd)
    return sd


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


def load_local_transencoder_and_tokenizer(
    lit_model_name: str,
    tokenizer_name: str,
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.nn.Module, AutoTokenizer, Optional[int], Any]:
    model_name = f"Diff_LLaMA_{lit_model_name}M"
    logging.info("Loading local TransEncoder config: %s", model_name)
    config = Config.from_name(model_name)

    logging.info("Instantiating TransEncoder...")
    model = TransEncoder(config).to(device)

    logging.info("Loading checkpoint (local): %s", ckpt_path)
    state_dict = load_state_dict_local(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.info("load_state_dict(strict=False): missing=%d unexpected=%d", len(missing), len(unexpected))
    if unexpected:
        logging.warning("Unexpected keys head: %s", unexpected[:20])
    if missing:
        logging.warning("Missing keys head: %s", missing[:20])

    model = model.to(device=device, dtype=dtype)
    model.eval()


    param_dtype = next(model.parameters()).dtype
    for name, buf in model.named_buffers():
        if torch.is_floating_point(buf) and buf.dtype != param_dtype:
            model._buffers[name] = buf.to(dtype=param_dtype)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    vocab_size = infer_vocab_size(model, config)
    return model, tokenizer, vocab_size, config


# ============================================================
# ============================================================

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    noise = noise.clamp_min(1e-12)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def gt_logprob_under_gumbel_sampling(
    logits_fp64: torch.Tensor, gt_ids: torch.Tensor, temperature: float
) -> torch.Tensor:

    if temperature <= 0:
        raise ValueError("gumbel_temperature must be > 0")
    scaled = logits_fp64 / float(temperature)
    log_probs = F.log_softmax(scaled, dim=-1)
    return log_probs.gather(1, gt_ids.unsqueeze(1)).squeeze(1)


def diff_recover_logprob_batch(
    model: torch.nn.Module,
    gt_tokens_1: torch.Tensor,      # [1, L]
    prompt_len: int,
    steps: int,
    alg: str,
    temperature: float,
    cfg_scale: float,
    eps: float,
    dim: int,
    device: torch.device,
    log_inv_fre: torch.Tensor,      # NEW: [V] float64, on device
    alpha_clip: float,              # NEW
 ) -> Tuple[torch.Tensor, torch.Tensor]:

    B = gt_tokens_1.shape[0]
    L = gt_tokens_1.shape[1]

    gt = gt_tokens_1  # [B, L]（外部已经 repeat）

    # ---- NEW: random mask positions across full text, per-trajectory independent ----
    mask_count = int(L - int(prompt_len))
    if mask_count < 0:
        raise ValueError(f"prompt_len invalid: prompt_len={prompt_len} > L={L}")

    if mask_count == 0:
        x = gt.clone()
    else:
        # sample exactly mask_count positions per row (uniform without replacement via topk on rand)
        r = torch.rand((B, L), device=device)
        _, idx = torch.topk(r, k=mask_count, dim=1, largest=True, sorted=False)  # [B, mask_count]
        mask_indices = torch.zeros((B, L), device=device, dtype=torch.bool)
        mask_indices.scatter_(1, idx, True)

        x = gt.clone()
        x[mask_indices] = dim

    # ---- NEW: timesteps start from t0 that matches the (fixed-count) mask ratio ----
    # treat p0 as masked fraction (mask_count/L), map to forward_process t: p = (1-eps)*t + eps
    p0 = float(mask_count) / float(L) if L > 0 else 0.0
    t0 = (p0 - eps) / (1.0 - eps) if (1.0 - eps) != 0 else 1.0
    if not math.isfinite(t0):
        t0 = 1.0
    t0 = max(min(t0, 1.0), eps)

    timesteps = torch.linspace(t0, eps, steps + 1, device=device)
    total_log = torch.zeros((B,), device=device, dtype=torch.float64)
    
    alpha_sum = torch.zeros((B,), device=device, dtype=torch.float64)
    alpha_cnt = torch.zeros((B,), device=device, dtype=torch.float64)

    use_amp = (device.type == "cuda")
    amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_amp else nullcontext()

    with torch.no_grad():
        for i in range(steps):
            mask_index = (x == dim)  # [B, L]
            if int(mask_index.sum().item()) == 0:
                break

            with amp_ctx:
                if cfg_scale > 0.:
                    un_x = x.clone()
                    if prompt_len > 0:

                        un_x[:, :prompt_len] = dim
                    x_ = torch.cat([x, un_x], dim=0)              # [2B, L]
                    logits_all = model(x_)                        # [2B, L, V]
                    logits_c, logits_u = torch.chunk(logits_all, 2, dim=0)
                    logits_full = logits_u + (cfg_scale + 1) * (logits_c - logits_u)  # [B, L, V]
                else:
                    logits_full = model(x)                        # [B, L, V]

            t = timesteps[i]
            s = timesteps[i + 1]

            if alg == "origin":

                p_transfer = 1 - s / t if i < steps - 1 else 1.0
                transfer_pos_mask = (torch.rand((B, L), device=device) < p_transfer) & mask_index  # [B,L]

                b_idx, pos_idx = transfer_pos_mask.nonzero(as_tuple=True)
                if b_idx.numel() > 0:
                    sel_logits = logits_full[b_idx, pos_idx, :].to(torch.float64)  # [N,V]
                    sel_gt = gt[b_idx, pos_idx]                                     # [N]
                    sel_lp = gt_logprob_under_gumbel_sampling(sel_logits, sel_gt, temperature=temperature)  # [N]
                    total_log.index_add_(0, b_idx, sel_lp)

                    # -----------------------------
                    # NEW: alpha = p * log(1/freq(token)), then clip per-token
                    # -----------------------------
                    sel_prob = torch.exp(sel_lp)  # [N] in (0,1]
                    sel_loginv = log_inv_fre[sel_gt]  # [N]
                    sel_alpha = sel_prob * sel_loginv  # [N]
                    if math.isfinite(alpha_clip) and alpha_clip > 0:
                        sel_alpha = torch.clamp(sel_alpha, max=float(alpha_clip))

                    alpha_sum.index_add_(0, b_idx, sel_alpha)
                    alpha_cnt.index_add_(0, b_idx, torch.ones_like(sel_alpha))


                    # teacher forcing
                    x[b_idx, pos_idx] = sel_gt

            elif alg == "greddy":
                logits_masked = logits_full[mask_index]  # [Nmask_total, V]
                logits_with_noise = add_gumbel_noise(logits_masked, temperature=temperature)
                x0_masked = torch.argmax(logits_with_noise, dim=-1)  # [Nmask_total]
                logits_masked_fp64 = logits_masked.to(torch.float64)
                p = F.softmax(logits_masked_fp64, dim=-1)
                confidence_masked = torch.gather(p, dim=-1, index=x0_masked.unsqueeze(-1)).squeeze(-1)  # [Nmask_total]
                confidence_full = torch.full((B, L), float("-inf"), device=device, dtype=torch.float64)
                confidence_full[mask_index] = confidence_masked

                num_mask = mask_index.sum(dim=1).to(torch.int64)  # [B]
                if i < steps - 1:
                    frac = (1 - s / t).item()  # scalar float
                    k = torch.floor(num_mask.to(torch.float64) * frac).to(torch.int64)  # [B]
                else:
                    k = num_mask

                max_k = int(k.max().item())
                if max_k <= 0:
                    continue

                _, top_pos = torch.topk(confidence_full, k=max_k, dim=1)  # [B, max_k]

                ar = torch.arange(max_k, device=device).unsqueeze(0)  # [1, max_k]
                take = ar < k.unsqueeze(1)                            # [B, max_k] bool

                b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_k)
                sel_b = b_idx[take]
                sel_pos = top_pos[take]

                if sel_b.numel() == 0:
                    continue


                sel_logits = logits_full[sel_b, sel_pos, :].to(torch.float64)  # [N,V]
                sel_gt = gt[sel_b, sel_pos]                                     # [N]
                sel_lp = gt_logprob_under_gumbel_sampling(sel_logits, sel_gt, temperature=temperature)
                total_log.index_add_(0, sel_b, sel_lp)

                x[sel_b, sel_pos] = sel_gt

            else:
                raise NotImplementedError(alg)
    alpha_mean = alpha_sum / torch.clamp(alpha_cnt, min=1.0)
    return total_log, alpha_mean  # [B]


def diff_recover_log_stats_for_text_mc(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    mask_id: int,
    mask_ratio: float,
    steps: int,
    alg: str,
    temperature: float,
    cfg_scale: float,
    eps: float,
    tail_len: int,
    seed: int,
    mc_samples: int,
    mc_batch_size: int,
    log_inv_fre: torch.Tensor,   # NEW
    alpha_clip: float,           # NEW
    weight_eps: float,           # NEW
) -> Dict[str, Any]:

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids = token_ids[-tail_len:]
    seq_len = len(token_ids)
    if seq_len < tail_len:
        raise ValueError(f"sequence too short after tail cut: {seq_len} < tail_len={tail_len}")

    mask_count = int(seq_len * mask_ratio)
    if mask_count <= 0 or mask_count > seq_len:
        raise ValueError(f"invalid mask_ratio={mask_ratio} for seq_len={seq_len}")

    prompt_len = seq_len - mask_count
    device = next(model.parameters()).device

    # GT: [1,L]
    gt_1 = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    all_logs: List[float] = []
        
    # NEW: weighted p_hat accumulation on probability space
    sum_w = 0.0
    sum_wP = 0.0


    alpha_all: List[float] = []

    for start in range(0, mc_samples, mc_batch_size):
        cur_bs = min(mc_batch_size, mc_samples - start)


        chunk_seed = int(seed + start * 1337 + steps * 17)
        torch.manual_seed(chunk_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(chunk_seed)

        gt = gt_1.repeat(cur_bs, 1)  # [B,L]

        total_log_b, alpha_b = diff_recover_logprob_batch(
            model=model,
            gt_tokens_1=gt,
            prompt_len=prompt_len,
            steps=steps,
            alg=alg,
            temperature=temperature,
            cfg_scale=cfg_scale,
            eps=eps,
            dim=mask_id,
            device=device,
            log_inv_fre=log_inv_fre,
            alpha_clip=alpha_clip,
        )  # [B]
        with torch.no_grad():
            min_log = float(total_log_b.min().detach().cpu().item())
            max_log = float(total_log_b.max().detach().cpu().item())

            like_b = torch.where(
                total_log_b > -745.0,
                torch.exp(total_log_b),
                torch.zeros_like(total_log_b),
            )
            min_like = float(like_b.min().detach().cpu().item())
            max_like = float(like_b.max().detach().cpu().item())
            # NEW: linear weights from alpha (already clipped per-token inside batch)
            # weight_j = alpha_mean_j + eps
            w_b = torch.clamp(alpha_b, min=0.0) + float(weight_eps)

            # probability-space weighted average accumulator
            sum_w += float(w_b.sum().detach().cpu().item())
            sum_wP += float((w_b * like_b).sum().detach().cpu().item())

            # optional: keep alpha for summary
            alpha_all.extend(alpha_b.detach().cpu().tolist())

        logging.info(
            "[micro-batch] steps=%d traj_chunk=[%d:%d] bs=%d | "
            "total_log min=%.6f max=%.6f | likelihood min=%.3e max=%.3e",
            int(steps), int(start), int(start + cur_bs), int(cur_bs),
            min_log, max_log, min_like, max_like,
        )

        all_logs.extend(total_log_b.detach().cpu().tolist())

    stats = summarize_finite(all_logs)
    
    p_hat_weight = (sum_wP / sum_w) if (sum_w > 0.0) else float("nan")
    alpha_stats = summarize_finite(alpha_all)

    return {
        "recover_total_log_list": all_logs,            
        "recover_total_log_mean": stats["mean"],
        "recover_total_log_stdev": stats["stdev"],
        "recover_total_log_min": stats["min"],
        "recover_total_log_max": stats["max"],
        "recover_token_count": int(mask_count),
        "recover_prompt_len": int(prompt_len),
        # NEW
        "recover_p_hat_weight": p_hat_weight,
        "recover_alpha_mean": alpha_stats["mean"],
        "recover_alpha_min": alpha_stats["min"],
        "recover_alpha_max": alpha_stats["max"],
    }


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=10_000_000)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=88666)

    # local model
    parser.add_argument("--lit_model_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)

    # diffusion/mask params
    parser.add_argument("--diff_mask_id", type=int, default=126336)
    parser.add_argument("--tail_len", type=int, default=100)
    parser.add_argument("--default_mask_ratio", type=float, default=0.2)
    parser.add_argument("--use_sample_mask_ratio", action="store_true", default=True)

    # steps list（
    parser.add_argument("--mc_list", type=str, default="1", help="diffusion steps list")

    parser.add_argument("--traj_list", type=str, default="64", help="MC trajectories per sample, e.g. 64,128,512")
    parser.add_argument("--traj_batch_size", type=int, default=16, help="micro-batch size for trajectories (VRAM)")

    # sampler params
    parser.add_argument("--alg", type=str, default="origin", choices=["origin", "greddy"])
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--gumbel_temperature", type=float, default=1.0)

    # np stats targets
    parser.add_argument("--p_targets", type=str, default="0.1,0.5,0.9,0.99")
    parser.add_argument("--n_targets", type=str, default="1,10,100")

    # device/dtype
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    # output/log
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--log_path", type=Path, default=None)

    parser.add_argument(
        "--freq_pt",
        type=str,
        default="smdm/frequent/1bpretrainmdm.pt",
        help="precomputed token frequency distribution .pt (with Laplace smoothing)",
    )
    parser.add_argument("--alpha_clip", type=float, default=0.01, help="clip threshold a for alpha per token")
    parser.add_argument("--weight_eps", type=float, default=1e-12, help="epsilon added to linear weights")

    args = parser.parse_args()

    steps_list = parse_int_list(args.mc_list)
    traj_list = parse_int_list(args.traj_list)
    args.p_targets = parse_float_list(args.p_targets)
    args.n_targets = parse_int_list(args.n_targets)

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    if args.log_path is not None and args.log_path.name.endswith(".log"):
        args.log_path = add_timestamp_to_path(args.log_path, ts=run_ts)
    if args.output_path is not None and args.output_path.name.endswith(".jsonl"):
        args.output_path = add_run_tags_to_path(args.output_path, run_ts, tag="recover_steps_traj")

    args.output_path = add_range_to_path(args.output_path, int(args.start_index), args.end_index)
    args.log_path = add_range_to_path(args.log_path, int(args.start_index), args.end_index)

    setup_logger(args.log_path)
    logging.info("Starting diffusion teacher-forcing recovery log eval (trajectory batched)")
    logging.info("Args: %s", vars(args))
    logging.info("steps_list: %s", steps_list)
    logging.info("traj_list: %s | traj_batch_size=%d", traj_list, int(args.traj_batch_size))

    if not args.input_path.exists():
        raise FileNotFoundError(f"input jsonl not found: {args.input_path}")

    samples_all = load_jsonl(args.input_path, args.max_samples, args.shuffle, args.seed)
    start_i = max(0, int(args.start_index))
    end_i = int(args.end_index) if args.end_index is not None else len(samples_all)
    end_i = min(len(samples_all), max(start_i, end_i))
    samples = samples_all[start_i:end_i]
    logging.info("Loaded %d lines (total_loaded=%d, range=[%d:%d])", len(samples), len(samples_all), start_i, end_i)

    device = torch.device(args.device)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    model, tokenizer, vocab_size, _config = load_local_transencoder_and_tokenizer(
        lit_model_name=args.lit_model_name,
        tokenizer_name=args.tokenizer_name,
        ckpt_path=args.ckpt_path,
        device=device,
        dtype=dtype,
    )
    logging.info("Model ready. vocab_size(inferred)=%s", str(vocab_size))

    # =========================
    # NEW: load frequency distribution and precompute log(1/f)
    # =========================
    logging.info("Loading token frequency distribution: %s", args.freq_pt)
    freq_obj = torch.load(args.freq_pt, map_location="cpu")

    if isinstance(freq_obj, dict):
        if "fre_dis" in freq_obj and torch.is_tensor(freq_obj["fre_dis"]):
            fre_dis = freq_obj["fre_dis"]
        elif "probs" in freq_obj and torch.is_tensor(freq_obj["probs"]):
            fre_dis = freq_obj["probs"]
        else:
            cand = None
            for v in freq_obj.values():
                if torch.is_tensor(v):
                    cand = v
                    break
            if cand is None:
                raise RuntimeError("freq_pt dict has no tensor field (expected fre_dis/probs)")
            fre_dis = cand
    elif torch.is_tensor(freq_obj):
        fre_dis = freq_obj
    else:
        raise RuntimeError(f"Unrecognized freq_pt type: {type(freq_obj)}")

    fre_dis = fre_dis.detach().to(torch.float64)

    tok_vocab = getattr(tokenizer, "vocab_size", None)
    if tok_vocab is not None and int(fre_dis.numel()) != int(tok_vocab):
        raise RuntimeError(f"freq vocab mismatch: fre_dis={fre_dis.numel()} vs tokenizer.vocab_size={tok_vocab}")

    import math

    # ---- Sanity check: token id -> frequent value ----
    sanity_tid = 29892
    sanity_freq = 0.03152924943  # from your CSV
    actual = float(fre_dis[sanity_tid].cpu().item())

    # check to ~5 decimals (abs tol)
    assert math.isclose(actual, sanity_freq, rel_tol=0.0, abs_tol=1e-5), \
        f"[FREQ MISMATCH] fre_dis[{sanity_tid}]={actual:.12g} != {sanity_freq:.12g} (abs_tol=1e-5)"

    # optional: also check decoded token text (allow minor whitespace)
    decoded = tokenizer.decode([sanity_tid], clean_up_tokenization_spaces=False)
    assert decoded.strip() == ",", f"[TOKEN MISMATCH] decode({sanity_tid})='{decoded}' expected ','"

    logging.info("[SANITY] fre_dis[%d]=%.10f OK | decode='%s'", sanity_tid, actual, decoded)

    fre_dis = torch.clamp(fre_dis, min=1e-30)
    log_inv_fre = torch.log(1.0 / fre_dis).to(device=device, dtype=torch.float64)

    logging.info("Frequency distribution ready. min_f=%.3e max_f=%.3e", float(fre_dis.min()), float(fre_dis.max()))



    results_f = None
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        results_f = open(args.output_path, "w", encoding="utf-8")

    # global summary（存 mean log）
    global_by_steps_traj: Dict[int, Dict[int, List[float]]] = {s: {m: [] for m in traj_list} for s in steps_list}

    for line_idx, sample in enumerate(samples, start=start_i):
        sample_index = sample.get("index", line_idx)
        set_name = sample.get("set_name", sample.get("setname", ""))

        text = sample.get("text", "")
        if not isinstance(text, str) or not text.strip():
            logging.warning("Line %d (sample_index=%s) missing/empty text, skipping", line_idx, str(sample_index))
            continue

        mask_ratio = args.default_mask_ratio
        if args.use_sample_mask_ratio and ("mask_ratio" in sample):
            try:
                mask_ratio = float(sample["mask_ratio"])
            except Exception:
                mask_ratio = args.default_mask_ratio

        logging.info(
            "Line %d | sample_index=%s | set=%s | mask_ratio=%.4f | alg=%s",
            line_idx, str(sample_index), str(set_name), mask_ratio, args.alg
        )

        per_steps_results: Dict[str, Any] = {}

        for steps_id, steps in enumerate(steps_list):
            per_traj_results: Dict[str, Any] = {}

            for traj_id, mc_samples in enumerate(traj_list):

                run_seed = int(args.seed + int(sample_index) * 1_000_000 + steps_id * 100_000 + traj_id * 10_000 + steps)

                t0 = time.time()
                try:
                    stats = diff_recover_log_stats_for_text_mc(
                        model=model,
                        tokenizer=tokenizer,
                        text=text,
                        mask_id=args.diff_mask_id,
                        mask_ratio=mask_ratio,
                        steps=int(steps),
                        alg=args.alg,
                        temperature=float(args.gumbel_temperature),
                        cfg_scale=float(args.cfg_scale),
                        eps=float(args.eps),
                        tail_len=int(args.tail_len),
                        seed=run_seed,
                        mc_samples=int(mc_samples),
                        mc_batch_size=int(args.traj_batch_size),
                        log_inv_fre=log_inv_fre,
                        alpha_clip=float(args.alpha_clip),
                        weight_eps=float(args.weight_eps),

                    )

                    logs = stats["recover_total_log_list"]
                    np_stats = compute_np_stats(logs, args.p_targets, args.n_targets)

                    per_traj_results[str(mc_samples)] = {
                        "recover_total_log_mean": stats["recover_total_log_mean"],
                        "recover_total_log_stdev": stats["recover_total_log_stdev"],
                        "recover_total_log_min": stats["recover_total_log_min"],
                        "recover_total_log_max": stats["recover_total_log_max"],
                        "recover_token_count": stats["recover_token_count"],
                        "recover_prompt_len": stats["recover_prompt_len"],
                        "recover_processable_count": np_stats["processable_count"],
                        "recover_p_hat": np_stats["p_hat"],
                        "recover_p_hat_weight": stats["recover_p_hat_weight"],
                        "recover_n_for_p": np_stats["n_for_p"],
                        "recover_p_for_n": np_stats["p_for_n"],
                        "recover_alpha_mean": stats.get("recover_alpha_mean", float("nan")),
                        "recover_alpha_min": stats.get("recover_alpha_min", float("nan")),
                        "recover_alpha_max": stats.get("recover_alpha_max", float("nan")),

                    }

                    if math.isfinite(stats["recover_total_log_mean"]):
                        global_by_steps_traj[int(steps)][int(mc_samples)].append(stats["recover_total_log_mean"])
                    
                    logging.info(
                        "steps=%d traj=%d | mean_log=%.6f std=%.6f | p_hat=%.3e | p_hat_w=%.3e | time=%.2fs",
                        int(steps), int(mc_samples),
                        per_traj_results[str(mc_samples)]["recover_total_log_mean"],
                        per_traj_results[str(mc_samples)]["recover_total_log_stdev"],
                        per_traj_results[str(mc_samples)]["recover_p_hat"],
                        per_traj_results[str(mc_samples)]["recover_p_hat_weight"],
                        time.time() - t0,
                    )

                except Exception as e:
                    logging.exception("Failed line %d steps=%d traj=%d: %s", line_idx, steps, mc_samples, str(e))
                    per_traj_results[str(mc_samples)] = {
                        "error": str(e),
                        "recover_total_log_mean": float("nan"),
                        "recover_total_log_stdev": 0.0,
                        "recover_total_log_min": float("nan"),
                        "recover_total_log_max": float("nan"),
                        "recover_token_count": 0,
                        "recover_prompt_len": 0,
                        "recover_processable_count": 0,
                        "recover_p_hat": float("nan"),
                        "recover_n_for_p": {},
                        "recover_p_for_n": {},
                        "recover_p_hat_weight": float("nan"),
                        "recover_alpha_mean": float("nan"),
                        "recover_alpha_min": float("nan"),
                        "recover_alpha_max": float("nan"),
                    }

            per_steps_results[str(steps)] = {"recover_by_traj": per_traj_results}

        out = {
            "line_index": line_idx,
            "index": sample_index,
            "set_name": set_name,
            "mask_ratio": mask_ratio,
            "alg": args.alg,
            "cfg_scale": args.cfg_scale,
            "eps": args.eps,
            "gumbel_temperature": args.gumbel_temperature,
            "steps_list": steps_list,
            "traj_list": traj_list,
            "recover_by_steps": per_steps_results,
        }

        if results_f is not None:
            results_f.write(json.dumps(out, ensure_ascii=False) + "\n")

    if results_f is not None:
        results_f.close()

    # global summary
    for steps in steps_list:
        for mc_samples in traj_list:
            vals = [v for v in global_by_steps_traj[int(steps)][int(mc_samples)] if math.isfinite(v)]
            if not vals:
                logging.info("Global summary steps=%d traj=%d | no finite results", int(steps), int(mc_samples))
                continue
            logging.info(
                "Global summary steps=%d traj=%d | n=%d mean=%.6f std=%.6f min=%.6f max=%.6f",
                int(steps), int(mc_samples),
                len(vals),
                statistics.fmean(vals),
                statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                min(vals),
                max(vals),
            )

    logging.info("Done.")


if __name__ == "__main__":
    main()
