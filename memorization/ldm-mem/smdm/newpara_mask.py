import argparse
import json
import logging
import math
import random
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

    # buffer dtype align
    param_dtype = next(model.parameters()).dtype
    for name, buf in model.named_buffers():
        if torch.is_floating_point(buf) and buf.dtype != param_dtype:
            model._buffers[name] = buf.to(dtype=param_dtype)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    vocab_size = infer_vocab_size(model, config)
    return model, tokenizer, vocab_size, config


# =========================
# Diffusion / gumbel helpers (aligned with your logic)
# =========================
def add_gumbel_noise_with_generator(
    logits: torch.Tensor, temperature: float, generator: torch.Generator
) -> torch.Tensor:
    """
    logits.exp() / ((-log(u))**temperature)
    (matching your add_gumbel_noise; but supports per-trajectory generator)
    """
    logits_fp64 = logits.to(torch.float64)
    noise = torch.rand(
        logits_fp64.shape,
        device=logits_fp64.device,
        dtype=torch.float64,
        generator=generator,
    ).clamp_min(1e-12)
    gumbel_noise = (-torch.log(noise)) ** float(temperature)
    return logits_fp64.exp() / gumbel_noise


def gt_logprob_under_gumbel_sampling(
    logits_fp64: torch.Tensor, gt_ids: torch.Tensor, temperature: float
) -> torch.Tensor:
    """
    Under your gumbel-max sampling, token distribution = softmax(logits/temperature).
    Return logprob of GT token.
    """
    if temperature <= 0:
        raise ValueError("gumbel_temperature must be > 0")
    scaled = logits_fp64 / float(temperature)
    log_probs = F.log_softmax(scaled, dim=-1)
    return log_probs.gather(1, gt_ids.unsqueeze(1)).squeeze(1)


def _safe_seed(x: int) -> int:
    return int(x) % (2**63 - 1)


def diffusion_recover_total_log_batch(
    model: torch.nn.Module,
    gt: torch.Tensor,                  # [B, L]
    x: torch.Tensor,                   # [B, L] initial masked (dim at masked positions)
    prompt_len: int,
    steps: int,
    alg: str,                          # "origin" or "greddy"
    temperature: float,
    cfg_scale: float,
    eps: float,
    dim: int,                          # mask token id
    generators: List[torch.Generator], # per-trajectory RNG
) -> torch.Tensor:
    """
    Run diffusion recover in parallel for B trajectories (same sample),
    and return total_log per trajectory: shape [B], float64.

    Notes:
    - initial mask pattern is provided by x (so we can record mask_positions / masked_input_ids)
    - all randomness uses per-trajectory generators for reproducibility
    """
    device = gt.device
    B, L = gt.shape

    # map current mask ratio -> t0 (as in your code)
    mask_counts = (x == dim).sum(dim=1).to(torch.float64)
    mask_count_ref = float(mask_counts.max().item()) if B > 0 else 0.0
    p0 = (mask_count_ref / float(L)) if L > 0 else 0.0
    t0 = (p0 - eps) / (1.0 - eps) if (1.0 - eps) != 0 else 1.0
    if not math.isfinite(t0):
        t0 = 1.0
    t0 = max(min(t0, 1.0), eps)

    timesteps = torch.linspace(t0, eps, steps + 1, device=device)
    total_log = torch.zeros((B,), device=device, dtype=torch.float64)

    model_dtype = next(model.parameters()).dtype
    use_amp = (device.type == "cuda") and (model_dtype in (torch.float16, torch.bfloat16))
    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=model_dtype) if device.type == "cuda" else nullcontext()

    with torch.no_grad():
        for i in range(steps):
            mask_index = (x == dim)  # [B, L]
            if int(mask_index.sum().item()) == 0:
                break

            with amp_ctx:
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    if prompt_len > 0:
                        un_x[:, :prompt_len] = dim
                    x_ = torch.cat([x, un_x], dim=0)  # [2B, L]
                    logits_all = model(x_)            # [2B, L, V]
                    logits_c, logits_u = torch.chunk(logits_all, 2, dim=0)
                    logits_full = logits_u + (cfg_scale + 1.0) * (logits_c - logits_u)  # [B, L, V]
                else:
                    logits_full = model(x)  # [B, L, V]

            t = timesteps[i]
            s = timesteps[i + 1]

            if alg == "origin":
                if i < steps - 1:
                    p_transfer = 1.0 - (s / t).item()
                    transfer_pos_mask = torch.zeros((B, L), device=device, dtype=torch.bool)
                    for b in range(B):
                        if not mask_index[b].any():
                            continue
                        r = torch.rand((L,), device=device, generator=generators[b])
                        transfer_pos_mask[b] = (r < p_transfer) & mask_index[b]
                else:
                    transfer_pos_mask = mask_index

                b_idx, pos_idx = transfer_pos_mask.nonzero(as_tuple=True)
                if b_idx.numel() == 0:
                    continue

                sel_logits = logits_full[b_idx, pos_idx, :].to(torch.float64)  # [N, V]
                sel_gt = gt[b_idx, pos_idx]                                     # [N]
                sel_lp = gt_logprob_under_gumbel_sampling(sel_logits, sel_gt, temperature=temperature)  # [N]
                total_log.index_add_(0, b_idx, sel_lp)
                x[b_idx, pos_idx] = sel_gt  # teacher forcing

            elif alg == "greddy":
                frac = (1.0 - (s / t)).item() if i < steps - 1 else 1.0

                for b in range(B):
                    mi = mask_index[b]
                    if not mi.any():
                        continue

                    mask_pos = mi.nonzero(as_tuple=False).squeeze(1)  # [Nmask]
                    logits_masked = logits_full[b, mask_pos, :]       # [Nmask, V]

                    logits_with_noise = add_gumbel_noise_with_generator(
                        logits_masked, temperature=temperature, generator=generators[b]
                    )
                    x0_masked = torch.argmax(logits_with_noise, dim=-1)  # [Nmask]

                    # confidence uses softmax(logits) (no /temperature), matching your greddy
                    logits_masked_fp64 = logits_masked.to(torch.float64)
                    p = F.softmax(logits_masked_fp64, dim=-1)
                    conf = torch.gather(p, dim=-1, index=x0_masked.unsqueeze(-1)).squeeze(-1)  # [Nmask]

                    num_mask = int(mask_pos.numel())
                    k = int(math.floor(num_mask * frac)) if i < steps - 1 else num_mask
                    if k <= 0:
                        continue

                    top = torch.topk(conf, k=k, largest=True).indices  # [k]
                    pos_to_transfer = mask_pos[top]                    # [k]

                    sel_logits = logits_full[b, pos_to_transfer, :].to(torch.float64)  # [k, V]
                    sel_gt = gt[b, pos_to_transfer]                                     # [k]
                    sel_lp = gt_logprob_under_gumbel_sampling(sel_logits, sel_gt, temperature=temperature)  # [k]
                    total_log[b] = total_log[b] + sel_lp.sum()

                    x[b, pos_to_transfer] = sel_gt  # teacher forcing

            else:
                raise NotImplementedError(f"Unknown alg: {alg}")

    return total_log


# =========================
# Core: collect good trajectories (NEW selection logic)
# =========================
def collect_good_trajectories_for_text(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    mask_id: int,
    mask_ratio: float,
    num_traj: int,
    traj_batch_size: int,
    base_seed: int,
    # diffusion/gumbel params
    steps: int,
    alg: str,
    gumbel_temperature: float,
    cfg_scale: float,
    eps: float,
    tail_len: int,
    logp_threshold: float,
) -> Dict[str, Any]:
    """
    NEW:
    - random mask pattern per trajectory (exactly mask_count positions across full length)
    - run diffusion recover (origin/greddy) to compute total_log per trajectory
    - keep trajectories with logp >= threshold
    """
    token_ids_all = tokenizer.encode(text, add_special_tokens=False)
    token_ids = token_ids_all[-tail_len:]
    L = len(token_ids)
    if L < tail_len:
        raise ValueError(f"sequence too short after tail cut: {L} < tail_len={tail_len}")

    mask_count = int(L * mask_ratio)
    if mask_count < 0 or mask_count > L:
        raise ValueError(f"invalid mask_ratio={mask_ratio} for seq_len={L} -> mask_count={mask_count}")

    # keep relation: prompt_len = L - mask_count (matching your diff code)
    prompt_len = L - mask_count

    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    device = next(model.parameters()).device
    gt_1 = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]

    seeds = [_safe_seed(base_seed + i) for i in range(num_traj)]
    good_trajs: List[Dict[str, Any]] = []

    with torch.no_grad():
        for start in range(0, num_traj, traj_batch_size):
            cur_bs = min(traj_batch_size, num_traj - start)
            gt = gt_1.repeat(cur_bs, 1)  # [B, L]

            gens: List[torch.Generator] = []
            for j in range(cur_bs):
                g = torch.Generator(device=device)
                g.manual_seed(int(seeds[start + j]))
                gens.append(g)

            # initial masks (recordable)
            x_init = gt.clone()
            mask_pos_list: List[List[int]] = []
            if mask_count > 0:
                for j in range(cur_bs):
                    scores = torch.rand((L,), device=device, generator=gens[j])
                    _, idx = torch.topk(scores, k=mask_count, largest=True, sorted=False)
                    pos = [int(p) for p in idx.detach().cpu().tolist()]
                    pos_sorted = sorted(pos)
                    mask_pos_list.append(pos_sorted)
                    x_init[j, idx] = mask_id
            else:
                mask_pos_list = [[] for _ in range(cur_bs)]

            total_log = diffusion_recover_total_log_batch(
                model=model,
                gt=gt,
                x=x_init.clone(),  # diffusion modifies x; keep x_init for output
                prompt_len=int(prompt_len),
                steps=int(steps),
                alg=str(alg),
                temperature=float(gumbel_temperature),
                cfg_scale=float(cfg_scale),
                eps=float(eps),
                dim=int(mask_id),
                generators=gens,
            )  # [B] float64

            total_log_cpu = total_log.detach().cpu().tolist()
            x_init_cpu = x_init.detach().cpu().tolist()

            for j in range(cur_bs):
                traj_id = start + j
                logp = float(total_log_cpu[j])
                if (not math.isfinite(logp)) or (logp < logp_threshold):
                    continue

                # joint prob may underflow for very negative logp
                p = math.exp(logp) if logp > -745 else 0.0
                p_percent = p * 100.0

                pos_list_sorted = mask_pos_list[j]
                masked_token_ids = [token_ids[p_] for p_ in pos_list_sorted]
                masked_tokens = [tokens[p_] for p_ in pos_list_sorted]

                good_trajs.append(
                    {
                        "traj_id": traj_id,
                        "seed": int(seeds[traj_id]),
                        "logp": logp,
                        "p": p,
                        "p_percent": p_percent,
                        "mask_positions": pos_list_sorted,
                        "masked_token_ids": masked_token_ids,
                        "masked_tokens": masked_tokens,
                        "masked_input_ids": x_init_cpu[j],

                        # reproducibility + debug
                        "prompt_len": int(prompt_len),
                        "steps": int(steps),
                        "alg": str(alg),
                        "gumbel_temperature": float(gumbel_temperature),
                        "cfg_scale": float(cfg_scale),
                        "eps": float(eps),
                    }
                )

    return {
        "token_ids": token_ids,
        "tokens": tokens,
        "seq_len": L,
        "mask_count": mask_count,
        "prompt_len": int(prompt_len),
        "good_trajectories": good_trajs,
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

    # local lit_gpt model
    parser.add_argument("--lit_model_name", type=str, required=True, help="e.g. 1028 -> Diff_LLaMA_1028M")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="local path or HF id for tokenizer")
    parser.add_argument("--ckpt_path", type=str, required=True)

    # mask id
    parser.add_argument(
        "--mask_id",
        type=int,
        default=32000,
        help="mask token id. Local small model often uses 32000; HF tokenizer might use 126336.",
    )

    # sequence + mask ratio
    parser.add_argument("--tail_len", type=int, default=100)
    parser.add_argument("--default_mask_ratio", type=float, default=0.2)
    parser.add_argument("--use_sample_mask_ratio", action="store_true", default=True)

    # trajectories
    parser.add_argument("--num_traj", type=int, default=64)
    parser.add_argument("--traj_batch_size", type=int, default=128)

    # keep your CLI args (gen_temperature/top_k) but gen_top_k is unused under new logic
    parser.add_argument("--gen_temperature", type=float, default=1.0)
    parser.add_argument("--gen_top_k", type=int, default=40)

    # NEW: diffusion params (defaults chosen to be safe; you can override)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--alg", type=str, default="greddy", choices=["origin", "greddy"])
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-3)

    # threshold
    parser.add_argument(
        "--logp_threshold",
        type=float,
        default=float(math.log(0.001)),
        help="keep trajectories with logp >= threshold (default ln(0.01) ~ -4.60517 => p>=1%)",
    )

    # device/dtype
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    # output/log
    parser.add_argument("--output_path", type=Path, required=True, help="output jsonl for good trajectories only")
    parser.add_argument("--log_path", type=Path, default=None)

    args = parser.parse_args()

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    if args.log_path is not None and args.log_path.name.endswith(".log"):
        args.log_path = add_timestamp_to_path(args.log_path, ts=run_ts)
    if args.output_path.name.endswith(".jsonl"):
        args.output_path = add_run_tags_to_path(args.output_path, run_ts, tag="goodtraj")

    args.output_path = add_range_to_path(args.output_path, int(args.start_index), args.end_index)
    args.log_path = add_range_to_path(args.log_path, int(args.start_index), args.end_index)

    setup_logger(args.log_path)
    logging.info(
        "Starting GOOD trajectory collection | num_traj=%d | logp_threshold=%.6f | alg=%s | steps=%d | gumbel_temp=%.4f",
        args.num_traj, args.logp_threshold, args.alg, args.steps, args.gen_temperature
    )
    logging.info("Args: %s", vars(args))
    if args.gen_top_k is not None:
        logging.info("Note: --gen_top_k is UNUSED under diffusion/gumbel trajectory logic (kept for CLI compatibility).")

    if not args.input_path.exists():
        raise FileNotFoundError(f"input jsonl not found: {args.input_path}")

    # load samples
    samples_all = load_jsonl(args.input_path, args.max_samples, args.shuffle, args.seed)
    start_i = max(0, int(args.start_index))
    end_i = int(args.end_index) if args.end_index is not None else len(samples_all)
    end_i = min(len(samples_all), max(start_i, end_i))
    samples = samples_all[start_i:end_i]
    logging.info("Loaded %d lines (total_loaded=%d, range=[%d:%d])", len(samples), len(samples_all), start_i, end_i)

    # device/dtype
    device = torch.device(args.device)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # load model/tokenizer
    model, tokenizer, vocab_size, _config = load_local_transencoder_and_tokenizer(
        lit_model_name=args.lit_model_name,
        tokenizer_name=args.tokenizer_name,
        ckpt_path=args.ckpt_path,
        device=device,
        dtype=dtype,
    )
    logging.info("Model ready. vocab_size(inferred)=%s", str(vocab_size))

    # output
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.output_path, "w", encoding="utf-8")

    # main loop
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
            "Line %d | sample_index=%s | set=%s | mask_ratio=%.4f | mask_id=%d",
            line_idx, str(sample_index), str(set_name), mask_ratio, args.mask_id
        )

        # per-sample base seed
        base_seed = _safe_seed(args.seed + int(sample_index) * 1_000_000)

        t0 = time.time()
        try:
            ret = collect_good_trajectories_for_text(
                model=model,
                tokenizer=tokenizer,
                text=text,
                mask_id=args.mask_id,
                mask_ratio=mask_ratio,
                num_traj=args.num_traj,
                traj_batch_size=args.traj_batch_size,
                base_seed=base_seed,
                steps=args.steps,
                alg=args.alg,
                gumbel_temperature=args.gen_temperature,
                cfg_scale=args.cfg_scale,
                eps=args.eps,
                tail_len=args.tail_len,
                logp_threshold=args.logp_threshold,
            )
        except Exception as e:
            logging.exception("Failed on line %d sample_index=%s: %s", line_idx, str(sample_index), str(e))
            continue

        good_trajs = ret["good_trajectories"]
        logging.info(
            "Done %d traj | good=%d | seq_len=%d mask_count=%d prompt_len=%d | time=%.2fs",
            args.num_traj, len(good_trajs), ret["seq_len"], ret["mask_count"], ret["prompt_len"], time.time() - t0
        )

        if not good_trajs:
            continue

        out_obj = {
            "line_index": line_idx,
            "index": sample_index,
            "set_name": set_name,
            "text": text,

            "tail_len": args.tail_len,
            "mask_ratio": mask_ratio,
            "mask_id": args.mask_id,

            # diffusion settings
            "prompt_len": ret["prompt_len"],
            "steps": args.steps,
            "alg": args.alg,
            "gumbel_temperature": args.gen_temperature,
            "cfg_scale": args.cfg_scale,
            "eps": args.eps,

            "logp_threshold": args.logp_threshold,
            "token_ids": ret["token_ids"],
            "tokens": ret["tokens"],
            "good_trajectories": good_trajs,
        }

        out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        out_f.flush()

    out_f.close()
    logging.info("All done. Output saved to: %s", str(args.output_path))


if __name__ == "__main__":
    main()
