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

from lit_gpt.diffmodel import TransEncoder, Config
from safetensors.torch import load_file


# =========================
# Logger
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
# Checkpoint helpers (local lit_gpt model)
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

    # buffer dtype align
    param_dtype = next(model.parameters()).dtype
    for name, buf in model.named_buffers():
        if torch.is_floating_point(buf) and buf.dtype != param_dtype:
            model._buffers[name] = buf.to(dtype=param_dtype)

    vocab_size = infer_vocab_size(model, config)
    return model, config, vocab_size


# =========================
# Utils
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
    “linear” 分配：剩余 token 尽量均匀分到剩余步数。
    k_t = ceil(rem / steps_left)
    steps=mask_count => 每步 1 个
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


# =========================
# Gumbel-Max sampling (STRICT step1 method)
# =========================
def sample_gumbel_argmax(
    logits: torch.Tensor,  # [..., V]
    temperature: float,
    torch_gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    严格按你的 step1：
      gumbel = -log(-log(U))
      scores = logits.float() + temperature * gumbel
      argmax(scores)
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
# Step1 generation on FIXED mask positions (from jsonl)
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
    一步生成：对输入里所有 mask 位置一次性采样。
    返回：
      hit_count_per_row: [B]  每条样本 mask 位置命中 GT 的 token 数
      exact_flags:       [B]  mask 位置是否全对
    """
    device = x_masked.device
    B, L = x_masked.shape

    mask_index = (x_masked == int(mask_id))             # [B, L]
    mask_count_per_row = mask_index.sum(dim=1)          # [B]
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
            x_ = torch.cat([x_masked, un_x], dim=0)     # [2B, L]
            logits_all = model(x_)                      # [2B, L, V]
            logits_c, logits_u = torch.chunk(logits_all, 2, dim=0)
            logits_full = logits_u + (float(cfg_scale) + 1.0) * (logits_c - logits_u)  # [B, L, V]
        else:
            logits_full = model(x_masked)               # [B, L, V]

    b_ar = torch.arange(B, device=device).unsqueeze(1).expand(B, max_mask)
    logits_sel = logits_full[b_ar, idx, :]              # [B, max_mask, V]
    sampled = sample_gumbel_argmax(logits_sel, temperature=float(temperature), torch_gen=torch_gen)  # [B, max_mask]
    gt_sel = gt.gather(1, idx)                          # [B, max_mask]

    use_mask = torch.arange(max_mask, device=device).unsqueeze(0) < mask_count_per_row.unsqueeze(1)  # [B, max_mask]
    correct = ((sampled == gt_sel) & use_mask).to(torch.long)
    hit_count = correct.sum(dim=1)                      # [B]
    exact = (hit_count == mask_count_per_row)           # [B]
    return hit_count, exact


# ============================================================
# Multi-step generation (steps=2/5/10/per_token) + hits
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

        mask_index = (x == int(mask_id))  # [B, L]
        remaining = mask_index.sum(dim=1)  # [B]
        max_rem = int(remaining.max().item())
        if max_rem <= 0:
            break

        k_sel = min(int(k_step), int(max_rem))

        # random select among remaining masks
        rand_scores = torch.rand((B, L), generator=torch_gen, device=device, dtype=torch.float32)
        rand_scores = torch.where(mask_index, rand_scores, torch.full_like(rand_scores, -torch.inf))
        _, select_pos = torch.topk(rand_scores, k=k_sel, dim=1, largest=True, sorted=False)  # [B, k_sel]

        with amp_ctx:
            if cfg_scale is not None and float(cfg_scale) > 0.0:
                un_x = x.clone()
                if prompt_len_for_cfg > 0:
                    un_x[:, : int(prompt_len_for_cfg)] = int(mask_id)
                x_ = torch.cat([x, un_x], dim=0)  # [2B, L]
                logits_all = model(x_)  # [2B, L, V]
                logits_c, logits_u = torch.chunk(logits_all, 2, dim=0)
                logits_full = logits_u + (float(cfg_scale) + 1.0) * (logits_c - logits_u)  # [B, L, V]
            else:
                logits_full = model(x)  # [B, L, V]

        b_ar = torch.arange(B, device=device).unsqueeze(1).expand(B, k_sel)
        logits_sel = logits_full[b_ar, select_pos, :]  # [B, k_sel, V]
        sampled = sample_gumbel_argmax(logits_sel, temperature=float(temperature), torch_gen=torch_gen)  # [B, k_sel]

        use_mask = torch.arange(k_sel, device=device).unsqueeze(0) < remaining.unsqueeze(1)  # [B, k_sel]
        if use_mask.any():
            bb = b_ar[use_mask]
            pp = select_pos[use_mask]
            tt = sampled[use_mask]
            x[bb, pp] = tt

    return x


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
) -> Tuple[List[int], float, float]:
    """
    固定 masked_input_ids，跑 runs 次，返回：
      - hit_list: 每次 run 命中多少个 mask token（长度=runs）
      - exact_rate: mask 全对比例
      - mean_hit: 均值
    step_key 仅用于日志/区分，不影响计算
    """
    assert len(gt_ids) == len(masked_input_ids)
    L = len(gt_ids)

    device = next(model.parameters()).device
    gt_1 = torch.tensor(gt_ids, dtype=torch.long, device=device).unsqueeze(0)        # [1, L]
    x0_1 = torch.tensor(masked_input_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]

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
        g.manual_seed(_safe_seed(int(base_seed) + int(done)))

        if int(steps_eff) == 1:
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
        else:
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

            # 每次 run 的 hit：mask_positions 上命中 GT 的数量
            # hit_count: [B]
            hit_count = (x_gen.index_select(1, mask_positions) == gt.index_select(1, mask_positions)).to(torch.long).sum(dim=1)
            exact_flags = (hit_count == int(mask_count))

        hit_cpu = hit_count.detach().to("cpu").tolist()
        hit_list.extend([int(v) for v in hit_cpu])
        exact_hits += int(exact_flags.sum().item())

        done += cur_bs

    exact_rate = exact_hits / float(runs) if runs > 0 else 0.0
    mean_hit = (sum(hit_list) / float(runs)) if runs > 0 else 0.0
    return hit_list, float(exact_rate), float(mean_hit)


# =========================
# Sampling helper: multi-round, no duplicate traj per sample
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

    parser.add_argument("--input_path", type=Path, required=True, help="goodtraj*.jsonl from previous script")
    parser.add_argument("--output_path", type=Path, required=True, help="write eval results jsonl here")
    parser.add_argument("--log_path", type=Path, default=None)

    parser.add_argument("--lit_model_name", type=str, required=True, help="e.g. 1028 -> Diff_LLaMA_1028M")
    parser.add_argument("--ckpt_path", type=str, required=True)

    # generation params (NO TOP-K)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--gen_temperature", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)

    # steps settings (保留)
    parser.add_argument("--steps_2", type=int, default=2)
    parser.add_argument("--steps_5", type=int, default=5)
    parser.add_argument("--steps_10", type=int, default=10)
    parser.add_argument("--enable_per_token_steps", action="store_true")

    # NEW: whether to actually evaluate multi-step
    parser.add_argument(
        "--enable_multistep_eval",
        action="store_true",
        help="If set, also evaluate steps_2/5/10 (+per_token if enabled). Otherwise, only step=1 is evaluated.",
    )

    # runtime
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16/fp16/fp32")

    parser.add_argument(
    "--traj_start",
    type=int,
    default=0,
    help="Start index (0-based) of selected trajectories to evaluate (after sampling).",
    )
    parser.add_argument(
        "--traj_end",
        type=int,
        default=-1,
        help="End index (exclusive) of selected trajectories to evaluate; -1 means till the end.",
    )

    # reproducibility
    parser.add_argument("--seed", type=int, default=12345)

    # multi-round sampling
    parser.add_argument("--target_trajs", type=int, default=200)
    parser.add_argument("--max_traj_per_sample", type=int, default=1)

    args = parser.parse_args()

    setup_logger(args.log_path)
    logging.info("Args: %s", vars(args))

    if not args.input_path.exists():
        raise FileNotFoundError(f"input_path not found: {args.input_path}")

    per_round = int(args.max_traj_per_sample) if args.max_traj_per_sample is not None else None
    if per_round is not None and per_round <= 0:
        raise ValueError("--max_traj_per_sample must be >= 1 (per-round).")

    # Load all samples
    raw_samples: List[Dict[str, Any]] = list(iter_jsonl(args.input_path))
    logging.info("Loaded %d lines from input jsonl.", len(raw_samples))

    # Pre-validate and build per-sample valid trajectory list
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
    # ---- slice selected trajectories: [traj_start:traj_end] ----
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

    # Load model
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    model, _config, vocab_size = load_local_diff_model(
        lit_model_name=args.lit_model_name,
        ckpt_path=args.ckpt_path,
        device=device,
        dtype=dtype,
    )
    logging.info("Model loaded. vocab_size(inferred)=%s", str(vocab_size))

    # Output
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
            "Eval traj: line=%s index=%s set=%s traj_id=%s mask_count=%d multistep=%s",
            str(line_no),
            str(index),
            str(set_name),
            str(traj_id),
            int(mask_count),
            "ON" if args.enable_multistep_eval else "OFF",
        )
        _flush_logs()

        # base seed for this trajectory
        base_seed = _safe_seed(
            int(args.seed)
            + int(index if index is not None else line_no) * 1_000_000
            + int(traj_id if traj_id is not None else 0) * 10_000
        )

        # build step plan: always include step1; optionally include others
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

        t0_all = time.time()
        for key, s in step_items:
            steps_eff = max(1, min(int(s), int(mask_count))) if mask_count > 0 else 1
            # decorrelate per-step seeds a bit
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

                # NEW: per-step per-run hit lists
                "hit_token": hit_by_step,                       # dict: step_key -> [runs]
                "mean_hit_token": mean_hit_by_step,             # dict: step_key -> float
                "success_rate_all_correct": exact_rate_by_step, # dict: step_key -> float

                # step settings (kept)
                "enable_multistep_eval": bool(args.enable_multistep_eval),
                "steps_2": int(args.steps_2),
                "steps_5": int(args.steps_5),
                "steps_10": int(args.steps_10),
                "enable_per_token_steps": bool(args.enable_per_token_steps),

                "target_trajs": int(args.target_trajs),
                "max_traj_per_sample_per_round": int(per_round) if per_round is not None else None,
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
