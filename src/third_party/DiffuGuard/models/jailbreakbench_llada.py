# models/jailbreakbench_llada.py
# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import time
import platform
import torch
import argparse
import logging
import subprocess
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utility.generate_function_llada import generate_llada
from defense_utils import Defender

# ===== Constants =====
DEFAULT_GEN_LENGTH = 128
DEFAULT_STEPS = 64
DEFAULT_BLOCK_LENGTH = 128
DEFAULT_MASK_ID = 126336
DEFAULT_MASK_COUNTS = 36
DEFAULT_TEMPERATURE = 0.5
# Default reference tail block length for detection equals block_length
DEFAULT_REF_TAIL_LEN = DEFAULT_BLOCK_LENGTH

MASK_TOKEN = "<|mdm_mask|>"
START_TOKEN = "<startoftext>"
END_TOKEN = "<endoftext>"
SPECIAL_TOKEN_PATTERN = r"<mask:(\d+)>"

_LOGGED_MASK_CHECK = False

COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"


def _patch_transformers_tied_weights_compat() -> None:
    """
    Compatibility shim for newer transformers versions that expect
    `all_tied_weights_keys` on custom trust_remote_code models.
    """
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return
    original_adjust = getattr(PreTrainedModel, "_adjust_tied_keys_with_tied_pointers", None)
    original_finalize = getattr(PreTrainedModel, "_finalize_model_loading", None)
    if original_adjust is None or original_finalize is None:
        return
    if getattr(original_adjust, "__name__", "") == "_safe_text_diffusion_adjust_tied_keys":
        return

    def _safe_text_diffusion_adjust_tied_keys(self, *args, **kwargs):
        current = getattr(self, "all_tied_weights_keys", None)
        if current is None:
            self.all_tied_weights_keys = {}
        elif isinstance(current, set):
            # Newer transformers expects a mapping with `.keys()`.
            self.all_tied_weights_keys = {k: True for k in current}
        return original_adjust(self, *args, **kwargs)

    @classmethod
    def _safe_text_diffusion_finalize_model_loading(cls, model, load_config, loading_info):
        """
        Transformers>=4.57 calls tie_weights with kwargs. Legacy LLaDA remote-code
        models define tie_weights(self) only, so adapt the bound method shape.
        """
        tie_fn = getattr(model, "tie_weights", None)
        if callable(tie_fn):
            try:
                import inspect
                import types

                sig = inspect.signature(tie_fn)
                if "missing_keys" not in sig.parameters:
                    original_tie_fn = tie_fn

                    def _compat_tie_weights(self, *args, **kwargs):
                        return original_tie_fn()

                    model.tie_weights = types.MethodType(_compat_tie_weights, model)
            except Exception:
                pass
        return original_finalize.__func__(cls, model, load_config, loading_info)

    PreTrainedModel._adjust_tied_keys_with_tied_pointers = _safe_text_diffusion_adjust_tied_keys
    PreTrainedModel._finalize_model_loading = _safe_text_diffusion_finalize_model_loading


def _get_total_ram_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
        total = page_size * phys_pages
        return int(total) if total > 0 else None
    except Exception:
        return None


def _collect_hardware_snapshot(device: torch.device) -> dict:
    snapshot = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": platform.processor() or platform.machine(),
        "cpu_logical_cores": int(os.cpu_count() or 0),
        "cpu_affinity_cores": None,
        "ram_total_bytes": _get_total_ram_bytes(),
        "gpu_device": str(device),
        "gpu_name": None,
        "gpu_vram_total_bytes": None,
        "gpu_vram_free_bytes": None,
        "gpu_vram_used_bytes": None,
    }
    try:
        if hasattr(os, "sched_getaffinity"):
            snapshot["cpu_affinity_cores"] = len(os.sched_getaffinity(0))
    except Exception:
        pass
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            idx = device.index if device.index is not None else torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            total = int(getattr(props, "total_memory", 0) or 0)
            free, _ = torch.cuda.mem_get_info(idx)
            free = int(free)
            used = int(total - free) if total > 0 else None
            snapshot["gpu_name"] = torch.cuda.get_device_name(idx)
            snapshot["gpu_vram_total_bytes"] = total if total > 0 else None
            snapshot["gpu_vram_free_bytes"] = free
            snapshot["gpu_vram_used_bytes"] = used
    except Exception:
        pass
    return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate responses using LLaDA(-Instruct) with a two-prompt schema")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--attack_prompt", type=str, required=True,
                        help="Path to JSON with 'vanilla prompt' / 'refined prompt'")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Optional limit on number of prompts to process.")

    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--gen_length", type=int, default=DEFAULT_GEN_LENGTH)
    parser.add_argument("--block_length", type=int, default=DEFAULT_BLOCK_LENGTH,
                        help="If you prefer tying to steps like older script, set equal to --steps")
    parser.add_argument("--mask_id", type=int, default=DEFAULT_MASK_ID)
    parser.add_argument("--mask_counts", type=int, default=DEFAULT_MASK_COUNTS)

    parser.add_argument("--attack_method", type=str, default="zeroshot",
                        choices=["zeroshot", "DIJA", "PAD", "other"])
    parser.add_argument(
        "--defense_method",
        type=str,
        default=None,
        choices=["self-reminder", "ppl", "para", "retok", "rpo", "diffuguard", "None"],
    )

    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--cfg_scale", type=float, default=0.1)
    parser.add_argument("--remasking", type=str, default="low_confidence",
                        choices=["low_confidence", "random", "rate", "adaptive", "adaptive_step"])
    parser.add_argument("--random_rate", type=float, default=0.0)
    parser.add_argument("--injection_step", type=int, default=None)
    parser.add_argument("--alpha0", type=float, default=0.3)

    parser.add_argument("--sp_mode", type=str, default="off", choices=["off", "logit", "hidden"])
    parser.add_argument("--sp_threshold", type=float, default=0.35)
    parser.add_argument("--refinement_steps", type=int, default=8)
    parser.add_argument("--remask_ratio", type=float, default=0.9)
    parser.add_argument("--suppression_value", type=float, default=1e6)
    parser.add_argument("--fill_all_masks", action="store_true")
    parser.add_argument("--debug_print", action="store_true")

    parser.add_argument("--correct_only_first_block", dest="correct_only_first_block", action="store_true")
    parser.add_argument("--no_correct_only_first_block", dest="correct_only_first_block", action="store_false")
    parser.set_defaults(correct_only_first_block=True)

    # ---- New: automatically pick a GPU (enabled by default) ----
    parser.add_argument("--auto_pick_gpu", dest="auto_pick_gpu", action="store_true")
    parser.add_argument("--no_auto_pick_gpu", dest="auto_pick_gpu", action="store_false")
    parser.set_defaults(auto_pick_gpu=True)

    # ---- New: reference tail length for detection only (does not affect generation) ----
    parser.add_argument("--ref_tail_len", type=int, default=DEFAULT_REF_TAIL_LEN,
                        help="Length of the reference tail masks used ONLY for detection.")

    return parser.parse_args()


# ---------- GPU selection utilities ----------
def _pick_gpu_by_torch() -> int | None:
    """
    Pick the GPU with the most free memory using torch.cuda.mem_get_info().
    Returns the GPU index; returns None on failure.
    """
    if not torch.cuda.is_available():
        return None
    try:
        best_i, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                free, total = torch.cuda.mem_get_info()  # bytes
            if free > best_free:
                best_free, best_i = free, i
        return best_i
    except Exception:
        return None


def _pick_gpu_by_nvidia_smi() -> int | None:
    """
    Fallback: call nvidia-smi to read each GPU's free memory (MiB) and pick the largest.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        frees = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
        if not frees:
            return None
        return max(range(len(frees)), key=lambda i: frees[i])
    except Exception:
        return None


def pick_best_gpu_index() -> int | None:
    """
    Combined strategy: prefer torch-based picking; fall back to nvidia-smi.
    """
    idx = _pick_gpu_by_torch()
    if idx is not None:
        return idx
    return _pick_gpu_by_nvidia_smi()


# ---------- Only process user-side text: expand <mask:x> / default tail ----------
def process_user_text(user_text: str, mask_counts: int) -> str:
    """
    Expand any inline tokens like <mask:N> into N copies of MASK_TOKEN on the user side.
    If no MASK_TOKEN occurs after expansion and mask_counts > 0, append a default tail:
      START_TOKEN + MASK_TOKEN * mask_counts + END_TOKEN
    """
    def repl(m):
        n = int(m.group(1))
        return MASK_TOKEN * max(n, 0)

    processed = re.sub(SPECIAL_TOKEN_PATTERN, repl, user_text)

    if (MASK_TOKEN not in processed) and mask_counts:
        processed = processed + START_TOKEN + (MASK_TOKEN * mask_counts) + END_TOKEN

    return processed


# ---------- Build chat prompt ----------
def build_chat_prompt(tokenizer, user_text_processed: str, is_instruct: bool, system_prompt: str | None = None) -> str:
    if is_instruct:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text_processed})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        if system_prompt:
            return f"{system_prompt}\n\n{user_text_processed}"
        return user_text_processed


def get_tokenized_input(prompt: str, tokenizer, device: torch.device):
    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs["input_ids"].to(device), inputs["attention_mask"].to(device)


def compute_baseline_hidden(
    vanilla_text: str,
    tokenizer,
    model,
    is_instruct: bool,
    system_prompt: str | None,
    *,
    debug_print: bool = False,
) -> torch.Tensor | None:
    """
    Build a baseline using the pure vanilla text WITHOUT appending a mask tail,
    then take the mean over the sequence of the last-layer hidden states.
    Returns a vector of shape [H]; if the model does not expose hidden_states, return None
    (the caller will fall back to logits-based self-protection).
    Note: this is only used by your existing sp_mode='hidden' branch. The new
    tail-only detection is independent of this logic.
    """
    try:
        # Do NOT append a default tail for baseline: mask_counts=0
        baseline_user = process_user_text(vanilla_text, mask_counts=0)
        baseline_prompt = build_chat_prompt(tokenizer, baseline_user, is_instruct, system_prompt)

        baseline_ids, _ = get_tokenized_input(baseline_prompt, tokenizer, model.device)
        out = model(baseline_ids, output_hidden_states=True, return_dict=True)

        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            h_last = out.hidden_states[-1]            # [1, L, H]
            baseline_hidden = h_last.mean(dim=1).squeeze(0).detach()  # [H]
            if debug_print:
                logging.info(f"[Baseline] hidden OK, dim={baseline_hidden.numel()}")
            return baseline_hidden
        else:
            if debug_print:
                logging.info("[Baseline] model.hidden_states is None; fallback to logits-only SP.")
            return None
    except Exception as e:
        logging.info(f"[Baseline] failed to compute hidden: {e}; fallback to logits-only SP.")
        return None


# ======== Added: utilities for a reference tail used only for detection (no effect on generation) ========

def attach_reference_tail(user_text: str, tail_counts: int) -> str:
    """Always append a reference tail at the end ONLY for detection, regardless of existing <mask:n>."""
    if tail_counts <= 0:
        return user_text
    tail = START_TOKEN + (MASK_TOKEN * tail_counts) + END_TOKEN
    return f"{user_text}{tail}"


def build_detection_prompt(tokenizer, user_text: str, is_instruct: bool,
                           system_prompt: str | None, tail_counts: int) -> str:
    """
    Detection prompt:
    1) Expand inline <mask:n> in the text (do NOT append default tail);
    2) Force-append a reference tail of the same length;
    3) Wrap with the chat template.
    """
    processed = process_user_text(user_text, mask_counts=0)  # expand inline <mask:n> only
    user_with_tail = attach_reference_tail(processed, tail_counts)
    return build_chat_prompt(tokenizer, user_with_tail, is_instruct, system_prompt)


@torch.no_grad()
def _find_tail_span(tok: torch.Tensor, mask_id: int, tail_len: int) -> tuple[int, int]:
    """
    In token sequence `tok`, find the span [start, end) that covers the last `tail_len`
    occurrences of `mask_id`. We assume the appended reference tail is a contiguous
    run of MASK_TOKEN repeated tail_len times.
    """
    mask_pos = (tok == mask_id).nonzero(as_tuple=False).squeeze(1)
    assert mask_pos.numel() >= tail_len, f"Reference tail not found or too short: have {mask_pos.numel()}, need {tail_len}"
    last = mask_pos[-tail_len:]                     # take the last tail_len mask positions
    start = int(last.min().item())
    end   = int(last.max().item()) + 1
    return start, end


@torch.no_grad()
def first_step_tail_mean_hidden(model, tokenizer, prompt: str, mask_id: int, tail_len: int) -> torch.Tensor:
    """
    Run a single forward pass and take the mean of the last-layer hidden states
    over the span corresponding to the appended reference tail (length = tail_len).
    For diffusion-LLMs, the last-layer hidden from the first pass approximates the "first step".
    Returns a tensor of shape [H].
    """
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)  # [1, L]
    out = model(ids, output_hidden_states=True, return_dict=True)
    if not hasattr(out, "hidden_states") or out.hidden_states is None:
        raise RuntimeError("Model did not return hidden_states; cannot compute hidden-based SP.")
    h_last = out.hidden_states[-1]                      # [1, L, H]
    tok = ids[0]
    tail_start, tail_end = _find_tail_span(tok, mask_id, tail_len)
    mu = h_last[:, tail_start:tail_end, :].mean(dim=1).squeeze(0).detach()  # [H]
    return mu


def cosine_distance_vec(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float32); b = b.to(torch.float32)
    a = a / (a.norm(p=2) + 1e-12)
    b = b / (b.norm(p=2) + 1e-12)
    return float(1.0 - torch.dot(a, b).item())


# ===================== Generation =====================

def generate_response(
    vanilla_prompt: str,
    prompt: str,
    tokenizer,
    model,
    args,
    baseline_hidden=None,
    system_prompt: str | None = None,
) -> tuple[str, str, list[int], int, str]:
    input_ids, attention_mask = get_tokenized_input(prompt, tokenizer, model.device)
    vanilla_ids, _ = get_tokenized_input(vanilla_prompt, tokenizer, model.device)
    global _LOGGED_MASK_CHECK
    if not _LOGGED_MASK_CHECK:
        mask_id = int(args.mask_id)
        try:
            mdm_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        except Exception:
            mdm_token_id = None
        mask_count = int((input_ids == mask_id).sum().item())
        mdm_count = (
            int((input_ids == mdm_token_id).sum().item())
            if isinstance(mdm_token_id, int)
            else -1
        )
        logging.info(
            "DiffuGuard DIJA mask check: mask_id=%s mdm_token_id=%s count_in_prompt=%d mdm_count=%d",
            mask_id,
            mdm_token_id,
            mask_count,
            mdm_count,
        )
        _LOGGED_MASK_CHECK = True
    # === Compute token length of system/self-reminder and build a protection mask ===
    protected_index = torch.zeros_like(input_ids, dtype=torch.bool, device=model.device)
    if system_prompt:
        # Tokenize system-only with the same template (no generation prompt)
        sys_only_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}],
            tokenize=True, add_generation_prompt=False, return_tensors="pt"
        ).to(model.device)
        sys_len = sys_only_ids.shape[1]
        sys_len = min(sys_len, input_ids.shape[1])  # safety clamp
        protected_index[:, :sys_len] = True

    # Count common prefix length (token-wise)
    matching_count = min(
        sum(a.item() == b.item() for a, b in zip(vanilla_ids[0], input_ids[0])),
        len(vanilla_ids[0]),
    )

    runtime_trace = {}
    output_ids = generate_llada(
        input_ids=input_ids,
        attention_mask=attention_mask,
        model=model,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        mask_id=args.mask_id,

        # —— Pass tokenizer + advanced options ——
        tokenizer=tokenizer,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        random_rate=args.random_rate,
        injection_step=args.injection_step,
        alpha0=args.alpha0,
        sp_mode=args.sp_mode,
        sp_threshold=args.sp_threshold,
        refinement_steps=args.refinement_steps,
        remask_ratio=args.remask_ratio,
        suppression_value=args.suppression_value,
        correct_only_first_block=args.correct_only_first_block,
        fill_all_masks=args.fill_all_masks,
        debug_print=args.debug_print,
        baseline_hidden=baseline_hidden,
        attack_method=args.attack_method,
        pad_anchors=["Step 1:", "Step 2:", "Step 3:"],
        pad_in_uncond=True,   #####
        protected_index=protected_index,
        runtime_trace=runtime_trace,
    )

    decode_start = int(matching_count)
    decode_start_reason = "matching_prefix"
    attack_method_key = str(getattr(args, "attack_method", "")).strip().lower()
    if attack_method_key in {"pad", "dija"}:
        first_mask_pos = (input_ids[0] == int(args.mask_id)).nonzero(as_tuple=False)
        if first_mask_pos.numel() > 0:
            decode_start = int(first_mask_pos.min().item())
            decode_start_reason = "first_mask_token"

    response = tokenizer.batch_decode(output_ids[:, decode_start:], skip_special_tokens=True)[0]
    if args.attack_method == "DIJA":
        # Some outputs may carry content after 'assistant\n'; do a hard split
        response = response.split("assistant\n")[0]

    runtime_prompt_ids: list[int] = []
    runtime_prompt_tokenized = ""
    runtime_inputs = runtime_trace.get("runtime_input_ids")
    if isinstance(runtime_inputs, list) and runtime_inputs:
        first = runtime_inputs[0]
        if isinstance(first, list):
            runtime_prompt_ids = [int(tok) for tok in first]
            try:
                runtime_prompt_tokenized = tokenizer.decode(runtime_prompt_ids, skip_special_tokens=False)
            except Exception:
                runtime_prompt_tokenized = ""

    return response, runtime_prompt_tokenized, runtime_prompt_ids, decode_start, decode_start_reason


def pick_two_prompt_fields(item) -> tuple[str, str]:
    if isinstance(item, str):
        return item, ""
    if not isinstance(item, dict):
        return "", ""
    vanilla = (
        item.get("vanilla prompt")
        or item.get("goal")
        or item.get("Behavior")
        or ""
    )
    refined = (
        item.get("refined prompt")
        or item.get("refined_goal")
        or item.get("Refined_behavior")
        or ""
    )
    return vanilla, refined


def should_use_refined(args, attack_prompt_path, refined_text):
    path_flag = ("refine" in os.path.basename(attack_prompt_path).lower())
    if args.attack_method.lower() == "dija" or path_flag:
        return bool(refined_text.strip())
    return False


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    args = parse_args()
    _patch_transformers_tied_weights_compat()

    # Paper-style DiffuGuard safety uses hidden audit + repair knobs rather than
    # prompt rewriting. If requested, apply conservative defaults unless overridden.
    if args.defense_method and args.defense_method.lower() == "diffuguard":
        if getattr(args, "sp_mode", "off") == "off":
            args.sp_mode = "hidden"
        # These already match the paper defaults in this wrapper, but we log them
        # explicitly to make the mode clear in job outputs.
        logging.info(
            "[DiffuGuard mode] sp_mode=%s sp_threshold=%.3f refinement_steps=%d remask_ratio=%.3f remasking=%s",
            args.sp_mode,
            float(args.sp_threshold),
            int(args.refinement_steps),
            float(args.remask_ratio),
            str(args.remasking),
        )

    # ---- Auto-pick the GPU with the most free memory (can be disabled with --no_auto_pick_gpu) ----
    if torch.cuda.is_available():
        if args.auto_pick_gpu:
            best_idx = pick_best_gpu_index()
            if best_idx is not None:
                try:
                    torch.cuda.set_device(best_idx)
                except Exception:
                    pass  # if set_device fails, fall back to the default CUDA device
                device = torch.device(f"cuda:{best_idx}")
                try:
                    with torch.cuda.device(device):
                        free, total = torch.cuda.mem_get_info()
                    logging.info(f"[GPU] Auto-picked cuda:{best_idx} "
                                 f"(free {free/1024/1024/1024:.1f} GB / total {total/1024/1024/1024:.1f} GB)")
                except Exception:
                    logging.info(f"[GPU] Auto-picked cuda:{best_idx}")
            else:
                device = torch.device("cuda")
                logging.info("[GPU] Auto-pick failed, fallback to default CUDA device.")
        else:
            device = torch.device("cuda")
            logging.info("[GPU] Auto-pick disabled, using default CUDA device.")
    else:
        device = torch.device("cpu")
        logging.info("[GPU] CUDA not available, using CPU.")

    # Defender
    defender = Defender(args.defense_method) if args.defense_method and args.defense_method.lower() != "none" else None

    # Heuristic: is this an instruct model?
    is_instruct = ("instruct" in args.model_path.lower()) or ("1.5" in args.model_path)

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    # Load data
    with open(args.attack_prompt, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        flattened = []
        for _, value in data.items():
            if isinstance(value, list):
                if not value:
                    continue
                # AutoDAN exports often store the usable prompt as the second entry.
                chosen = value[1] if len(value) > 1 else value[0]
                if isinstance(chosen, str):
                    flattened.append(chosen)
            elif isinstance(value, str):
                flattened.append(value)
        data = flattened
    if isinstance(data, list) and args.max_prompts is not None:
        data = data[: max(0, int(args.max_prompts))]

    results = []
    hardware_snapshot = _collect_hardware_snapshot(device)

    with torch.no_grad():
        for item in tqdm(data, desc="Processing data", total=len(data)):
            sample_t0 = time.perf_counter()
            vanilla_text, refined_text = pick_two_prompt_fields(item)
            use_refined = should_use_refined(args, args.attack_prompt, refined_text)

            chosen_text = refined_text if use_refined and refined_text.strip() else vanilla_text
            prompt_type = "refined" if (use_refined and refined_text.strip()) else "vanilla"

            logging.info(
                "[run] attack_method=%s defense_method=%s prompt_type=%s",
                str(args.attack_method),
                str(args.defense_method),
                prompt_type,
            )

            # --- Defense: inject self-reminder into system; otherwise rewrite user text ---
            defense_t0 = time.perf_counter()
            system_for_both = None
            chosen_text_defended = chosen_text
            vanilla_text_defended = vanilla_text
            if defender and getattr(defender, "defense", None) == "self-reminder":
                logging.info(
                    "[defense] applying defense=%s with attack=%s (system injection)",
                    str(getattr(defender, "defense", args.defense_method)),
                    str(args.attack_method),
                )
                ret = defender.handler(chosen_text)
                if isinstance(ret, tuple) and len(ret) >= 1:
                    system_for_both = ret[0]
                elif isinstance(ret, str):
                    system_for_both = ret
            elif defender:
                logging.info(
                    "[defense] applying defense=%s with attack=%s (user rewrite)",
                    str(getattr(defender, "defense", args.defense_method)),
                    str(args.attack_method),
                )
                chosen_text_defended = defender.handler(chosen_text)
                vanilla_text_defended = defender.handler(vanilla_text)
            defense_time_sec = time.perf_counter() - defense_t0

            # --- Expand masks / append default tail ONLY on user-side text (for generation) ---
            prompt_build_t0 = time.perf_counter()
            chosen_user = process_user_text(chosen_text_defended, args.mask_counts)
            vanilla_user = process_user_text(vanilla_text_defended, args.mask_counts)

            # --- Build final prompts (vanilla and chosen share the same system) ---
            prompt = build_chat_prompt(tokenizer, chosen_user, is_instruct, system_for_both)
            vanilla_prompt = build_chat_prompt(tokenizer, vanilla_user, is_instruct, system_for_both)
            prompt_build_time_sec = time.perf_counter() - prompt_build_t0

            # ========= Added: dual forward passes for template-jailbreak detection (tail-only) =========
            detection_t0 = time.perf_counter()
            ref_tail_len = getattr(args, "ref_tail_len", DEFAULT_REF_TAIL_LEN)
            det_vanilla_prompt = build_detection_prompt(tokenizer, vanilla_text, is_instruct, system_for_both, ref_tail_len)
            det_refined_prompt = build_detection_prompt(tokenizer, chosen_text,  is_instruct, system_for_both, ref_tail_len)

            sp_hid_tail = None
            template_attack = None
            try:
                van_mu = first_step_tail_mean_hidden(model, tokenizer, det_vanilla_prompt, args.mask_id, ref_tail_len)  # [H]
                ref_mu = first_step_tail_mean_hidden(model, tokenizer, det_refined_prompt, args.mask_id, ref_tail_len)  # [H]
                sp_hid_tail = cosine_distance_vec(van_mu, ref_mu)  # 1 - cos
                template_attack = (sp_hid_tail >= args.sp_threshold)
            except Exception as e:
                logging.info(f"[TailDetection] skipped due to: {e}")

            # --- Compute baseline_hidden for the legacy sp_mode='hidden' path (independent of tail-only detection) ---
            baseline_hidden = None
            if args.sp_mode == "hidden":
                baseline_hidden = compute_baseline_hidden(
                    vanilla_text=vanilla_text,
                    tokenizer=tokenizer,
                    model=model,
                    is_instruct=is_instruct,
                    system_prompt=system_for_both,
                    debug_print=args.debug_print,
                )
            detection_time_sec = time.perf_counter() - detection_t0

            # --- Debug previews (optional via env flags) ---
            if os.getenv("DEBUG_SHOW_PROMPT", "0") == "1":
                logging.info(("PROMPT_PREVIEW=" + prompt[:1200]).replace("\n", "\\n"))
            if os.getenv("DEBUG_SHOW_TAIL", "0") == "1":
                logging.info(("TAIL_VAN=" + det_vanilla_prompt[:600]).replace("\n", "\\n"))
                logging.info(("TAIL_REF=" + det_refined_prompt[:600]).replace("\n", "\\n"))

            # --- Generate ---
            generation_t0 = time.perf_counter()
            response, runtime_prompt_tokenized, runtime_prompt_ids, decode_start, decode_start_reason = generate_response(
                vanilla_prompt,
                prompt,
                tokenizer,
                model,
                args,
                baseline_hidden=baseline_hidden,
                system_prompt=system_for_both,
            )
            generation_time_sec = time.perf_counter() - generation_t0
            total_time_sec = time.perf_counter() - sample_t0

            logging.info(f"{COLOR_BLUE}Response: {response}{COLOR_RESET}\n")

            # --- Save record: include final_prompt & detection metrics ---
            rec = {
                "vanilla prompt": vanilla_text,
                "refined prompt": refined_text,
                "used_prompt_type": "refined" if (use_refined and refined_text.strip()) else "vanilla",
                "final_prompt": prompt,          # actual model input (for generation)
                "runtime_prompt_tokenized": runtime_prompt_tokenized,
                "runtime_prompt_token_ids": runtime_prompt_ids,
                "decode_start_token_index": decode_start,
                "decode_start_reason": decode_start_reason,
                "response": response,
                # Added: detection metrics (tail-only hidden distance)
                "sp_hid_tail": sp_hid_tail,
                "template_attack": template_attack,
                "ref_tail_len": ref_tail_len,
                # Per-sample timing (seconds)
                "timing_total_sec": total_time_sec,
                "timing_defense_sec": defense_time_sec,
                "timing_prompt_build_sec": prompt_build_time_sec,
                "timing_detection_sec": detection_time_sec,
                "timing_generation_sec": generation_time_sec,
                "hardware": hardware_snapshot,
            }
            if isinstance(item, dict):
                for k in ("source", "category", "goal", "refined_goal", "Behavior", "Refined_behavior", "target"):
                    if k in item:
                        rec[k] = item[k]

            results.append(rec)

    out_dir = os.path.dirname(args.output_json) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logging.info(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
