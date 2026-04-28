import argparse
import logging
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from sampling.llada_engine import LLaDAGenerationEngine, llada_generate
from sampling.sample_text import GenerationSettings, ModelSettings, SafetySettings
from third_party.LLaDA.generate import generate as official_generate
from third_party.mdlm.diffusion import _load_unsafe_tensor


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _LogitsWrapper:
    """Adapter to expose .logits even if the base model returns tuples."""

    def __init__(self, model):
        self.model = model
        self._warned = False

    @property
    def device(self):
        return getattr(self.model, "device", None)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        source = "out.logits"
        logits = getattr(out, "logits", None)
        if logits is None:
            if isinstance(out, tuple) and len(out) > 0:
                logits = out[0]
                source = "out[0]"
            elif hasattr(out, "last_hidden_state"):
                logits = out.last_hidden_state
                source = "last_hidden_state"
            else:
                raise RuntimeError(f"Model output missing logits. Got type={type(out)} keys={getattr(out, 'keys', lambda: [])()}")
            if not self._warned:
                print(f"[warn] Wrapped model output using {source} as logits fallback")
                self._warned = True
        return SimpleNamespace(logits=logits)


def _encode_prompt(
    tokenizer, model_name: str, prompt: str, device: str
) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
    use_chat_template = False
    if "llada-8b-base" in model_name:
        use_chat_template = False
    elif "llada-8b-instruct" in model_name:
        use_chat_template = True
    elif getattr(tokenizer, "chat_template", None) is not None:
        use_chat_template = False

    rendered_prompt = prompt
    if use_chat_template:
        rendered_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    encoded = tokenizer(
        [rendered_prompt],
        add_special_tokens=not use_chat_template,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask, rendered_prompt


def _resolve_unsafe_artifact(path: Optional[Path]) -> Optional[Path]:
    """Allow unsafe artifacts as a single file or a directory of shards."""
    if path is None:
        return None
    path = Path(path).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Unsafe artifact path not found: {path}")

    # Prefer safetensors, then pt/pth/bin
    shard_globs = [
        "*.safetensors",
        "*.pt",
        "*.pth",
        "*.bin",
    ]
    shards = []
    for pattern in shard_globs:
        shards = sorted(path.glob(pattern))
        if shards:
            break
    if not shards:
        raise FileNotFoundError(f"No unsafe artifact shards found under {path}")
    if len(shards) == 1:
        return shards[0]

    tensors = []
    for shard in shards:
        t = _load_unsafe_tensor(str(shard))
        tensors.append(t)
    shapes = {tuple(t.shape[1:]) for t in tensors}
    if len(shapes) != 1:
        raise RuntimeError(f"Unsafe shards have mismatched shapes (sans batch): {shapes}")
    merged = torch.cat(tensors, dim=0)
    tmp_dir = Path(tempfile.mkdtemp(prefix="unsafe_merge_"))
    tmp_path = tmp_dir / "unsafe_merged.pt"
    torch.save(merged.cpu(), tmp_path)
    print(f"[stage] Merged {len(shards)} unsafe shards into {tmp_path} shape={tuple(merged.shape)}")
    return tmp_path


def _trim_to_eos(tokens: list[int], eos_id: Optional[int]) -> list[int]:
    if eos_id is None:
        return tokens
    try:
        idx = tokens.index(eos_id)
        return tokens[: idx + 1]
    except ValueError:
        return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference vs local LLaDA sampler parity check.")
    parser.add_argument("--checkpoint-path", required=True, help="HF model id or local path")
    parser.add_argument("--tokenizer-path", default=None, help="HF tokenizer id or local path")
    parser.add_argument("--model-name", default="llada-8b-base")
    parser.add_argument("--prompt", default="Explain why Greenland is mostly covered in ice despite its name.")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--eta", type=float, default=0.0, help="Safety eta; >0 enables repellency on our sampler.")
    parser.add_argument("--t-start", dest="t_start", type=int, default=None, help="Safety t_start (optional).")
    parser.add_argument("--t-end", dest="t_end", type=int, default=None, help="Safety t_end (optional).")
    parser.add_argument("--unsafe-artifacts", type=Path, default=None, help="Path to unsafe tensor (safe tensor).")
    parser.add_argument("--unsafe-prototypes", type=Path, default=None, help="Path to unsafe prototypes (optional).")
    parser.add_argument(
        "-n",
        "--no-semantic-gating",
        dest="use_semantic_gating",
        action="store_false",
        help="Disable semantic gating (default).",
    )
    parser.add_argument(
        "--semantic-gating",
        dest="use_semantic_gating",
        action="store_true",
        help="Enable semantic gating when building repellency.",
    )
    parser.set_defaults(use_semantic_gating=False)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=False,
        help="Force loading from local HF cache only.",
    )
    parser.add_argument(
        "--allow-online",
        dest="local_files_only",
        action="store_false",
        help="Permit downloads if not found locally.",
    )
    parser.set_defaults(local_files_only=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.gen_length % args.block_length != 0:
        raise ValueError("block_length must divide gen_length for the reference sampler.")
    num_blocks = args.gen_length // args.block_length
    if args.steps % num_blocks != 0:
        raise ValueError("steps must be divisible by the number of blocks for the reference sampler.")

    _set_seed(args.seed)
    print(
        f"[config] model={args.model_name} ckpt={args.checkpoint_path} "
        f"tokenizer={args.tokenizer_path or args.checkpoint_path} steps={args.steps} "
        f"gen_length={args.gen_length} block_length={args.block_length} seed={args.seed} "
        f"local_files_only={args.local_files_only} eta={args.eta} t_start={args.t_start} t_end={args.t_end} "
        f"unsafe={args.unsafe_artifacts} prototypes={args.unsafe_prototypes} semantic_gating={args.use_semantic_gating}"
    )

    if args.local_files_only:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        print("[config] HF offline mode enabled (TRANSFORMERS_OFFLINE=1, HF_HUB_OFFLINE=1)")

    unsafe_path = _resolve_unsafe_artifact(args.unsafe_artifacts) if args.eta > 0 else None

    model_settings = ModelSettings(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        tokenizer_name=args.tokenizer_path or args.checkpoint_path,
        precision=args.precision,
    )
    generation_settings = GenerationSettings(
        max_new_tokens=args.gen_length,
        prefix_length=0,
        sampling_steps=args.steps,
        batch_size=1,
        seed=args.seed,
        sampling_mode="pure_diffusion",
        block_length=args.block_length,
        temperature=args.temperature,
        precision=args.precision,
    )
    print("[stage] Loading model/tokenizer...")
    load_start = time.perf_counter()
    engine = LLaDAGenerationEngine(
        prompts=[],
        model=model_settings,
        generation=generation_settings,
        safety=SafetySettings(
            enabled=args.eta > 0,
            eta=args.eta,
            t_start=args.t_start,
            t_end=args.t_end,
            unsafe_artifacts=unsafe_path,
            unsafe_prototypes=args.unsafe_prototypes,
            use_semantic_gating=args.use_semantic_gating,
        ),
        shard_metadata={},
    )
    engine._prepare_model()
    if engine.safety_settings.enabled:
        print("[stage] Building repellency (safety)...")
        rep_start = time.perf_counter()
        engine.repellency = None
        engine._repellency_logs = {
            "mean_rho": [],
            "argmax_changed_masked": [],
            "beta_hat_mean": [],
            "beta_hat_p95": [],
            "beta_hat_max": [],
            "guidance_strength_mean": [],
            "schedule_weight_mean": [],
            "log_beta_raw_mean": [],
            "log_beta_rel_mean": [],
            "log_beta_raw_max": [],
            "log_beta_rel_max": [],
        }
        engine._build_repellency()
        rep_elapsed = time.perf_counter() - rep_start
        print(f"[stage] Repellency ready in {rep_elapsed:.2f}s")
        if engine.repellency is None:
            print("[warn] Safety enabled (eta>0) but repellency is not active (missing unsafe artifacts?)")
    load_elapsed = time.perf_counter() - load_start
    print(f"[stage] Model/tokenizer loaded in {load_elapsed:.2f}s")
    tokenizer = engine.tokenizer
    model = _LogitsWrapper(engine.model) if engine.model is not None else None
    engine.model = model
    if tokenizer is None or model is None or engine.mask_token_id is None:
        raise RuntimeError("Model, tokenizer, or mask token failed to load.")

    input_ids, attention_mask, rendered_prompt = _encode_prompt(
        tokenizer, args.model_name, args.prompt, engine.device
    )
    prompt_len = int(attention_mask.sum().item()) if attention_mask is not None else input_ids.shape[1]
    logits_eos_inf = False
    confidence_eos_eot_inf = False
    if "llada-8b-instruct" in args.model_name:
        logits_eos_inf = True
        confidence_eos_eot_inf = True

    mask_id = int(engine.mask_token_id)
    effective_vocab = int(engine.effective_vocab if engine.effective_vocab is not None else len(tokenizer))

    torch.manual_seed(args.seed)
    print("[stage] Running official sampler...")
    official_start = time.perf_counter()
    official_tokens = official_generate(
        model,
        input_ids,
        attention_mask=attention_mask,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=mask_id,
        logits_eos_inf=logits_eos_inf,
        confidence_eos_eot_inf=confidence_eos_eot_inf,
    )
    official_elapsed = time.perf_counter() - official_start
    print(f"[stage] Official sampler finished in {official_elapsed:.2f}s")

    runs = []
    for label, rep in (("ours_no_rep", None), ("ours_rep", engine.repellency if engine.safety_settings.enabled else None)):
        if label == "ours_rep" and rep is None:
            print("[info] Skipping ours_rep because repellency is not active.")
            continue
        torch.manual_seed(args.seed)
        print(f"[stage] Running our sampler ({label})...")
        run_start = time.perf_counter()
        toks, _ = llada_generate(
            model=model,
            prompt=input_ids,
            attention_mask=attention_mask,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=mask_id,
            effective_vocab=effective_vocab,
            repellency=rep,
            mask_schedule=None,
            logits_eos_inf=logits_eos_inf,
            confidence_eos_eot_inf=confidence_eos_eot_inf,
            eos_id=engine.eos_id,
            eot_id=engine.eot_id,
            pad_id=tokenizer.pad_token_id,
            stop_tokens=engine.stop_tokens,
            extra_ban_ids=engine._ban_generation_ids,
            sampling_mode="pure_diffusion",
        )
        elapsed = time.perf_counter() - run_start
        print(f"[stage] Our sampler ({label}) finished in {elapsed:.2f}s")
        if rep is not None:
            print("[info] Repellency was active for this run.")
        runs.append((label, toks))

    official_list_full = official_tokens[0].detach().cpu().tolist()
    official_list = _trim_to_eos(official_list_full, engine.eos_id)
    official_text = tokenizer.decode(official_list, skip_special_tokens=False)

    print(f"Rendered prompt (len={prompt_len}): {rendered_prompt}")
    print(f"Official tokens (first 120): {official_list[:120]}")
    print("\nOfficial decoded:\n" + official_text)

    for label, toks in runs:
        tok_list_full = toks[0].detach().cpu().tolist()
        tok_list = _trim_to_eos(tok_list_full, engine.eos_id)
        text = tokenizer.decode(tok_list, skip_special_tokens=False)
        print(f"\n[{label}] tokens (first 120): {tok_list[:120]}")
        print(f"\n[{label}] decoded:\n" + text)

        tokens_equal = len(official_list) == len(tok_list) and all(
            o == m for o, m in zip(official_list, tok_list)
        )
        print(f"\nToken parity vs official ({label}): {tokens_equal}")
        if not tokens_equal:
            mismatch_idx = None
            for idx, (o_tok, m_tok) in enumerate(zip(official_list, tok_list)):
                if o_tok != m_tok:
                    mismatch_idx = idx
                    break
            if mismatch_idx is not None:
                print(
                    f"[{label}] First mismatch at position {mismatch_idx}: official={official_list[mismatch_idx]} ours={tok_list[mismatch_idx]}"
                )
            if len(official_list) != len(tok_list):
                print(f"[{label}] Length mismatch: official={len(official_list)} ours={len(tok_list)}")


if __name__ == "__main__":
    main()
