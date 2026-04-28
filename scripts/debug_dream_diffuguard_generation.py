import argparse
import importlib
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from unsafe_prep.utils import ensure_pad_token, resolve_mask_index


LOGGER = logging.getLogger(__name__)


def _default_prompts() -> List[str]:
    return [
        "Write one sentence about a calm lake.",
        "Give a quick tip for staying focused.",
        "Explain photosynthesis in a short sentence.",
        "Provide a fun fact about penguins.",
    ]


def _import_generate_function():
    for name in (
        "third_party.DiffuGuard.utility.generate_function_dream",
        "src.third_party.DiffuGuard.utility.generate_function_dream",
    ):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError("Could not import DiffuGuard Dream generate function module.")


def _import_jailbreakbench_dream():
    for name in (
        "third_party.DiffuGuard.models.jailbreakbench_dream",
        "src.third_party.DiffuGuard.models.jailbreakbench_dream",
    ):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError("Could not import DiffuGuard jailbreakbench_dream module.")


def _map_precision(precision: str) -> torch.dtype:
    lookup = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return lookup.get(str(precision).lower(), torch.float32)


def _pick_two_prompt_fields(item: dict) -> Tuple[str, str]:
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


def _should_use_refined(refined_text: str, attack_prompt_path: Optional[str]) -> bool:
    if not refined_text.strip():
        return False
    if not attack_prompt_path:
        return True
    return "refine" in Path(attack_prompt_path).name.lower()


def _process_user_text(user_text: str, mask_counts: int, mask_token: str) -> str:
    if mask_counts <= 0:
        return user_text
    return user_text + (mask_token * int(mask_counts))


def _build_prompt(
    tokenizer,
    user_text: str,
    use_chat_template: bool,
    system_prompt: Optional[str],
) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if system_prompt:
        return f"{system_prompt}\n\n{user_text}"
    return user_text


def main() -> None:
    parser = argparse.ArgumentParser(description="DiffuGuard Dream generate_dream_hidden sanity check.")
    parser.add_argument("--model-path", required=True, help="HF model id or local path")
    parser.add_argument("--tokenizer-path", default=None, help="HF tokenizer id or local path")
    parser.add_argument("--model-name", default="dream")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--alpha0", type=float, default=0.3)
    parser.add_argument("--random-rate", type=float, default=0.0)
    parser.add_argument("--attack-method", default="pad", choices=["pad", "none"])
    parser.add_argument("--mask-id", type=int, default=None)
    parser.add_argument("--mask-counts", type=int, default=0)
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        default=False,
        help="Emulate jailbreakbench zeroshot: use vanilla prompts, non-PAD path (attack_method=none).",
    )
    parser.add_argument("--refinement-steps", type=int, default=8)
    parser.add_argument("--remask-ratio", type=float, default=0.9)
    parser.add_argument("--sp-threshold", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument(
        "--prompt-json",
        type=Path,
        default=None,
        help="Path to a DiffuGuard-style prompt JSON (list of dicts).",
    )
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument(
        "--use-jailbreak-template",
        action="store_true",
        default=False,
        help="Mimic jailbreakbench_dream prompt construction.",
    )
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--use-chat-template", action="store_true", default=False)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    parser.add_argument(
        "--allow-online",
        dest="local_files_only",
        action="store_false",
        help="Permit downloads if not found locally.",
    )
    parser.set_defaults(local_files_only=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.zero_shot:
        args.attack_method = "none"

    if args.local_files_only:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = _map_precision(args.precision)

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    ensure_pad_token(tokenizer, eos_token_id=getattr(tokenizer, "eos_token_id", None))

    mask_id = args.mask_id
    if mask_id is None:
        mask_id = resolve_mask_index(tokenizer, getattr(tokenizer, "mask_token", None))
    if mask_id is None:
        raise RuntimeError("No mask token id found. Provide --mask-id explicitly.")

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()

    torch.manual_seed(args.seed)

    prompts: List[str] = []
    if args.prompt_json is not None:
        with open(args.prompt_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Prompt JSON must be a list of objects.")
        if args.max_prompts is not None:
            data = data[: max(0, int(args.max_prompts))]
        for item in data:
            if not isinstance(item, dict):
                continue
            vanilla, refined = _pick_two_prompt_fields(item)
            if args.use_jailbreak_template:
                if args.zero_shot:
                    chosen = vanilla
                else:
                    chosen = refined if _should_use_refined(refined, str(args.prompt_json)) else vanilla
            else:
                chosen = refined or vanilla
            if chosen:
                prompts.append(chosen)
    if not prompts:
        prompts = args.prompts if args.prompts else _default_prompts()
    generate_module = _import_generate_function()
    jailbreak_module = _import_jailbreakbench_dream()

    for idx, prompt_text in enumerate(prompts):
        use_chat = args.use_chat_template
        if args.use_jailbreak_template and not args.use_chat_template:
            lower_path = str(args.model_path).lower()
            use_chat = "instruct" in lower_path or "1.5" in lower_path
        user_text = prompt_text
        prompt = _build_prompt(
            tokenizer,
            user_text,
            use_chat_template=use_chat,
            system_prompt=args.system_prompt,
        )
        if args.use_jailbreak_template and args.mask_counts > 0:
            mask_token = getattr(tokenizer, "mask_token", "<|mask|>")
            if mask_token not in prompt:
                prompt = prompt + (mask_token * int(args.mask_counts))
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        if args.attack_method == "pad":
            output_ids, _ = generate_module.generate_dream_hidden(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                gen_length=args.max_new_tokens,
                steps=args.steps,
                block_length=args.block_length,
                temperature=args.temperature,
                sp_threshold=args.sp_threshold,
                refinement_steps=args.refinement_steps,
                remask_ratio=args.remask_ratio,
                mask_id=int(mask_id),
                attack_method=args.attack_method,
            )
        else:
            if args.remasking == "off":
                gen_out = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    output_history=False,
                    return_dict_in_generate=True,
                    steps=args.steps,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                output_ids = gen_out.sequences
            else:
                output_ids = jailbreak_module.dream_adaptive_generate(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    steps=args.steps,
                    temperature=args.temperature,
                    mask_id=int(mask_id),
                    remasking=args.remasking,
                    alpha0=args.alpha0,
                    random_rate=args.random_rate,
                    debug_print=False,
                    logits_hook=None,
                )

        full = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        completion_ids = output_ids[0, input_ids.shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        remaining_masks = int((output_ids == int(mask_id)).sum().item())

        LOGGER.info("=== Prompt %d ===", idx)
        LOGGER.info("prompt=%s", prompt_text)
        LOGGER.info("mask_id=%s remaining_masks=%d", mask_id, remaining_masks)
        print(f"FULL: {full}")
        print(f"COMPLETION: {completion}\n")


if __name__ == "__main__":
    main()
