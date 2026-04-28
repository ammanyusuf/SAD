import argparse
import logging
import os
from pathlib import Path
from typing import List

import torch

from sampling.llada_engine import LLaDAGenerationEngine
from sampling.sample_text import (
    GenerationSettings,
    ModelSettings,
    PromptRecord,
    SafetySettings,
)


LOGGER = logging.getLogger(__name__)


def _default_prompts() -> List[str]:
    return [
        "Explain how rainbows form in one sentence.",
        "Give a short recipe idea for a quick lunch.",
        "Summarize the plot of a classic novel.",
        "Provide a fun fact about space.",
        "Write a single line of poetic imagery.",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick LLaDA generation sanity check.")
    parser.add_argument("--checkpoint-path", required=True, help="HF model id or local path")
    parser.add_argument("--model-name", default="llada-8b-base")
    parser.add_argument("--tokenizer-name", default=None, help="HF tokenizer id or local path")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sampling-steps", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eta-values", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    parser.add_argument("--unsafe-artifacts", type=Path, default=None)
    parser.add_argument("--unsafe-prototypes", type=Path, default=None)
    parser.add_argument(
        "--sampling-mode",
        choices=["pure_diffusion", "block_diffusion", "ar"],
        default="pure_diffusion",
    )
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--prompts", nargs="*", default=None)
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

    logging.basicConfig(level=logging.INFO)

    if args.local_files_only:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    prompts_text = args.prompts if args.prompts else _default_prompts()
    prompt_records = [
        PromptRecord(prompt_id=str(idx), prompt=text) for idx, text in enumerate(prompts_text)
    ]

    model_settings = ModelSettings(
        model_name=args.model_name,
        checkpoint_path=Path(args.checkpoint_path),
        tokenizer_name=args.tokenizer_name or args.checkpoint_path,
        precision=args.precision,
    )
    generation_settings = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        prefix_length=0,
        sampling_steps=args.sampling_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        block_length=args.block_length,
        sampling_mode=args.sampling_mode,
    )

    engine = LLaDAGenerationEngine(
        prompts=prompt_records,
        model=model_settings,
        generation=generation_settings,
        safety=SafetySettings(enabled=False),
        shard_metadata={},
    )

    # Load model/tokenizer once, then rebuild repellency per eta.
    engine._prepare_model()
    tokenizer = engine.tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer failed to load.")
    vocab_size = len(tokenizer)

    for eta in args.eta_values:
        LOGGER.info("=== Running safety eta=%.2f ===", eta)
        safety_cfg = SafetySettings(
            enabled=eta > 0,
            eta=float(eta),
            unsafe_artifacts=args.unsafe_artifacts,
            unsafe_prototypes=args.unsafe_prototypes,
        )
        engine.safety_settings = safety_cfg
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

        with torch.inference_mode():
            results = engine._generate_conditioned()

        for res in results:
            token_ids = res.token_ids
            max_token = max(token_ids) if token_ids else -1
            oov = sum(1 for tid in token_ids if tid >= vocab_size)
            completion = tokenizer.decode(
                token_ids[res.prompt_length:], skip_special_tokens=False
            )
            print(
                f"[eta={eta}] prompt_id={res.prompt_id} max_id={max_token} "
                f"oov_count={oov} completion={completion}"
            )
            assert oov == 0, f"OOV tokens detected for prompt {res.prompt_id}"


if __name__ == "__main__":
    main()
