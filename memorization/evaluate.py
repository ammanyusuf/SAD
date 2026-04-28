"""
evaluate.py — Safety-signal evaluation on extracted trajectories.

Reads a JSONL file produced by extract.py and computes safety signals
at each denoising timestep.  Outputs one JSONL record per prompt:

  {
    "prompt_id": "...",
    "signals": {
      "mask_frac":  [0.95, 0.88, ...],    # already in extraction output
      "perplexity": [42.1, 39.0, ...],    # GPT-2-large per timestep
      "llamaguard": [null, ..., "unsafe"] # sparse — only last N steps
    }
  }

Usage:
  python memorization/evaluate.py \
    --input-jsonl  /tmp/memorization_test.jsonl \
    --output-jsonl /tmp/memorization_eval.jsonl \
    --signals mask_frac,perplexity \
    --perplexity-model gpt2 \
    --device cpu
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s %(levelname)s %(name)s — %(message)s",
  datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("memorization.evaluate")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
  records = []
  with open(path, encoding="utf-8") as fh:
    for line in fh:
      line = line.strip()
      if line:
        records.append(json.loads(line))
  return records


# ─────────────────────────────────────────────────────────────────────────────
# Signal: mask_frac  (pass-through — already computed during extraction)
# ─────────────────────────────────────────────────────────────────────────────

def signal_mask_frac(record: Dict[str, Any]) -> List[float]:
  return list(record.get("timestep_mask_fracs", []))


# ─────────────────────────────────────────────────────────────────────────────
# Signal: perplexity  (GPT-2-large causal LM, per-timestep decoded text)
# ─────────────────────────────────────────────────────────────────────────────

class PerplexityEvaluator:
  """Wraps a GPT-2 (or similar) causal LM for per-sequence perplexity."""

  def __init__(self, model_name: str = "gpt2-large", device: str = "cpu") -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer as _AT
    LOGGER.info("Loading perplexity model: %s on %s", model_name, device)
    self.tokenizer = _AT.from_pretrained(model_name)
    self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    self.model.eval()
    self.device = device
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

  @torch.no_grad()
  def compute(self, texts: Sequence[str]) -> List[Optional[float]]:
    results: List[Optional[float]] = []
    for text in texts:
      if not text or not text.strip():
        results.append(None)
        continue
      try:
        enc = self.tokenizer(
          text, return_tensors="pt", truncation=True, max_length=512
        )
        input_ids = enc["input_ids"].to(self.device)
        if input_ids.shape[1] < 2:
          results.append(None)
          continue
        outputs = self.model(input_ids, labels=input_ids)
        ppl = float(math.exp(outputs.loss.item()))
        results.append(ppl)
      except Exception as exc:
        LOGGER.warning("Perplexity failed for text (len=%d): %s", len(text), exc)
        results.append(None)
    return results


def _decode_gen_region(
  tok_ids: List[int],
  fixed_mask: List[int],
  tokenizer,
  mask_id: int = 126336,
  pad_id: int = 126081,
) -> str:
  """Decode only the generated (non-fixed) positions; drop mask and pad tokens."""
  gen_ids = [
    tid for tid, fm in zip(tok_ids, fixed_mask)
    if fm == 0 and tid != mask_id and tid != pad_id
  ]
  if not gen_ids:
    return ""
  try:
    return tokenizer.decode(gen_ids, skip_special_tokens=True)
  except Exception:
    return ""


def signal_perplexity(
  record: Dict[str, Any],
  evaluator: "PerplexityEvaluator",
  tokenizer_for_decode,
) -> List[Optional[float]]:
  """Decode the generated region at each timestep and compute GPT-2 perplexity.

  Uses prompt_fixed_mask to select only generated (non-prompt) positions,
  then drops mask tokens and pad tokens before decoding.
  """
  timestep_tokens: List[List[int]] = record.get("timestep_tokens", [])
  fixed_mask: List[int] = record.get("prompt_fixed_mask", [])
  if not timestep_tokens:
    return []
  texts = []
  for tok_ids in timestep_tokens:
    text = _decode_gen_region(tok_ids, fixed_mask, tokenizer_for_decode)
    texts.append(text)
  return evaluator.compute(texts)


# ─────────────────────────────────────────────────────────────────────────────
# Signal: llamaguard  (sparse — only last `llamaguard_last_n` steps)
# ─────────────────────────────────────────────────────────────────────────────

def _load_llamaguard(model_path: str, device: str):
  try:
    from src.safety_eval.classifiers.llamaguard import LlamaGuardClassifier
    return LlamaGuardClassifier(model_path, device=device)
  except ImportError as exc:
    LOGGER.error("Could not import LlamaGuardClassifier: %s", exc)
    raise


def signal_llamaguard(
  record: Dict[str, Any],
  classifier,
  tokenizer_for_decode,
  last_n: int = 10,
  batch_size: int = 4,
  mode: str = "gen_only",
) -> List[Optional[str]]:
  """Run LlamaGuard on the last `last_n` timestep partial completions.

  mode="gen_only"  (default, recommended):
    Sends only the decoded generated region as the user turn.
    Steps where the generation is all-masked → empty string → "safe".
    This answers: "does the partial completion itself look harmful?"
    and gives a clean step-by-step signal from safe→unsafe (or safe throughout).

  mode="with_prompt":
    Sends (user=original_prompt, assistant=partial_gen) as a two-turn conversation.
    This answers: "is this conversation harmful at step T?"
    Early steps return "unsafe" purely because the prompt is harmful, even with
    an empty assistant turn — so it's harder to locate the generation transition.

  Mask tokens and pad tokens in the generated region are always stripped.

  Returns a list of length == number of timesteps, with None for skipped steps
  and the raw classifier output string for evaluated steps.
  """
  timestep_tokens: List[List[int]] = record.get("timestep_tokens", [])
  fixed_mask: List[int] = record.get("prompt_fixed_mask", [])
  prompt_text: str = record.get("prompt", "")
  n_steps = len(timestep_tokens)
  if n_steps == 0:
    return []

  results: List[Optional[str]] = [None] * n_steps
  eval_indices = list(range(max(0, n_steps - last_n), n_steps))

  texts_to_eval = []
  lg_tok = classifier.tokenizer  # LlamaGuard's own tokenizer

  for idx in eval_indices:
    partial_completion = _decode_gen_region(
      timestep_tokens[idx], fixed_mask, tokenizer_for_decode
    )

    if mode == "gen_only":
      # Pass just the partial completion; LlamaGuard's classify_texts will
      # wrap it in {"role": "user"} via use_chat_template (no override needed).
      texts_to_eval.append(partial_completion)
    else:
      # mode == "with_prompt": pre-format as a two-turn conversation using
      # LlamaGuard's own chat template, then pass with use_chat_template=False
      # to avoid double-wrapping.
      if getattr(lg_tok, "chat_template", None):
        try:
          formatted = lg_tok.apply_chat_template(
            [
              {"role": "user",      "content": prompt_text},
              {"role": "assistant", "content": partial_completion},
            ],
            tokenize=False,
            add_generation_prompt=True,
          )
          texts_to_eval.append(formatted)
          continue
        except Exception:
          pass
      texts_to_eval.append(f"User: {prompt_text}\nAssistant: {partial_completion}")

  # For with_prompt mode the text is already fully formatted; disable the
  # classifier's own chat template to avoid double-wrapping.
  override_chat = (mode == "with_prompt")
  orig_use_chat = classifier.use_chat_template
  if override_chat:
    classifier.use_chat_template = False
  try:
    raw_outputs, _ = classifier.classify_texts(
      texts_to_eval,
      batch_size=batch_size,
      max_new_tokens=32,
    )
  finally:
    if override_chat:
      classifier.use_chat_template = orig_use_chat

  for local_i, global_i in enumerate(eval_indices):
    # Strip leading "assistant\n\n" that LlamaGuard sometimes prepends.
    # Use removeprefix-style logic, not lstrip (which strips individual chars).
    raw = raw_outputs[local_i]
    if isinstance(raw, str):
      for prefix in ("assistant\n\n", "assistant\n", "assistant"):
        if raw.startswith(prefix):
          raw = raw[len(prefix):]
          break
      raw = raw.strip()
    results[global_i] = raw

  return results


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
  p = argparse.ArgumentParser(
    description="Offline evaluation of denoising trajectories.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  p.add_argument("--input-jsonl", required=True, help="Input JSONL from extract.py.")
  p.add_argument("--output-jsonl", required=True, help="Output JSONL path.")
  p.add_argument(
    "--signals",
    default="mask_frac,perplexity",
    help="Comma-separated signals to compute: mask_frac, perplexity, llamaguard.",
  )
  p.add_argument("--llamaguard-model", default=None,
                 help="Path to LlamaGuard model (required if 'llamaguard' in --signals).")
  p.add_argument("--llamaguard-last-n", type=int, default=10,
                 help="Number of final timesteps to classify with LlamaGuard (saves compute).")
  p.add_argument(
    "--llamaguard-mode",
    default="gen_only",
    choices=["gen_only", "with_prompt"],
    help=(
      "gen_only (default): classify only the partial generated text — empty at early steps → safe. "
      "Answers 'does the completion itself look harmful?' "
      "with_prompt: classify (user=prompt, assistant=partial_gen) as a conversation — "
      "harmful prompts return unsafe even when the generation is empty."
    ),
  )
  p.add_argument("--perplexity-model", default="gpt2-large",
                 help="HuggingFace model name for perplexity scoring.")
  p.add_argument("--device", default=None,
                 help="Torch device (default: cuda if available, else cpu).")
  p.add_argument("--batch-size", type=int, default=4,
                 help="Batch size for LlamaGuard classification.")
  p.add_argument("--decode-tokenizer", default=None,
                 help="Path to tokenizer for decoding timestep tokens. "
                      "If omitted, tries to infer from extraction records (not always possible).")
  return p.parse_args(argv)


def _infer_decode_tokenizer(records: List[Dict[str, Any]], provided_path: Optional[str]):
  """Load the tokenizer used during extraction for decoding timestep tokens."""
  if provided_path:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(provided_path, trust_remote_code=True)

  LOGGER.warning(
    "--decode-tokenizer not provided; falling back to 'gpt2' for decoding. "
    "This may produce garbled text for LLaDA models. "
    "Pass --decode-tokenizer /path/to/LLaDA-8B-Instruct for accurate decoding."
  )
  from transformers import AutoTokenizer
  return AutoTokenizer.from_pretrained("gpt2")


def main(argv=None) -> None:
  args = parse_args(argv)
  signals = [s.strip() for s in args.signals.split(",") if s.strip()]
  LOGGER.info("Signals to compute: %s", signals)

  device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
  LOGGER.info("Using device: %s", device)

  records = _load_jsonl(args.input_jsonl)
  LOGGER.info("Loaded %d records from %s", len(records), args.input_jsonl)

  if not records:
    LOGGER.info("No records to evaluate.")
    return

  # ── Load evaluation resources ─────────────────────────────────────────────
  decode_tokenizer = _infer_decode_tokenizer(records, args.decode_tokenizer)

  ppl_evaluator: Optional[PerplexityEvaluator] = None
  if "perplexity" in signals:
    ppl_evaluator = PerplexityEvaluator(model_name=args.perplexity_model, device=device)

  llamaguard_classifier = None
  if "llamaguard" in signals:
    if not args.llamaguard_model:
      LOGGER.error("--llamaguard-model is required when 'llamaguard' is in --signals.")
      sys.exit(1)
    llamaguard_classifier = _load_llamaguard(args.llamaguard_model, device)

  # ── Per-record evaluation ─────────────────────────────────────────────────
  out_path = Path(args.output_jsonl)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  n_written = 0
  with out_path.open("w", encoding="utf-8") as fout:
    for i, record in enumerate(records):
      if i % 10 == 0:
        LOGGER.info("Evaluating record %d/%d", i, len(records))

      signal_values: Dict[str, Any] = {}

      if "mask_frac" in signals:
        signal_values["mask_frac"] = signal_mask_frac(record)

      if "perplexity" in signals and ppl_evaluator is not None:
        signal_values["perplexity"] = signal_perplexity(record, ppl_evaluator, decode_tokenizer)

      if "llamaguard" in signals and llamaguard_classifier is not None:
        signal_values["llamaguard"] = signal_llamaguard(
          record,
          llamaguard_classifier,
          decode_tokenizer,
          last_n=args.llamaguard_last_n,
          batch_size=args.batch_size,
          mode=args.llamaguard_mode,
        )

      out_record = {
        "prompt_id": record.get("prompt_id"),
        "model": record.get("model"),
        "steps": record.get("steps"),
        "signals": signal_values,
      }
      fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
      fout.flush()
      n_written += 1

  LOGGER.info("Done. Wrote %d evaluation records to %s", n_written, out_path)


if __name__ == "__main__":
  main()
