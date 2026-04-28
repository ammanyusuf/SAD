"""
extract.py — Denoising trajectory extractor for LLaDA / Dream.

Runs the generation loop for a given engine and records the full token state
(entire sequence) at every denoising step.  Supports both standard generation
(masks appended after a clean prompt) and infill mode (masks embedded inside
the prompt, as in DIJA / PAD attacks or arbitrary infill experiments).

Output format — one JSONL record per prompt:

  {
    "prompt_id": "...",
    "prompt": "...",
    "model": "llada-instruct",
    "engine": "llada_local",
    "steps": 128,
    "infill": false,
    "seq_len": 384,
    "prompt_fixed_mask": [1, 1, ..., 0, 0, ...],
    "timestep_tokens":    [[tok_id_0, ...], ...],
    "timestep_mask_fracs": [0.95, 0.88, ...],
    "final_completion": "..."
  }

``prompt_fixed_mask[i] == 1`` means position i held a fixed prompt token;
``== 0`` means it was a denoising target.  Downstream analysis should always
use this bitmask to index into ``timestep_tokens`` rather than assuming any
prefix/suffix split.

Supported engines  (--engine):
  llada_local    — src/sampling/llada_engine.py :: llada_generate()
                   Native step_callback and infill_prompt_masks support.
  llada_upstream — src/third_party/LLaDA/generate.py :: generate()
                   logits_hook only; step state captured via hook on x.
                   Infill not supported (prompt region frozen upstream).
  dream          — model.diffusion_generate() via DreamBackend
                   generation_logits_hook_func; step state captured similarly.
                   Infill not supported.

Usage:
  # Standard, local engine (default)
  python memorization/extract.py \\
    --model llada-instruct \\
    --checkpoint /path/to/LLaDA-8B-Instruct \\
    --tokenizer  /path/to/LLaDA-8B-Instruct \\
    --prompts-json /path/to/prompts.json \\
    --output-jsonl /tmp/out.jsonl \\
    --steps 128 --max-new-tokens 256 --batch-size 4

  # Infill (only with llada_local)
  python memorization/extract.py ... --infill

  # Upstream LLaDA reference implementation
  python memorization/extract.py ... --engine llada_upstream

  # Dream
  python memorization/extract.py ... --engine dream --model dream-instruct
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

# ── PYTHONPATH guard ─────────────────────────────────────────────────────────
try:
  from src.utils.constants import LLADA_MASK_TOKEN_ID, LLADA_EOS_TOKEN_ID, LLADA_EOT_TOKEN_ID
  from src.unsafe_prep.utils import ensure_pad_token
except ImportError as _e:
  print(
    f"[ERROR] Import failed: {_e}\n"
    "Make sure PYTHONPATH includes the repo root, src/, and src/third_party/mdlm/.\n"
    "Example: export PYTHONPATH=$(pwd):$(pwd)/src:$(pwd)/src/third_party/mdlm",
    file=sys.stderr,
  )
  raise

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s %(levelname)s %(name)s — %(message)s",
  datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("memorization.extract")


# ─────────────────────────────────────────────────────────────────────────────
# Engine adapters
#
# Each adapter exposes a single method:
#
#   generate(
#       model, tokenizer, input_ids, attention_mask,
#       steps, gen_length, block_length, temperature,
#       mask_id, effective_vocab, eos_id, eot_id,
#       infill,
#       step_callback: Callable[[int, Tensor], None],
#   ) -> Tensor          # full output sequence (B, L)
#
# The step_callback is called after every denoising step with
# (step_index: int, x: Tensor[B, L]) — x is the *post-step* state.
# ─────────────────────────────────────────────────────────────────────────────

class _LLaDALocalAdapter:
  """Wraps sampling/llada_engine.py :: llada_generate().

  This is the richest adapter: native step_callback and infill_prompt_masks.
  """

  name = "llada_local"

  def generate(
    self,
    model, tokenizer, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor],
    steps: int, gen_length: int, block_length: int, temperature: float,
    mask_id: int, effective_vocab: int, eos_id: int, eot_id: int,
    infill: bool,
    step_callback: Callable[[int, torch.Tensor], None],
  ) -> torch.Tensor:
    from src.sampling.llada_engine import llada_generate

    gen_kwargs: Dict[str, Any] = dict(
      model=model,
      prompt=input_ids,
      attention_mask=attention_mask,
      steps=steps,
      gen_length=gen_length,
      block_length=block_length,
      temperature=temperature,
      cfg_scale=0.0,
      remasking="low_confidence",
      mask_id=mask_id,
      effective_vocab=effective_vocab,
      eos_id=eos_id,
      eot_id=eot_id,
      pad_id=tokenizer.pad_token_id,
      step_callback=step_callback,
    )
    if infill:
      gen_kwargs["infill_prompt_masks"] = True

    output_ids, _ = llada_generate(**gen_kwargs)
    return output_ids


class _LLaDAUpstreamAdapter:
  """Wraps third_party/LLaDA/generate.py :: generate().

  The upstream implementation does not expose a post-step callback; it exposes
  a logits_hook that fires *before* sampling (receives logits and current x).
  We capture the pre-step x there — one step behind — and flush the last state
  from the final output tensor.  This gives an identical trajectory to the
  local engine (both record x *after* each transfer step).

  Infill (masks in prompt) is not supported by the upstream implementation;
  passing --infill with this adapter raises an error.
  """

  name = "llada_upstream"

  def generate(
    self,
    model, tokenizer, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor],
    steps: int, gen_length: int, block_length: int, temperature: float,
    mask_id: int, effective_vocab: int, eos_id: int, eot_id: int,
    infill: bool,
    step_callback: Callable[[int, torch.Tensor], None],
  ) -> torch.Tensor:
    if infill:
      raise ValueError("llada_upstream engine does not support --infill (prompt region is frozen).")

    from src.third_party.LLaDA.generate import generate as _upstream_generate

    # The upstream logits_hook is called at the *start* of each step, before
    # sampling.  The `x` argument is the sequence entering that step, which
    # equals the sequence *after* the previous step — exactly what we want.
    # We therefore capture x inside the hook and tag it with (step - 1).
    # After the loop ends we emit the final state using the output tensor.
    step_counter: List[int] = [0]

    def _hook(logits, *, x, t, mask_index, prompt_index, attention_mask, extra):
      # x here is the state *before* this step's sampling — i.e. the state
      # *after* the previous step.  step_counter[0] == 0 on the first call,
      # so this correctly emits the state after step 0, 1, ...
      if step_counter[0] > 0:
        step_callback(step_counter[0] - 1, x)
      step_counter[0] += 1
      return logits  # pass logits through unmodified

    output_ids = _upstream_generate(
      model=model,
      prompt=input_ids,
      attention_mask=attention_mask,
      steps=steps,
      gen_length=gen_length,
      block_length=block_length,
      temperature=temperature,
      cfg_scale=0.0,
      remasking="low_confidence",
      mask_id=mask_id,
      logits_hook=_hook,
    )
    # Emit the final post-step state (the last step's x == output_ids).
    if step_counter[0] > 0:
      step_callback(step_counter[0] - 1, output_ids)
    return output_ids


class _DreamAdapter:
  """Wraps DreamBackend / model.diffusion_generate().

  Dream exposes generation_logits_hook_func, which fires before each step's
  sampling — same timing as the upstream LLaDA hook.  We apply the same
  one-step-lag pattern to recover the post-step state.

  Infill is not supported.
  """

  name = "dream"

  def generate(
    self,
    model, tokenizer, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor],
    steps: int, gen_length: int, block_length: int, temperature: float,
    mask_id: int, effective_vocab: int, eos_id: int, eot_id: int,
    infill: bool,
    step_callback: Callable[[int, torch.Tensor], None],
  ) -> torch.Tensor:
    if infill:
      raise ValueError("dream engine does not support --infill.")

    step_counter: List[int] = [0]

    def _hook(logits, x):
      # Same one-step-lag pattern as _LLaDAUpstreamAdapter.
      if step_counter[0] > 0:
        step_callback(step_counter[0] - 1, x)
      step_counter[0] += 1
      return logits

    result = model.diffusion_generate(
      input_ids=input_ids,
      attention_mask=attention_mask,
      max_new_tokens=gen_length,
      output_history=False,
      return_dict_in_generate=True,
      steps=steps,
      temperature=temperature,
      generation_logits_hook_func=_hook,
    )
    output_ids = result.sequences if hasattr(result, "sequences") else result
    if step_counter[0] > 0:
      step_callback(step_counter[0] - 1, output_ids)
    return output_ids


_ENGINES: Dict[str, Any] = {
  "llada_local":    _LLaDALocalAdapter,
  "llada_upstream": _LLaDAUpstreamAdapter,
  "dream":          _DreamAdapter,
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt loading
# ─────────────────────────────────────────────────────────────────────────────

def load_prompts(path: str) -> List[Dict[str, Any]]:
  """Load prompts from a JSON / JSONL file.

  Accepted formats:
  - JSON list: [{...}, ...]
  - JSONL: one JSON object per line

  Each record must have at least a ``goal`` or ``prompt`` text field.
  A ``BehaviorID`` or ``id`` field is used as ``prompt_id``.
  """
  p = Path(path)
  if not p.exists():
    raise FileNotFoundError(f"Prompts file not found: {path}")
  raw = p.read_text(encoding="utf-8")
  try:
    data = json.loads(raw)
    if isinstance(data, list):
      return data
    if isinstance(data, dict):
      return [data]
  except json.JSONDecodeError:
    pass
  # Try JSONL
  records = []
  for line in raw.splitlines():
    line = line.strip()
    if line:
      records.append(json.loads(line))
  return records


def _extract_text(record: Dict[str, Any]) -> str:
  for key in ("goal", "prompt", "behavior", "text", "instruction"):
    if key in record and isinstance(record[key], str):
      return record[key]
  raise KeyError(f"No text field found in prompt record: {list(record.keys())}")


def _extract_id(record: Dict[str, Any], fallback_idx: int) -> str:
  for key in ("BehaviorID", "behavior_id", "id", "prompt_id"):
    if key in record:
      return str(record[key])
  return str(fallback_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
  checkpoint: str,
  tokenizer_path: str,
  device: str,
  precision: str = "bf16",
) -> Tuple[Any, Any, int, int, int, int]:
  """Returns (model, tokenizer, mask_id, eos_id, eot_id, effective_vocab)."""
  LOGGER.info("Loading tokenizer from %s", tokenizer_path)
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
  if tokenizer.padding_side != "left":
    tokenizer.padding_side = "left"

  eos_id = tokenizer.eos_token_id or LLADA_EOS_TOKEN_ID
  eot_id = LLADA_EOT_TOKEN_ID
  pad_id = ensure_pad_token(tokenizer, eos_token_id=eos_id)
  tokenizer.pad_token_id = pad_id

  # Resolve mask token id
  upstream_token = tokenizer.convert_ids_to_tokens(LLADA_MASK_TOKEN_ID)
  if upstream_token and upstream_token != tokenizer.unk_token:
    mask_id = LLADA_MASK_TOKEN_ID
  elif tokenizer.mask_token_id is not None:
    mask_id = int(tokenizer.mask_token_id)
  else:
    raise SystemExit("Could not resolve mask token id.")

  dtype_map = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
  }
  dtype = dtype_map.get(precision.lower(), torch.bfloat16)

  LOGGER.info("Loading model from %s (dtype=%s, device=%s)", checkpoint, dtype, device)
  model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, torch_dtype=dtype).to(device)
  model.eval()

  effective_vocab = len(tokenizer)
  LOGGER.info(
    "Model loaded. effective_vocab=%d  mask_id=%d  eos_id=%s  eot_id=%s",
    effective_vocab, mask_id, eos_id, eot_id,
  )
  return model, tokenizer, mask_id, eos_id, eot_id, effective_vocab


# ─────────────────────────────────────────────────────────────────────────────
# Chat-template formatting
# ─────────────────────────────────────────────────────────────────────────────

def _apply_chat_template(tokenizer, prompt_text: str) -> str:
  """Wrap plain text in an instruct chat template if available."""
  if not getattr(tokenizer, "chat_template", None):
    return prompt_text
  messages = [{"role": "user", "content": prompt_text}]
  try:
    return tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      tokenize=False,
    )
  except Exception as exc:
    LOGGER.warning("chat_template failed (%s); using raw prompt.", exc)
    return prompt_text


# ─────────────────────────────────────────────────────────────────────────────
# Extraction loop
# ─────────────────────────────────────────────────────────────────────────────

def extract_batch(
  engine,
  model,
  tokenizer,
  prompts: List[Dict[str, Any]],
  mask_id: int,
  eos_id: int,
  eot_id: int,
  effective_vocab: int,
  steps: int,
  max_new_tokens: int,
  temperature: float,
  use_chat_template: bool,
  infill: bool,
  device: str,
) -> List[Dict[str, Any]]:
  """Run one batch through the engine, recording the full sequence at every step.

  Standard mode (infill=False):
    input_ids contains clean prompt tokens; the engine appends max_new_tokens
    mask tokens before denoising.
    prompt_fixed_mask = [1]*prompt_len + [0]*max_new_tokens

  Infill mode (infill=True, llada_local only):
    input_ids already contains mask tokens at arbitrary positions.  No tokens
    are appended.  prompt_fixed_mask is derived from which positions were NOT
    mask_id at input time.
  """
  texts = [_extract_text(p) for p in prompts]
  if use_chat_template:
    formatted = [_apply_chat_template(tokenizer, t) for t in texts]
  else:
    formatted = texts

  encoded = tokenizer(
    formatted,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=1024,
  )
  input_ids = encoded["input_ids"].to(device)
  attention_mask = encoded.get("attention_mask")
  if attention_mask is not None:
    attention_mask = attention_mask.to(device)

  batch_size = input_ids.shape[0]
  prompt_len  = input_ids.shape[1]

  if infill:
    # fixed mask: 1 where the token was already set (not mask_id), 0 = denoising target
    fixed_mask_tensor = (input_ids != mask_id).long()  # (B, L)
    gen_length   = 0
    block_length = prompt_len  # denoise the whole sequence as one block
  else:
    # all prompt positions are fixed; the engine appends the gen region
    fixed_mask_tensor = torch.ones(batch_size, prompt_len, dtype=torch.long, device=device)
    gen_length   = max_new_tokens
    block_length = max_new_tokens  # pure diffusion: single block

  total_len = prompt_len + gen_length

  # Serialisable fixed mask per item (full sequence length).
  # In standard mode the gen region is all-denoised (0), appended after prompt.
  fixed_mask_lists: List[List[int]] = []
  for b in range(batch_size):
    fm = fixed_mask_tensor[b].cpu().tolist()
    if not infill:
      fm = fm + [0] * gen_length
    fixed_mask_lists.append(fm)

  # Number of denoised positions per item (positions where fixed_mask == 0).
  denoised_counts = [sum(1 for v in fm if v == 0) for fm in fixed_mask_lists]

  # Per-item trajectory buffers.
  all_tokens:     List[List[List[int]]] = [[] for _ in range(batch_size)]
  all_mask_fracs: List[List[float]]     = [[] for _ in range(batch_size)]

  def _step_callback(step_idx: int, x: torch.Tensor) -> None:
    # x shape: (B, total_len)  — full sequence after this step's token transfers.
    x_cpu = x.cpu()
    for b in range(batch_size):
      all_tokens[b].append(x_cpu[b].tolist())
      n_denoised = denoised_counts[b]
      if n_denoised == 0:
        frac = 0.0
      elif infill:
        denoised_pos = (fixed_mask_tensor[b] == 0)
        n_masked = int((x[b][denoised_pos] == mask_id).sum().item())
        frac = n_masked / n_denoised
      else:
        n_masked = int((x[b, prompt_len:] == mask_id).sum().item())
        frac = n_masked / n_denoised
      all_mask_fracs[b].append(frac)

  output_ids = engine.generate(
    model=model,
    tokenizer=tokenizer,
    input_ids=input_ids,
    attention_mask=attention_mask,
    steps=steps,
    gen_length=gen_length,
    block_length=block_length,
    temperature=temperature,
    mask_id=mask_id,
    effective_vocab=effective_vocab,
    eos_id=eos_id,
    eot_id=eot_id,
    infill=infill,
    step_callback=_step_callback,
  )

  # Decode final completions from denoised positions only.
  final_texts: List[str] = []
  for b in range(batch_size):
    seq = output_ids[b]
    if infill:
      denoised_ids = seq[fixed_mask_tensor[b] == 0]
    else:
      denoised_ids = seq[prompt_len:]
    final_texts.append(tokenizer.decode(denoised_ids.tolist(), skip_special_tokens=True))

  records = []
  for b, p in enumerate(prompts):
    records.append({
      "prompt_id":          _extract_id(p, -1),
      "prompt":             _extract_text(p),
      "model":              p.get("_model_tag", "llada"),
      "engine":             engine.name,
      "steps":              steps,
      "infill":             infill,
      "seq_len":            total_len,
      "prompt_fixed_mask":  fixed_mask_lists[b],
      "timestep_tokens":    all_tokens[b],
      "timestep_mask_fracs": all_mask_fracs[b],
      "final_completion":   final_texts[b],
    })
  return records


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
  p = argparse.ArgumentParser(
    description="Extract denoising trajectories from LLaDA/Dream.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  p.add_argument("--model", default="llada-instruct",
                 choices=["llada-instruct", "llada-base", "dream-instruct", "dream-base"],
                 help="Model family tag (labels the output, does not affect loading).")
  p.add_argument("--engine", default="llada_local",
                 choices=list(_ENGINES.keys()),
                 help=(
                   "Generation engine.  "
                   "llada_local: sampling/llada_engine.py (native step_callback + infill). "
                   "llada_upstream: third_party/LLaDA/generate.py (no infill). "
                   "dream: model.diffusion_generate() via DreamBackend (no infill)."
                 ))
  p.add_argument("--checkpoint", required=True, help="Path to model checkpoint directory.")
  p.add_argument("--tokenizer",  required=True, help="Path to tokenizer directory.")
  p.add_argument("--prompts-json", required=True, help="Path to prompts JSON/JSONL file.")
  p.add_argument("--output-jsonl", required=True, help="Output JSONL path.")
  p.add_argument("--steps",           type=int,   default=128,  help="Denoising steps.")
  p.add_argument("--max-new-tokens",  type=int,   default=256,  help="Generation length (standard mode).")
  p.add_argument("--batch-size",      type=int,   default=4,    help="Prompts per forward pass.")
  p.add_argument("--temperature",     type=float, default=0.0,  help="Gumbel noise temperature.")
  p.add_argument("--prompt-limit",    type=int,   default=None, help="Process only first N prompts.")
  p.add_argument("--shard-id",        type=int,   default=0,    help="Zero-based shard index.")
  p.add_argument("--num-shards",      type=int,   default=1,    help="Total shards.")
  p.add_argument("--seed",            type=int,   default=1)
  p.add_argument("--precision",       default="bf16", choices=["bf16", "bfloat16", "fp16", "fp32"])
  p.add_argument("--no-chat-template", action="store_true", help="Disable chat template even for instruct models.")
  p.add_argument("--infill", action="store_true",
                 help=(
                   "Infill mode (llada_local only): the prompt text already contains mask "
                   "tokens at positions to be denoised.  The model denoises in-place over "
                   "the full sequence; --max-new-tokens is ignored."
                 ))
  p.add_argument("--device", default=None,
                 help="Torch device (default: cuda if available, else cpu).")
  return p.parse_args(argv)


def main(argv=None) -> None:
  args = parse_args(argv)

  if args.infill and args.engine != "llada_local":
    raise SystemExit(f"--infill requires --engine llada_local (got {args.engine!r}).")

  random.seed(args.seed)
  torch.manual_seed(args.seed)

  device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
  LOGGER.info("Using device: %s  engine: %s", device, args.engine)

  engine = _ENGINES[args.engine]()

  # ── Load prompts ─────────────────────────────────────────────────────────
  all_prompts = load_prompts(args.prompts_json)
  LOGGER.info("Loaded %d prompts from %s", len(all_prompts), args.prompts_json)
  if args.prompt_limit is not None:
    all_prompts = all_prompts[: args.prompt_limit]

  # ── Sharding ──────────────────────────────────────────────────────────────
  if args.num_shards > 1:
    shard_size = math.ceil(len(all_prompts) / args.num_shards)
    start = args.shard_id * shard_size
    end   = min(start + shard_size, len(all_prompts))
    all_prompts = all_prompts[start:end]
    LOGGER.info("Shard %d/%d: [%d, %d)", args.shard_id, args.num_shards, start, end)

  if not all_prompts:
    LOGGER.info("No prompts to process.")
    return

  for p in all_prompts:
    p["_model_tag"] = args.model

  # ── Load model ────────────────────────────────────────────────────────────
  model, tokenizer, mask_id, eos_id, eot_id, effective_vocab = load_model_and_tokenizer(
    checkpoint=args.checkpoint,
    tokenizer_path=args.tokenizer,
    device=device,
    precision=args.precision,
  )

  use_chat_template = (not args.no_chat_template) and ("instruct" in args.model.lower())

  if args.infill:
    LOGGER.info("Infill mode: mask tokens in prompt will be denoised in-place. --max-new-tokens ignored.")

  # ── Output ────────────────────────────────────────────────────────────────
  out_path = Path(args.output_jsonl)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  n_written = 0
  with out_path.open("w", encoding="utf-8") as fout:
    for batch_start in range(0, len(all_prompts), args.batch_size):
      batch = all_prompts[batch_start: batch_start + args.batch_size]
      LOGGER.info("Batch [%d, %d) / %d", batch_start, batch_start + len(batch), len(all_prompts))
      try:
        records = extract_batch(
          engine=engine,
          model=model,
          tokenizer=tokenizer,
          prompts=batch,
          mask_id=mask_id,
          eos_id=eos_id,
          eot_id=eot_id,
          effective_vocab=effective_vocab,
          steps=args.steps,
          max_new_tokens=args.max_new_tokens,
          temperature=args.temperature,
          use_chat_template=use_chat_template,
          infill=args.infill,
          device=device,
        )
      except Exception as exc:
        LOGGER.error("Batch [%d, %d) failed: %s", batch_start, batch_start + len(batch), exc, exc_info=True)
        continue

      for rec in records:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        n_written += 1

  LOGGER.info("Done. Wrote %d records to %s", n_written, out_path)


if __name__ == "__main__":
  main()
