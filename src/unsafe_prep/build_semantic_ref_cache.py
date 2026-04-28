"""Build and cache semantic embeddings for unsafe references offline.

This keeps sampling fast: the MDLM repellency module can load the cached
`semantic_ref_embeddings` via `semantic_ref_path` without recomputing.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from unsafe_prep import utils
from unsafe_prep.semantic_utils import (
    EmbeddingProvider,
    CallableEmbeddingProvider,
    HFEmbeddingProvider,
    MDLMEmbeddingProvider,
    masked_mean_pool,
)

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache semantic embeddings for unsafe refs.")
    parser.add_argument(
        "--artifact-root",
        required=True,
        help="Root directory containing unsafe artifacts (index.json or shards).",
    )
    parser.add_argument(
        "--artifact-name",
        dest="artifact_names",
        action="append",
        default=None,
        help="Artifact name; repeat for multiple. Required if index has multiple entries.",
    )
    parser.add_argument(
        "--provider",
        default="hf",
        choices=["hf", "mdlm", "callable"],
        help="Embedding provider: 'hf' (default AutoModel), 'mdlm' (import callable from MDLM module), "
             "or 'callable' (explicit module:function).",
    )
    parser.add_argument(
        "--encoder",
        default=None,
        help="HF encoder name/path (e.g., bert-base-uncased) for semantic pooling. "
             "Used when provider=hf or as fallback.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to save semantic_ref_embeddings (default: <artifact_root>/semantic_ref_embeddings.pt).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda", help="Device for encoding (e.g., cuda or cpu).")
    parser.add_argument(
        "--embed-fn",
        default=None,
        help="Python path 'module:function' returning embeddings for input_ids. "
        "Required when provider=callable. Callable must accept LongTensor [B, L] and "
        "return [B, D] or [B, L, D].",
    )
    parser.add_argument(
        "--mdlm-fn",
        default=None,
        help="Optional python path 'module:function' for provider=mdlm.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to MDLM checkpoint when using provider=mdlm.",
    )
    parser.add_argument(
        "--mdlm-embed-attr",
        default=None,
        help="Optional backbone embedding attribute name (e.g., vocab_embed). If unset, will try get_input_embeddings/vocab_embed/embedding/word_embeddings.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer name/path to load for MDLM checkpoints (falls back to TOKENIZER_PATH env var).",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer override (path/name) for pad/mask resolution; "
        "defaults to the tokenizer recorded in the unsafe artifact index.",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Path to MDLM Hydra config used for checkpoint loading (falls back to MODEL_CONFIG_PATH env var).",
    )
    return parser.parse_args()


@torch.no_grad()
def _pool_embeddings(
    ids: torch.LongTensor,
    model: torch.nn.Module | None,
    pad_id: int,
    mask_id: Optional[int],
    batch_size: int,
    device: torch.device,
    embed_fn=None,
) -> torch.FloatTensor:
    """Mean-pool hidden states over non-pad/non-mask tokens."""
    ids = ids.to(device)
    outputs = []
    total = ids.size(0)
    with tqdm(total=total, desc="semantic_cache", unit="seq") as pbar:
        for start in range(0, total, batch_size):
            batch = ids[start : start + batch_size]
            attn_mask = (batch != pad_id).long()
            if mask_id is not None:
                attn_mask = attn_mask * (batch != mask_id).long()
            if embed_fn is not None:
                hidden = embed_fn(batch)  # expected [B,D] or [B,L,D]
            else:
                model_out = model(
                    input_ids=batch,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = getattr(model_out, "last_hidden_state", None)
                if hidden is None and getattr(model_out, "hidden_states", None):
                    hidden = model_out.hidden_states[-1]
                if hidden is None and getattr(model_out, "logits", None) is not None:
                    hidden = model_out.logits
                if hidden is None:
                    raise RuntimeError("Embedding model did not return hidden states or logits.")
            pooled = masked_mean_pool(hidden, batch, pad_id=pad_id, mask_id=mask_id)
            outputs.append(pooled.detach().cpu())
            pbar.update(batch.size(0))
    return torch.cat(outputs, dim=0)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)

    artifact_root = Path(args.artifact_root).expanduser()
    # Resolve tokenizer/pad/mask from index.json (prefer parent if artifact_root has no index)
    index_root = artifact_root if (artifact_root / "index.json").exists() else artifact_root.parent
    index = utils.load_index(index_root)
    tokenizer_name = args.tokenizer or index.get("tokenizer") or index.get("tokenizer_alias")
    if not tokenizer_name:
        raise ValueError("Tokenizer name missing from unsafe index; cannot resolve pad/mask ids.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    pad_id = utils.ensure_pad_token(tokenizer)
    mask_id = index.get("mask_index", utils.resolve_mask_index(tokenizer, mask_token=None))

    device = torch.device(args.device)

    provider: EmbeddingProvider
    if args.provider == "callable":
        fn_path = args.embed_fn
        if not fn_path:
            raise ValueError("provider=callable requires --embed-fn")
        provider = CallableEmbeddingProvider(fn_path, label="callable")
    elif args.provider == "mdlm":
        provider = MDLMEmbeddingProvider(
            args.mdlm_fn,
            args.encoder,
            device,
            checkpoint=args.checkpoint or os.getenv("CHECKPOINT_PATH"),
            embed_attr=args.mdlm_embed_attr,
            tokenizer_path=args.tokenizer_path or os.getenv("TOKENIZER_PATH"),
            model_config_path=args.model_config or os.getenv("MODEL_CONFIG_PATH"),
        )
    else:
        encoder_name = args.encoder
        if not encoder_name:
            raise ValueError("provider=hf requires --encoder")
        provider = HFEmbeddingProvider(encoder_name, device)

    embed_fn, model, provider_label = provider.resolve()

    artifact_names = args.artifact_names or []
    if not artifact_names:
        entries = index.get("unsafe_artifacts") or []
        if len(entries) == 1:
            artifact_names = [entries[0].get("name") or Path(entries[0].get("path") or artifact_root).name]
        else:
            raise ValueError("Multiple artifacts in index; specify --artifact-name (can repeat).")

    for artifact_name in artifact_names:
        entry = utils.find_unsafe_artifact(artifact_root, artifact_name)
        resolved_name = artifact_name or entry.get("name") or Path(entry.get("path") or artifact_root).name
        tensor_path = utils.materialize_artifact(Path(entry["path"]), entry["storage"], overwrite=False)
        payload = torch.load(tensor_path, map_location="cpu")
        if isinstance(payload, dict):
            if "input_ids" in payload:
                unsafe_ids = payload["input_ids"].long()
            else:
                raise ValueError(f"Unsafe artifact dict missing 'input_ids' at {tensor_path}")
        elif torch.is_tensor(payload):
            unsafe_ids = payload.long()
        else:
            raise ValueError(f"Unsupported unsafe artifact payload type: {type(payload)} at {tensor_path}")

        LOGGER.info(
            "Encoding %d unsafe references with %s on %s (batch=%d) for %s...",
            unsafe_ids.size(0),
            provider_label,
            device,
            args.batch_size,
            resolved_name,
        )
        semantic_ref_embeddings = _pool_embeddings(
            unsafe_ids,
            model,
            pad_id,
            mask_id,
            batch_size=args.batch_size,
            device=device,
            embed_fn=embed_fn,
        )

        out_dir = artifact_root / "semantic_refs"
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.output and len(artifact_names) == 1:
            out_path = Path(args.output)
        else:
            out_path = out_dir / f"semantic_ref_embeddings_{resolved_name}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(semantic_ref_embeddings, out_path)
        LOGGER.info("Saved semantic_ref_embeddings to %s", out_path)


if __name__ == "__main__":
    main()
