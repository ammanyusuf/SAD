"""Unsafe prototype container and builders for clustered unsafe denoising."""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from unsafe_prep import utils

LOGGER = logging.getLogger(__name__)


@dataclass
class UnsafePrototypes:
    centroids: Tensor           # [K, L] LongTensor, prototype sequences (medoids)
    cluster_sizes: Tensor       # [K]   LongTensor
    token_histograms: Tensor    # [K, V] FloatTensor, contrastive or raw histos
    max_length: int
    vocab_size: int
    tokenizer_name_or_path: str

    # semantic prototypes (not implemented yet)
    # intended: [K, D] FloatTensor, embeddings for each unsafe concept
    cluster_embeddings: Optional[Tensor] = None


def load_unsafe_matrix(artifact_root: Path, artifact_name: Optional[str]) -> torch.LongTensor:
    """Materialize and load the unsafe tensor from an artifact root."""
    artifact_root = Path(artifact_root).expanduser()
    entry = utils.find_unsafe_artifact(artifact_root, artifact_name)
    logging.info("Found unsafe artifact entry: %s", entry)
    artifact_dir = Path(entry["path"])
    storage = entry["storage"]
    logging.info("Loading unsafe artifact from %s", artifact_dir)
    tensor_path = utils.materialize_artifact(artifact_dir, storage, overwrite=False)
    payload = torch.load(tensor_path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("input_ids", "ref_data", "unsafe_reference", "data"):
            if key in payload:
                payload = payload[key]
                break
        else:
            raise TypeError(f"Unsupported unsafe artifact payload keys: {list(payload.keys())}")
    if isinstance(payload, (list, tuple)):
        payload = torch.tensor(payload)
    if not torch.is_tensor(payload):
        raise TypeError(f"Unsafe artifact at {tensor_path} must be a tensor or mapping.")
    return payload.long()


def save_unsafe_prototypes(path: Path, obj: UnsafePrototypes) -> None:
    payload = {
        "centroids": obj.centroids,
        "cluster_sizes": obj.cluster_sizes,
        "token_histograms": obj.token_histograms,
        "max_length": obj.max_length,
        "vocab_size": obj.vocab_size,
        "tokenizer": obj.tokenizer_name_or_path,
    }
    if obj.cluster_embeddings is not None:
        payload["cluster_embeddings"] = obj.cluster_embeddings
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_unsafe_prototypes(path: Path) -> UnsafePrototypes:
    payload = torch.load(Path(path), map_location="cpu")
    return UnsafePrototypes(
        centroids=payload["centroids"].long(),
        cluster_sizes=payload["cluster_sizes"].long(),
        token_histograms=payload["token_histograms"].float(),
        max_length=int(payload["max_length"]),
        vocab_size=int(payload["vocab_size"]),
        tokenizer_name_or_path=str(payload.get("tokenizer", "")),
        cluster_embeddings=payload.get("cluster_embeddings", None),
    )


def build_unsafe_prototypes(
    unsafe_ids: Tensor,
    tokenizer: PreTrainedTokenizerBase,
    num_prototypes: int,
    vocab_size: int,
    max_length: Optional[int] = None,
    seed: int = 1,
) -> UnsafePrototypes:
    """Build cluster-compressed unsafe prototypes from a matrix of unsafe ids."""
    if unsafe_ids.ndim != 2:
        raise ValueError(f"unsafe_ids must be 2D [N, L], got {unsafe_ids.shape}")
    unsafe_ids = unsafe_ids.long()
    if max_length is not None:
        unsafe_ids = unsafe_ids[:, :max_length]
    max_length = unsafe_ids.shape[1]
    pad_id = utils.ensure_pad_token(tokenizer)
    torch.manual_seed(seed)

    def _hashed_bow_embeddings(ids: torch.LongTensor, dim: int) -> torch.FloatTensor:
        """Simple hashed bag-of-words embedding per sequence."""
        N, _ = ids.shape
        emb = torch.zeros((N, dim), device=ids.device, dtype=torch.float32)
        mask = ids != pad_id
        if mask.any():
            indices = torch.repeat_interleave(
                torch.arange(N, device=ids.device),
                mask.sum(dim=1)
            )
            token_idx = ids[mask].remainder(dim)
            emb.index_put_(
                (indices, token_idx),
                torch.ones_like(token_idx, dtype=torch.float32),
                accumulate=True,
            )
        lengths = mask.sum(dim=1, keepdim=True).clamp_min(1)
        emb = emb / lengths
        return emb

    def _torch_kmeans(x: torch.FloatTensor, k: int, iters: int = 25) -> torch.LongTensor:
        """Lightweight torch k-means fallback."""
        N, _ = x.shape
        k = min(k, N)
        centroids = x[torch.randperm(N)[:k]].clone()
        labels = torch.zeros(N, dtype=torch.long, device=x.device)
        for _ in range(iters):
            distances = torch.cdist(x, centroids)
            labels = distances.argmin(dim=1)
            for idx in range(k):
                members = labels == idx
                if members.any():
                    centroids[idx] = x[members].mean(dim=0)
                else:
                    centroids[idx] = x[torch.randint(0, N, (1,), device=x.device)]
        return labels

    def _choose_medoid(
        ids: torch.LongTensor,
        member_indices: torch.LongTensor,
        max_candidates: int = 256,
    ) -> torch.LongTensor:
        """Pick sequence with minimal mean Hamming distance ignoring PAD."""
        if member_indices.numel() == 0:
            return torch.zeros_like(ids[0])
        cluster_sequences = ids[member_indices]
        max_cand = (
            max_candidates if member_indices.numel() > max_candidates else member_indices.numel()
        )
        candidate_indices = member_indices[:max_cand]
        best_seq = ids[candidate_indices[0]]
        best_score = math.inf
        for cand_idx in candidate_indices:
            cand_seq = ids[cand_idx]
            valid = (cand_seq != pad_id) & (cluster_sequences != pad_id)
            mismatch = valid & (cluster_sequences != cand_seq)
            distances = mismatch.sum(dim=1, dtype=torch.float32)
            score = distances.mean().item()
            if score < best_score:
                best_score = score
                best_seq = cand_seq
        return best_seq.clone()

    def _compute_token_histogram(cluster_sequences: torch.LongTensor) -> torch.FloatTensor:
        if cluster_sequences.numel() == 0:
            return torch.zeros(vocab_size, dtype=torch.float32)
        valid_tokens = cluster_sequences[cluster_sequences != pad_id]
        if valid_tokens.numel() == 0:
            return torch.zeros(vocab_size, dtype=torch.float32)
        counts = torch.bincount(valid_tokens, minlength=vocab_size).to(torch.float32)
        counts = counts / counts.sum().clamp_min(1e-12)
        return counts

    embed_dim = int(min(vocab_size, 2048))
    embeddings = _hashed_bow_embeddings(unsafe_ids, embed_dim)
    try:
        from sklearn.cluster import MiniBatchKMeans  # type: ignore

        model = MiniBatchKMeans(
            n_clusters=num_prototypes,
            batch_size=1024,
            random_state=seed,
            n_init=5,
        )
        labels = model.fit_predict(embeddings.cpu().numpy())
        labels = torch.from_numpy(labels).long().to(unsafe_ids.device)
    except Exception as exc:  # pragma: no cover - sklearn optional
        LOGGER.warning("Falling back to torch k-means due to: %s", exc)
        labels = _torch_kmeans(embeddings, num_prototypes)

    cluster_sizes = torch.bincount(labels, minlength=num_prototypes)
    centroids = torch.zeros(
        (num_prototypes, max_length), dtype=torch.long, device=unsafe_ids.device
    )
    token_histograms = torch.zeros(
        (num_prototypes, vocab_size), dtype=torch.float32, device=unsafe_ids.device
    )

    for k in range(num_prototypes):
        members = (labels == k).nonzero(as_tuple=False).squeeze(-1)
        if members.numel() == 0:
            LOGGER.warning("Cluster %d is empty; assigning zero prototype.", k)
            continue
        centroids[k] = _choose_medoid(unsafe_ids, members)
        token_histograms[k] = _compute_token_histogram(unsafe_ids[members])

    return UnsafePrototypes(
        centroids=centroids.cpu(),
        cluster_sizes=cluster_sizes.cpu(),
        token_histograms=token_histograms.cpu(),
        max_length=max_length,
        vocab_size=vocab_size,
        tokenizer_name_or_path=str(tokenizer.name_or_path),
        cluster_embeddings=None,
    )


def compute_background_token_distribution(
    unsafe_ids: Tensor,
    vocab_size: int,
) -> Tensor:
    """Compute a background token frequency p_bg(v)."""
    flat = unsafe_ids.reshape(-1)
    counts = torch.bincount(flat, minlength=vocab_size).to(torch.float32)
    return counts / counts.sum().clamp_min(1e-12)


def _load_tokenizer_for_artifact(root: Path, override: Optional[str]) -> PreTrainedTokenizerBase:
    if override:
        return AutoTokenizer.from_pretrained(override)
    index_path = root / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"index.json missing under {root}; set --tokenizer."
        )
    index = utils.load_index(index_path)
    tokenizer_name = index.get("tokenizer")
    if not tokenizer_name:
        raise ValueError(
            "Tokenizer field missing in unsafe artifact index; specify --tokenizer."
        )
    return AutoTokenizer.from_pretrained(str(tokenizer_name))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build clustered unsafe prototypes.")
    parser.add_argument(
        "--unsafe-artifact-root",
        type=str,
        required=True,
        help="Root containing unsafe artifacts or index.json.",
    )
    parser.add_argument(
        "--unsafe-artifact-name",
        type=str,
        default=None,
        help="Optional specific artifact name from index.json.",
    )
    parser.add_argument(
        "--unsafe-artifact-names",
        nargs="+",
        default=None,
        help="List of artifact names; defaults to all in index.json.",
    )
    parser.add_argument(
        "--num-prototypes",
        type=int,
        required=True,
        help="Number of prototypes to build.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional directory to store prototypes (default: <artifact_root>/prototypes).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer path/name to override index.json.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional max length to truncate continuations.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Optional vocab size (defaults to tokenizer vocab).",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for clustering.")
    return parser


def _cli_build(args: argparse.Namespace) -> None:
    artifact_root = Path(args.unsafe_artifact_root).expanduser()
    tokenizer = _load_tokenizer_for_artifact(artifact_root, args.tokenizer)
    vocab_size = args.vocab_size or tokenizer.vocab_size
    if tokenizer.mask_token_id is None:
        vocab_size += 1

    if args.unsafe_artifact_names:
        artifact_names = list(args.unsafe_artifact_names)
    elif args.unsafe_artifact_name:
        artifact_names = [args.unsafe_artifact_name]
    else:
        index = utils.load_index(artifact_root)
        artifact_names = [entry["name"] for entry in index.get("unsafe_artifacts", [])]
    if not artifact_names:
        raise ValueError("No unsafe artifacts specified or found in index.")

    output_root = Path(args.output_root) if args.output_root else artifact_root / "prototypes"
    output_root.mkdir(parents=True, exist_ok=True)

    for name in artifact_names:
        unsafe_artifact_path = artifact_root / name
        LOGGER.info("Building prototypes for artifact '%s', %s", name, unsafe_artifact_path)
        unsafe_ids = load_unsafe_matrix(artifact_root, name)
        prototypes = build_unsafe_prototypes(
            unsafe_ids=unsafe_ids,
            tokenizer=tokenizer,
            num_prototypes=args.num_prototypes,
            vocab_size=int(vocab_size),
            max_length=args.max_length,
            seed=args.seed,
        )
        output_path = output_root / f"{name}_k{args.num_prototypes}.pt"
        save_unsafe_prototypes(output_path, prototypes)
        LOGGER.info("Saved prototypes to %s", output_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO)
    parser = _build_arg_parser()
    _cli_build(parser.parse_args(argv))


if __name__ == "__main__":
    main()
