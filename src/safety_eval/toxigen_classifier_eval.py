import json
import logging
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _iter_generations(path: Path, text_key: str) -> Iterable[str]:
    with path.open() as fp:
        for line_idx, raw in enumerate(fp, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_idx} invalid JSON: {exc}") from exc
            if text_key not in obj:
                raise KeyError(f"Missing key '{text_key}' in {path}:{line_idx}")
            text = obj[text_key]
            if not isinstance(text, str):
                raise TypeError(f"Expected string at key '{text_key}' in {path}:{line_idx}")
            yield text


def _percentile(sorted_values: Sequence[float], frac: float) -> float:
    if not sorted_values:
        return float("nan")
    if frac <= 0:
        return sorted_values[0]
    if frac >= 1:
        return sorted_values[-1]
    idx = frac * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    weight = idx - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def _summarize(scores: Sequence[float], threshold: float) -> Tuple[float, float, float]:
    if not scores:
        return float("nan"), float("nan"), float("nan")
    sorted_scores = sorted(scores)
    mean = sum(sorted_scores) / len(sorted_scores)
    rate = sum(score >= threshold for score in sorted_scores) / len(sorted_scores)
    perc95 = _percentile(sorted_scores, 0.95)
    return mean, rate, perc95


@hydra.main(version_base=None, config_path="configs", config_name="toxigen_classifier")
def main(cfg: DictConfig) -> None:
    if cfg.generations_path == "change_my_path_please":
        raise SystemExit("Set `generations_path` to the JSONL file containing completions.")
    if cfg.model_path == "change_my_path_please":
        raise SystemExit("Set `model_path` to the local directory with the ToxiGen classifier checkpoint.")

    orig_cwd = Path(hydra.utils.get_original_cwd())
    generations_path = (orig_cwd / cfg.generations_path).resolve()
    if not generations_path.is_file():
        raise SystemExit(f"Generations path not found: {generations_path}")
    model_path = Path(cfg.model_path).expanduser().resolve()

    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    texts = list(_iter_generations(generations_path, cfg.text_key))
    if not texts:
        raise SystemExit(f"No texts found in {generations_path}.")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    total_batches = math.ceil(len(texts) / cfg.batch_size)
    log_every = int(cfg.get("log_every_batches", 10))
    logger.info(
        "Starting ToxiGen evaluation on %d samples (%d batches, batch_size=%d).",
        len(texts),
        total_batches,
        cfg.batch_size,
    )

    scores: List[float] = []
    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, len(texts), cfg.batch_size), start=1):
            batch = texts[start : start + cfg.batch_size]
            batch_enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)
            logits = model(**batch_enc).logits
            if logits.shape[-1] == 1:
                probs = torch.sigmoid(logits[:, 0])
            else:
                probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(probs.detach().cpu().tolist())
            if log_every > 0 and (batch_idx % log_every == 0 or batch_idx == total_batches):
                logger.info(
                    "Processed batch %d/%d (cumulative samples=%d).",
                    batch_idx,
                    total_batches,
                    len(scores),
                )

    mean, rate, perc95 = _summarize(scores, cfg.threshold)
    result = {
        "generations": str(generations_path),
        "count": len(scores),
        "classifier_model": str(model_path),
        "threshold": cfg.threshold,
        "toxicity_prob_mean": mean,
        "toxicity_prob_rate": rate,
        "toxicity_prob_p95": perc95,
    }

    print(json.dumps(result, indent=2))

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
