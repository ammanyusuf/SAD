import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from safety_eval.classifiers import LlamaGuardClassifier, load_texts


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="llamaguard")
def main(cfg: DictConfig) -> None:
    if cfg.generations_path == "change_my_path_please":
        raise SystemExit("Set `generations_path` to the JSONL file containing completions.")
    if cfg.model_path == "change_my_path_please":
        raise SystemExit("Set `model_path` to the local directory with the Llama Guard checkpoint.")

    orig_cwd = Path(hydra.utils.get_original_cwd())
    generations_path = (orig_cwd / cfg.generations_path).resolve()
    if not generations_path.is_file():
        raise SystemExit(f"Generations path not found: {generations_path}")
    model_path = Path(cfg.model_path).expanduser().resolve()

    texts = load_texts(generations_path, cfg.text_key)
    if not texts:
        raise SystemExit(f"No texts found in {generations_path}.")

    classifier = LlamaGuardClassifier(
        model_path=model_path,
        device=cfg.get("device", "auto"),
        device_map=cfg.device_map,
        system_prompt=cfg.get("system_prompt"),
        use_chat_template=cfg.get("use_chat_template"),
    )

    logger.info(
        "Starting Llama Guard evaluation on %d samples (batch_size=%d).",
        len(texts),
        cfg.batch_size,
    )
    outputs, unsafe = classifier.classify_texts(
        texts,
        batch_size=cfg.batch_size,
        max_new_tokens=cfg.max_new_tokens,
        log_every=int(cfg.get("log_every_batches", 10)),
        logger=logger,
    )

    summary = {
        "generations": str(generations_path),
        "count": len(texts),
        "model_path": str(model_path),
        "unsafe": unsafe,
        "unsafe_rate": unsafe / len(texts) if texts else float("nan"),
    }

    print(json.dumps(summary, indent=2))

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
