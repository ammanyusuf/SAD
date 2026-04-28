import glob
import os
import sys
from pathlib import Path
from typing import List, Optional
from multiprocessing import Process, cpu_count

import numpy as np
from tqdm import tqdm

# Support running without installing as a package.
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Tokenizer


def iter_enron_mail_files(maildir_root: Path) -> List[str]:
    """
    Scan all email file paths under a Maildir-style directory.

    Expected structure:
      maildir/<person>/<folder>/<message_file>
    """
    files = glob.glob(str(Path(maildir_root) / "*" / "*" / "*"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    return sorted(files)


def read_email_text(filepath: str) -> str:
    """
    Read a single email as text (headers + body). Uses 'replace' to avoid decode failures.
    """
    try:
        with open(filepath, "rb") as f:
            raw = f.read()
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def prepare_full(
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    prefix: str,
    mail_files_subset: List[str],
    process_id: int,
) -> None:
    destination_path = Path(destination_path)
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)
    proc_prefix = f"{prefix}_{process_id}"

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=proc_prefix,
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    for fp in tqdm(mail_files_subset, desc=f"[proc {process_id}] files", leave=False):
        text = read_email_text(fp)
        if not text.strip():
            continue
        text_ids = tokenizer.encode(text)
        builder.add_array(np.array(text_ids, dtype=builder.dtype))

    # The builder intentionally drops any trailing remainder (no write_reminder).


def prepare(
    source_path: Path = Path("smdm/dataset/enron_kaggle/enron_mail_20110402/maildir"),
    tokenizer_path: Path = Path("smdm/dataset/TinyLlama/checkpoints"),
    destination_path: Path = Path("smdm/workspace/icml/SMDM-main/dataset/finetuning_enron_combined"),
    chunk_size: int = 2049 * 1024,
    prefix: str = "train_enron",
    percentage: float = 1.0,
    num_processes: Optional[int] = 128,
) -> None:
    import time

    source_path = Path(source_path)
    tokenizer_path = Path(tokenizer_path)
    destination_path = Path(destination_path)

    mail_files = iter_enron_mail_files(source_path)

    # Select a prefix of the dataset by percentage (ensuring k>=1 when 0<percentage<1).
    if percentage <= 0:
        selected: List[str] = []
    elif percentage >= 1.0:
        selected = mail_files
    else:
        k = int(len(mail_files) * float(percentage))
        k = max(1, k)
        selected = mail_files[:k]

    if not selected:
        raise RuntimeError(
            f"No files selected. total={len(mail_files)}, percentage={percentage}, source={source_path}"
        )

    if num_processes is None:
        num_processes = cpu_count()

    # Clamp process count to avoid empty workers.
    num_processes = max(1, min(int(num_processes), len(selected)))

    chunked = np.array_split(selected, num_processes)

    processes: List[Process] = []
    start_time = time.time()

    for i, subset in enumerate(chunked):
        p = Process(
            target=prepare_full,
            args=(
                tokenizer_path,
                destination_path,
                chunk_size,
                prefix,
                list(subset),
                i,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Output dir: {destination_path.resolve()}")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
