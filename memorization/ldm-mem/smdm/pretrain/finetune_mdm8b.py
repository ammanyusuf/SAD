import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from functools import partial

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader

# =========================
# 审计相关 import
# =========================
import os
import json
from collections import defaultdict
import random
import argparse

from swanlab.integration.pytorch_lightning import SwanLabLogger
from flash_attn.losses.cross_entropy import CrossEntropyLoss

# HuggingFace
from transformers import AutoConfig, AutoModelForCausalLM

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.utils import get_default_supported_precision, num_parameters, step_csv_logger

from torch.distributed.elastic.multiprocessing.errors import record


# =========================
# 词表/Mask 相关（按你给的事实固定）
# =========================
BASE_TOKENIZER_VOCAB = 126080      # 你 tokenizer 里看到的 vocab_size
MASK_TOKEN_ID = 126336            # 你明确的 mask_id
# HF config.vocab_size 你已经确认是 126464（还有额外 token 槽）
# 这里不写死 126464，以 HF config 为准，只做一致性检查


def parse_args():
    parse = argparse.ArgumentParser()

    # =========================
    # HuggingFace 模型加载
    # =========================
    parse.add_argument(
        '--hf_model_name',
        type=str,
        default='GSAI-ML/LLaDA-8B-Base',
        help='HuggingFace model name or local path'
    )
    parse.add_argument(
        '--hf_trust_remote_code',
        action='store_true',
        default=True,
        help='trust_remote_code for HF model'
    )
    parse.add_argument(
        '--hf_dtype',
        type=str,
        default='bfloat16',
        choices=['bfloat16', 'float16', 'float32'],
        help='dtype for HF model weights'
    )

    # =========================
    # 训练超参（固定学习率）
    # =========================
    parse.add_argument('--learning_rate', type=float, default=1e-5, help='fixed learning rate')
    parse.add_argument('--nodes_num', type=int, default=1, help='number of nodes')
    parse.add_argument('--devices', type=int, default=6, help='number of devices per node (Fabric devices)')
    parse.add_argument('--batch_size', type=int, default=192, help='global_batch_size (across devices on this node)')
    parse.add_argument('--micro_batch_size', type=int, default=1, help='micro batch size per device')
    parse.add_argument('--max_steps', type=int, default=10000, help='optimizer steps (ignored when finetune_enable)')

    # =========================
    # 审计开关与参数
    # =========================
    parse.add_argument('--audit_enable', action='store_true', help='enable audit logging (low-overhead)')
    parse.add_argument('--audit_every', type=int, default=100, help='log every N optimizer steps (window)')
    parse.add_argument('--audit_level', type=str, default='bin', choices=['bin', 'block'],
                       help="audit granularity: 'bin' or '(bin, block_id)'")
    parse.add_argument('--audit_fsync', action='store_true',
                       help='fsync on every audit flush (slower but safer)')

    # =========================
    # finetune：按“跑完一遍可用 bins”停止（与代码1对齐）
    # - finetune_epochs：表示“跑完一遍可用 bins”算 1 epoch
    # =========================
    parse.add_argument('--finetune_enable', action='store_true',
                       help='enable finetune on new bins, stop by bin coverage (by dataloader exhaustion)')
    parse.add_argument('--finetune_dir', type=str,
                       default='smdm/dataset/8bslim_enron_combined',
                       help='finetune bins folder')
    parse.add_argument('--finetune_pattern', type=str, default='validation_slimpajama_*.bin',
                       help='glob pattern inside finetune_dir')
    parse.add_argument('--finetune_epochs', type=int, default=1,
                       help='how many epochs by bin-coverage (global union over ranks)')

    args = parse.parse_args()
    return args


# =========================
# HF Wrapper：保证 forward 返回 logits tensor
# =========================
class HFModelWrapper(torch.nn.Module):
    def __init__(self, hf_model: torch.nn.Module, block_size: int):
        super().__init__()
        self.hf_model = hf_model
        self.config = hf_model.config
        if not hasattr(self.config, "block_size"):
            self.config.block_size = int(block_size)

    def forward(self, input_ids: torch.Tensor):
        out = self.hf_model(input_ids=input_ids)
        if hasattr(out, "logits") and out.logits is not None:
            return out.logits
        if torch.is_tensor(out):
            return out
        if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
            return out[0]
        raise RuntimeError("HF model forward() did not produce logits. Make sure you load AutoModelForCausalLM.")


# =========================
# 审计窗口聚合器
# =========================
class AuditWindow:
    def __init__(self, path: Path, every_steps: int = 100, level: str = "bin",
                 enable: bool = False, fsync: bool = False):
        self.enable = bool(enable)
        self.every_steps = max(1, int(every_steps))
        self.level = level
        self.fsync = bool(fsync)

        self.path = Path(path)
        self._fp = None

        self._bins = set()
        self._bin_blocks = defaultdict(set)
        self._win_start_step = None

        if self.enable:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = open(self.path, "a", encoding="utf-8", buffering=1024 * 1024)

    def reset_window(self):
        if not self.enable:
            return
        self._bins.clear()
        self._bin_blocks.clear()
        self._win_start_step = None

    def add_batch(self, filenames, block_ids=None):
        if not self.enable or filenames is None:
            return

        if self.level == "bin" or block_ids is None:
            self._bins.update(list(filenames))
            return

        if torch.is_tensor(block_ids):
            block_ids = block_ids.tolist()

        for fn, bid in zip(filenames, block_ids):
            self._bins.add(fn)
            self._bin_blocks[fn].add(int(bid))

    def maybe_flush(self, fabric, iter_num: int, step_count: int, lr: float, loss: float, force: bool = False):
        if not self.enable or self._fp is None:
            return

        if self._win_start_step is None:
            self._win_start_step = int(step_count)

        if (not force) and (step_count % self.every_steps != 0):
            return

        record = {
            "step_start": int(self._win_start_step),
            "step_end": int(step_count),
            "iter_num": int(iter_num),
            "global_rank": int(getattr(fabric, "global_rank", 0)),
            "world_size": int(getattr(fabric, "world_size", 1)),
            "lr": float(lr),
            "loss": float(loss),
            "bins": sorted(self._bins),
        }

        if self.level == "block":
            record["bin_blocks"] = {k: sorted(v) for k, v in self._bin_blocks.items()}

        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fp.flush()

        if self.fsync:
            os.fsync(self._fp.fileno())

        self._bins.clear()
        self._bin_blocks.clear()
        self._win_start_step = int(step_count) + 1

    def close(self, fabric=None, iter_num: int = 0, step_count: int = 0, lr: float = 0.0, loss: float = 0.0):
        if not self.enable or self._fp is None:
            return
        if len(self._bins) > 0:
            self.maybe_flush(fabric=fabric, iter_num=iter_num, step_count=step_count, lr=lr, loss=loss, force=True)
        try:
            self._fp.close()
        except Exception:
            pass


def forward_process(batch: torch.Tensor, mask_id: int = MASK_TOKEN_ID, eps: float = 1e-3):
    """
    batch: (B, L) int token ids
    mask_id: 替换用的 mask token id（固定 126336）
    """
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)

    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    mask_indices = torch.rand((b, l), device=batch.device) < p_mask

    # 用 full_like 确保 dtype/device 一致
    mask_fill = torch.full_like(batch, int(mask_id))
    noisy_batch = torch.where(mask_indices, mask_fill, batch)
    return noisy_batch, mask_indices, p_mask


# Treat all dataset equally by their size.
train_data_config = [("train_slim", 1.0)]
val_data_config = [("validation", 1.0)]


def create_dataloader(
    batch_size: int,
    block_size: int,
    data_dir: Path,
    fabric,
    shuffle: bool = True,
    seed: int = 12345,
    split: str = "train",
    wrap: bool = True,  # ✅ 与代码1对齐：finetune 时 wrap=False
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*.bin")))
        random.seed(seed)
        random.shuffle(filenames)

        dataset = PackedDataset(
            filenames,
            n_chunks=8 if split == "train" else 1,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed + fabric.global_rank,
            wrap=wrap,  # ✅ 关键：finetune 用 False，跑完一遍就 StopIteration
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(f"No data found at {data_dir}. Make sure your bin pattern is correct.")

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights) if len(weights) else 1.0
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
    train_wrap: bool = True,  # ✅ 与代码1对齐
) -> Tuple[DataLoader, DataLoader]:
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train",
        wrap=train_wrap,
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation",
            wrap=False,
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


def _infer_block_size_from_hf_config(hf_cfg) -> int:
    for k in ["max_position_embeddings", "model_max_length", "n_positions", "seq_length", "max_seq_len"]:
        v = getattr(hf_cfg, k, None)
        if isinstance(v, int) and 0 < v < 1_000_000:
            return int(v)
    return 2048


def setup(
    args,
    train_data_dir: Path = Path("smdm/dataset/8bfinetuning_enron_combined"),
    val_data_dir: Path = Path("smdm/dataset/8bfinetuning_enron_combined"),
    precision: Optional[str] = None,
    tpu: bool = False,
) -> None:
    out_root = Path('workdir')
    hf_name_safe = args.hf_model_name.replace("/", "_").replace(":", "_")
    hp_name = f"hf-{hf_name_safe}-lr{args.learning_rate}-gb{int(args.batch_size/args.nodes_num)}"
    out_dir = out_root / "scaling_debug" / hp_name

    wandb_logger = SwanLabLogger(name=f'{hp_name}-mc', save_dir=out_dir, project='scaling')

    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if args.devices > 1:
        if tpu:
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
            auto_wrap = partial(size_based_auto_wrap_policy, min_num_params=100_000_000)
            strategy = FSDPStrategy(
                auto_wrap_policy=auto_wrap,
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
        devices = args.devices
    else:
        strategy = "auto"
        devices = args.devices

    logger = step_csv_logger("out", hf_name_safe, flush_logs_every_n_steps=10)

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
    fabric.print({"hf_model_name": args.hf_model_name, "learning_rate": args.learning_rate})

    main(fabric, args, out_dir, train_data_dir, val_data_dir)


def main(fabric, args, out_dir: Path, train_data_dir: Path, val_data_dir: Optional[Path]):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=10)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    audit_dir = out_dir / "audit"
    audit_local_path = audit_dir / f"audit_rank{fabric.global_rank:04d}.jsonl"
    if args.audit_enable:
        if fabric.global_rank == 0:
            audit_dir.mkdir(parents=True, exist_ok=True)
        fabric.barrier()

    # =========================
    # finetune 模式：切换数据目录 + prefix
    # =========================
    if args.finetune_enable:
        global train_data_config, val_data_config
        train_data_dir = Path(args.finetune_dir)
        val_data_dir = None

        pat = args.finetune_pattern
        prefix = pat.split("*")[0] if "*" in pat else pat
        if prefix.endswith(".bin"):
            prefix = prefix[:-4]
        prefix = Path(prefix).name

        train_data_config = [(prefix, 1.0)]
        val_data_config = []

        if fabric.global_rank == 0:
            fabric.print(f"[finetune] data_dir={train_data_dir} pattern={args.finetune_pattern} -> prefix={prefix}")

    # =========================
    # 从 HF config 取 block_size + vocab/mask 一致性检查
    # =========================
    hf_cfg = AutoConfig.from_pretrained(args.hf_model_name, trust_remote_code=args.hf_trust_remote_code)
    block_size = _infer_block_size_from_hf_config(hf_cfg)

    cfg_vocab = int(getattr(hf_cfg, "vocab_size", 0))
    cfg_mask = getattr(hf_cfg, "mask_token_id", None)

    if cfg_mask is not None and int(cfg_mask) != int(MASK_TOKEN_ID):
        raise RuntimeError(f"[vocab] HF config.mask_token_id={cfg_mask} != expected {MASK_TOKEN_ID}")

    if cfg_vocab <= MASK_TOKEN_ID:
        raise RuntimeError(f"[vocab] HF config.vocab_size={cfg_vocab} does not cover mask_id={MASK_TOKEN_ID}")

    reserved_slots = cfg_vocab - BASE_TOKENIZER_VOCAB
    if fabric.global_rank == 0:
        fabric.print(
            f"[vocab] base_tokenizer_vocab={BASE_TOKENIZER_VOCAB}, "
            f"hf_config_vocab={cfg_vocab}, reserved_slots={reserved_slots}, "
            f"mask_id={MASK_TOKEN_ID}"
        )

    # ✅ 与代码1对齐：finetune 用 wrap=False（跑完一遍可用 bins 就 dataloader exhausted）
    train_wrap = (not args.finetune_enable)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=args.micro_batch_size,
        block_size=block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
        train_wrap=train_wrap,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)

    # =========================
    # HF 模型加载（权重 dtype 只影响浮点精度）
    # =========================
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    hf_dtype = dtype_map[args.hf_dtype]

    fabric.print(f"Loading HF model: {args.hf_model_name} dtype={args.hf_dtype} trust_remote_code={args.hf_trust_remote_code}")
    t0 = time.perf_counter()

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_name,
        trust_remote_code=args.hf_trust_remote_code,
        torch_dtype=hf_dtype,
        low_cpu_mem_usage=True,
    )

    # 确保 embedding 覆盖 mask_id（理论上 cfg_vocab 已经覆盖，这里再保险）
    emb_rows = int(hf_model.get_input_embeddings().weight.shape[0])
    if emb_rows <= MASK_TOKEN_ID:
        hf_model.resize_token_embeddings(MASK_TOKEN_ID + 1)
        if fabric.global_rank == 0:
            fabric.print(f"[vocab] resized token embeddings: {emb_rows} -> {MASK_TOKEN_ID + 1}")

    hf_model.config.block_size = int(block_size)

    if getattr(hf_model.config, "mask_token_id", MASK_TOKEN_ID) != MASK_TOKEN_ID:
        raise RuntimeError(f"[vocab] model.config.mask_token_id != {MASK_TOKEN_ID} after load")

    model = HFModelWrapper(hf_model, block_size=block_size)

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)

    # =========================
    # 优化器：固定学习率
    # =========================
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        foreach=False,
    )
    optimizer = fabric.setup_optimizers(optimizer)

    # 训练步数相关（沿用你的 iter/accum 结构）
    num_of_devices = int(args.devices)
    global_batch_size = int(args.batch_size / args.nodes_num)

    batch_size = global_batch_size // num_of_devices
    gradient_accumulation_steps = batch_size // int(args.micro_batch_size)
    assert gradient_accumulation_steps > 0, (
        f"gradient_accumulation_steps <= 0 (batch_size={batch_size}, micro_batch_size={args.micro_batch_size})"
    )

    max_step = int(args.max_steps)                  # optimizer steps
    max_iters = max_step * gradient_accumulation_steps  # micro iters（与代码1一致）

    state = {
        "model": model,
        "optimizer": optimizer,
        "hparams": {
            "hf_model_name": args.hf_model_name,
            "learning_rate": float(args.learning_rate),
            "global_batch_size": int(global_batch_size),
            "micro_batch_size": int(args.micro_batch_size),
            "devices": int(args.devices),
            "nodes_num": int(args.nodes_num),
            "max_steps": int(args.max_steps),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "block_size": int(block_size),
            "mask_token_id": int(MASK_TOKEN_ID),
            "base_tokenizer_vocab": int(BASE_TOKENIZER_VOCAB),
            "hf_vocab_size": int(cfg_vocab),
        },
        "iter_num": 0,
        "step_count": 0,
    }

    train_time = time.perf_counter()
    train(
        fabric=fabric,
        args=args,
        out_dir=out_dir,
        state=state,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        monitor=monitor,
        audit_local_path=audit_local_path,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_iters=max_iters,
        max_step=max_step,
        grad_clip=grad_clip,
        hf_vocab_size=cfg_vocab,
    )
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(
    fabric,
    args,
    out_dir: Path,
    state,
    train_dataloader,
    val_dataloader,
    monitor,
    audit_local_path: Path,
    gradient_accumulation_steps: int,
    max_iters: int,
    max_step: int,
    grad_clip: float,
    hf_vocab_size: int,
):
    model = state["model"]
    optimizer = state["optimizer"]

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()

    loss_func = CrossEntropyLoss(reduction='none')

    audit = AuditWindow(
        path=audit_local_path,
        every_steps=args.audit_every,
        level=args.audit_level,
        enable=args.audit_enable,
        fsync=args.audit_fsync,
    )

    last_lr = float(args.learning_rate)
    last_loss = 0.0

    # 固定 mask id
    mask_token_id = int(MASK_TOKEN_ID)

    # ✅ 与代码1对齐：finetune 只按 dataloader exhausted 作为 “跑完一遍可用 bins”
    finetune_epochs_target = max(1, int(args.finetune_epochs)) if args.finetune_enable else 0

    if args.finetune_enable and fabric.global_rank == 0:
        num_workers = getattr(train_dataloader, "num_workers", 0)
        num_workers = 1 if int(num_workers) == 0 else int(num_workers)
        total = len(sorted(glob.glob(str(Path(args.finetune_dir) / args.finetune_pattern))))
        group = fabric.world_size * num_workers * 8
        usable = (total // group) * group
        fabric.print(f"[finetune] total_bins={total} group={group} -> usable_bins={usable} (your 144 comes from here)")
        fabric.print(f"[finetune] epochs={finetune_epochs_target} lr_fixed={float(args.learning_rate)} wd_fixed=0.1")

    save_step_interval = 200
    eval_step_interval = 999999999999
    eval_iters = int(100 * 1024 / int(args.batch_size / args.nodes_num)) if int(args.batch_size / args.nodes_num) > 0 else 0

    checked_vocab_once = False

    def run_one_epoch():
        nonlocal total_lengths, total_t0, last_lr, last_loss, checked_vocab_once

        for train_data in train_dataloader:
            # ✅ 非 finetune：预算停止（与代码1一致：按 max_iters/micro-iter 截止）
            if (not args.finetune_enable) and (state["iter_num"] >= max_iters):
                return False  # stop training

            # 固定 lr（finetune/non-finetune 都是固定值，这里保持一致）
            lr = float(args.learning_rate)
            last_lr = float(lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            iter_t0 = time.perf_counter()

            # 兼容 Tensor / (Tensor, filename) / (Tensor, filename, block_id)
            train_tokens = None
            train_filenames = None
            train_block_ids = None

            if isinstance(train_data, (tuple, list)):
                if len(train_data) == 3:
                    train_tokens, train_filenames, train_block_ids = train_data
                elif len(train_data) == 2:
                    train_tokens, train_filenames = train_data
                else:
                    train_tokens = train_data
            else:
                train_tokens = train_data

            train_tokens = train_tokens.to(dtype=torch.long)
            input_ids = train_tokens[:, 0: model.config.block_size].contiguous()

            if not checked_vocab_once:
                local_max = int(input_ids.max().item())
                local_min = int(input_ids.min().item())
                if local_min < 0 or local_max >= int(hf_vocab_size):
                    raise RuntimeError(
                        f"[vocab] batch token id out of range: min={local_min}, max={local_max}, hf_vocab_size={hf_vocab_size}"
                    )
                if fabric.global_rank == 0:
                    fabric.print(
                        f"[vocab] first-batch id range OK: min={local_min}, max={local_max}, hf_vocab_size={hf_vocab_size}"
                    )
                checked_vocab_once = True

            # 审计：收集 bin
            if args.audit_enable and train_filenames is not None:
                audit.add_batch(train_filenames, train_block_ids)

            noisy_input, mask_indices, p_mask = forward_process(input_ids, mask_id=mask_token_id)
            is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0

            with fabric.no_backward_sync(model, enabled=is_accumulating):
                logits = model(noisy_input)
                loss = loss_func(logits[mask_indices], input_ids[mask_indices]) / p_mask[mask_indices]
                loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
                last_loss = float(loss.item())
                fabric.backward(loss / gradient_accumulation_steps)

            did_optimizer_step = False
            if not is_accumulating:
                fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                state["step_count"] += 1
                did_optimizer_step = True
            elif fabric.device.type == "xla":
                import torch_xla.core.xla_model as xm
                xm.mark_step()

            state["iter_num"] += 1

            # 审计：按窗口落盘
            if did_optimizer_step and args.audit_enable:
                audit.maybe_flush(
                    fabric=fabric,
                    iter_num=state["iter_num"],
                    step_count=state["step_count"],
                    lr=last_lr,
                    loss=last_loss,
                )

            total_lengths += input_ids.size(1)
            t1 = time.perf_counter()

            fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )

            monitor.on_train_batch_end(
                state["iter_num"] * int(args.micro_batch_size),
                t1 - total_t0,
                fabric.world_size,
                state["step_count"],
                flops_per_batch=0.0,
                lengths=total_lengths,
                train_loss=loss.item(),
            )

            # 验证逻辑保留（finetune 默认 val=None）
            if val_dataloader is not None and (not is_accumulating) and (
                state["step_count"] % eval_step_interval == 0 or state["step_count"] == max_step
            ):
                t0 = time.perf_counter()
                val_loss = validate(fabric, model, val_dataloader, eval_iters=eval_iters)
                t1v = time.perf_counter() - t0
                monitor.eval_end(t1v)
                fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1v * 1000:.2f}ms")
                fabric.barrier()

            # 保存 ckpt（与代码1对齐：也在 step_count==max_step 触发）
            if (not is_accumulating) and (
                state["step_count"] % save_step_interval == 0 or state["step_count"] == max_step
            ):
                checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
                fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
                fabric.save(checkpoint_path, state)

        # ✅ dataloader exhausted（finetune wrap=False 会到这里）
        return True

    try:
        if args.finetune_enable:
            for ep in range(finetune_epochs_target):
                if fabric.global_rank == 0:
                    fabric.print(f"[finetune] ===== epoch {ep+1}/{finetune_epochs_target} start =====")
                run_one_epoch()
                fabric.barrier()
                if fabric.global_rank == 0:
                    fabric.print(f"[finetune] ===== epoch {ep+1}/{finetune_epochs_target} done (dataloader exhausted) =====")
        else:
            run_one_epoch()

    finally:
        final_ckpt = out_dir / f"iter-{state['iter_num']:06d}-final.pth"
        fabric.print(f"Saving final checkpoint to {str(final_ckpt)!r}")
        fabric.save(final_ckpt, state)

        audit.close(
            fabric=fabric,
            iter_num=state.get("iter_num", 0),
            step_count=state.get("step_count", 0),
            lr=last_lr,
            loss=last_loss,
        )


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, eval_iters: int) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    if eval_iters <= 0:
        out = torch.tensor(0.0, device=fabric.device)
        model.train()
        return out

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break

        if isinstance(val_data, (tuple, list)):
            val_tokens = val_data[0]
        else:
            val_tokens = val_data

        val_tokens = val_tokens.to(dtype=torch.long)

        mc_loss = torch.zeros(128, device=fabric.device)
        for i in range(128):
            input_ids = val_tokens[:, 0: model.config.block_size].contiguous()
            noisy_input, mask_indices, p_mask = forward_process(input_ids, mask_id=int(MASK_TOKEN_ID))
            logits = model(noisy_input)
            loss = torch.nn.functional.cross_entropy(
                logits[mask_indices],
                input_ids[mask_indices],
                reduction='none',
            ) / p_mask[mask_indices]
            loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
            mc_loss[i] = loss

        losses[k] = mc_loss.mean().item()

    losses = fabric.all_reduce(losses, reduce_op="mean")
    out = losses.mean()

    model.train()
    return out


@record
def _main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    setup(args=args)


if __name__ == "__main__":
    _main()
