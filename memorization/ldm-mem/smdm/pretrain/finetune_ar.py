# finetune_ar.py
import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial

# =========================
# 审计相关 import
# =========================
import os
import json
from collections import defaultdict

# ckpt 搜索
import re

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.model import GPT, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger
from swanlab.integration.pytorch_lightning import SwanLabLogger
from lit_gpt import FusedCrossEntropyLoss

import random
import argparse


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=int, required=True, help='model parameters (M)')
    parse.add_argument('--nodes_num', type=int, default=1, help='number of nodes')
    parse.add_argument('--devices', type=int, default=8, help='number of devices')
    parse.add_argument('--flops', type=float, default=0.0, help='FLOPs budget, *e18 (pretrain only)')
    parse.add_argument('--batch_size', type=int, default=256, help='global_batch_size')

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
    # finetune：不按 flops 截止，改为按“跑完一遍可用 bins”停止
    # - finetune_epochs：表示“跑完一遍可用 bins”算 1 epoch
    # =========================
    parse.add_argument('--finetune_enable', action='store_true',
                       help='enable finetune on new bins, stop by bin coverage')
    parse.add_argument('--finetune_dir', type=str,
                       default='smdm/dataset/slim_star_combined',
                       help='finetune bins folder')
    parse.add_argument('--finetune_pattern', type=str, default='validation_slimpajama_*.bin',
                       help='glob pattern inside finetune_dir')
    parse.add_argument('--finetune_epochs', type=int, default=1,
                       help='how many epochs by bin-coverage (global union over ranks)')

    return parse.parse_args()


args = parse_args()

model_name = f'Diff_LLaMA_{args.model}M'
out_dir = Path('workdir')

model_para_config = {
    '6': 6.294784, '19': 18.880896, '34': 33.563136, '48': 47.786688, '66': 65.54944,
    '85': 85.21408, '75': 75.38752, '113': 113.265408, '142': 141.581568, '170': 169.897728,
    '180': 179.856768, '206': 205.550464, '231': 231.24416, '268': 268.469248, '302': 302.027776,
    '336': 335.586304, '472': 471.90656, '551': 550.55744, '571': 571.001728, '629': 629.20832,
    '666': 666.168448, '717': 717.285888, '761': 761.335168, '831': 830.541312, '944': 943.796736,
    '1028': 1027.677952, '1233': 1233.213184, '1476': 1476.487168, '1678': 1677.826048, '2121': 2121.39328
}

# =========================
# Hyperparameters
# =========================
num_of_devices = args.devices
global_batch_size = int(args.batch_size / args.nodes_num)

# 预训练默认学习率（finetune 会覆盖为固定值）
learning_rate = 2e-4

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 2e-5  # 终末段 lr（finetune 固定用这个）

# finetune 固定超参：固定 lr=终末段(min_lr)，重置 optimizer（状态清空）
FINETUNE_LR = 2e-5
FINETUNE_WD = 0.1

if args.model <= 20:
    micro_batch_size = 32
elif args.model <= 50:
    micro_batch_size = 16
elif args.model <= 1280:
    micro_batch_size = 8
elif args.model <= 2000:
    micro_batch_size = 4
else:
    micro_batch_size = 2

# pretrain 才用 flops 预算；finetune 不用
max_step = 0
if (not args.finetune_enable) and (args.flops is None or float(args.flops) <= 0.0):
    raise RuntimeError("[pretrain] --flops 必须 > 0（finetune 模式不需要 --flops）")

if (not args.finetune_enable) and float(args.flops) > 0.0:
    max_step = int(
        args.flops * 1e12 / (6 * model_para_config[f'{args.model}'] * global_batch_size * 2048) / args.nodes_num
    )

warmup_steps = int(max_step / 100) if int(max_step / 100) > 100 else 100
log_step_interval = 10
eval_iters = int(100 * 1024 / global_batch_size) if global_batch_size > 0 else 0

# finetune 这里也用更频繁的保存（对齐你 mdm finetune 的改法）
save_step_interval = 100
eval_step_interval = 999999999999  # inf

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps

max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps

# dataset config（finetune 会在 main() 里改）
train_data_config = [
    ("train_slim", 1.0),
]
val_data_config = [
    ("validation", 1.0),
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", model_name, flush_logs_every_n_steps=log_iter_interval)


def auto_find_latest_ckpt_or_die(model_m: int) -> Path:
    """
    在 workdir/scaling_debug 下搜索 arm-{model}M-* 的 iter-*-ckpt.pth，取 iter 最大的。
    finetune 模式必须成功找到，否则直接报错退出（避免默默从头训）。
    """
    root = Path("workdir/scaling_debug")
    best = None  # (iter, path)
    for p in root.glob(f"arm-{model_m}M-*/iter-*-ckpt.pth"):
        m = re.search(r"iter-(\d+)-ckpt\.pth$", str(p))
        if not m:
            continue
        it = int(m.group(1))
        if best is None or it > best[0]:
            best = (it, p)
    if best is None:
        raise RuntimeError(
            f"[finetune] No checkpoint found under {root}/arm-{model_m}M-*/iter-*-ckpt.pth"
        )
    return best[1]


class AuditWindow:
    """
    低侵入、低开销的数据审计记录器：
    - 训练中只做内存聚合（set/dict），不频繁写盘
    - 每 N 个 optimizer step 写一次 jsonl
    - 文件句柄保持打开 + 大 buffer
    """
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


def setup(
    devices: int = 8,
    train_data_dir: Path = Path("smdm/dataset/slim_star_combined"),
    val_data_dir: Path = Path("smdm/dataset/slim_star_combined"),
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = True,
) -> None:
    global out_dir
    hp_name = f'arm-{args.model}M-{args.flops}'
    out_dir = Path('workdir/scaling_debug') / hp_name
    swan_logger = SwanLabLogger(name=hp_name, save_dir=out_dir, project='scaling')

    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if devices > 1:
        if tpu:
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices="auto", strategy=strategy, precision=precision, loggers=[logger, swan_logger])
    fabric.print(hparams)
    main(fabric, train_data_dir, val_data_dir, resume)


def main(fabric, train_data_dir, val_data_dir, resume):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 审计日志目录与文件名（每 rank 单独写）
    audit_dir = out_dir / "audit"
    audit_local_path = audit_dir / f"audit_rank{fabric.global_rank:04d}.jsonl"
    if args.audit_enable:
        if fabric.global_rank == 0:
            audit_dir.mkdir(parents=True, exist_ok=True)
        fabric.barrier()

    # finetune：切换数据目录 + prefix；val=None；wrap=False（跑完一遍可用 bins 就 StopIteration）
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

    config = Config.from_name(model_name)

    train_wrap = (not args.finetune_enable)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
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

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)

    # =========================
    # resume / load：
    # - finetune：自动找最新 AR ckpt，只加载 model 权重（optimizer 重置）
    # - 非 finetune：加载全量 state
    # =========================
    if resume is True:
        if args.finetune_enable:
            resume = auto_find_latest_ckpt_or_die(args.model)
        else:
            def extract_number(filename):
                m = re.search(r'iter-(\d+)-ckpt\.pth', str(filename))
                return int(m.group(1)) if m else 0
            try:
                resume = sorted(out_dir.glob("*.pth"), key=extract_number)[-1]
            except Exception:
                resume = False

    if resume:
        fabric.print(f"Resuming from {resume}")
        if args.finetune_enable:
            # ✅ 只 load model（optimizer/state 全重置）
            fabric.load(resume, {"model": model})
            if fabric.global_rank == 0:
                fabric.print(f"[finetune] loaded model weights only; optimizer reset (lr_fixed={FINETUNE_LR}, wd_fixed={FINETUNE_WD})")
        else:
            # full load later after optimizer/state created
            pass
    else:
        if args.finetune_enable:
            raise RuntimeError("[finetune] resume checkpoint not found (should not happen).")
        fabric.print("No checkpoint found, training from scratch.")

    # =========================
    # optimizer：
    # - finetune：重置 AdamW，固定 lr=终末段(min_lr=2e-5)，wd=0.1
    # - 非 finetune：按原设置
    # =========================
    if args.finetune_enable:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=FINETUNE_LR, weight_decay=FINETUNE_WD, betas=(beta1, beta2), foreach=False
        )
        optimizer = fabric.setup_optimizers(optimizer)
        state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
        )
        optimizer = fabric.setup_optimizers(optimizer)
        state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}
        if resume:
            fabric.load(resume, state)

    # finetune：不做 dataloader resume-skip
    resume_skip_loader = bool((not args.finetune_enable) and bool(resume))

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume_skip_loader, audit_local_path)
    fabric.print(f"Training time: {(time.perf_counter() - train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, monitor, resume_skip_loader, audit_local_path: Path):
    model = state["model"]
    optimizer = state["optimizer"]

    with torch.device("meta"):
        meta_model = GPT(model.config)
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()

    initial_iter = state["iter_num"]
    curr_iter = 0

    loss_func = FusedCrossEntropyLoss()

    audit = AuditWindow(
        path=audit_local_path,
        every_steps=args.audit_every,
        level=args.audit_level,
        enable=args.audit_enable,
        fsync=args.audit_fsync,
    )

    last_lr = float(FINETUNE_LR if args.finetune_enable else learning_rate)
    last_loss = 0.0

    finetune_epochs_target = max(1, int(args.finetune_epochs)) if args.finetune_enable else 0
    finetune_lr = FINETUNE_LR if args.finetune_enable else None

    if args.finetune_enable and fabric.global_rank == 0:
        num_workers = getattr(train_dataloader, "num_workers", 0)
        num_workers = 1 if int(num_workers) == 0 else int(num_workers)
        total = len(sorted(glob.glob(str(Path(args.finetune_dir) / args.finetune_pattern))))
        group = fabric.world_size * num_workers * 8
        usable = (total // group) * group
        fabric.print(f"[finetune] total_bins={total} group={group} -> usable_bins={usable}")
        fabric.print(f"[finetune] epochs={finetune_epochs_target} lr_fixed={FINETUNE_LR} wd_fixed={FINETUNE_WD}")

    def run_one_epoch():
        nonlocal curr_iter, last_lr, last_loss, total_lengths, total_t0, resume_skip_loader
        for train_data in train_dataloader:
            # 仅非 finetune 才可能需要跳过 dataloader 来对齐断点
            if resume_skip_loader:
                if curr_iter < initial_iter:
                    curr_iter += 1
                    continue
                resume_skip_loader = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
                audit.reset_window()

            # 非 finetune：预算停止
            if (not args.finetune_enable) and (state["iter_num"] >= max_iters):
                return False  # stop training

            # 学习率
            if args.finetune_enable:
                lr = finetune_lr  # 固定终末段 2e-5
            else:
                lr = get_lr(state["iter_num"]) if decay_lr else learning_rate

            last_lr = float(lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

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
                    train_tokens = train_data[0]
            else:
                train_tokens = train_data

            input_ids = train_tokens[:, 0: model.config.block_size].contiguous()
            targets = train_tokens[:, 1: model.config.block_size + 1].contiguous()

            # 审计：收集 bin
            if args.audit_enable and train_filenames is not None:
                audit.add_batch(train_filenames, train_block_ids)

            is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                logits = model(input_ids)
                loss = loss_func(logits, targets)
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
                state["iter_num"] * micro_batch_size,
                t1 - total_t0,
                fabric.world_size,
                state["step_count"],
                flops_per_batch=estimated_flops,
                lengths=total_lengths,
                train_loss=loss.item()
            )

            # 验证（finetune 默认 None）
            if val_dataloader is not None and (not is_accumulating) and (
                state["step_count"] % eval_step_interval == 0 or state["step_count"] == max_step
            ):
                t0v = time.perf_counter()
                val_loss = validate(fabric, model, val_dataloader)
                t1v = time.perf_counter() - t0v
                monitor.eval_end(t1v)
                fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1v * 1000:.2f}ms")
                fabric.barrier()

            # 保存 ckpt
            if (not is_accumulating) and (
                state["step_count"] % save_step_interval == 0 or state["step_count"] == max_step
            ):
                checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
                fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
                fabric.save(checkpoint_path, state)

        # dataloader exhausted（finetune wrap=False 会到这里）
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
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break

        # 兼容 Tensor / (Tensor, filename) / (Tensor, filename, block_id)
        if isinstance(val_data, (tuple, list)):
            val_tokens = val_data[0]
        else:
            val_tokens = val_data

        input_ids = val_tokens[:, 0: model.config.block_size].contiguous()
        targets = val_tokens[:, 1: model.config.block_size + 1].contiguous()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        losses[k] = loss.item()

    losses = fabric.all_reduce(losses, reduce_op="mean")
    out = losses.mean()

    model.train()
    return out


def create_dataloader(
    batch_size: int,
    block_size: int,
    data_dir: Path,
    fabric,
    shuffle: bool = True,
    seed: int = 12345,
    split: str = "train",
    wrap: bool = True,
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
            wrap=wrap,
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
    train_wrap: bool = True,
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


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


from torch.distributed.elastic.multiprocessing.errors import record

@record
def _main():
    torch.set_float32_matmul_precision("high")
    setup(devices=args.devices)

if __name__ == "__main__":
    _main()
