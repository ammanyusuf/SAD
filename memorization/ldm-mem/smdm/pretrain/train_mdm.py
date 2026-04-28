import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial

# =========================
# 【新增】审计相关 import（中文注释：用于低开销写 jsonl + 窗口聚合 + 可选 fsync）
# =========================
import os
import json
from collections import defaultdict
import torch.distributed as dist  # 中文注释：可选用于分布式场景判断（默认不做 gather，避免同步开销）

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.diffmodel import TransEncoder, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
#from pytorch_lightning.loggers import WandbLogger
from swanlab.integration.pytorch_lightning import SwanLabLogger
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import random
import argparse


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=int, help='model parameters')
    parse.add_argument('--nodes_num', type=int, default=1, help='number of nodes')
    parse.add_argument('--flops', type=float, help='FLOPs, *e18')
    parse.add_argument('--batch_size', type=int, default=192, help='global_batch_size')

    # =========================
    # 【新增】审计开关与参数（中文注释：可配置、低 IO、按窗口写入，避免影响训练吞吐）
    # =========================
    parse.add_argument('--audit_enable', action='store_true', help='enable audit logging (low-overhead)')
    parse.add_argument('--audit_every', type=int, default=100, help='log every N optimizer steps (window)')
    parse.add_argument('--audit_level', type=str, default='bin', choices=['bin', 'block'],
                       help="audit granularity: 'bin' or '(bin, block_id)'")
    parse.add_argument('--audit_fsync', action='store_true',
                       help='fsync on every audit flush (slower but safer)')

    args = parse.parse_args()
    return args


args = parse_args()
model_name = f'Diff_LLaMA_{args.model}M'  # config
out_dir = Path('workdir')

model_para_config = {
    '6': 6.294784, '19': 18.880896, '34': 33.563136, '48': 47.786688, '66': 65.54944,
    '85': 85.21408, '75': 75.38752, '113': 113.265408, '142': 141.581568, '170': 169.897728,
    '180': 179.856768, '206': 205.550464, '231': 231.24416, '268': 268.469248, '302': 302.027776,
    '336': 335.586304, '472': 471.90656, '551': 550.55744, '571': 571.001728, '629': 629.20832,
    '666': 666.168448, '717': 717.285888, '761': 761.335168, '831': 830.541312, '944': 943.796736,
    '1028': 1027.677952, '1233': 1233.213184, '1476': 1476.487168, '1678': 1677.826048, '2121': 2121.39328
}

# Hyperparameters
num_of_devices = 6
global_batch_size = int(args.batch_size / args.nodes_num)
learning_rate = 2e-4
if args.model <= 20:
    micro_batch_size = 32
elif args.model <= 50:
    micro_batch_size = 16
elif args.model <= 1280:
    micro_batch_size = 8
else:
    micro_batch_size = 4
max_step = int(args.flops * 1e12 / (6 * model_para_config[f'{args.model}'] * global_batch_size * 2048) / args.nodes_num)
warmup_steps = int(max_step / 100) if int(max_step / 100) > 100 else 100
log_step_interval = 10
eval_iters = int(100 * 1024 / global_batch_size)
save_step_interval = 1000
eval_step_interval = 999999999999 #inf

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 2e-5

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps

max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps

# Treat all dataset equally by their size. If you want to use a different weight for a dataset, add it to the list with the weight.
train_data_config = [
    ("train_slim", 1.0),
    #("train_star", 0.),
]

val_data_config = [
    ("validation", 1.0),
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", model_name, flush_logs_every_n_steps=log_iter_interval)


# =========================
# 【新增】审计窗口聚合器（中文注释：只在每 audit_every 个 optimizer step 写一次，最大化减少 IO）
# =========================
class AuditWindow:
    """
    低侵入、低开销的数据审计记录器：
    - 训练中只做内存聚合（set/dict），不频繁写盘
    - 每 N 个 optimizer step（step_count）写一次 jsonl
    - 文件句柄保持打开 + 大 buffer，减少 open/close 和系统调用
    """
    def __init__(self, path: Path, every_steps: int = 100, level: str = "bin",
                 enable: bool = False, fsync: bool = False):
        self.enable = bool(enable)
        self.every_steps = max(1, int(every_steps))
        self.level = level
        self.fsync = bool(fsync)

        self.path = Path(path)
        self._fp = None  # 中文注释：未开启审计时不打开文件，完全无 IO 影响

        self._bins = set()
        self._bin_blocks = defaultdict(set)  # filename -> set(block_id)
        self._win_start_step = None

        if self.enable:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # 中文注释：buffering=1MB，减少写盘 syscalls；全程保持打开，避免反复 open/close
            self._fp = open(self.path, "a", encoding="utf-8", buffering=1024 * 1024)

    def reset_window(self):
        """中文注释：重置窗口（例如 resume 跳过阶段结束后，避免把跳过前后的数据混在一个窗口里）"""
        if not self.enable:
            return
        self._bins.clear()
        self._bin_blocks.clear()
        self._win_start_step = None

    def add_batch(self, filenames, block_ids=None):
        """中文注释：收集一个 micro-batch 的审计信息（只做内存聚合，不写盘）"""
        if not self.enable or filenames is None:
            return

        # 中文注释：audit_level=bin 或者没有 block_ids 时，只记录 bin 文件名集合
        if self.level == "bin" or block_ids is None:
            self._bins.update(list(filenames))
            return

        # 中文注释：audit_level=block 时，记录 (bin, block_id) 去重集合
        if torch.is_tensor(block_ids):
            # block_ids 一般在 CPU 上（DataLoader collate），tolist 很快
            block_ids = block_ids.tolist()

        for fn, bid in zip(filenames, block_ids):
            self._bins.add(fn)
            self._bin_blocks[fn].add(int(bid))

    def maybe_flush(self, fabric, iter_num: int, step_count: int, lr: float, loss: float, force: bool = False):
        """
        中文注释：
        - 仅在每 every_steps 个 optimizer step 时写盘一次（或 force=True 强制写）
        - 写入 jsonl 一行，包含窗口内见过的 bins（以及可选 blocks）
        """
        if not self.enable or self._fp is None:
            return

        if self._win_start_step is None:
            self._win_start_step = int(step_count)

        # 中文注释：不满足窗口边界且不强制时，不写盘
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
            # 中文注释：可选记录每个 bin 覆盖到的 block_id 列表（去重后排序）
            record["bin_blocks"] = {k: sorted(v) for k, v in self._bin_blocks.items()}

        # 中文注释：写入单行 jsonl
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fp.flush()  # 中文注释：仅窗口边界 flush，频率低，对训练影响小

        # 中文注释：默认不 fsync（性能最好）；需要更强持久化时再开 --audit_fsync
        if self.fsync:
            os.fsync(self._fp.fileno())

        # 中文注释：写完一窗后清空，开始下一窗
        self._bins.clear()
        self._bin_blocks.clear()
        self._win_start_step = int(step_count) + 1

    def close(self, fabric=None, iter_num: int = 0, step_count: int = 0, lr: float = 0.0, loss: float = 0.0):
        """中文注释：训练结束时，强制把最后不足一窗的残留数据也写出去（force flush）"""
        if not self.enable or self._fp is None:
            return
        # 强制刷一次，把最后没到窗口边界的数据也落盘（依然只写 1 行）
        if len(self._bins) > 0:
            self.maybe_flush(fabric=fabric, iter_num=iter_num, step_count=step_count, lr=lr, loss=loss, force=True)
        try:
            self._fp.close()
        except Exception:
            pass


def forward_process(batch, total_dim=32000, eps=1e-3):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)

    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim, batch)
    return noisy_batch, mask_indices, p_mask


def setup(
    devices: int = 6,
    train_data_dir: Path = Path("smdm/dataset/slim_star_combined"),
    val_data_dir: Path = Path("smdm/dataset/slim_star_combined"),
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = True,
) -> None:
    global out_dir
    hp_name = f'mdm-{args.model}M-{args.flops}'
    out_dir = Path('workdir/scaling_debug') / hp_name
    wandb_logger = SwanLabLogger(name=f'{hp_name}-mc', save_dir=out_dir, project='scaling')

    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
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

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
    fabric.print(hparams)
    #fabric.launch(main, train_data_dir, val_data_dir, resume)
    main(fabric, train_data_dir, val_data_dir, resume)


def main(fabric, train_data_dir, val_data_dir, resume):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 【新增】审计日志目录与文件名（中文注释：每个 rank 单独写一个本地日志，避免跨 rank 同步影响性能）
    # =========================
    audit_dir = out_dir / "audit"
    audit_local_path = audit_dir / f"audit_rank{fabric.global_rank:04d}.jsonl"
    if args.audit_enable:
        if fabric.global_rank == 0:
            audit_dir.mkdir(parents=True, exist_ok=True)
        # 中文注释：保证所有 rank 看到目录存在（只在开启审计时 barrier，减少不必要同步）
        fabric.barrier()

    config = Config.from_name(model_name)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = TransEncoder(config)
        model.apply(partial(model._init_weights ,n_layer=config.n_layer))

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        import re
        def extract_number(filename):
            match = re.search(r'iter-(\d+)-ckpt\.pth', str(filename))
            return int(match.group(1)) if match else 0
        try:
            resume = sorted(out_dir.glob("*.pth"), key=extract_number)[-1]
        except:
            resume = False
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()

    # =========================
    # 【修改】把 audit_local_path 传给 train（中文注释：最小侵入，只多传一个参数）
    # =========================
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume, audit_local_path)

    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


# =========================
# 【修改】train() 增加一个 audit_local_path 参数（中文注释：最小侵入改动）
# =========================
def train(fabric, state, train_dataloader, val_dataloader, monitor, resume, audit_local_path: Path):
    model = state["model"]
    optimizer = state["optimizer"]

    # if val_dataloader is not None:
    #     validate(fabric, model, val_dataloader)  # sanity check

    with torch.device("meta"):
        meta_model = TransEncoder(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        # measured_flos run in meta. Will trigger fusedRMSNorm error
        #measured_flops = measure_flops(meta_model, x)
        #fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()

    initial_iter = state["iter_num"]
    curr_iter = 0

    loss_func = CrossEntropyLoss(reduction='none')

    # =========================
    # 【新增】创建审计窗口对象（中文注释：开启审计时才会真正打开文件句柄；否则完全 no-op）
    # =========================
    audit = AuditWindow(
        path=audit_local_path,
        every_steps=args.audit_every,
        level=args.audit_level,
        enable=args.audit_enable,
        fsync=args.audit_fsync,
    )

    # 中文注释：用于训练结束 close 时传入最后一次 lr/loss（如果训练中断，也尽量写出最后残留窗口）
    last_lr = float(learning_rate)
    last_loss = 0.0

    try:
        for train_data in train_dataloader:
            # resume loader state. This is not elegant but it works. Should rewrite it in the future.
            if resume:
                if curr_iter < initial_iter:
                    curr_iter += 1
                    continue
                else:
                    resume = False
                    curr_iter = -1
                    fabric.barrier()
                    fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
                    # =========================
                    # 【新增】resume 跳过结束后重置审计窗口（中文注释：避免把跳过前后混在一个窗口里）
                    # =========================
                    audit.reset_window()

            if state["iter_num"] >= max_iters:
                break

            # determine and set the learning rate for this iteration
            lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
            last_lr = float(lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            iter_t0 = time.perf_counter()

            # =========================
            # 【修改】适配新返回格式（中文注释：兼容 Tensor / (Tensor, filename) / (Tensor, filename, block_id)）
            # =========================
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

            input_ids = train_tokens[:, 0: model.config.block_size].contiguous()

            # =========================
            # 【新增】收集审计信息（中文注释：仅内存聚合，不写盘；写盘发生在每 audit_every 个 optimizer step）
            # =========================
            if args.audit_enable and train_filenames is not None:
                # 中文注释：如果你当前 iterator 只返回 (tensor, filename)，train_block_ids=None，会自动退化为只记录 bin
                audit.add_batch(train_filenames, train_block_ids)

            noisy_input, mask_indices, p_mask = forward_process(input_ids)
            is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0

            with fabric.no_backward_sync(model, enabled=is_accumulating):
                logits = model(noisy_input)
                loss = loss_func(logits[mask_indices], input_ids[mask_indices]) / p_mask[mask_indices]
                loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
                last_loss = float(loss.item())

                # loss = chunked_cross_entropy(logits, targets, chunk_size=0)
                fabric.backward(loss / gradient_accumulation_steps)

            did_optimizer_step = False
            if not is_accumulating:
                fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                state["step_count"] += 1
                did_optimizer_step = True
            elif fabric.device.type == "xla":
                xm.mark_step()

            state["iter_num"] += 1

            # =========================
            # 【新增】在 optimizer step 时按窗口落盘（中文注释：每 N 个 step 写一次，尽量不影响训练性能）
            # =========================
            if did_optimizer_step and args.audit_enable:
                audit.maybe_flush(
                    fabric=fabric,
                    iter_num=state["iter_num"],
                    step_count=state["step_count"],
                    lr=last_lr,
                    loss=last_loss,
                )

            # input_id: B L
            total_lengths += input_ids.size(1)
            t1 = time.perf_counter()
            fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. "
                # print days as well
                f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
            )

            monitor.on_train_batch_end(
                state["iter_num"] * micro_batch_size,
                t1 - total_t0,
                # this assumes that device FLOPs are the same and that all devices have the same batch size
                fabric.world_size,
                state["step_count"],
                flops_per_batch=estimated_flops,
                lengths=total_lengths,
                train_loss=loss.item()
            )

            if val_dataloader is not None and not is_accumulating and (state["step_count"] % eval_step_interval == 0 or state["step_count"] == max_step):
                t0 = time.perf_counter()
                val_loss = validate(fabric, model, val_dataloader)
                t1 = time.perf_counter() - t0
                monitor.eval_end(t1)
                fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
                fabric.log_dict({"metric/val_loss": val_loss.item(), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
                fabric.log_dict({"metric/val_ppl": math.exp(val_loss.item()), "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
                fabric.barrier()

            if not is_accumulating and (state["step_count"] % save_step_interval == 0 or state["step_count"] == max_step):
                checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
                fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
                fabric.save(checkpoint_path, state)

    finally:
        # =========================
        # 【新增】训练结束/异常退出时强制刷掉最后残留窗口（中文注释：保证最后一段数据也可审计）
        # =========================
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

        # =========================
        # 【修改】验证也要适配新返回格式（中文注释：val_data 可能是 Tensor / (Tensor, filename) / (Tensor, filename, block_id)）
        # =========================
        if isinstance(val_data, (tuple, list)):
            val_tokens = val_data[0]
        else:
            val_tokens = val_data

        mc_loss = torch.zeros(128, device=fabric.device)  # mc_num=128
        for i in range(128):
            input_ids = val_tokens[:, 0: model.config.block_size].contiguous()
            noisy_input, mask_indices, p_mask = forward_process(input_ids)
            logits = model(noisy_input)
            loss = torch.nn.functional.cross_entropy(
                logits[mask_indices],
                input_ids[mask_indices],
                reduction='none'
            ) / p_mask[mask_indices]
            loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
            mc_loss[i] = loss

        losses[k] = mc_loss.mean().item()

    losses = fabric.all_reduce(losses, reduce_op="mean")
    out = losses.mean()

    model.train()
    return out


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*.bin")))
        random.seed(seed)
        random.shuffle(filenames)

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size.
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=8 if split == "train" else 1,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed + fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
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
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
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
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)



from torch.distributed.elastic.multiprocessing.errors import record

@record
def _main():
    torch.set_float32_matmul_precision("high")
    setup()

if __name__ == "__main__":
    _main()

