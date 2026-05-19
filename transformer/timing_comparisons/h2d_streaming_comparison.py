import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

from transformer_pipeline import Transformer
import loss
import optimizer


METHODS = (
    "sequential",
    "shuffle",
    "sorted_batch",
    "block_shuffle",
    "round_robin_block_shuffle",
)

METHOD_ALIASES = {
    "sorted": "sorted_batch",
    "block": "block_shuffle",
    "roundrobin": "round_robin_block_shuffle",
    "round_robin": "round_robin_block_shuffle",
    "round_robin_block": "round_robin_block_shuffle",
}

MODES = ("no_stream", "stream")


class UncachedMMapDataset(torch.utils.data.Dataset):
    """Uncached mmap-backed token windows without an explicit per-item copy."""

    def __init__(self, filepath, context_len=1024):
        try:
            self.contents = np.load(filepath, mmap_mode="r+")
        except (OSError, PermissionError, ValueError):
            self.contents = np.load(filepath, mmap_mode="r")
        self.context_len = context_len
        self.num_sequences = max(len(self.contents) - context_len, 0)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        x_val = self.contents[idx:idx + self.context_len]
        y_val = self.contents[idx + 1:idx + 1 + self.context_len]
        return torch.from_numpy(x_val), torch.from_numpy(y_val)


class SortedRandomBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, drop_last=True, generator=None):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = generator

    def __iter__(self):
        indices = torch.randperm(self.dataset_len, generator=self.generator).tolist()
        for start in range(0, self.dataset_len, self.batch_size):
            batch = indices[start:start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield sorted(batch)

    def __len__(self):
        if self.drop_last:
            return self.dataset_len // self.batch_size
        return math.ceil(self.dataset_len / self.batch_size)


class BlockShuffleBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, block_size, drop_last=True, generator=None):
        if block_size < batch_size:
            raise ValueError("block_size must be >= batch_size")

        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.block_size = block_size
        self.drop_last = drop_last
        self.generator = generator

    def __iter__(self):
        block_starts = list(range(0, self.dataset_len, self.block_size))
        block_order = torch.randperm(len(block_starts), generator=self.generator).tolist()

        for block_idx in block_order:
            block_start = block_starts[block_idx]
            block_end = min(block_start + self.block_size, self.dataset_len)
            batch_starts = list(range(block_start, block_end, self.batch_size))
            batch_order = torch.randperm(len(batch_starts), generator=self.generator).tolist()

            for batch_idx in batch_order:
                batch_start = batch_starts[batch_idx]
                batch_end = min(batch_start + self.batch_size, block_end)
                if self.drop_last and batch_end - batch_start < self.batch_size:
                    continue
                yield list(range(batch_start, batch_end))

    def __len__(self):
        num_batches = 0
        for block_start in range(0, self.dataset_len, self.block_size):
            block_len = min(self.block_size, self.dataset_len - block_start)
            if self.drop_last:
                num_batches += block_len // self.batch_size
            else:
                num_batches += math.ceil(block_len / self.batch_size)
        return num_batches


class RoundRobinBlockShuffleBatchSampler(BlockShuffleBatchSampler):
    def __iter__(self):
        block_starts = list(range(0, self.dataset_len, self.block_size))
        block_order = torch.randperm(len(block_starts), generator=self.generator).tolist()

        blocks = []
        for block_idx in range(len(block_starts)):
            block_start = block_starts[block_idx]
            block_end = min(block_start + self.block_size, self.dataset_len)
            block_len = block_end - block_start
            if self.drop_last:
                num_batches = block_len // self.batch_size
            else:
                num_batches = math.ceil(block_len / self.batch_size)
            blocks.append((block_start, block_end, num_batches))

        max_batches = max((num_batches for _, _, num_batches in blocks), default=0)
        for batch_idx in range(max_batches):
            for block_idx in block_order:
                block_start, block_end, num_batches = blocks[block_idx]
                if batch_idx >= num_batches:
                    continue

                batch_start = block_start + batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, block_end)
                yield list(range(batch_start, batch_end))


class BatchSource:
    def __init__(self, loader, max_epochs):
        self.loader = loader
        self.max_epochs = max_epochs
        self.epoch = 0
        self.iterator = iter(loader)

    def next(self):
        while self.epoch < self.max_epochs:
            try:
                return next(self.iterator)
            except StopIteration:
                self.epoch += 1
                if self.epoch >= self.max_epochs:
                    return None
                self.iterator = iter(self.loader)
        return None


@dataclass
class DeviceBatch:
    x: torch.Tensor
    y: torch.Tensor
    host_x: torch.Tensor
    host_y: torch.Tensor


def normalize_method(method):
    return METHOD_ALIASES.get(method, method)


def parse_methods(value):
    requested = [method.strip() for method in value.split(",") if method.strip()]
    if requested == ["all"]:
        return list(METHODS)

    methods = [normalize_method(method) for method in requested]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown methods: {', '.join(unknown)}")
    return methods


def parse_modes(value):
    modes = [mode.strip() for mode in value.split(",") if mode.strip()]
    if modes == ["both"]:
        return list(MODES)

    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown modes: {', '.join(unknown)}")
    return modes


def make_dataloader(dataset, method, args, block_size):
    generator = torch.Generator().manual_seed(args.seed)
    common_args = {
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        common_args["persistent_workers"] = True
        common_args["prefetch_factor"] = args.prefetch_factor

    if method == "sequential":
        return DataLoader(
            dataset,
            batch_size=args.batchsize,
            shuffle=False,
            drop_last=True,
            **common_args,
        )

    if method == "shuffle":
        return DataLoader(
            dataset,
            batch_size=args.batchsize,
            shuffle=True,
            drop_last=True,
            generator=generator,
            **common_args,
        )

    if method == "sorted_batch":
        sampler = SortedRandomBatchSampler(
            len(dataset),
            args.batchsize,
            drop_last=True,
            generator=generator,
        )
    elif method == "block_shuffle":
        sampler = BlockShuffleBatchSampler(
            len(dataset),
            args.batchsize,
            block_size=block_size,
            drop_last=True,
            generator=generator,
        )
    elif method == "round_robin_block_shuffle":
        sampler = RoundRobinBlockShuffleBatchSampler(
            len(dataset),
            args.batchsize,
            block_size=block_size,
            drop_last=True,
            generator=generator,
        )
    else:
        raise ValueError(f"unknown method: {method}")

    return DataLoader(dataset, batch_sampler=sampler, **common_args)


def resolve_dff(d_ff, d_model):
    if d_ff != -1:
        return d_ff
    d_ff = round((8 / 3 * d_model) / 64) * 64
    return max(64, d_ff)


def make_model_and_optimizer(args, device):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    d_ff = resolve_dff(args.dff, args.d_model)
    model = Transformer(
        d_model=args.d_model,
        num_heads=args.heads,
        d_ff=d_ff,
        vocab_size=args.vocabsize,
        num_layers=args.num_layers,
        theta=args.rope_theta,
        max_seq_len=args.seqlen,
        device=device,
        dtype=torch.float32,
    )
    model.to(device)

    if args.tcompile:
        model = torch.compile(model)

    opt = optimizer.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=1e-8,
        weight_decay=1e-2,
    )
    return model, opt


def move_to_device(host_batch, device):
    host_x, host_y = host_batch
    return DeviceBatch(
        x=host_x.to(device, non_blocking=True),
        y=host_y.to(device, non_blocking=True),
        host_x=host_x,
        host_y=host_y,
    )


def enqueue_to_device(host_batch, device, copy_stream):
    host_x, host_y = host_batch
    with torch.cuda.stream(copy_stream):
        device_x = host_x.to(device, non_blocking=True)
        device_y = host_y.to(device, non_blocking=True)
    return DeviceBatch(device_x, device_y, host_x, host_y)


def train_step(model, opt, x, y):
    opt.zero_grad()
    y_hat = model(x)
    computed_loss = loss.getCrossEntropyLossFromClass(y, y_hat)
    computed_loss.backward()
    optimizer.gradientClipping(list(model.parameters()), 1)
    opt.step()


def required_batch(source, step):
    batch = source.next()
    if batch is None:
        raise RuntimeError(
            f"dataloader exhausted before step {step}; increase --epochs or reduce --num_steps"
        )
    return batch


def summarize_runtime(runtime, measured_steps, measured_tokens, batchsize):
    step_ms = 1e3 * runtime / max(measured_steps, 1)
    return {
        "measured_steps": measured_steps,
        "runtime_s": runtime,
        "step_ms": step_ms,
        "samples_per_s": batchsize * measured_steps / max(runtime, 1e-12),
        "tokens_per_s": measured_tokens / max(runtime, 1e-12),
    }


def benchmark_no_stream(loader, args, device):
    model, opt = make_model_and_optimizer(args, device)
    source = BatchSource(loader, args.epochs)
    start = None
    measured_steps = 0
    measured_tokens = 0

    for step in range(args.num_steps):
        host_batch = required_batch(source, step)
        batch = move_to_device(host_batch, device)
        train_step(model, opt, batch.x, batch.y)

        if step == args.warmups - 1:
            torch.cuda.synchronize()
            start = time.perf_counter()
        elif step >= args.warmups:
            measured_steps += 1
            measured_tokens += batch.y.numel()

    torch.cuda.synchronize()
    runtime = time.perf_counter() - start
    return summarize_runtime(runtime, measured_steps, measured_tokens, args.batchsize)


def benchmark_stream(loader, args, device):
    model, opt = make_model_and_optimizer(args, device)
    source = BatchSource(loader, args.epochs)
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.current_stream()

    pending = enqueue_to_device(required_batch(source, 0), device, copy_stream)
    start = None
    measured_steps = 0
    measured_tokens = 0

    for step in range(args.num_steps):
        compute_stream.wait_stream(copy_stream)
        batch = pending
        batch.x.record_stream(compute_stream)
        batch.y.record_stream(compute_stream)

        if step + 1 < args.num_steps and step != args.warmups - 1:
            pending = enqueue_to_device(required_batch(source, step + 1), device, copy_stream)
        else:
            pending = None

        train_step(model, opt, batch.x, batch.y)

        if step == args.warmups - 1:
            torch.cuda.synchronize()
            start = time.perf_counter()
            if step + 1 < args.num_steps:
                pending = enqueue_to_device(required_batch(source, step + 1), device, copy_stream)
        elif step >= args.warmups:
            measured_steps += 1
            measured_tokens += batch.y.numel()

    torch.cuda.synchronize()
    runtime = time.perf_counter() - start
    return summarize_runtime(runtime, measured_steps, measured_tokens, args.batchsize)


def print_result(method, mode, result):
    print(
        f"{method:<26} {mode:<10} "
        f"{result['measured_steps']:>8d} "
        f"{result['runtime_s']:>10.3f} "
        f"{result['step_ms']:>10.3f} "
        f"{result['samples_per_s']:>14.1f} "
        f"{result['tokens_per_s']:>14.1f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare H2D CUDA stream overlap across uncached disk-read orders"
    )
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning Rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for adamw")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for adamw")
    parser.add_argument("--d_model", type=int, default=384, help="dmodel for transformer")
    parser.add_argument("--seqlen", type=int, default=1024, help="context length")
    parser.add_argument("--heads", type=int, default=12, help="number of heads")
    parser.add_argument("--batchsize", type=int, default=1, help="batchsize")
    parser.add_argument("--num_layers", type=int, default=1, help="num of layers")
    parser.add_argument("--num_steps", type=int, default=20, help="num of steps incl. warmups")
    parser.add_argument("--warmups", type=int, default=5, help="num of warmup steps before timing")
    parser.add_argument("--rope_theta", type=int, default=10000, help="parameter for rope")
    parser.add_argument("--dff", type=int, default=-1, help="ffsize")
    parser.add_argument("--vocabsize", type=int, default=50304, help="vocab size of the tokenizer used")
    parser.add_argument("--epochs", type=int, default=1, help="max dataloader epochs")
    parser.add_argument(
        "--tcompile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch.compile (default: true)",
    )

    parser.add_argument("--tfile", type=str, default="temp/temp.npy", help="token .npy file")
    parser.add_argument("--block_size", type=int, default=10_000_000, help="block size in dataset indices")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch factor when workers > 0")
    parser.add_argument("--seed", type=int, default=42, help="random seed for model init and samplers")
    parser.add_argument(
        "--methods",
        type=parse_methods,
        default=list(METHODS),
        help=(
            "comma-separated subset of all,sequential,shuffle,sorted_batch,"
            "block_shuffle,round_robin_block_shuffle"
        ),
    )
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default=list(MODES),
        help="comma-separated subset of both,no_stream,stream",
    )
    args = parser.parse_args()

    if args.warmups <= 0:
        raise ValueError("--warmups must be > 0")
    if args.warmups >= args.num_steps:
        raise ValueError("--warmups must be < --num_steps")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark is CUDA-only")
    device = torch.device("cuda")

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset = UncachedMMapDataset(args.tfile, args.seqlen)
    if len(dataset) < args.batchsize:
        raise ValueError(f"dataset has {len(dataset)} sequences, fewer than batchsize={args.batchsize}")

    d_ff = resolve_dff(args.dff, args.d_model)
    print(f"file={args.tfile}")
    print(
        f"dataset=uncached_no_copy pin_memory=True async_xfer=True "
        f"tensor_cores=high torch_compile={args.tcompile}"
    )
    print(
        f"d_model={args.d_model} d_ff={d_ff} heads={args.heads} layers={args.num_layers} "
        f"seqlen={args.seqlen} batchsize={args.batchsize}"
    )
    print(
        f"num_steps={args.num_steps} warmups={args.warmups} epochs={args.epochs} "
        f"num_workers={args.num_workers} block_size={args.block_size}"
    )
    print("Note: OS page cache is not flushed between methods.")
    print()
    print(
        f"{'method':<26} {'mode':<10} "
        f"{'steps':>8} {'total_s':>10} {'ms/step':>10} "
        f"{'samples/s':>14} {'tokens/s':>14}"
    )
    print("-" * 99)

    for method in args.methods:
        for mode in args.modes:
            loader = make_dataloader(dataset, method, args, args.block_size)
            if mode == "no_stream":
                result = benchmark_no_stream(loader, args, device)
            elif mode == "stream":
                result = benchmark_stream(loader, args, device)
            else:
                raise ValueError(f"unknown mode: {mode}")
            print_result(method, mode, result)


if __name__ == "__main__":
    main()
