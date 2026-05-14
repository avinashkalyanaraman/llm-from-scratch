import argparse
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

import filehandler


METHODS = (
    "sequential",
    "shuffle",
    "sorted_batch",
    "block_shuffle",
    "round_robin_block_shuffle",
)


class EagerCachedDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, context_len=1024, cache_size=10_000_000):
        self.contents = np.load(filepath, mmap_mode="r")
        self.context_len = context_len
        self.num_sequences = max(len(self.contents) - context_len, 0)
        self.cache = None
        self.cache_size = cache_size
        self.cachestart = -1

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        cache_start = (idx // self.cache_size) * self.cache_size

        if cache_start != self.cachestart:
            cache_end = min(cache_start + self.cache_size + self.context_len, len(self.contents))
            self.cache = self.contents[cache_start:cache_end].copy()
            self.cachestart = cache_start

        rel_start = idx % self.cache_size
        x_val = self.cache[rel_start:rel_start + self.context_len]
        y_val = self.cache[rel_start + 1:rel_start + 1 + self.context_len]

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


def parse_methods(value):
    methods = [method.strip() for method in value.split(",") if method.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown methods: {', '.join(unknown)}")
    return methods


def make_dataset(dataset_kind, args):
    if dataset_kind == "cached":
        return filehandler.myDataset(args.tfile, args.seqlen, cache_size=args.cache_size)
    if dataset_kind == "copy":
        return EagerCachedDataset(args.tfile, args.seqlen, cache_size=args.cache_size)
    if dataset_kind == "uncached":
        return filehandler.myDatasetInefficient2(args.tfile, args.seqlen)
    raise ValueError(f"unknown dataset kind: {dataset_kind}")


def make_loader(dataset, method, args, block_size):
    generator = torch.Generator().manual_seed(args.seed)
    common_args = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        common_args["persistent_workers"] = True

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


def percentile(values, pct):
    if not values:
        return float("nan")

    values = sorted(values)
    rank = (len(values) - 1) * pct
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return values[lo]
    weight = rank - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def next_batch(loader_iter, loader):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def benchmark_loader(loader, args):
    total_steps = args.warmups + args.steps
    loader_iter = iter(loader)
    timings = []
    samples = 0
    tokens = 0

    for step in range(total_steps):
        start = time.perf_counter()
        (x, y), loader_iter = next_batch(loader_iter, loader)
        elapsed = time.perf_counter() - start

        if step >= args.warmups:
            timings.append(elapsed)
            samples += x.shape[0]
            tokens += x.numel() + y.numel()

    total = sum(timings)
    return {
        "batches": len(timings),
        "mean_ms": statistics.mean(timings) * 1e3,
        "p50_ms": percentile(timings, 0.50) * 1e3,
        "p95_ms": percentile(timings, 0.95) * 1e3,
        "total_s": total,
        "samples_per_s": samples / max(total, 1e-12),
        "tokens_per_s": tokens / max(total, 1e-12),
    }


def print_result(dataset_kind, method, result):
    print(
        f"{dataset_kind:<9} {method:<26} "
        f"{result['batches']:>8d} "
        f"{result['mean_ms']:>10.3f} "
        f"{result['p50_ms']:>10.3f} "
        f"{result['p95_ms']:>10.3f} "
        f"{result['total_s']:>10.3f} "
        f"{result['samples_per_s']:>14.1f} "
        f"{result['tokens_per_s']:>14.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Compare mmap dataset read orders")
    parser.add_argument("--tfile", type=str, default="temp/temp.npy", help="mmap-backed token .npy file")
    parser.add_argument("--seqlen", type=int, default=1024, help="context length")
    parser.add_argument("--batchsize", type=int, default=16, help="batch size")
    parser.add_argument("--steps", type=int, default=100, help="measured batches")
    parser.add_argument("--warmups", type=int, default=10, help="unmeasured warmup batches")
    parser.add_argument("--dataset", choices=("cached", "uncached", "both"), default="cached")
    parser.add_argument("--copy", action="store_true", help="also benchmark an eager copy-backed cache")
    parser.add_argument("--cache_size", type=int, default=10_000_000, help="cache size for filehandler.myDataset")
    parser.add_argument("--block_size", type=int, default=None, help="block size in dataset indices; defaults to cache_size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--pin_memory", action="store_true", help="enable pinned-memory collation")
    parser.add_argument("--seed", type=int, default=42, help="sampler seed")
    parser.add_argument(
        "--methods",
        type=parse_methods,
        default=list(METHODS),
        help=f"comma-separated subset of: {', '.join(METHODS)}",
    )
    args = parser.parse_args()

    block_size = args.block_size if args.block_size is not None else args.cache_size
    dataset_kinds = ["cached", "uncached"] if args.dataset == "both" else [args.dataset]
    if args.copy:
        insert_at = 1 if "cached" in dataset_kinds else len(dataset_kinds)
        dataset_kinds.insert(insert_at, "copy")

    print(f"file={args.tfile}")
    print(
        f"seqlen={args.seqlen} batchsize={args.batchsize} "
        f"steps={args.steps} warmups={args.warmups} block_size={block_size}"
    )
    print("Note: OS page cache is not flushed between methods.")
    print()
    print(
        f"{'dataset':<9} {'method':<26} "
        f"{'batches':>8} {'mean_ms':>10} {'p50_ms':>10} {'p95_ms':>10} "
        f"{'total_s':>10} {'samples/s':>14} {'tokens/s':>14}"
    )
    print("-" * 119)

    for dataset_kind in dataset_kinds:
        for method in args.methods:
            dataset = make_dataset(dataset_kind, args)
            #print (f'dataset len (#of training samples) = {len(dataset)}')
            if len(dataset) < args.batchsize:
                raise ValueError(
                    f"dataset has {len(dataset)} sequences, fewer than batchsize={args.batchsize}"
                )
            loader = make_loader(dataset, method, args, block_size)
            result = benchmark_loader(loader, args)
            print_result(dataset_kind, method, result)


if __name__ == "__main__":
    main()
