import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

from transformer_pipeline import Transformer
import loss
import optimizer


D2H_OUTPUT_PATH = "/dev/null"


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
class PendingD2HCopy:
    host_tensor: torch.Tensor
    ready_event: torch.cuda.Event
    num_bytes: int


@dataclass
class WriterBarrier:
    done: threading.Event


class AsyncD2HWriter:
    def __init__(self, output_path):
        self.output_path = output_path
        self.pending = queue.Queue()
        self.bytes_written = 0
        self.copies_written = 0
        self.error = None
        self.thread = threading.Thread(target=self._run, name="async-d2h-writer")
        self.thread.start()

    def submit(self, pending_copy):
        if self.error is not None:
            raise RuntimeError("D2H writer thread failed") from self.error
        self.pending.put(pending_copy)

    def drain(self):
        barrier = WriterBarrier(threading.Event())
        self.pending.put(barrier)
        while not barrier.done.wait(timeout=0.1):
            if self.error is not None:
                raise RuntimeError("D2H writer thread failed") from self.error
        if self.error is not None:
            raise RuntimeError("D2H writer thread failed") from self.error

    def close(self):
        self.pending.put(None)
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("D2H writer thread failed") from self.error

    def _run(self):
        fd = None
        try:
            fd = os.open(self.output_path, os.O_WRONLY)
            while True:
                pending_copy = self.pending.get()
                try:
                    if isinstance(pending_copy, WriterBarrier):
                        pending_copy.done.set()
                        continue
                    if pending_copy is None:
                        return

                    pending_copy.ready_event.synchronize()
                    byte_view = pending_copy.host_tensor.numpy().view(np.uint8)
                    write_all(fd, memoryview(byte_view))
                    self.bytes_written += pending_copy.num_bytes
                    self.copies_written += 1
                finally:
                    self.pending.task_done()
        except BaseException as exc:
            self.error = exc
        finally:
            if fd is not None:
                os.close(fd)


@dataclass
class D2HTransferState:
    source: torch.Tensor
    copy_stream: torch.cuda.Stream
    writer: AsyncD2HWriter


def write_all(fd, data):
    while len(data) > 0:
        written = os.write(fd, data)
        if written == 0:
            raise RuntimeError("write returned 0 bytes")
        data = data[written:]


def parse_size(value):
    multipliers = {
        "": 1,
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
    }
    text = value.strip().lower().replace("_", "")
    suffix = ""
    for candidate in ("kb", "mb", "gb", "b", "k", "m", "g"):
        if text.endswith(candidate):
            suffix = candidate
            text = text[:-len(candidate)]
            break
    if suffix not in multipliers:
        raise argparse.ArgumentTypeError(f"unknown size suffix in {value!r}")
    try:
        return int(float(text) * multipliers[suffix])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid size: {value!r}") from exc


def resolve_dff(d_ff, d_model):
    if d_ff != -1:
        return d_ff
    d_ff = round((8 / 3 * d_model) / 64) * 64
    return max(64, d_ff)


def make_dataloader(dataset, args):
    generator = torch.Generator().manual_seed(args.seed)
    common_args = {
        "batch_size": args.batchsize,
        "shuffle": True,
        "drop_last": True,
        "generator": generator,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        common_args["persistent_workers"] = True
        common_args["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **common_args)


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


def move_to_device(host_batch, device):
    host_x, host_y = host_batch
    return (
        host_x.to(device, non_blocking=True),
        host_y.to(device, non_blocking=True),
    )


def make_d2h_state(args, device):
    if args.d2h_xfer_freq == 0:
        return None

    source = torch.zeros(args.tensor_size, dtype=torch.uint8, device=device)
    return D2HTransferState(
        source=source,
        copy_stream=torch.cuda.Stream(),
        writer=AsyncD2HWriter(D2H_OUTPUT_PATH),
    )


def maybe_enqueue_d2h_copy(state, step, freq):
    if state is None or (step + 1) % freq != 0:
        return None

    host_tensor = torch.empty_like(state.source, device="cpu", pin_memory=True)
    ready_event = torch.cuda.Event(blocking=True)
    compute_stream = torch.cuda.current_stream()

    with torch.cuda.stream(state.copy_stream):
        state.copy_stream.wait_stream(compute_stream)
        host_tensor.copy_(state.source, non_blocking=True)
        ready_event.record(state.copy_stream)

    pending_copy = PendingD2HCopy(
        host_tensor=host_tensor,
        ready_event=ready_event,
        num_bytes=host_tensor.numel() * host_tensor.element_size(),
    )
    state.writer.submit(pending_copy)
    return pending_copy.num_bytes


def summarize_runtime(runtime, measured_steps, measured_tokens, measured_d2h_copies, measured_d2h_bytes, batchsize):
    step_ms = 1e3 * runtime / max(measured_steps, 1)
    return {
        "measured_steps": measured_steps,
        "runtime_s": runtime,
        "step_ms": step_ms,
        "samples_per_s": batchsize * measured_steps / max(runtime, 1e-12),
        "tokens_per_s": measured_tokens / max(runtime, 1e-12),
        "d2h_copies": measured_d2h_copies,
        "d2h_bytes": measured_d2h_bytes,
        "d2h_mb": measured_d2h_bytes / 1024**2,
        "d2h_mb_per_s": (measured_d2h_bytes / 1024**2) / max(runtime, 1e-12),
    }


def benchmark(loader, args, device):
    model, opt = make_model_and_optimizer(args, device)
    source = BatchSource(loader, args.epochs)
    d2h_state = make_d2h_state(args, device)
    compute_stream = torch.cuda.current_stream()

    start = None
    measured_steps = 0
    measured_tokens = 0
    measured_d2h_copies = 0
    measured_d2h_bytes = 0

    try:
        for step in range(args.num_steps):
            batch = move_to_device(required_batch(source, step), device)
            train_step(model, opt, batch[0], batch[1])

            d2h_bytes = maybe_enqueue_d2h_copy(d2h_state, step, args.d2h_xfer_freq)

            if step == args.warmups - 1:
                torch.cuda.synchronize()
                if d2h_state is not None:
                    d2h_state.writer.drain()
                start = time.perf_counter()
            elif step >= args.warmups:
                measured_steps += 1
                measured_tokens += batch[1].numel()
                if d2h_bytes is not None:
                    measured_d2h_copies += 1
                    measured_d2h_bytes += d2h_bytes

        compute_stream.synchronize()
        runtime = time.perf_counter() - start
        result = summarize_runtime(
            runtime,
            measured_steps,
            measured_tokens,
            measured_d2h_copies,
            measured_d2h_bytes,
            args.batchsize,
        )
    finally:
        if d2h_state is not None:
            flush_start = time.perf_counter()
            d2h_state.writer.close()
            flush_s = time.perf_counter() - flush_start
        else:
            flush_s = 0.0

    result["d2h_flush_s"] = flush_s
    return result


def print_result(result):
    print(
        f"{result['measured_steps']:>8d} "
        f"{result['runtime_s']:>10.3f} "
        f"{result['step_ms']:>10.3f} "
        f"{result['samples_per_s']:>14.1f} "
        f"{result['tokens_per_s']:>14.1f} "
        f"{result['d2h_copies']:>10d} "
        f"{result['d2h_mb']:>12.1f} "
        f"{result['d2h_mb_per_s']:>14.1f} "
        f"{result['d2h_flush_s']:>12.3f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train with shuffled batches while asynchronously streaming synthetic D2H copies"
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
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch factor when workers > 0")
    parser.add_argument("--seed", type=int, default=42, help="random seed for model init and dataloader shuffle")
    parser.add_argument(
        "--tensor_size",
        type=parse_size,
        default=parse_size("64mb"),
        help="D2H tensor size in bytes; suffixes b/kb/mb/gb are accepted",
    )
    parser.add_argument(
        "--d2h_xfer_freq",
        type=int,
        default=1,
        help="copy tensor GPU->CPU every N training steps; 0 disables D2H copies",
    )
    args = parser.parse_args()

    if args.warmups <= 0:
        raise ValueError("--warmups must be > 0")
    if args.warmups >= args.num_steps:
        raise ValueError("--warmups must be < --num_steps")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.d2h_xfer_freq < 0:
        raise ValueError("--d2h_xfer_freq must be >= 0")
    if args.d2h_xfer_freq > 0 and args.tensor_size <= 0:
        raise ValueError("--tensor_size must be > 0 when D2H copies are enabled")

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark is CUDA-only")
    device = torch.device("cuda")

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset = UncachedMMapDataset(args.tfile, args.seqlen)
    if len(dataset) < args.batchsize:
        raise ValueError(f"dataset has {len(dataset)} sequences, fewer than batchsize={args.batchsize}")
    loader = make_dataloader(dataset, args)

    d_ff = resolve_dff(args.dff, args.d_model)
    print(f"file={args.tfile}")
    print(
        f"dataset=uncached_no_copy shuffle=True pin_memory=True h2d_non_blocking=True "
        f"tensor_cores=high torch_compile={args.tcompile}"
    )
    print(
        f"d_model={args.d_model} d_ff={d_ff} heads={args.heads} layers={args.num_layers} "
        f"seqlen={args.seqlen} batchsize={args.batchsize}"
    )
    print(
        f"num_steps={args.num_steps} warmups={args.warmups} epochs={args.epochs} "
        f"num_workers={args.num_workers}"
    )
    print(
        f"d2h_tensor_size={args.tensor_size} bytes d2h_xfer_freq={args.d2h_xfer_freq} "
        f"d2h_output={D2H_OUTPUT_PATH}"
    )
    print("Note: measured runtime synchronizes the default compute stream only; d2h_flush_s is the tail drain.")
    print()
    print(
        f"{'steps':>8} {'total_s':>10} {'ms/step':>10} "
        f"{'samples/s':>14} {'tokens/s':>14} "
        f"{'d2h_copies':>10} {'d2h_mb':>12} {'d2h_mb/s':>14} {'d2h_flush_s':>12}"
    )
    print("-" * 121)

    result = benchmark(loader, args, device)
    print_result(result)


if __name__ == "__main__":
    main()
