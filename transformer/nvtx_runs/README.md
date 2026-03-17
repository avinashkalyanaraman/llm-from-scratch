# NVTX Runs

This directory contains CUDA profiling scripts for the transformer codebase. The tracked files here focus on annotating training steps with NVTX ranges, comparing stream-overlap behavior, and profiling custom DDP implementations.

## Tracked Files

- [`run.py`](run.py): single-GPU transformer training loop instrumented with `torch.cuda.nvtx` ranges. Supports a baseline path and a copy-stream overlap path.
- [`stream_comparison.py`](stream_comparison.py): single-GPU throughput comparison for default-stream execution versus explicit host-to-device overlap on a separate CUDA stream.
- [`ddp_nvtx_runs_transformer.py`](ddp_nvtx_runs_transformer.py): multi-process CUDA benchmark for profiling the repository's custom DDP variants with NVTX markers around forward, backward, and gradient synchronization.
- [`nvtx_profiling_notes.pptx`](nvtx_profiling_notes.pptx): presentation notes summarizing profiling observations for this directory's experiments.

## Environment

- Run commands from [`transformer/nvtx_runs/`](./).
- CUDA is required for all profiling scripts in this directory.
- The scripts rely on local imports from the parent transformer package, so the working directory matters.
- For Nsight Systems traces, run the commands through `nsys profile -o <trace_name> python ...`.

```bash
cd transformer/nvtx_runs
```

## Data Expectations

[`run.py`](run.py) and [`stream_comparison.py`](stream_comparison.py) expect a tokenized `.npy` file passed through `--tfile`. The file should contain a flat token ID array compatible with [`transformer/filehandler.py`](../filehandler.py).

One straightforward way to create that `.npy` input is with [`tokenizer/off_the_shelf_tokenizer.py`](../../tokenizer/off_the_shelf_tokenizer.py), which tokenizes a UTF-8 text corpus with GPT-2 tokenization and writes the token IDs to disk.

Example:

```bash
cd ../../tokenizer

python off_the_shelf_tokenizer.py \
  --input path/to/corpus.txt \
  --out-npy ../transformer/temp/temp.npy \
  --special "<|endoftext|>" "<|assistant|>"
```

[`ddp_nvtx_runs_transformer.py`](ddp_nvtx_runs_transformer.py) does not read a dataset file. It generates random token inputs and targets on each worker for profiling communication and synchronization behavior.

## Single-GPU NVTX Trace

[`run.py`](run.py) is the main profiling entry point for annotating a training step with NVTX ranges. It labels host-to-device transfer, forward, loss, learning-rate update, backward, clipping, optimizer step, and a post-warmup measurement window.

Example:

```bash
nsys profile -o transformer_train_trace python run.py --tfile ../temp/temp.npy --steps 20
nsys profile -o transformer_train_stream_trace python run.py --tfile ../temp/temp.npy --steps 20 --stream
```

Useful arguments:

- `--tfile`: tokenized training file
- `--steps`: number of measured steps after startup
- `--stream`: enables the copy-stream overlap path
- `--torchcompile`: wraps the model in `torch.compile`
- `--d_model --seqlen --heads --num_layers --batchsize --vocabsize`: model and batch configuration

Notes:

- This script asserts that the selected device is CUDA.
- The `--stream` path preloads the next batch on a separate CUDA stream and uses NVTX markers for both the initial and overlapped transfers.
- In the non-stream path, the script prints mean post-warmup step time and throughput.

## Stream Overlap Comparison

[`stream_comparison.py`](stream_comparison.py) is a simpler benchmark intended to isolate the effect of overlapping data transfer with compute. It injects an artificial delay with `torch.cuda._sleep(...)` to make transfer overlap easier to observe.

Example:

```bash
nsys profile -o stream_comparison_trace python stream_comparison.py --tfile ../temp/temp.npy --steps 20
nsys profile -o stream_comparison_stream_trace python stream_comparison.py --tfile ../temp/temp.npy --steps 20 --stream
```

Useful arguments are the same as [`run.py`](run.py), including `--stream`, `--steps`, and the transformer shape parameters.

Notes:

- The non-stream path performs transfer and compute serially.
- The stream path prefetches the next batch onto a copy stream while compute proceeds on the default stream.
- The script prints aggregate throughput and elapsed time after warmup.

## DDP NVTX Benchmark

[`ddp_nvtx_runs_transformer.py`](ddp_nvtx_runs_transformer.py) profiles distributed execution for the custom DDP implementations under [`transformer/distributed_training/`](../distributed_training/). It annotates forward, backward, and explicit synchronization phases with NVTX ranges.

Example:

```bash
nsys profile -o ddp_nvtx_trace python ddp_nvtx_runs_transformer.py \
  --num_workers 2 \
  --batchsize 8 \
  --epochs 20 \
  --type bucketedoverlap \
  --bucketsize 10
```

Useful arguments:

- `--num_workers`: world size
- `--batchsize`: global batch size, which must be divisible by `num_workers`
- `--epochs`: number of profiling iterations
- `--type`: one of `naive`, `flattened`, `overlap`, or `bucketedoverlap`
- `--bucketsize`: bucket size in MB for the bucketed-overlap variant
- `--tcompile`: enables `torch.compile`

Notes:

- The script uses `torch.multiprocessing.spawn` and initializes NCCL with `MASTER_ADDR=localhost` and a fixed `MASTER_PORT`.
- Each worker binds to `cuda:{rank}`.
- The benchmark uses synthetic random inputs rather than reading from disk, which keeps the focus on model execution and communication behavior.

## Scope

This README intentionally documents only the files currently tracked in Git for this directory. Locally generated profiler outputs such as `.nsys-rep` or `.sqlite` traces are not cataloged here.
