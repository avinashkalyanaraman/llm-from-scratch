# Timing Comparisons

This directory contains micro-benchmarks and timing harnesses for selected transformer components, along with tracked output snapshots for the custom DDP timing study.

## Tracked Files

- [`linear_comparison.py`](linear_comparison.py): compares three equivalent implementations of a linear projection.
- [`mha_comparison.py`](mha_comparison.py): benchmarks forward and backward runtimes for the repository's transformer implementation against a baseline model.
- `ddp_comparisons.py`: CUDA DDP timing harness for the repository's custom distributed training variants.
- `ddp_comparisons.sh`: shell script that runs the DDP timing harness across several communication strategies and bucket sizes.
- `ddp_outputs/run1/` through `ddp_outputs/run3/`: tracked text outputs from repeated DDP comparison runs.

## Environment

- Run commands from [`transformer/timing_comparisons/`](./).
- The scripts rely on local imports from the parent transformer package, so the working directory matters.
- `linear_comparison.py` can run on CUDA, MPS, or CPU depending on availability.
- `mha_comparison.py` can run on CUDA, MPS, or CPU, but its baseline mode also expects a reference code that isn't git-tracked here. Please modify it accordingly.
- `ddp_comparisons.py` is CUDA-only and requires multiple visible GPUs for meaningful multi-worker runs.

```bash
cd transformer/timing_comparisons
```

## Linear Projection Micro-Benchmark

[`linear_comparison.py`](linear_comparison.py) compares three implementations of the same linear transform:

- reshape plus `@`
- direct `@`
- `torch.matmul`

The script generates synthetic inputs, runs each method repeatedly, checks that the outputs match exactly, and prints mean runtime after warmup.

Example:

```bash
python linear_comparison.py
```

Output includes mean timings in milliseconds for each implementation.

## Transformer Timing Benchmark

[`mha_comparison.py`](mha_comparison.py) benchmarks a full transformer forward and backward pass on synthetic token inputs. It supports two modes:

- `comparison`: uses this repository's transformer implementation
- `baseline`: uses a reference implementation

Example:

```bash
python mha_comparison.py --type comparison --epochs 5 --warmups 1
python mha_comparison.py --type baseline --epochs 5 --warmups 1
```

Useful arguments:

- `--type`: `comparison` or `baseline`
- `--epochs`: number of iterations
- `--warmups`: number of untimed warmup iterations
- `--d_model --seqlen --heads --num_layers --batchsize --vocabsize`: model shape
- `--dff`: feed-forward hidden size; defaults to an internal multiple-of-64 calculation when set to `-1`
- `--tcompile`: enables `torch.compile`

The script reports mean and standard deviation for forward and backward pass times after warmup.

## DDP Timing Harness

The tracked DDP comparison scripts are stored at the top level of this directory in Git:

- `ddp_comparisons.py`
- `ddp_comparisons.sh`

`ddp_comparisons.py` benchmarks the custom DDP implementations under [`transformer/distributed_training/`](../distributed_training/) using synthetic random inputs. It measures per-rank forward, backward, synchronization, and end-to-end iteration time after warmup.

Example:

```bash
python ddp_comparisons.py \
  --num_workers 2 \
  --batchsize 64 \
  --epochs 30 \
  --warmups 5 \
  --type bucketedoverlap \
  --bucketsize 10
```

Useful arguments:

- `--num_workers`: world size
- `--batchsize`: global batch size, which must be divisible by `num_workers`
- `--epochs`: total iterations
- `--warmups`: number of untimed warmup iterations
- `--type`: one of `naive`, `flattened`, `overlap`, or `bucketedoverlap`
- `--bucketsize`: bucket size in MB for `bucketedoverlap`

`ddp_comparisons.sh` automates three repeated runs of the DDP harness and writes outputs under `ddp_outputs/run1`, `run2`, and `run3`. It covers:

- `naive`
- `flattened`
- `overlap`
- `bucketedoverlap` with `1 MB`, `10 MB`, and `100 MB` bucket sizes

## Tracked DDP Outputs

The repository includes tracked output snapshots under:

- `ddp_outputs/run1/`
- `ddp_outputs/run2/`
- `ddp_outputs/run3/`

These files record per-rank summaries such as:

- mean and standard deviation of forward pass time
- mean and standard deviation of backward pass time
- mean and standard deviation of synchronization time
- mean and standard deviation of total iteration time

They are useful as reference results for the DDP timing study and for comparing reruns against a previously recorded baseline.

## Scope

This README intentionally documents only the files currently tracked in Git for this directory. If you have local reorganizations or additional generated artifacts, they are outside the scope of this document.
