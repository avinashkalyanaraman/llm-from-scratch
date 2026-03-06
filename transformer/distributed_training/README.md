# Distributed Training Experiments

This directory contains hands-on PyTorch distributed training experiments, starting from basic collectives and building up to custom DDP overlap and optimizer sharding.

## What Is Inside

- `msg_passing_basics/`: small single-node examples of `all_reduce`, `broadcast`, `reduce_scatter`, and timing sweeps.
- `naive_ddp/`: manual DDP-style training with per-parameter gradient `all_reduce`.
- `flattened_ddp/`: same training loop, but gradients are flattened into one buffer before `all_reduce`.
- `overlap_commnxn_compn/`: overlap communication with backprop via post-grad hooks.
- `bucketed_overlap_commnxn_compn/`: overlap with gradient buckets (similar idea to production DDP bucketing).
- `optimizer_sharding/`: combines bucketed gradient sync with a custom optimizer that shards parameter updates across ranks.

## Requirements

- Python 3.10+
- `torch`
- `torchvision`
- `matplotlib` (for `msg_passing_basics/single_node_time_sweep.py`)

Install (example):

```bash
pip install torch torchvision matplotlib
```

## Important Runtime Notes

- All scripts use `torch.multiprocessing.spawn(...)`; you run one Python script, and it launches all ranks.
- Process group backend is hard-coded to `gloo`.
- Rendezvous is hard-coded to:
  - `MASTER_ADDR=localhost`
  - `MASTER_PORT=25131`
- Run one experiment at a time to avoid port collisions.
- The MNIST scripts use relative dataset roots; run commands from the indicated subdirectory.

## Run Guide

### 1) Collective communication basics

From `transformer/distributed_training/msg_passing_basics`:

```bash
python single_node_toy.py
python single_node_tensor_bcast.py
python single_node_reduce_scatter.py
python single_node_time_sweep.py
```

`single_node_time_sweep.py` benchmarks multiple world sizes and tensor sizes, then plots timing bars (an example PNG is included in this folder).

### 2) Naive manual DDP

From `transformer/distributed_training/naive_ddp`:

```bash
python run.py
```

Behavior:

- `world_size=6`, `num_epochs=33`, `worker_batch_size=32`
- each rank trains on a `DistributedSampler` partition
- gradients are synchronized with per-parameter `dist.all_reduce`
- rank 0 verifies final params against a non-DDP baseline (`assert torch.allclose(...)`)

### 3) Flattened gradient DDP

From `transformer/distributed_training/flattened_ddp`:

```bash
python run.py
```

Behavior:

- same toy MNIST model/training setup
- collects all parameter gradients, flattens into one tensor, does one `all_reduce`, then unflattens
- reduces many small collectives into one larger collective per step

### 4) Overlap communication with gradient computation

From `transformer/distributed_training/overlap_commnxn_compn`:

```bash
python run.py
python run2.py
```

Behavior:

- `run.py`: direct hook-based implementation using `register_post_accumulate_grad_hook`
- `run2.py`: same idea wrapped in custom `DDP` class (`ddp_overlap_indiv_params.py`)
- each gradient is asynchronously reduced as soon as it is produced in backward

### 5) Bucketed overlap DDP

From `transformer/distributed_training/bucketed_overlap_commnxn_compn`:

```bash
python run.py
```

Behavior:

- custom `DDP` in `ddp_overlap_bucketed.py`
- groups gradients into buckets (`bucket_size_MB` in `run.py`, default `10`)
- flattens per bucket, all-reduces asynchronously, then unflattens and copies back

### 6) Optimizer sharding + bucketed overlap

From `transformer/distributed_training/optimizer_sharding`:

```bash
python sharded_optimizer.py
python run.py
```

Behavior:

- `sharded_optimizer.py`: standalone test of `MyShardedOptimizer`
- `run.py`: full training with:
  - bucketed gradient synchronization (custom DDP)
  - sharded optimizer ownership (`param_index % world_size == rank`)
  - post-step broadcasts so all ranks stay in sync

## Implementation Pattern Across Training Scripts

Most training experiments follow this sequence:

1. initialize process group (`gloo`)
2. build same model on all ranks
3. broadcast rank-0 parameters to synchronize start state
4. shard input batches with `DistributedSampler`
5. backward pass + custom gradient synchronization strategy
6. optimizer step
7. rank-0 correctness check against single-process reference model

## Caveats

- These are educational prototypes, not production wrappers.
- Some scripts use private torch helpers (`torch._utils._flatten_dense_tensors`), which can change across PyTorch versions.
- `MyShardedOptimizer.add_param_group(...)` is not implemented for runtime extension.
