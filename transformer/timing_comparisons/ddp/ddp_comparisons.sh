#!/bin/bash

for itn in {1..3}
do
    outdir="ddp_outputs/run${itn}"
    mkdir -p "$outdir"

    python3 ddp_comparisons.py --d_model=768 --heads=16 --batchsize=64 --num_layers=10 --type=naive --num_workers=2 --seqlen=600 --epochs=30 > "$outdir/naive.out"

    python3 ddp_comparisons.py --d_model=768 --heads=16 --batchsize=64 --num_layers=10 --type=flattened --num_workers=2 --seqlen=600 --epochs=30 > "$outdir/flattened.out"

    python3 ddp_comparisons.py --d_model=768 --heads=16 --batchsize=64 --num_layers=10 --type=overlap --num_workers=2 --seqlen=600 --epochs=30 > "$outdir/overlap.out"

    python3 ddp_comparisons.py --d_model=768 --heads=16 --batchsize=64 --num_layers=10 --type=bucketedoverlap --num_workers=2 --seqlen=600 --epochs=30 --bucketsize=1 > "$outdir/bucketedoverlap_1MB.out"

    python3 ddp_comparisons.py --d_model=768 --heads=16 --batchsize=64 --num_layers=10 --type=bucketedoverlap --num_workers=2 --seqlen=600 --epochs=30 --bucketsize=10 > "$outdir/bucketedoverlap_10MB.out"

    python3 ddp_comparisons.py --d_model=768 --heads=16 --batchsize=64 --num_layers=10 --type=bucketedoverlap --num_workers=2 --seqlen=600 --epochs=30 --bucketsize=100 > "$outdir/bucketedoverlap_100MB.out"

done