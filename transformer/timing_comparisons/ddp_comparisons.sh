#!/bin/sh

python3 ddp_comparisons.py --d_model=768 --heads=16 --batchsize=64 --num_layers=11 --type=naive --num_workers=2  --seqlen=800 --epochs=30 > ddp_outputs/naive.out
python3 ddp_comparisons.py --d_model=768 --heads=16 --batchsize=64 --num_layers=11 --type=flattened --num_workers=2  --seqlen=800 --epochs=30 > ddp_outputs/flattened.out
python3 ddp_comparisons.py --d_model=768 --heads=16 --batchsize=64 --num_layers=11 --type=overlap --num_workers=2  --seqlen=800 --epochs=30 > ddp_outputs/overlap.out
python3 ddp_comparisons.py --d_model=768 --heads=16 --batchsize=64 --num_layers=11 --type=bucketedoverlap --num_workers=2  --seqlen=800 --epochs=30 > ddp_outputs/bucketedoverlap.out