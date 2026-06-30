#!/bin/bash
# Fire the dist-sample reduce microbench. Run INSIDE ww25 container, 8 XPUs FREE.
# Decomposes the suspicious ~3.8ms cross-rank reduce: int64-MAX slowpath (a) vs
# all-reduce hop count (b) vs raw-pg-vs-optimized-communicator.
set -eu

source /opt/intel/oneapi/ccl/2021.15/env/vars.sh
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29555}

B=${B:-768}
ITERS=${ITERS:-2000}
OUT=${OUT:-/host/Workspace/gpt-oss/bench_reduce.log}

echo "[bench] B=$B iters=$ITERS -> $OUT"
torchrun --nproc_per_node=8 \
  /host/Workspace/vllm-public/bench_dist_sample_reduce.py \
  --b "$B" --iters "$ITERS" 2>&1 | tee "$OUT"
