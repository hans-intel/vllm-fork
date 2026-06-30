# Dist-Sampling Port to ww25 - Testing Plan

This document describes how to test the dist-sampling feature ported from ww16 to ww25 (vllm-public).

## What was ported

The vocabulary-parallel Gumbel-max distributed sampling optimization has been ported from ww16 (vllm-xpu) to ww25 (vllm-public):

### Files changed:
1. **vllm/envs.py** - Added three env vars:
   - `VLLM_XPU_DIST_SAMPLE` (default: off) - Enable the feature
   - `VLLM_XPU_DIST_SAMPLE_FUSED` (default: on) - Use fused Triton kernel
   - `VLLM_XPU_DIST_SAMPLE_ALLGATHER` (default: off) - Use broken all-gather (debug only)

2. **vllm/v1/worker/gpu/sample/gumbel.py** - Added fused kernel:
   - `_dist_gumbel_packed_kernel` - Triton kernel for in-kernel packing
   - `dist_gumbel_local_packed` - Python wrapper for the fused kernel

3. **vllm/model_executor/layers/logits_processor.py** - Added three methods:
   - `gumbel_argmax_tokens()` - Main entry point (dispatcher)
   - `_gumbel_argmax_tokens_fused()` - Fast path (default)
   - `_gumbel_argmax_tokens_proto()` - Prototype path (fallback)

4. **vllm/v1/worker/gpu_model_runner.py** - Added:
   - `_dist_sample_eligible()` - Check if batch is eligible for dist-sampling

## Testing Steps

### Step 1: Prepare the environment

Inside the ww25 container:

```bash
cd /workspace/code  # This is where your repo is mounted

# Source the stable CCL to avoid 2021.17 profiling hangs
source /opt/intel/oneapi/ccl/2021.15/env/vars.sh

# Initialize environment
source init_env.sh gpt-oss-120b offline acc
```

### Step 2: Run baseline accuracy (stock sampler, VLLM_XPU_DIST_SAMPLE=0)

Run a small accuracy test to establish the baseline:

```bash
# Set sample count to 220 (or smaller for quick test)
export DATASET_PATH=/data/dataset/gpt-oss-120b/v4/acc/mixed_subset220.pkl
export TOTAL_SAMPLE_COUNT=220
export OUTPUT_DIR=/workspace/code/logs-ww25-baseline
export VLLM_XPU_DIST_SAMPLE=0
export SAMPLING_MAX_TOKENS=10240

mkdir -p $OUTPUT_DIR
./run_local.sh

# Extract accuracy from mlperf_log_accuracy.json
python3 -c "
import json
with open('$OUTPUT_DIR/mlperf_log_accuracy.json') as f:
    data = json.load(f)
    acc = data['results']['gpt-oss-120b']['accuracy']
    print(f'Baseline (DIST_SAMPLE=0): {acc:.1%}')
"
```

### Step 3: Run with dist-sampling enabled (VLLM_XPU_DIST_SAMPLE=1, fused path)

```bash
export OUTPUT_DIR=/workspace/code/logs-ww25-dist-sample-fused
export VLLM_XPU_DIST_SAMPLE=1
export VLLM_XPU_DIST_SAMPLE_FUSED=1

mkdir -p $OUTPUT_DIR
./run_local.sh

# Extract accuracy
python3 -c "
import json
with open('$OUTPUT_DIR/mlperf_log_accuracy.json') as f:
    data = json.load(f)
    acc = data['results']['gpt-oss-120b']['accuracy']
    print(f'Dist-sample fused (DIST_SAMPLE=1, FUSED=1): {acc:.1%}')
"
```

### Step 4: Run with dist-sampling prototype path (VLLM_XPU_DIST_SAMPLE=1, FUSED=0)

```bash
export OUTPUT_DIR=/workspace/code/logs-ww25-dist-sample-proto
export VLLM_XPU_DIST_SAMPLE=1
export VLLM_XPU_DIST_SAMPLE_FUSED=0

mkdir -p $OUTPUT_DIR
./run_local.sh

# Extract accuracy
python3 -c "
import json
with open('$OUTPUT_DIR/mlperf_log_accuracy.json') as f:
    data = json.load(f)
    acc = data['results']['gpt-oss-120b']['accuracy']
    print(f'Dist-sample proto (DIST_SAMPLE=1, FUSED=0): {acc:.1%}')
"
```

## Expected Results

For the ww25 container with GPT-OSS 120B on a mixed accuracy subset (220 samples):

1. **Baseline (VLLM_XPU_DIST_SAMPLE=0)**: ~61% accuracy
   - Stock sampler, all-gather full logits

2. **Dist-sample fused (FUSED=1, default)**: ~61% accuracy (±0.5%)
   - Fused Triton kernel + all_reduce(MAX)
   - Should match baseline (pure perf optimization, no accuracy impact)
   - Latency: decode period should be 3-5% faster than baseline

3. **Dist-sample proto (FUSED=0)**: ~61% accuracy (±0.5%)
   - torch.rand_like + all-gather (float32 pairs) + host argmax
   - Also validated == baseline (reference/fallback path)
   - Latency: similar to baseline (no perf benefit)

### Success Criteria

✅ All three paths (baseline, fused, proto) achieve accuracy within 0.5% of each other
✅ Fused path shows latency improvement vs baseline (at least 1-2% on decode period)
✅ No crashes or hangs during runs
✅ All `[unixts]` timestamps in logs are present (no truncation)

## Troubleshooting

### If accuracy is significantly different (>1% off)

1. Check CCL version: `source /opt/intel/oneapi/ccl/2021.15/env/vars.sh` BEFORE init_env
2. Verify DATASET_PATH is correct and readable
3. Check batch sizes haven't changed (should be constant with same dataset/sample_count)
4. Check if `sample_concatenate_permutation = 1` in user.conf (ensures proper shuffling)

### If tests hang or crash

1. Check memory usage (dist-sample shouldn't increase memory)
2. Check GPU utilization (should be >80%)
3. Look for XPU-specific errors in logs (search for ERROR, WARN)
4. Try with smaller sample count first (e.g., 50 samples for quick validation)

### If latency doesn't improve

1. Check that decode period is actually changing (not an outlier run)
2. Verify sampling_max_tokens isn't too small (should be ~10000 to see steady state)
3. Check if batch is actually reaching expected batch size (look at batch_size in logs)

## Performance Measurement

To extract latency:

```bash
# Extract decode-step period from logs (median of [unixts] timestamp deltas)
python3 << 'EOF'
import json
import re
from statistics import median

log_file = "logs-ww25-dist-sample-fused/mlperf_log_detail.json"
with open(log_file) as f:
    data = json.load(f)

# Find all decode step timestamps
decode_timestamps = []
for entry in data:
    if 'detail' in entry and 'unixts' in entry['detail']:
        ts = entry['detail']['unixts']
        decode_timestamps.append(ts)

# Compute median period (last 100 steps to avoid startup)
if len(decode_timestamps) > 100:
    deltas = [decode_timestamps[i+1] - decode_timestamps[i] for i in range(-100, -1)]
    median_period_ms = median(deltas) * 1000
    print(f"Median decode period (last 100 steps): {median_period_ms:.1f} ms")
EOF
```

Expected improvement: 3-5% on decode period (47-48ms for ww16, measure actual for ww25).
