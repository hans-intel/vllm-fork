#!/bin/bash
# Test harness for dist-sampling on ww25 container
# Usage: ./run_ww25_tests.sh [sample_count] [dataset_subset]
# Example: ./run_ww25_tests.sh 220 mixed_subset220.pkl

set -e

SAMPLE_COUNT="${1:-220}"
DATASET_SUBSET="${2:-mixed_subset220.pkl}"
DATASET_PATH="/data/dataset/gpt-oss-120b/v4/acc/${DATASET_SUBSET}"

if [ ! -f "$DATASET_PATH" ]; then
  echo "ERROR: Dataset not found at $DATASET_PATH"
  echo "Available subsets: mixed_subset220.pkl, lcb_subset180.pkl"
  exit 1
fi

echo "=========================================="
echo "Dist-Sampling WW25 Test Harness"
echo "=========================================="
echo "Dataset: $DATASET_PATH"
echo "Sample count: $SAMPLE_COUNT"
echo ""

# Source stable CCL before init_env
echo "[1] Setting up environment..."
source /opt/intel/oneapi/ccl/2021.15/env/vars.sh 2>/dev/null || true
cd /workspace/code
source init_env.sh gpt-oss-120b offline acc

# Configure user.conf for the sample count
sed -i "s/^gpt-oss-120b.\*.accuracy_sample_count_override.*/gpt-oss-120b.*.accuracy_sample_count_override = $SAMPLE_COUNT/" user.conf
sed -i "s/^gpt-oss-120b.\*.performance_sample_count_override.*/gpt-oss-120b.*.performance_sample_count_override = $SAMPLE_COUNT/" user.conf
sed -i "s/^gpt-oss-120b.Offline.min_query_count.*/gpt-oss-120b.Offline.min_query_count = $SAMPLE_COUNT/" user.conf

# Ensure permutation is enabled for proper shuffling
sed -i "s/.*sample_concatenate_permutation.*/gpt-oss-120b.*.sample_concatenate_permutation = 1/" user.conf

# Run baseline (DIST_SAMPLE=0)
echo ""
echo "[2] Running baseline (DIST_SAMPLE=0, stock sampler)..."
OUTDIR="/workspace/code/logs-ww25-test-baseline-$(date +%s)"
export DATASET_PATH
export TOTAL_SAMPLE_COUNT=$SAMPLE_COUNT
export OUTPUT_DIR=$OUTDIR
export VLLM_XPU_DIST_SAMPLE=0
export VLLM_XPU_DIST_SAMPLE_FUSED=1
export SAMPLING_MAX_TOKENS=10240

mkdir -p $OUTPUT_DIR
./run_local.sh | tail -50
BASELINE_ACC=$(python3 -c "
import json
try:
  with open('$OUTPUT_DIR/mlperf_log_accuracy.json') as f:
    data = json.load(f)
    acc = data['results']['gpt-oss-120b']['accuracy']
    print(f'{acc*100:.1f}')
except:
  print('ERROR')
" 2>/dev/null)

if [ "$BASELINE_ACC" = "ERROR" ]; then
  echo "WARNING: Could not extract baseline accuracy, skipping comparison"
  BASELINE_ACC=""
else
  echo "✓ Baseline accuracy: ${BASELINE_ACC}%"
fi

# Run fused dist-sample (DIST_SAMPLE=1, FUSED=1)
echo ""
echo "[3] Running dist-sample fused (DIST_SAMPLE=1, FUSED=1)..."
OUTDIR="/workspace/code/logs-ww25-test-fused-$(date +%s)"
export OUTPUT_DIR=$OUTDIR
export VLLM_XPU_DIST_SAMPLE=1
export VLLM_XPU_DIST_SAMPLE_FUSED=1

mkdir -p $OUTPUT_DIR
./run_local.sh | tail -50
FUSED_ACC=$(python3 -c "
import json
try:
  with open('$OUTPUT_DIR/mlperf_log_accuracy.json') as f:
    data = json.load(f)
    acc = data['results']['gpt-oss-120b']['accuracy']
    print(f'{acc*100:.1f}')
except:
  print('ERROR')
" 2>/dev/null)

if [ "$FUSED_ACC" = "ERROR" ]; then
  echo "ERROR: Could not extract fused accuracy"
  FUSED_ACC="ERROR"
else
  echo "✓ Dist-sample fused accuracy: ${FUSED_ACC}%"
fi

# Run proto dist-sample (DIST_SAMPLE=1, FUSED=0)
echo ""
echo "[4] Running dist-sample prototype (DIST_SAMPLE=1, FUSED=0)..."
OUTDIR="/workspace/code/logs-ww25-test-proto-$(date +%s)"
export OUTPUT_DIR=$OUTDIR
export VLLM_XPU_DIST_SAMPLE=1
export VLLM_XPU_DIST_SAMPLE_FUSED=0

mkdir -p $OUTPUT_DIR
./run_local.sh | tail -50
PROTO_ACC=$(python3 -c "
import json
try:
  with open('$OUTPUT_DIR/mlperf_log_accuracy.json') as f:
    data = json.load(f)
    acc = data['results']['gpt-oss-120b']['accuracy']
    print(f'{acc*100:.1f}')
except:
  print('ERROR')
" 2>/dev/null)

if [ "$PROTO_ACC" = "ERROR" ]; then
  echo "ERROR: Could not extract proto accuracy"
  PROTO_ACC="ERROR"
else
  echo "✓ Dist-sample proto accuracy: ${PROTO_ACC}%"
fi

# Summary
echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo "Baseline (DIST_SAMPLE=0):      $BASELINE_ACC%"
echo "Fused (DIST_SAMPLE=1,FUSED=1): $FUSED_ACC%"
echo "Proto (DIST_SAMPLE=1,FUSED=0): $PROTO_ACC%"
echo ""

# Check if results are valid
if [ "$BASELINE_ACC" != "" ] && [ "$FUSED_ACC" != "ERROR" ] && [ "$PROTO_ACC" != "ERROR" ]; then
  BASE_NUM=$(echo "$BASELINE_ACC" | cut -d. -f1)
  FUSED_NUM=$(echo "$FUSED_ACC" | cut -d. -f1)
  PROTO_NUM=$(echo "$PROTO_ACC" | cut -d. -f1)

  # Allow 1% margin for variance
  if [ $((FUSED_NUM - BASE_NUM)) -gt 1 ] || [ $((BASE_NUM - FUSED_NUM)) -gt 1 ]; then
    echo "⚠ WARNING: Fused accuracy differs from baseline by >1%"
  else
    echo "✓ Accuracies are within 1% of each other"
  fi
fi

echo ""
echo "Next steps:"
echo "1. Verify all three accuracies are within 0.5-1.0% of each other"
echo "2. Measure latency by comparing decode-step periods"
echo "3. Check logs for any errors or warnings"
echo ""
