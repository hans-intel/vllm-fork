# gpt-oss-120b TP=8 XPU — TTFT optimization investigation

**Date:** 2026-06-27
**Hardware:** 8× Intel XPU (24 GB each), **PCIe-only** interconnect (no Xe-Link)
**Model:** gpt-oss-120b-mxfp4, TP=8, `enforce_eager=True`, fp8 KV cache
**Workload:** MLPerf Offline, steady-state step = 3072 tokens (~2.5k prefill + 0.5k decode)
**Branch:** `hans/ww25` (built in worktree `/home/saehanse/Workspace/vllm-ww25`)

## TL;DR

The TTFT bottleneck is the **TP all-reduce: 42% of every step (122 ms / ~290 ms)**, running
over PCIe. The three obvious ways to attack it were each ruled out by direct
measurement; the one surviving lever is a **quantized (fp8-transport) all-reduce**,
worth a measured ~25–50% of the all-reduce cost (~30–50 ms/step), gated on a
numerics/accuracy check.

## How we got here

Profiled with the in-tree `PROFILE`/`PROFILE_TABLE`/`VLLM_STEP_LOG` harness
(`gpu_model_runner.py`). Steady prefill step, Self-XPU time:

| Kernel | Self XPU | % step |
|---|---|---|
| `c10d::allreduce_` → `oneccl_allreduce_pcie` | 122 ms | **42%** |
| `cutlass_grouped_gemm` (sycl-tla MoE) | 82 ms | 28% |
| `varlen_fwd` (triton XPU attn) | 27–47 ms | 9–15% |
| dense `addmm` | 13 ms | 4% |
| pointwise tail (rmsnorm/rope/swiglu/gather) | ~45 ms | ~15% |

73 all-reduces/step = 2 per layer × 36 layers (attn `o_proj` + MoE `down_proj`),
~1.68 ms each.

Why prior experiments did nothing: dist-sampling is <1% of the step; DP-attention
kept the attn all-reduce and *added* MoE comm — neither reduced per-layer
all-reduce volume.

## What was ruled out (by measurement, not argument)

1. **Lower TP (TP=4/DP).** Works in Server, but **not Offline**: 61 GB weights /
   24 GB cards leaves no KV headroom at TP=4 to hold the 3k-token steady state.

2. **Faster interconnect.** Cards are PCIe-only; no Xe-Link fabric exists.

3. **Comm/compute overlap (DBO and async-TP).** The decisive result.
   `overlap_microbench.py` (8-rank, real 3072×2880 bf16 payload): issuing the
   all-reduce on a side XPU stream alongside a GEMM gives **0% overlap**
   (serial 2.36 ms, overlapped 2.36/2.37 ms), stable across `async_op=True` and
   `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`. **oneCCL monopolizes the XPU
   engine while the collective runs.** This kills *both* comm-hiding strategies,
   which share the same false premise:
   - **Async-TP** — additionally blocked by CUDA-only `torch._symmetric_memory`
     + FlashInfer kernels; months of kernel work even if overlap existed.
   - **DBO** — I cleared 4 blockers and got it to *engage* in pure TP on XPU
     (see "DBO XPU enablement" below), but its overlap is cooperative and lives
     only in the EP all2all MoE path; it has **no hooks on the TP all-reduce**,
     and even if instrumented, the overlap is physically impossible here.

4. **oneCCL algorithm tuning.** `allreduce_cost_microbench.py` swept
   `CCL_ALLREDUCE`. The **default (a SYCL/GPU path) is best at 1.68 ms**; every
   forced algorithm is 8–17× worse (ring 17.1, rabenseifner 14.4,
   recursive_doubling 26.9, double_tree 28.0, nreduce 14.1, ring_rma 16.8 ms).
   **Never set `CCL_ALLREDUCE`.**

## The surviving lever: quantized all-reduce

`allreduce_cost_microbench.py` shows the all-reduce is **purely bandwidth-bound**
(fp32 = 3.33 ms = 2× bf16's 1.68 ms; fp16 = 1.67 ms). So fewer transported bytes
= proportionally less time. Measured:

| Variant | Time | vs bf16 all-reduce |
|---|---|---|
| all-reduce bf16 (baseline) | 1.68 ms | — |
| reduce_scatter(bf16) + all_gather(**fp8**) | 1.26 ms | **−25%** |

Narrowing only the all-gather half already saved 25%; quantizing both halves
should approach the ~2× ceiling. At 42% of the step, a ~40% byte reduction is
roughly **~50 ms off a 290 ms step** — a real single-digit-% TTFT win.

**Correctness risk (the gating issue, not perf):** SUM-in-fp8 across 8 ranks is
inaccurate. The viable shape is **reduce in bf16 (accurate sum), transport the
result shard in fp8** so the only precision loss is a *single* fp8 quantization
of the final summed output (not an accumulation), with per-shard scaling to
avoid e4m3 saturation. Must A/B against the MLPerf accuracy target — a quantized
all-reduce that fails accuracy is worthless.

## fp8 all-reduce prototype — correct, but needs a fused kernel to win

Implemented behind `VLLM_XPU_FP8_ALLREDUCE=1` in `XpuCommunicator.all_reduce`:
bf16 `reduce_scatter` (accurate sum) → quantize the summed shard to fp8 with a
per-shard fp32 scale → fp8 `all_gather` → bf16 dequant. Validated 8-rank
(`fp8_allreduce_test.py`):

- **Correctness: rel L2 = 2.7%** vs the bf16 all-reduce. Only one fp8
  quantization (of the already-summed shard, not an accumulation). Per-row
  scaling does not improve it — the error is fp8 mantissa-bound, not
  dynamic-range-bound. Whether 2.7%/call × 72 calls/step holds MLPerf accuracy
  is **untested — that's the real gate.**
- **Perf breakdown (3072×2880, TP=8):**

  | Path | Time |
  |---|---|
  | bf16 all-reduce (baseline) | 1.79 ms |
  | fp8 **collectives only** (rs+ag, no quant math) | **1.27 ms (−29%)** |
  | fp8 **full path** (with quant/dequant) | 2.15 ms (**+20% slower**) |

  The wire win is real (−0.5 ms), but **5 un-fused eager elementwise kernels**
  (upcast / divide / to-fp8 / dequant / rescale over a 17 MB tensor) add ~0.87 ms
  and overrun the savings. `enforce_eager=True` → no fusion.

- **Eager verdict:** a net loss; the un-fused quant/dequant erases the wire win.

### Making it win: fused dequant + static scale

Two changes turn the −20% eager loss into a real win:

| Path | Time | vs bf16 |
|---|---|---|
| bf16 all-reduce | 1.78 ms | 1.00× |
| eager full | 2.15 ms | 0.83× |
| torch.compile(quant+dequant) | 1.72 ms | 1.04× |
| **Triton dequant + static scale (single gather)** | **1.38 ms** | **1.29×** |
| collectives-only floor | 1.26 ms | 1.42× |

1. **Fused Triton dequant on the full gathered tensor: 0.089 ms vs 0.598 ms
   eager (6.7×).** This is the expensive side and where fusion pays. (A
   hand-rolled Triton *quant* with `grid=(1,)` was 0.995 ms — terrible; the
   small summed shard is quantized in eager at 0.064 ms instead. Only the big
   dequant needs a kernel.)
2. **Static scale, single all-gather.** A dynamic per-rank scale requires a
   second all-gather; its collective-launch penalty is **~0.34 ms for 8 floats**
   and erases the win (1.03×). A fixed `VLLM_XPU_FP8_ALLREDUCE_SCALE` keeps it to
   one collective → 1.29×. (Packing the scale into the fp8 buffer via uint8
   views cost 4.6 ms — don't; the view/copy overhead dominates.)

**Net:** ~0.4 ms/call × 72 calls/step ≈ **~28 ms off a ~290 ms step**. Implemented
behind `VLLM_XPU_FP8_ALLREDUCE=1` (+ `VLLM_XPU_FP8_ALLREDUCE_SCALE`, default
0.05) in `XpuCommunicator._fp8_all_reduce`.

### End-to-end A/B (validated on the real model)

gpt-oss-120b TP=8, 32 requests, steady 3072-token step:

| Config | median | mean | min |
|---|---|---|---|
| bf16 all-reduce | 294.4 ms | 299.1 ms | 284.3 ms |
| **fp8 all-reduce** | **267.6 ms** | 272.3 ms | 258.6 ms |
| **delta** | **−26.8 ms (−9.1%)** | −26.8 ms | −25.7 ms |

Matches the microbench prediction (72 calls × ~0.4 ms ≈ 28 ms). All 32 requests
completed, no NaN/inf, clean shutdown.

**Remaining gate: MLPerf accuracy.** The perf win is confirmed; the static scale
(0.05) is the open risk — it must cover the activation range (amax/448) per
model. If a fixed value fails accuracy, replace with a slow running-max updated
off the critical path. Run the accuracy dataset with `VLLM_XPU_FP8_ALLREDUCE=1`
before shipping.

- Gotchas: the fp8 scale must be **fp32** (amax on a bf16 shard yields a bf16
  scale → XCCL `all_gather` raises `TypeError: output tensor must have the same
  type as input`). And `dist.all_gather(list, …)` is buggy on XCCL/XPU (per the
  dp-attn patches) — use `all_gather_into_tensor`.

### oneCCL version gotcha: 8-bit collectives + the int32-packing fix

The MLPerf harness sources **oneCCL 2021.15** (`run_local.sh` →
`/opt/intel/oneapi/ccl/2021.15/env/vars.sh`); the default container env is
2021.17. The first MLPerf run with `VLLM_XPU_FP8_ALLREDUCE=1` crashed at engine
init (`profile_run`, the vocab-embedding all-reduce):

```
RuntimeError: oneCCL: sycl_coll_base.hpp:473 invoke_collective_type:
              EXCEPTION: unsupported datatype UINT8
```

Reproduced under each version: **2021.15 rejects every 8-bit collective dtype**
(fp8/uint8/int8 all fail; int16 → "Short"); only int32/bf16/fp16/fp32 work.
2021.17 happens to accept fp8. So an int8 bitcast does **not** help under 2021.15.

**Fix:** pack 4 fp8 bytes into one int32 for the all-gather
(`shard_fp8.view(-1).view(torch.int32)`), gather as int32 (accepted), then
`gathered_i32.view(fp8)` back. Bit-exact, same 1 byte/element on the wire,
oneCCL-version-agnostic. Requires per-rank shard `numel % 4 == 0` (guarded).

Verified under CCL 2021.15: UINT8 error gone, 8/8 requests complete, steady
step **~259–263 ms vs ~294 ms bf16 — the ~30 ms / −9% win holds in the MLPerf
CCL env.** (Full `main.py` needs its own module-path setup via `run_local.sh`;
a bare `main.py` launch hits an unrelated `utils_model` import error.)

### Accuracy collapse → fix: the static scale was saturating fp8

First MLPerf accuracy run with the fp8 all-reduce: perf rose (1800→2000) but
**accuracy fell from 80–90% to <10%.** Root cause was **not** quantization
noise — it was **fp8 saturation**:

- The default static scale `0.05` covers e4m3 only up to `0.05 × 448 = 22.4`.
- Instrumenting the real run (`VLLM_XPU_FP8_ALLREDUCE_AMAX_LOG=1`) showed
  summed-shard amax reaching **~48,000–57,000**. So nearly every all-reduce
  clipped to the fp8 max, corrupting the residual stream from layer 0.
- The output tokens were **coherent-looking but systematically wrong** (proper
  sentences, wrong answers), so answer parsing failed — livecodebench 0%,
  multiple-choice degraded. (A separate scare — "garbage tokens" — was a decode
  artifact: `mlperf_log_accuracy.json` stores token ids as **int64**; reading
  them as int32 interleaves a spurious `0` after every real token.)

**Fix:** raise the default scale to **128** (covers amax up to 57,344). Because
e4m3 is floating-point, its ~2.6% relative error is roughly constant across
magnitudes, so a generously large scale costs **no** precision on small values
while eliminating saturation — and keeps the single fast all-gather. Verified
end-to-end at scale=128: output fully coherent (structured summaries, correct
analysis traces), no saturation (max amax 48,128 < 57,344), perf win intact.
A per-tensor *dynamic* scale would be marginally more accurate but needs a
second all-gather (~0.34 ms, kills the win) or an uneven packed payload (0.85×);
static-but-large is the right trade. If a model's amax exceeds 57,344, raise
`VLLM_XPU_FP8_ALLREDUCE_SCALE`.

## DBO XPU enablement (byproduct — correct, but won't help here)

Four fixes make DBO *engage* in pure TP on XPU (left in the worktree; useful as
upstream XPU-enablement, but cannot improve TTFT here because overlap is
impossible — see ruling #3):
1. `SMControlContextManager` CUDA/ROCm assert → self-disable on other platforms.
2. `should_ubatch` was DP-gated (`gpu_model_runner.py`) → pure-TP threshold branch.
3. `assert dp_metadata is not None` (`gpu_ubatch_wrapper.py`) → `None` per-ubatch when DP=1.
4. `VllmConfig` validator forcing deepep/nixl backend (`config/vllm.py`) → gated on EP.

## Repro

```bash
# profile (in container hans-gpt-oss-ww25-dev)
PROFILE=1 PROFILE_INTERVAL=80 PROFILE_TABLE=1 VLLM_STEP_LOG=1 \
  python -u evaluate_dataset.py -w /model/gpt-oss-120b-mxfp4 -d bfloat16 -t 8 \
  --max_num_batched_tokens 3072 -b 128 --gpu_memory_utilization 0.80 \
  --dataset .../perf_eval_ref.pkl -n 256 --max_output_len 10240 \
  --max_model_len 102400 --kv_cache_dtype fp8

# microbenches (mlperf_llm_large/)
torchrun --nproc-per-node=8 overlap_microbench.py
torchrun --nproc-per-node=8 allreduce_cost_microbench.py
```
