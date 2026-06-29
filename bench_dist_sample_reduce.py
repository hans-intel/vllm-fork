#!/usr/bin/env python3
"""Microbench: isolate WHY dist-sample's cross-rank reduce costs ~3.8ms on XPU.

The dist-sample fast path replaces a 204MB logits all-gather with a tiny cross-rank
reduce over [B] packed int64 winners. That reduce measures ~3.8ms (ww16) / ~4ms
(ww25 live) even though the payload is ~4KB -> sub-100us of actual wire time. So the
3.8ms is FIXED per-collective cost, not bandwidth. This bench decomposes it.

Candidate causes (need different workarounds):
  (a) int64-MAX on a dumb XCCL/oneCCL reduction path (no tuned int64-MAX kernel;
      shipped code uses RAW pg dist.all_reduce(op=MAX) because vLLM's
      tensor_model_parallel_all_reduce is SUM-only).
  (b) all-reduce is inherently 2x the hops of all-gather (ring AR = 2(N-1), AG = N-1).

Variants timed (payload [B] or [B,2], B configurable), each sync'd, warmed, median
over N iters, MAX over ranks reported (the bottleneck rank sets step latency):

  ar_max_i64   : dist.all_reduce(packed[B] int64, op=MAX, raw pg)        <- SHIPPED
  ar_sum_i64   : dist.all_reduce([B] int64, op=SUM, raw pg)              isolate MAX vs SUM
  ar_sum_f32   : dist.all_reduce([B] f32,  op=SUM, raw pg)               isolate int64 dtype
  ar_sum_f32_vllm : tensor_model_parallel_all_reduce([B] f32)           isolate raw-pg vs optimized
  ag_f32_pair  : tensor_model_parallel_all_gather([B,2] f32) + argmax    WORKAROUND 2 (prototype reduce)
  ag_i64_b1    : tensor_model_parallel_all_gather([B,1] i64) + max       (ww16-broken; timing only)
  sum0fill_f32 : WORKAROUND 1 -- [B,TP,2] f32 0-fill, in-place SUM AR, local argmax

Reading the result:
  ar_sum_f32 << ar_max_i64  => cause (a): int64-MAX path is slow. sum0fill_f32 wins.
  ar_sum_f32 ~= ar_max_i64  => cause (b): hop count. Only ag_* (all-gather) wins.
  ar_sum_f32_vllm << ar_sum_f32 (raw) => optimized communicator matters; route through it.

Run INSIDE the ww25 container, 8 XPUs free:
  source /opt/intel/oneapi/ccl/2021.15/env/vars.sh
  torchrun --nproc_per_node=8 /host/Workspace/vllm-public/bench_dist_sample_reduce.py
  # optional: --b 768 --iters 2000
"""
import argparse
import os
import time

import torch
import torch.distributed as dist


def _sync():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


def _time_op(fn, iters, warmup):
    """Median per-call ms over `iters`, after `warmup`. Barrier+sync around the
    whole loop so the timer captures real collective latency, not enqueue."""
    for _ in range(warmup):
        fn()
    _sync()
    dist.barrier()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    dt = (time.perf_counter() - t0) / iters * 1000.0
    return dt


def _per_call_samples(fn, iters, warmup):
    """Per-call timings with a sync EACH call -> upper-bound latency incl. launch.
    Captures the real per-step cost (the engine pays one synchronized reduce/step)."""
    for _ in range(warmup):
        fn()
    _sync()
    dist.barrier()
    samples = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=768, help="batch (rows of packed winners)")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(local_rank)
        device = torch.device(f"xpu:{local_rank}")
        backend = "xccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend, rank=rank, world_size=world)

    B, TP = args.b, world

    # ---- payloads ----
    packed_i64 = torch.randint(0, 2**60, (B,), dtype=torch.int64, device=device)
    vec_i64 = torch.randint(0, 1000, (B,), dtype=torch.int64, device=device)
    vec_f32 = torch.rand(B, dtype=torch.float32, device=device)
    pair_f32 = torch.rand(B, 2, dtype=torch.float32, device=device)
    b1_i64 = packed_i64.clone().unsqueeze(-1)  # [B,1]
    # 0-fill SUM tensor for workaround 1: [B, TP, 2], this rank's slot filled.
    local_val = torch.rand(B, dtype=torch.float32, device=device)
    local_idx = torch.randint(0, 201088, (B,), dtype=torch.float32, device=device)

    # vLLM optimized collectives (import lazily; needs a real TP group). The bench
    # uses the WORLD group as the TP group: init a minimal vLLM dist env if available.
    vllm_ar = vllm_ag = None
    try:
        from vllm.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
            tensor_model_parallel_all_reduce,
            tensor_model_parallel_all_gather,
        )
        init_distributed_environment(
            world_size=world, rank=rank, local_rank=local_rank, backend=backend
        )
        initialize_model_parallel(tensor_model_parallel_size=world)
        vllm_ar = tensor_model_parallel_all_reduce
        vllm_ag = tensor_model_parallel_all_gather
    except Exception as e:  # noqa: BLE001
        if rank == 0:
            print(f"[warn] vLLM optimized collectives unavailable: {e}\n"
                  f"       ar_sum_f32_vllm / ag_f32_pair / ag_i64_b1 will be skipped.")

    pg = dist.group.WORLD

    # ---- ops ----
    def ar_max_i64():
        x = packed_i64.clone()
        dist.all_reduce(x, op=dist.ReduceOp.MAX, group=pg)
        return x

    def ar_sum_i64():
        x = vec_i64.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=pg)
        return x

    def ar_sum_f32():
        x = vec_f32.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=pg)
        return x

    def ar_sum_f32_vllm():
        return vllm_ar(vec_f32.clone())

    def ag_f32_pair():
        g = vllm_ag(pair_f32, dim=-1).view(B, TP, 2)
        r = g[:, :, 0].argmax(dim=-1, keepdim=True)
        return g[:, :, 1].gather(-1, r).squeeze(-1)

    def ag_i64_b1():
        g = vllm_ag(b1_i64, dim=-1)  # [B, TP]
        return g.max(dim=-1).values

    def sum0fill_f32():
        # [B, TP, 2]; only this rank's slot nonzero -> SUM yields the gathered table.
        t = torch.zeros(B, TP, 2, dtype=torch.float32, device=device)
        t[:, rank, 0] = local_val
        t[:, rank, 1] = local_idx
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=pg)
        r = t[:, :, 0].argmax(dim=-1, keepdim=True)
        return t[:, :, 1].gather(-1, r).squeeze(-1)

    variants = [
        ("ar_max_i64  (SHIPPED)", ar_max_i64, True),
        ("ar_sum_i64           ", ar_sum_i64, True),
        ("ar_sum_f32           ", ar_sum_f32, True),
        ("ar_sum_f32_vllm      ", ar_sum_f32_vllm, vllm_ar is not None),
        ("ag_f32_pair  (WA #2) ", ag_f32_pair, vllm_ag is not None),
        ("ag_i64_b1 (ww16-brkn)", ag_i64_b1, vllm_ag is not None),
        ("sum0fill_f32 (WA #1) ", sum0fill_f32, True),
    ]

    rows = []
    for name, fn, enabled in variants:
        if not enabled:
            rows.append((name, None, None, None))
            continue
        loop_ms = _time_op(fn, args.iters, args.warmup)
        s = _per_call_samples(fn, max(200, args.iters // 5), 50)
        med = s[len(s) // 2]
        p99 = s[min(len(s) - 1, int(len(s) * 0.99))]
        # reduce across ranks: report the SLOWEST rank (sets step latency).
        stats = torch.tensor([loop_ms, med, p99], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.MAX, group=pg)
        rows.append((name, stats[0].item(), stats[1].item(), stats[2].item()))

    if rank == 0:
        print("\n" + "=" * 74)
        print(f"dist-sample reduce microbench  B={B} TP={TP} backend={backend} "
              f"iters={args.iters}")
        print("=" * 74)
        print(f"{'variant':<22} {'loop_ms':>9} {'sync_med':>9} {'sync_p99':>9}  "
              f"(slowest rank)")
        print("-" * 74)
        base = None
        for name, loop_ms, med, p99 in rows:
            if loop_ms is None:
                print(f"{name:<22} {'(skipped)':>9}")
                continue
            if base is None:
                base = loop_ms
            rel = f"{loop_ms / base:5.2f}x" if base else ""
            print(f"{name:<22} {loop_ms:9.3f} {med:9.3f} {p99:9.3f}  {rel}")
        print("-" * 74)
        print("READ: ar_sum_f32 << ar_max_i64 => int64-MAX path slow (cause a) -> WA#1 helps.")
        print("      ar_sum_f32 ~= ar_max_i64 => hop count (cause b) -> only all-gather (WA#2) helps.")
        print("      ar_sum_f32_vllm << ar_sum_f32(raw) => optimized communicator matters.")
        print("      Compare sync_med to the live run's dist sample_ms (~4ms incl. matmul+kernel).")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
