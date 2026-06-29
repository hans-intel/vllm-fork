# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

# Smallest positive normal fp32 value. Used to clamp the uniform draw so that
# `log(u)` cannot produce -inf (and thus `-log(-log(u))` stays finite).
#
# Triton requires globals accessed from `@triton.jit` functions to be wrapped
# in `tl.constexpr(...)`. We can only do that when Triton is actually
# available — on the CPU worker path `tl` is a placeholder whose `constexpr`
# attribute is `None`, and `tl.constexpr(...)` would crash at import time.
_FP32_TINY = (
    tl.constexpr(float.fromhex("0x1p-126")) if HAS_TRITON else float.fromhex("0x1p-126")
)


@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def tl_rand64(seed, offset, includes_zero: tl.constexpr):
    lo, hi, _, _ = tl.randint4x(seed, offset)
    lo = lo.to(tl.uint32, bitcast=True).to(tl.uint64)
    hi = hi.to(tl.uint32, bitcast=True).to(tl.uint64)
    r = (hi << 32) | lo

    # 1 / 2**64
    scale = 5.421010862427522170037e-20
    u = r.to(tl.float64) * scale
    if not includes_zero:
        u = tl.maximum(u, 2.2250738585072014e-308)  # float64 tiny
    return u


@triton.jit
def gumbel_block_argmax(
    logits,
    block,
    mask,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    vocab_size,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr = False,
):
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp != 0.0 and APPLY_TEMPERATURE:
        # Apply temperature.
        # NOTE(woosuk): Match the behavior of _temperature_kernel.
        # E.g., if the kernel uses tl.div_rn, we should use tl.div_rn here too.
        logits = logits / temp

    if processed_logits_ptr is not None:
        # Store the temperature-applied logits.
        if processed_logits_col_ptr is not None:
            if PER_TOKEN_COL:
                col = tl.load(processed_logits_col_ptr + token_idx)
            else:
                col = tl.load(processed_logits_col_ptr)
        else:
            col = 0
        tl.store(
            processed_logits_ptr
            + req_state_idx * processed_logits_stride
            + col * vocab_size
            + block,
            logits,
            mask=mask,
        )

    # fp32 is the default reduction dtype; fp64 is ~1/32–1/64x the throughput
    # on H100/Ada/Blackwell and empirically indistinguishable for Gumbel-max.
    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        # Calculate the seed for gumbel noise.
        seed = tl.load(seeds_ptr + req_state_idx)
        pos = tl.load(pos_ptr + token_idx)
        gumbel_seed = tl.randint(seed, pos)

        if USE_FP64:
            u = tl_rand64(gumbel_seed, block, includes_zero=False)
        else:
            u = tl.rand(gumbel_seed, block)
            u = tl.maximum(u, _FP32_TINY)
        gumbel_noise = -tl.log(-tl.log(u))

        # Apply gumbel noise.
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    value, idx = tl.max(logits, axis=0, return_indices=True)
    return value, idx


@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    value, idx = gumbel_block_argmax(
        logits,
        block,
        mask,
        token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        processed_logits_ptr,
        processed_logits_stride,
        processed_logits_col_ptr,
        vocab_size,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        USE_FP64=USE_FP64,
        PER_TOKEN_COL=PER_TOKEN_COL,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    output_processed_logits: torch.Tensor | None = None,
    output_processed_logits_col: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max_dtype = torch.float64 if use_fp64 else torch.float32
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=local_max_dtype)
    per_token_col = (
        output_processed_logits_col is not None
        and output_processed_logits_col.dim() > 0
    )
    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        output_processed_logits,
        output_processed_logits.stride(0) if output_processed_logits is not None else 0,
        output_processed_logits_col,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
        USE_FP64=use_fp64,
        PER_TOKEN_COL=per_token_col,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled


@triton.jit
def _dist_gumbel_packed_kernel(
    logits_ptr,
    logits_stride,
    out_packed_ptr,
    out_packed_stride,
    seed,
    vocab_start,
    total_vocab,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Gumbel-max over one TP vocab shard -> packed int64, one pass.

    For each token and BLOCK_SIZE-wide block of the local shard, loads logits,
    adds Gumbel(0,1) noise generated on the fly (no materialized noise tensor),
    reduces to the per-block argmax, and PACKS (value, global_index) into one
    positive sortable int64 directly in-kernel:
        bits [61:31] : 31-bit unsigned-monotonic key of the float32 value
        bits [30:0]  : global vocab index
    so a plain int64 max over blocks (host) and across ranks (collective) yields
    the value-then-index winner with no separate argmax/gather/pack ops.

    The noise counter is keyed by GLOBAL vocab position (vocab_start + local pos)
    and token, with `seed` shared across TP ranks, so the Gumbel field is one
    coherent i.i.d. draw over the full vocab — required for distributional
    correctness (disjoint shards must not reuse the same noise value).
    """
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float64)

    # Counter unique per (token, GLOBAL vocab position); same seed on all ranks.
    global_pos = (block + vocab_start).to(tl.int64)
    offset = token_idx.to(tl.int64) * total_vocab + global_pos
    u = tl_rand64(seed, offset, includes_zero=False)
    gumbel = -tl.log(-tl.log(u))
    perturbed = tl.where(mask, logits + gumbel, float("-inf"))

    value, idx = tl.max(perturbed, axis=0, return_indices=True)
    global_idx = (block_idx * BLOCK_SIZE + idx + vocab_start).to(tl.int64)

    # Pack (value, index) -> positive sortable int64 (matches the host-side layout
    # in LogitsProcessor.gumbel_argmax_tokens). float32 monotonic key in high bits.
    val_i32 = value.to(tl.float32).to(tl.int32, bitcast=True)
    key_i32 = tl.where(val_i32 >= 0, val_i32 | (-2147483648), ~val_i32)
    key_u32 = key_i32.to(tl.int64) & 0xFFFFFFFF  # unsigned-monotonic image
    key31 = key_u32 >> 1  # 31-bit key, positive
    packed = (key31 << 31) | (global_idx & 0x7FFFFFFF)
    tl.store(out_packed_ptr + token_idx * out_packed_stride + block_idx, packed)


@triton.jit
def _dist_gumbel_validx_kernel(
    logits_ptr,
    logits_stride,
    out_val_ptr,
    out_val_stride,
    out_idx_ptr,
    out_idx_stride,
    seed,
    vocab_start,
    total_vocab,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Same fused Gumbel-max over a TP vocab shard as _dist_gumbel_packed_kernel,
    but stores the per-block winner as SEPARATE (value fp32, global_index fp32)
    instead of a packed int64. Lets the cross-rank reduce run in fp32 (a tuned,
    ~13x faster XCCL path than int64) via a 0-fill SUM all-reduce. global_index
    <= ~201087 < 2**24 is fp32-exact; value is the raw float32 logit+gumbel.
    Noise keying is IDENTICAL to the packed kernel (same seed/offset) so the
    distribution is unchanged."""
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float64)

    global_pos = (block + vocab_start).to(tl.int64)
    offset = token_idx.to(tl.int64) * total_vocab + global_pos
    u = tl_rand64(seed, offset, includes_zero=False)
    gumbel = -tl.log(-tl.log(u))
    perturbed = tl.where(mask, logits + gumbel, float("-inf"))

    value, idx = tl.max(perturbed, axis=0, return_indices=True)
    global_idx = (block_idx * BLOCK_SIZE + idx + vocab_start).to(tl.int64)

    tl.store(
        out_val_ptr + token_idx * out_val_stride + block_idx,
        value.to(tl.float32),
    )
    tl.store(
        out_idx_ptr + token_idx * out_idx_stride + block_idx,
        global_idx.to(tl.float32),
    )


def dist_gumbel_local_validx(
    logits: torch.Tensor,  # [num_tokens, shard_vocab_size], float32
    seed: int,
    vocab_start: int,
    total_vocab: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vocab-parallel Gumbel-max local winner, returned as separate fp32 tensors.

    Returns (value[num_tokens], global_idx[num_tokens]) both float32: per token,
    this shard's winning logit+gumbel value and its global vocab index. Feed into
    a 0-fill SUM all-reduce + local argmax for the cross-rank reduce (fp32 path,
    no slow int64 collective). Noise keying matches dist_gumbel_local_packed, so
    accuracy is unchanged; only the reduce dtype differs.
    """
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    out_val = logits.new_empty(num_tokens, num_blocks, dtype=torch.float32)
    out_idx = logits.new_empty(num_tokens, num_blocks, dtype=torch.float32)
    _dist_gumbel_validx_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        out_val,
        out_val.stride(0),
        out_idx,
        out_idx.stride(0),
        seed,
        vocab_start,
        total_vocab,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Per-token local winner across blocks (fp32 argmax, exact gather of index).
    best_block = out_val.argmax(dim=-1, keepdim=True)
    value = out_val.gather(dim=-1, index=best_block).view(-1)
    global_idx = out_idx.gather(dim=-1, index=best_block).view(-1)
    return value, global_idx


def dist_gumbel_local_packed(
    logits: torch.Tensor,  # [num_tokens, shard_vocab_size], float32
    seed: int,
    vocab_start: int,
    total_vocab: int,
) -> torch.Tensor:
    """Vocab-parallel Gumbel-max local argmax, returned as a packed int64.

    Returns packed[num_tokens] (int64): per token, the shard-local winner packed
    as (31-bit value key << 31) | global_index. A plain int64 MAX across TP ranks
    then yields the global winner; extract the index with `packed & 0x7FFFFFFF`.
    `seed` must be identical across TP ranks for the step; `total_vocab` keys the
    per-(token, position) noise counter collision-free.
    """
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    out_packed = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    _dist_gumbel_packed_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        out_packed,
        out_packed.stride(0),
        seed,
        vocab_start,
        total_vocab,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # int64 max over blocks (monotonic packing => value-then-index winner).
    return out_packed.max(dim=-1).values
