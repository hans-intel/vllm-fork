# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

try:
    from vllm.triton_utils import tl, triton
except ImportError as e:
    logger.debug("Import error msg: %s", e)

USE_TRITON_XPU_ATTN = os.environ.get("VLLM_USE_TRITON_XPU_ATTN", "0") == "1"

if USE_TRITON_XPU_ATTN:
    torch._dynamo.config.recompile_limit = 16


@triton.jit
def fp8_e4m3_to_fp16(x):
    # x is fp8 casted to uint8 before function call
    x_i8 = x.to(tl.uint16)
    sign = (x_i8 & 0x80) << 8
    payload = (x_i8 & 0x7F) << 7
    unscaled_i16 = sign | payload
    result = unscaled_i16.to(tl.float16, bitcast=True)
    # Rebias exponent from e4m3 to fp16
    result *= 256.0
    return result


@triton.jit
def fp8_e4m3_to_bf16(x):
    x_i8 = x.to(tl.uint16)
    sign = (x_i8 & 0x80) << 8
    payload = (x_i8 & 0x7F) << 4
    unscaled_i16 = sign | payload
    result = unscaled_i16.to(tl.bfloat16, bitcast=True)
    # Rebias exponent from e4m3 to bf16. Same as fp32
    result *= 2.0**120
    return result


@triton.jit
def fp8_e5m2_to_fp16(x):
    x_i8 = x.to(tl.uint16)
    unscaled_i16 = (x_i8 & 0xFF) << 8
    result = unscaled_i16.to(tl.float16, bitcast=True)
    # No rebias needed
    return result


@triton.jit
def fp8_e5m2_to_bf16(x):
    x_i8 = x.to(tl.uint16)
    sign = (x_i8 & 0x80) << 8
    payload = (x_i8 & 0x7F) << 5
    unscaled_i16 = sign | payload
    result = unscaled_i16.to(tl.bfloat16, bitcast=True)
    # Rebias exponent from e5m2 to bf16
    result *= 2.0**112
    return result


@triton.jit
def fp8_e4m3_to_fp32(x):
    x_i8 = x.to(tl.uint32)
    sign = (x_i8 & 0x80) << 24
    payload = (x_i8 & 0x7F) << 20
    unscaled_f32 = (sign | payload).to(tl.float32, bitcast=True)
    # Rebias exponent from e4m3 to fp32
    result = unscaled_f32 * (2.0**120)
    return result


@triton.jit
def fp8_e5m2_to_fp32(x):
    x_f8 = x.to(tl.float8e5, bitcast=True)
    result = x_f8.to(tl.float32)
    return result


@triton.jit
def convert_to_dtype(x, src_dtype: tl.constexpr, target_dtype: tl.constexpr):
    if src_dtype == tl.float8e4nv:  # float8_e4m3fn
        if target_dtype == tl.float16:
            return fp8_e4m3_to_fp16(x)
        elif target_dtype == tl.bfloat16:
            return fp8_e4m3_to_bf16(x)
        else:
            return fp8_e4m3_to_fp32(x).to(target_dtype)
    elif src_dtype == tl.float8e5:  # float8_e5m2
        if target_dtype == tl.float16:
            # Yes, this is faster than direct conversion
            # No, I don't know why
            return fp8_e5m2_to_bf16(x).to(tl.float16)
        elif target_dtype == tl.bfloat16:
            return fp8_e5m2_to_bf16(x)
        else:
            return fp8_e5m2_to_fp32(x).to(target_dtype)
    elif target_dtype == tl.bfloat16:
        if src_dtype == target_dtype:
            return x
        else:
            return x.to(src_dtype, bitcast=True).to(tl.float32).to(tl.bfloat16)
    else:
        return x.to(src_dtype, bitcast=True).to(target_dtype)


@triton.jit
def _fwd_grouped_kernel_decode(
    # Input/Output Tensors
    Output,
    Query,
    K_Buffer,
    V_Buffer,
    KEY_SCRATCH,
    VALUE_SCRATCH,
    Req_to_tokens,
    cu_seqlens_q,
    seqused_k,
    flat_offset,
    Flat_Att_Out,
    Flat_Att_Single,
    num_splits,
    num_splits_real,
    # Parameters
    sm_scale,
    num_decodes,
    total_active_tasks,
    # Strides for tensor access
    stride_req_to_tokens_b: tl.constexpr,
    stride_qbs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_buf_kbs: tl.constexpr,
    stride_buf_kh: tl.constexpr,
    stride_buf_vbs: tl.constexpr,
    stride_buf_vh: tl.constexpr,
    stride_buf_kblock: tl.constexpr,
    stride_buf_vblock: tl.constexpr,
    # KV scales
    k_scale,
    v_scale,
    # sink
    sink,
    USE_SINKS: tl.constexpr,
    # Constexprs for kernel specialization
    window_size_left: tl.constexpr,
    kv_group_num: tl.constexpr,
    q_block_head: tl.constexpr,
    num_heads_kv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    # Autotune parameters
    BLOCK_N: tl.constexpr = 16,
    BLOCK_DMODEL_SLICE: tl.constexpr = 8,
    BLOCKS_PER_SPLIT: tl.constexpr = 1,
):
    """
    Stage 1 Kernel using static mapping. Each block processes one task.
    It reads task info from Task_metadata and writes its partial result
    to a unique slot in the flat Flat_Att_Out buffer.
    """
    task_id = tl.program_id(0)
    if task_id >= total_active_tasks:
        return

    cur_batch = task_id // (num_heads_kv * num_splits)
    cur_head = task_id % (num_heads_kv * num_splits) // num_splits
    cur_split = task_id % num_splits

    cur_batch_seq_len = tl.load(seqused_k + cur_batch)
    # This is where the current q starts in the batch
    cur_q_start = tl.load(cu_seqlens_q + cur_batch)

    seq_len_q = tl.load(cu_seqlens_q + cur_batch + 1) - tl.load(
        cu_seqlens_q + cur_batch
    )

    # Perform dequant for prompt pass in decode mode
    perform_dequant = False

    if seq_len_q > 1:
        # Decrement num_decodes for the prompt kernel
        if (cur_head == 0) and (cur_split == 0):
            tl.atomic_min(num_decodes, cur_batch)
        if K_Buffer.dtype.element_ty != Query.dtype.element_ty:
            perform_dequant = True
        else:
            # Early exit for prompt tokens
            return

    seq_start = 0
    if window_size_left > 0:
        seq_start = tl.maximum(0, cur_batch_seq_len - window_size_left - 1)
        seq_start = (seq_start // PAGE_SIZE) * PAGE_SIZE

    split_start = seq_start + cur_split * PAGE_SIZE * BLOCKS_PER_SPLIT
    split_end = tl.minimum(
        split_start + PAGE_SIZE * BLOCKS_PER_SPLIT, cur_batch_seq_len
    )

    dot_dtype = Query.dtype.element_ty

    cur_kv_head = cur_head
    k_head_offset = cur_kv_head * stride_buf_kh
    v_head_offset = cur_kv_head * stride_buf_vh

    k_scale_value = tl.load(k_scale)
    v_scale_value = tl.load(v_scale)
    sm_scale *= 1.44269504 * k_scale_value
    k_scale_dot = k_scale_value.to(dot_dtype)
    v_scale_dot = v_scale_value.to(dot_dtype)

    kv_dtype: tl.constexpr = K_Buffer.dtype.element_ty
    # When loading from 8-bit
    if kv_dtype == tl.float16 or kv_dtype == tl.bfloat16:
        K_Buffer_i8 = K_Buffer
        V_Buffer_i8 = V_Buffer
    else:
        K_Buffer_i8 = K_Buffer.to(tl.pointer_type(tl.uint8))
        V_Buffer_i8 = V_Buffer.to(tl.pointer_type(tl.uint8))

    if perform_dequant:  # Dequant only pass for prompt tokens
        if window_size_left > 0:
            seq_start = tl.maximum(0, cur_batch_seq_len - window_size_left - seq_len_q)
            seq_start = (seq_start // PAGE_SIZE) * PAGE_SIZE
            split_start = seq_start + cur_split * PAGE_SIZE * BLOCKS_PER_SPLIT
            split_end = tl.minimum(
                split_start + PAGE_SIZE * BLOCKS_PER_SPLIT, cur_batch_seq_len
            )

        num_kv_pages = (split_end - split_start + PAGE_SIZE - 1) // PAGE_SIZE

        # When kv_group_num is large (>8), more than 1 warp is used for attn comp
        # In that case, increase the block size to process more rows per warp
        if q_block_head > 8:
            if BLOCK_N * (q_block_head // 8) > PAGE_SIZE:
                BLOCK_N_DEQUANT: tl.constexpr = PAGE_SIZE
            else:
                BLOCK_N_DEQUANT: tl.constexpr = BLOCK_N * (q_block_head // 8)
        else:
            BLOCK_N_DEQUANT: tl.constexpr = BLOCK_N

        for kv_page in range(num_kv_pages):
            kv_page_number_scalar = tl.load(
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch
                + split_start // PAGE_SIZE
                + kv_page
            )
            desc_k = tl.make_tensor_descriptor(
                K_Buffer_i8
                # real cache: block jump uses the true per-block stride
                + kv_page_number_scalar * stride_buf_kblock
                + k_head_offset,
                shape=(PAGE_SIZE, BLOCK_DMODEL),
                strides=(stride_buf_kbs, 1),
                block_shape=(BLOCK_N_DEQUANT, BLOCK_DMODEL),
            )
            desc_k_quant = tl.make_tensor_descriptor(
                KEY_SCRATCH
                # scratch is contiguous: block stride == PAGE_SIZE * row stride
                + (kv_page_number_scalar * PAGE_SIZE) * stride_buf_kbs
                + k_head_offset,
                shape=(PAGE_SIZE, BLOCK_DMODEL),
                strides=(stride_buf_kbs, 1),
                block_shape=(BLOCK_N_DEQUANT, BLOCK_DMODEL),
            )
            desc_v = tl.make_tensor_descriptor(
                V_Buffer_i8
                # real cache: block jump uses the true per-block stride
                + kv_page_number_scalar * stride_buf_vblock
                + v_head_offset,
                shape=(PAGE_SIZE, BLOCK_DV),
                strides=(stride_buf_vbs, 1),
                block_shape=(BLOCK_N_DEQUANT, BLOCK_DV),
            )
            desc_v_quant = tl.make_tensor_descriptor(
                VALUE_SCRATCH
                # scratch is contiguous: block stride == PAGE_SIZE * row stride
                + (kv_page_number_scalar * PAGE_SIZE) * stride_buf_vbs
                + v_head_offset,
                shape=(PAGE_SIZE, BLOCK_DV),
                strides=(stride_buf_vbs, 1),
                block_shape=(BLOCK_N_DEQUANT, BLOCK_DV),
            )

            for start_n in tl.range(0, PAGE_SIZE, BLOCK_N_DEQUANT):
                k = desc_k.load([start_n, 0])
                k = convert_to_dtype(k, kv_dtype, dot_dtype)
                k *= k_scale_dot
                desc_k_quant.store([start_n, 0], k)
                v = desc_v.load([start_n, 0])
                v = convert_to_dtype(v, kv_dtype, dot_dtype)
                v *= v_scale_dot
                desc_v_quant.store([start_n, 0], v)
        return

    if split_start >= split_end:
        return

    last_split = split_end == cur_batch_seq_len
    # Update num_splits_real for stage 2
    if last_split and (cur_head == 0):
        tl.store(num_splits_real + cur_batch, cur_split + 1)

    splits_per_split = PAGE_SIZE * BLOCKS_PER_SPLIT
    numerator = cur_batch_seq_len - seq_start + splits_per_split - 1
    num_splits_real_value = numerator // splits_per_split
    output_base_idx = (
        tl.load(flat_offset + cur_batch) * num_heads_kv * kv_group_num
        + cur_head * kv_group_num * num_splits_real_value
        + cur_split
    )

    desc_q_d = tl.make_tensor_descriptor(
        Query + cur_q_start * stride_qbs + cur_head * kv_group_num * stride_qh,
        shape=(kv_group_num, BLOCK_DMODEL),
        strides=(stride_qh, 1),
        block_shape=(q_block_head, BLOCK_DMODEL // BLOCK_DMODEL_SLICE),
    )

    q_d_0 = desc_q_d.load([0, 0 * BLOCK_DMODEL // BLOCK_DMODEL_SLICE]).to(dot_dtype)
    if BLOCK_DMODEL_SLICE > 1:
        q_d_1 = desc_q_d.load([0, 1 * BLOCK_DMODEL // BLOCK_DMODEL_SLICE]).to(dot_dtype)
    if BLOCK_DMODEL_SLICE > 2:
        q_d_2 = desc_q_d.load([0, 2 * BLOCK_DMODEL // BLOCK_DMODEL_SLICE]).to(dot_dtype)
        q_d_3 = desc_q_d.load([0, 3 * BLOCK_DMODEL // BLOCK_DMODEL_SLICE]).to(dot_dtype)
    if BLOCK_DMODEL_SLICE > 4:
        q_d_4 = desc_q_d.load([0, 4 * BLOCK_DMODEL // BLOCK_DMODEL_SLICE]).to(dot_dtype)
        q_d_5 = desc_q_d.load([0, 5 * BLOCK_DMODEL // BLOCK_DMODEL_SLICE]).to(dot_dtype)
        q_d_6 = desc_q_d.load([0, 6 * BLOCK_DMODEL // BLOCK_DMODEL_SLICE]).to(dot_dtype)
        q_d_7 = desc_q_d.load([0, 7 * BLOCK_DMODEL // BLOCK_DMODEL_SLICE]).to(dot_dtype)

    # In case kv_group_num < q_block_head
    mask_head = tl.arange(0, q_block_head) < kv_group_num

    if USE_SINKS and (cur_split == 0):
        # Sink is only used for the first split (segment 0)
        e_max_d = (
            (
                tl.load(
                    sink + cur_kv_head * kv_group_num + tl.arange(0, q_block_head),
                    mask=mask_head,
                )
                .reshape([q_block_head, 1])
                .to(tl.float32)
            )
            * 1.44269504
        ).to(tl.float32)
    else:
        # e_max_d = tl.full([kv_group_num, 1], -float("inf"), dtype=tl.float32)
        e_max_d = tl.full([q_block_head, 1], -1e6, dtype=tl.float32)  # More stable

    e_sum_d = tl.full([q_block_head, 1], 1.0, dtype=tl.float32)

    acc_d_0 = tl.zeros([q_block_head, BLOCK_DV], dtype=tl.float32)
    # Load k/v scale. Only support per layer scale now

    for start_n in range(split_start, split_end, BLOCK_N):
        kv_page_number_scalar = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch + start_n // PAGE_SIZE
        )

        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < split_end

        if window_size_left > 0:
            # Sliding window mask
            mask_n = mask_n & (offs_n >= cur_batch_seq_len - window_size_left - 1)

        desc_k = tl.make_tensor_descriptor(
            K_Buffer_i8
            + kv_page_number_scalar * stride_buf_kblock
            + (start_n % PAGE_SIZE) * stride_buf_kbs
            + k_head_offset,
            shape=(BLOCK_DMODEL, PAGE_SIZE),
            strides=(1, stride_buf_kbs),
            block_shape=(BLOCK_DMODEL // BLOCK_DMODEL_SLICE, BLOCK_N),
        )
        k = desc_k.load([0, 0])
        # There is a bug when doing e4m3->fp16/bf16 followed by dot
        # hence going to fp32 first
        k = convert_to_dtype(k, kv_dtype, tl.float32).to(dot_dtype)
        qk_d = tl.dot(q_d_0, k)

        if BLOCK_DMODEL_SLICE > 1:
            k = desc_k.load([BLOCK_DMODEL // BLOCK_DMODEL_SLICE, 0])
            k = convert_to_dtype(k, kv_dtype, tl.float32).to(dot_dtype)
            qk_d += tl.dot(q_d_1, k)

        if BLOCK_DMODEL_SLICE > 2:
            k = desc_k.load([2 * BLOCK_DMODEL // BLOCK_DMODEL_SLICE, 0])
            k = convert_to_dtype(k, kv_dtype, tl.float32).to(dot_dtype)
            qk_d += tl.dot(q_d_2, k)

            k = desc_k.load([3 * BLOCK_DMODEL // BLOCK_DMODEL_SLICE, 0])
            k = convert_to_dtype(k, kv_dtype, tl.float32).to(dot_dtype)
            qk_d += tl.dot(q_d_3, k)

        if BLOCK_DMODEL_SLICE > 4:
            k = desc_k.load([4 * BLOCK_DMODEL // BLOCK_DMODEL_SLICE, 0])
            k = convert_to_dtype(k, kv_dtype, tl.float32).to(dot_dtype)
            qk_d += tl.dot(q_d_4, k)

            k = desc_k.load([5 * BLOCK_DMODEL // BLOCK_DMODEL_SLICE, 0])
            k = convert_to_dtype(k, kv_dtype, tl.float32).to(dot_dtype)
            qk_d += tl.dot(q_d_5, k)

            k = desc_k.load([6 * BLOCK_DMODEL // BLOCK_DMODEL_SLICE, 0])
            k = convert_to_dtype(k, kv_dtype, tl.float32).to(dot_dtype)
            qk_d += tl.dot(q_d_6, k)

            k = desc_k.load([7 * BLOCK_DMODEL // BLOCK_DMODEL_SLICE, 0])
            k = convert_to_dtype(k, kv_dtype, tl.float32).to(dot_dtype)
            qk_d += tl.dot(q_d_7, k)

        qk_d = (qk_d * sm_scale).to(tl.float32)

        # Causal mask
        qk_d = tl.where(mask_n, qk_d, tl.full([], float("-inf"), tl.float32))
        n_e_max_d = tl.maximum(tl.max(qk_d, 1, keep_dims=True), e_max_d)
        re_scale_d = tl.math.exp2(e_max_d - n_e_max_d)
        p_d = tl.math.exp2(qk_d - n_e_max_d)
        e_sum_d = e_sum_d * re_scale_d + tl.sum(p_d, 1, keep_dims=True)
        e_max_d = n_e_max_d
        p_d = p_d.to(dot_dtype)

        # 2d load for V
        desc_v = tl.make_tensor_descriptor(
            V_Buffer_i8
            + kv_page_number_scalar * stride_buf_vblock
            + (start_n % PAGE_SIZE) * stride_buf_vbs
            + v_head_offset,
            shape=(PAGE_SIZE, BLOCK_DV),
            strides=(stride_buf_vbs, 1),
            block_shape=(BLOCK_N, BLOCK_DV),
        )
        v = desc_v.load([0, 0])
        # There is a bug when doing e4m3->fp16/bf16 followed by dot
        # hence going to fp32 first
        v = convert_to_dtype(v, kv_dtype, tl.float32) * v_scale_value
        acc_d_0 *= re_scale_d
        acc_d_0 += tl.dot(p_d, v.to(dot_dtype))
    # Store result in the flat intermediate buffer
    e_max_d = e_max_d + tl.math.log2(e_sum_d)
    e_max_flat_d = tl.reshape(e_max_d, [q_block_head])

    acc_norm_d_0 = acc_d_0 / e_sum_d

    offs_h_d = output_base_idx + tl.arange(0, q_block_head) * num_splits_real_value
    offs_mid_o_1_d = offs_h_d
    tl.store(Flat_Att_Single + offs_mid_o_1_d, e_max_flat_d, mask=mask_head)

    if last_split and (cur_split == 0):
        # When there is only one split for this q position,
        # directly write to output
        output_base_idx = (
            cur_q_start * num_heads_kv * kv_group_num + cur_head * kv_group_num
        )
        desc_o = tl.make_tensor_descriptor(
            Output + output_base_idx * Lv,
            shape=(kv_group_num, BLOCK_DV),
            strides=(Lv, 1),
            block_shape=(q_block_head, BLOCK_DV),
        )
        acc_norm_d = acc_norm_d_0.to(Output.dtype.element_ty)
        desc_o.store([0, 0], acc_norm_d)
    else:
        # Otherwise, store to the flat intermediate buffer for stage 2
        desc_o = tl.make_tensor_descriptor(
            Flat_Att_Out + output_base_idx * Lv,
            shape=(kv_group_num, BLOCK_DV),
            strides=(num_splits_real_value * Lv, 1),
            block_shape=(q_block_head, BLOCK_DV),
        )
        acc_norm_d = acc_norm_d_0.to(Flat_Att_Out.dtype.element_ty)
        desc_o.store([0, 0], acc_norm_d)


@triton.jit
def _fwd_grouped_kernel_prompt(
    # Input/Output Tensors
    Output,
    Query,
    K_Buffer,
    V_Buffer,
    Req_to_tokens,
    cu_seqlens_q,
    seqused_k,
    Flat_Att_Out,
    Flat_Att_Single,
    num_splits_q,
    num_splits_real,
    # Parameters
    sm_scale,
    num_decodes,
    num_seqs,
    prompt_counter,
    # Strides for tensor access
    stride_req_to_tokens_b: tl.constexpr,
    stride_qbs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_buf_kbs: tl.constexpr,
    stride_buf_kh: tl.constexpr,
    stride_buf_vbs: tl.constexpr,
    stride_buf_vh: tl.constexpr,
    stride_buf_kblock: tl.constexpr,
    stride_buf_vblock: tl.constexpr,
    # KV scales
    k_scale,
    v_scale,
    # sink
    sink,
    USE_SINKS: tl.constexpr,
    # Constexprs for kernel specialization
    window_size_left: tl.constexpr,
    kv_group_num: tl.constexpr,
    num_heads_kv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    # Autotune parameters
    BLOCK_N: tl.constexpr = 16,
):
    """
    Stage 1 Kernel using static mapping. Each block processes one task.
    It reads task info from Task_metadata and writes its partial result
    to a unique slot in the flat Flat_Att_Out buffer.
    """
    num_decodes_value = tl.load(num_decodes)
    # If num_decodes_value==16384, there are no prompt requests
    if num_decodes_value == 16384:
        return
    global_id = tl.program_id(0)
    cur_batch = (
        global_id // (num_heads_kv * kv_group_num * num_splits_q) + num_decodes_value
    )

    while cur_batch < num_seqs:
        local_id_0 = global_id % (num_heads_kv * kv_group_num * num_splits_q)
        cur_head = local_id_0 // num_splits_q
        local_id_1 = local_id_0 % num_splits_q
        cur_seq_len_start = local_id_1 * BLOCK_Q
        cur_split: tl.constexpr = 0  # Disable split for prompt attention

        seq_len_q = tl.load(cu_seqlens_q + cur_batch + 1) - tl.load(
            cu_seqlens_q + cur_batch
        )

        # Skip for decode tokens and out-of-bound q positions
        if not ((seq_len_q <= 1) or (cur_seq_len_start >= seq_len_q)):
            cur_batch_seq_len = tl.load(seqused_k + cur_batch)
            # This is where the current q starts in the batch
            cur_q_start = tl.load(cu_seqlens_q + cur_batch)

            seq_start = 0
            if window_size_left > 0:
                seq_start = tl.maximum(
                    0,
                    cur_batch_seq_len
                    - window_size_left
                    - seq_len_q
                    + cur_seq_len_start,
                )
                seq_start = (seq_start // PAGE_SIZE) * PAGE_SIZE

            split_start = seq_start
            # q limit
            split_end = tl.minimum(
                cur_batch_seq_len,
                cur_batch_seq_len - seq_len_q + cur_seq_len_start + BLOCK_Q,
            )
            split_end_q = tl.minimum(seq_len_q, cur_seq_len_start + BLOCK_Q)

            # split_k disabled for prompt. num_splits_real is per batch
            if cur_head == 0 and cur_seq_len_start == 0:
                tl.store(num_splits_real + cur_batch, 1)

            dot_dtype = Query.dtype.element_ty

            # perform_dequant = K_Buffer_Orig.dtype.element_ty != Query.dtype.element_ty

            cur_kv_head = cur_head // kv_group_num
            k_head_offset = cur_kv_head * stride_buf_kh
            v_head_offset = cur_kv_head * stride_buf_vh

            # For attention masking
            offs_n_q = (
                cur_batch_seq_len
                - seq_len_q
                + cur_seq_len_start
                + tl.arange(0, BLOCK_Q)
            )

            if USE_SINKS and (cur_split == 0):
                sink_value = tl.load(sink + cur_head)
                # Convert sink from natural log space to log2 space
                # for consistency with exp2/log2 computations
                e_max_p = (
                    (sink_value * 1.44269504).broadcast_to([BLOCK_Q]).to(tl.float32)
                )  # multiply by 1/ln(2)
            else:
                e_max_p = tl.full([BLOCK_Q], -1e6, dtype=tl.float32)  # More stable
            e_sum_p = tl.zeros([BLOCK_Q], dtype=tl.float32) + 1.0

            acc_p_0 = tl.zeros([BLOCK_Q, BLOCK_DV], dtype=tl.float32)

            # Block ptr
            desc_q = tl.make_tensor_descriptor(
                Query
                + (cur_q_start + cur_seq_len_start) * stride_qbs
                + cur_head * BLOCK_DMODEL,
                shape=(split_end_q - cur_seq_len_start, BLOCK_DMODEL),
                strides=(stride_qbs, 1),
                block_shape=(BLOCK_Q, BLOCK_DMODEL),
            )
            q_p_0 = desc_q.load([0, 0]).to(dot_dtype)

            num_kv_pages = (split_end - split_start + PAGE_SIZE - 1) // PAGE_SIZE

            Req_to_tokens_base = (
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch
                + split_start // PAGE_SIZE
            )

            K_Buffer_base = K_Buffer + k_head_offset
            V_Buffer_base = V_Buffer + v_head_offset

            # Per-block jump = true block stride of the buffer being read
            # (contiguous scratch for fp8, interleaved real cache for bf16).
            stride_page_kbs = stride_buf_kblock
            stride_page_vbs = stride_buf_vblock

            qk_scale = sm_scale * 1.44269504  # 1/log(2)

            for kv_page in tl.range(num_kv_pages):
                page_start = kv_page * PAGE_SIZE + split_start
                kv_page_number_scalar = tl.load(Req_to_tokens_base + kv_page)

                desc_k = tl.make_tensor_descriptor(
                    K_Buffer_base + kv_page_number_scalar * stride_page_kbs,
                    shape=(BLOCK_DMODEL, PAGE_SIZE),
                    strides=(1, stride_buf_kbs),
                    block_shape=(BLOCK_DMODEL, BLOCK_N),
                )
                desc_v = tl.make_tensor_descriptor(
                    V_Buffer_base + kv_page_number_scalar * stride_page_vbs,
                    shape=(PAGE_SIZE, BLOCK_DV),
                    strides=(stride_buf_vbs, 1),
                    block_shape=(BLOCK_N, BLOCK_DV),
                )

                for start_n in tl.range(0, PAGE_SIZE, BLOCK_N):
                    k = desc_k.load([0, start_n]).to(dot_dtype)
                    qk_p = tl.dot(q_p_0, k)

                    # Causal mask
                    offs_n = page_start + start_n + tl.arange(0, BLOCK_N)
                    mask_qk = offs_n_q[:, None] >= offs_n[None, :]

                    if window_size_left > 0:
                        # Sliding window mask
                        mask_qk = mask_qk & (
                            offs_n[None, :] >= offs_n_q[:, None] - window_size_left
                        )
                    qk_p = (qk_p * qk_scale + tl.where(mask_qk, 0.0, -1e6)).to(
                        tl.float32
                    )

                    n_e_max_p = tl.maximum(tl.max(qk_p, 1), e_max_p)
                    qk_p -= n_e_max_p[:, None]

                    p_p = tl.math.exp2(qk_p)
                    re_scale_p = tl.math.exp2(e_max_p - n_e_max_p)
                    p_p_1 = tl.sum(p_p, 1)

                    acc_p_0 = acc_p_0 * re_scale_p[:, None]
                    v = desc_v.load([start_n, 0]).to(dot_dtype)
                    p_p = p_p.to(dot_dtype)
                    acc_p_0 = tl.dot(p_p, v, acc_p_0)
                    e_sum_p = e_sum_p * re_scale_p + p_p_1
                    e_max_p = n_e_max_p

            # Store result in the flat intermediate buffer
            acc_norm_p_0 = acc_p_0 / e_sum_p[:, None]

            # When there is only one split for this q position,
            # directly write to output
            output_base_idx = (
                cur_q_start + cur_seq_len_start
            ) * num_heads_kv * kv_group_num + cur_head
            desc_o = tl.make_tensor_descriptor(
                Output + output_base_idx * Lv,
                shape=(split_end_q - cur_seq_len_start, BLOCK_DV),
                strides=(kv_group_num * num_heads_kv * Lv, 1),
                block_shape=(BLOCK_Q, BLOCK_DV),
            )
            acc_norm_p = acc_norm_p_0.to(Output.dtype.element_ty)
            desc_o.store([0, 0], acc_norm_p)
        # Increment global counter
        global_id = tl.atomic_add(prompt_counter, 1)
        cur_batch = (
            global_id // (num_heads_kv * kv_group_num * num_splits_q)
            + num_decodes_value
        )


@triton.jit
def _fwd_kernel_stage2(
    # Input/Output Tensors
    Flat_Att_Out,
    Flat_Att_Single,
    o,
    flat_offset,
    num_splits,
    num_splits_real,
    cu_seqlens_q,
    # Parameters
    q_head_num: tl.constexpr,
    stride_obs,
    stride_oh,
    # Constexprs for kernel specialization
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    """
    Stage 2 Kernel. Reads from the flat intermediate buffer.
    It uses Stage2_metadata to find the slice of partial results
    it needs to reduce for its assigned (batch, head) pair.
    """
    cur_batch = tl.program_id(0)
    cur_head_group = tl.program_id(1)

    num_splits_real_value = tl.load(num_splits_real + cur_batch)

    if num_splits_real_value == 1:
        return  # Nothing to do

    flat_offset_value = tl.load(flat_offset + cur_batch)
    offs_head = cur_head_group * HEADS_PER_BLOCK + tl.arange(0, HEADS_PER_BLOCK)
    offs_logic = flat_offset_value * q_head_num + offs_head * num_splits_real_value
    mask_head = offs_head < q_head_num

    e_max = tl.full([HEADS_PER_BLOCK], -float("inf"), dtype=tl.float32)  # More stable
    e_sum = tl.zeros([HEADS_PER_BLOCK], dtype=tl.float32)
    acc = tl.zeros([HEADS_PER_BLOCK, BLOCK_DV], dtype=tl.float32)
    output_dtype: tl.constexpr = o.dtype.element_ty

    head_end = tl.minimum(q_head_num, (cur_head_group + 1) * HEADS_PER_BLOCK)
    # Batch, head, split, Lv
    desc_a = tl.make_tensor_descriptor(
        Flat_Att_Out
        + flat_offset_value * q_head_num * Lv
        + cur_head_group * HEADS_PER_BLOCK * num_splits_real_value * Lv,
        shape=(head_end - cur_head_group * HEADS_PER_BLOCK, num_splits_real_value * Lv),
        strides=(num_splits_real_value * Lv, 1),
        block_shape=(HEADS_PER_BLOCK, BLOCK_DV),
    )

    Flat_Att_Single_c = Flat_Att_Single + offs_logic
    for i in range(0, num_splits_real_value):
        tv = desc_a.load([0, i * Lv]).to(tl.float32)
        tlogic = tl.load(Flat_Att_Single_c + i, mask=mask_head)

        n_e_max = tl.maximum(tlogic, e_max)
        old_scale = tl.math.exp2(e_max - n_e_max)
        acc *= old_scale[:, None]
        exp_logic = tl.math.exp2(tlogic - n_e_max)
        acc += exp_logic[:, None] * tv
        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max

    # Map batch (seq) to token location
    cur_q_start = tl.load(cu_seqlens_q + cur_batch)
    # Store final result
    desc_o = tl.make_tensor_descriptor(
        o + cur_q_start * stride_obs + cur_head_group * HEADS_PER_BLOCK * stride_oh,
        shape=(head_end - cur_head_group * HEADS_PER_BLOCK, BLOCK_DV),
        strides=(stride_oh, 1),
        block_shape=(HEADS_PER_BLOCK, BLOCK_DV),
    )
    desc_o.store([0, 0], (acc / e_sum[:, None]).to(output_dtype))


try:
    UNIT_SCALE = torch.tensor(1.0, device="xpu")
except Exception:
    logger.debug(
        "XPU device not available or failed to create XPU tensor, falling back to CPU."
    )
    UNIT_SCALE = torch.tensor(1.0, device=torch.device("cpu"))

# May need to be increased for more powerful GPUs
PROMPT_NUM_BLOCKS = 512
# Persistent buffers
KEY_SCRATCH = None
VALUE_SCRATCH = None
FLAT_ATTN_LOGITS = None
FLAT_ATTN_SINGLE = None
NUM_SPLITS_REAL = None


def initialize_triton_attention_buffers(
    max_num_seqs: int,
    num_heads_q: int,
    head_dim: int,
    kv_cache_shape: tuple,
    kv_cache_dtype: torch.dtype,
    query_dtype: torch.dtype,
    device: torch.device,
) -> None:
    """
    Pre-allocate attention buffers to maximum size to avoid OOM during execution.

    This should be called once during model initialization, after KV cache is allocated.

    Args:
        max_num_seqs: Maximum number of sequences in a batch
        num_heads_q: Number of query heads
        head_dim: Dimension of each attention head
        kv_cache_shape: Shape of the KV cache tensors
        kv_cache_dtype: Data type of KV cache
        query_dtype: Data type of query tensors
        device: Device to allocate buffers on
    """
    global KEY_SCRATCH, VALUE_SCRATCH
    global FLAT_ATTN_LOGITS, FLAT_ATTN_SINGLE, NUM_SPLITS_REAL

    # Allocate FLAT_ATTN_LOGITS and FLAT_ATTN_SINGLE
    max_logits_size = kv_cache_shape[0] * num_heads_q
    BLOCK_DV = head_dim  # Assuming BLOCK_DV == head_dim

    FLAT_ATTN_LOGITS = torch.empty(
        max_logits_size, BLOCK_DV, dtype=query_dtype, device=device
    )
    FLAT_ATTN_SINGLE = torch.empty(max_logits_size, dtype=torch.float32, device=device)

    logger.info(
        "Allocated FLAT_ATTN_LOGITS: %s (%.2f GB)",
        FLAT_ATTN_LOGITS.shape,
        FLAT_ATTN_LOGITS.numel() * FLAT_ATTN_LOGITS.element_size() / 1e9,
    )

    # Allocate NUM_SPLITS_REAL
    NUM_SPLITS_REAL = torch.empty(max_num_seqs, dtype=torch.int32, device=device)

    # Allocate KEY_SCRATCH and VALUE_SCRATCH if needed for dtype conversion
    if kv_cache_dtype != query_dtype:
        KEY_SCRATCH = torch.zeros(kv_cache_shape, dtype=query_dtype, device=device)
        VALUE_SCRATCH = torch.zeros(kv_cache_shape, dtype=query_dtype, device=device)
        logger.info(
            "Allocated KEY/VALUE_SCRATCH: %s (%.2f GB total)",
            KEY_SCRATCH.shape,
            2 * KEY_SCRATCH.numel() * KEY_SCRATCH.element_size() / 1e9,
        )

    logger.info(
        "Triton attention buffers initialized: max_num_seqs=%d, num_pages=%d",
        max_num_seqs,
        kv_cache_shape[0],
    )


@torch.compile
def calculate_flat_offset(seqused_k: torch.Tensor, page_size: int):
    """
    Compute a flat page-offset vector from per-sequence key usage.
    Given the number of used keys per sequence and the page size, this function
    computes how many pages are needed for each sequence and returns the
    cumulative sum of these page counts. The resulting tensor can be used as
    a "flat offset" index when laying out or accessing paged KV-cache storage
    across multiple sequences.
    Args:
        seqused_k: 1D tensor where each element is the number of used keys
            (e.g., tokens) for a sequence.
        page_size: Number of keys stored in a single page.
    Returns:
        A 1D integer tensor of length ``seqused_k.numel() + 1`` containing the
        cumulative number of pages. The first element is 0, and each subsequent
        element gives the starting page index for the corresponding sequence in
        a flattened paged layout.
    """
    k_pages = (seqused_k + page_size - 1) // page_size
    k_pages = torch.cat(
        [torch.zeros(1, device=seqused_k.device, dtype=k_pages.dtype), k_pages], dim=0
    )
    return torch.cumsum(k_pages, dim=0, dtype=k_pages.dtype)


#@torch.compile
def flash_attn_varlen_func_triton(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    is_causal: bool,
    block_table: torch.Tensor,
    alibi_slopes: torch.Tensor = None,  # Not implemented
    sink: torch.Tensor = None,
    kv_cache_dtype: torch.dtype = torch.float16,
    window_size_left: int = 0,
    window_size_right: int = 0,  # Not implemented
    k_scale: torch.Tensor = UNIT_SCALE,
    v_scale: torch.Tensor = UNIT_SCALE,
):
    global KEY_SCRATCH
    global VALUE_SCRATCH
    global FLAT_ATTN_LOGITS
    global FLAT_ATTN_SINGLE
    global NUM_SPLITS_REAL

    batch_size, num_heads_q, _ = query.shape
    _, PAGE_SIZE, num_heads_kv, BLOCK_DMODEL = key_cache.shape
    _, _, _, BLOCK_DV = value_cache.shape
    kv_group_num = num_heads_q // num_heads_kv
    # Split k for decode
    BLOCKS_PER_SPLIT = 4 if key_cache.dtype.itemsize == 2 else 2  # Can be tuned
    BLOCK_Q = max(64, kv_group_num)  # Can be tuned
    SPLIT_SIZE = PAGE_SIZE * BLOCKS_PER_SPLIT
    num_seqs = cu_seqlens_q.shape[0] - 1

    # If num_decodes=16384, there are no prompt requests
    # Not using a global constant as triton doesn't like that
    num_decodes = torch.full((1,), 16384, dtype=torch.int32, device=query.device)

    num_splits_q = int((max_seqlen_q + BLOCK_Q - 1) // BLOCK_Q)

    assert window_size_right <= 0, "window_size_right > 0 is not supported yet"

    window_size = window_size_left + 1
    num_splits_k = int((max_seqlen_k + SPLIT_SIZE - 1) // SPLIT_SIZE)

    if window_size > 0:
        # Pad to page
        # Need to pad both in the front and back to pages
        # Worst case:
        #   p0   p1   p2
        # |----|----|----|
        # |---w|wwwq|q---|
        # window size is 4, q size is 2, but 3 pages are needed due to alignment
        num_splits_k_page = (
            window_size + max_seqlen_q + PAGE_SIZE - 1 + PAGE_SIZE - 1
        ) // PAGE_SIZE
        # Pad to split
        num_splits_k = min(
            num_splits_k, (num_splits_k_page * PAGE_SIZE + SPLIT_SIZE - 1) // SPLIT_SIZE
        )

    decode_active_tasks = num_seqs * num_heads_kv * num_splits_k

    # Separate logits and e_max allows block_2d loads and stores of logits
    # Check if buffers are pre-allocated, otherwise allocate dynamically
    # logits_size = num_kv_pages * num_heads_q * BLOCK_DV
    # Decode will not use more than the entire kv cache for intermediate logits
    logits_size = key_cache.shape[0] * num_heads_q
    if (
        FLAT_ATTN_LOGITS is None
        or FLAT_ATTN_LOGITS.numel() < logits_size * BLOCK_DV
        or FLAT_ATTN_LOGITS.dtype != output.dtype
    ):
        if FLAT_ATTN_LOGITS is not None:
            logger.info(
                "FLAT_ATTN_LOGITS buffer size insufficient or dtype mismatch. "
                "Required: %d, Available: %d. "
                "Consider calling initialize_triton_attention_buffers()"
                " during model init.",
                logits_size * BLOCK_DV,
                FLAT_ATTN_LOGITS.numel(),
            )
        FLAT_ATTN_LOGITS = torch.empty(
            logits_size, BLOCK_DV, dtype=output.dtype, device=query.device
        )
        FLAT_ATTN_SINGLE = torch.empty(
            logits_size,
            dtype=torch.float32,
            device=query.device,
        )
    if NUM_SPLITS_REAL is None or NUM_SPLITS_REAL.size(0) < num_seqs:
        if NUM_SPLITS_REAL is not None:
            logger.info(
                "NUM_SPLITS_REAL buffer size insufficient. "
                "Required: %d, Available: %d. "
                "Consider calling initialize_triton_attention_buffers()"
                " during model init.",
                num_seqs,
                NUM_SPLITS_REAL.size(0),
            )
        NUM_SPLITS_REAL = torch.empty(num_seqs, dtype=torch.int32, device=query.device)

    HEADS_PER_BLOCK = 8
    num_head_blocks = (num_heads_q + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK

    needs_dequant = key_cache.dtype != query.dtype or value_cache.dtype != query.dtype
    if needs_dequant:
        # The dequant pass indexes scratch by physical page number
        # (kv_page_number_scalar * PAGE_SIZE), so KEY/VALUE_SCRATCH MUST match
        # key_cache/value_cache shape exactly. Re-allocate on any mismatch of
        # shape OR dtype (not just None/dtype) — otherwise a wrongly-sized
        # pre-allocated buffer causes an out-of-bounds write and DEVICE_LOST.
        # The pre-allocation in initialize_triton_attention_buffers() is only
        # an optimization to avoid this realloc on the hot path.
        if (
            KEY_SCRATCH is None
            or KEY_SCRATCH.dtype != query.dtype
            or KEY_SCRATCH.shape != key_cache.shape
        ):
            if KEY_SCRATCH is not None:
                logger.info(
                    "KEY_SCRATCH buffer mismatch (have shape=%s dtype=%s, "
                    "need shape=%s dtype=%s); reallocating. Consider passing the "
                    "correct kv_cache_shape to initialize_triton_attention_buffers().",
                    tuple(KEY_SCRATCH.shape),
                    KEY_SCRATCH.dtype,
                    tuple(key_cache.shape),
                    query.dtype,
                )
            KEY_SCRATCH = torch.zeros(
                key_cache.shape, dtype=query.dtype, device=query.device
            )
        if (
            VALUE_SCRATCH is None
            or VALUE_SCRATCH.dtype != query.dtype
            or VALUE_SCRATCH.shape != value_cache.shape
        ):
            VALUE_SCRATCH = torch.zeros(
                value_cache.shape, dtype=query.dtype, device=query.device
            )

    # Decode
    # Not using num_decodes to launch kernel in case there are prompt requests
    # with 1 token.
    # E.g.: cu_seqlens_q = [0,1,3,4], num_decodes=1, but the 2nd prompt request
    # will be processed as decode since it only has 1 token
    if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        key_scratch = KEY_SCRATCH
        value_scratch = VALUE_SCRATCH
    else:
        key_scratch = key_cache
        value_scratch = value_cache

    flat_offset = calculate_flat_offset(seqused_k, PAGE_SIZE)
    prompt_counter = torch.full(
        (1,), PROMPT_NUM_BLOCKS, dtype=torch.int32, device=query.device
    )

    num_warps_d = max(1, triton.next_power_of_2(kv_group_num) // 8)
    _fwd_grouped_kernel_decode[(decode_active_tasks,)](
        output,
        query,
        key_cache,
        value_cache,
        key_scratch,
        value_scratch,
        block_table,
        cu_seqlens_q,
        seqused_k,
        flat_offset,
        FLAT_ATTN_LOGITS,
        FLAT_ATTN_SINGLE,
        num_splits_k,
        NUM_SPLITS_REAL,
        softmax_scale,
        num_decodes,
        decode_active_tasks,
        block_table.stride(0),
        query.stride(0),
        query.stride(1),
        key_cache.stride(-3),
        key_cache.stride(-2),
        value_cache.stride(-3),
        value_cache.stride(-2),
        key_cache.stride(0),
        value_cache.stride(0),
        k_scale,
        v_scale,
        sink=sink,
        USE_SINKS=(sink is not None),
        window_size_left=window_size_left,
        kv_group_num=kv_group_num,
        q_block_head=triton.next_power_of_2(kv_group_num),
        num_heads_kv=num_heads_kv,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_Q=1,
        BLOCKS_PER_SPLIT=BLOCKS_PER_SPLIT,
        PAGE_SIZE=PAGE_SIZE,
        Lk=BLOCK_DMODEL,
        Lv=BLOCK_DV,
        # Prior autotune parameters
        BLOCK_N=16,
        BLOCK_DMODEL_SLICE=BLOCK_DMODEL // 16,  # Issue handling using autotune
        num_warps=num_warps_d,
        num_stages=1,
        # NOTE: grf_mode="256" (large-GRF launch) overruns the work-group
        # resource budget on the ww25 torch-2.12 level-zero submit path
        # (UR_RESULT_ERROR_OUT_OF_RESOURCES at launch). Default GRF works.
    )

    if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        key_cache_p = KEY_SCRATCH
        value_cache_p = VALUE_SCRATCH
    else:
        key_cache_p = key_cache
        value_cache_p = value_cache

    # Prompt
    if max_seqlen_q > 1:
        _fwd_grouped_kernel_prompt[(PROMPT_NUM_BLOCKS,)](
            output,
            query,
            key_cache_p,
            value_cache_p,
            block_table,
            cu_seqlens_q,
            seqused_k,
            FLAT_ATTN_LOGITS,
            FLAT_ATTN_SINGLE,
            num_splits_q,
            NUM_SPLITS_REAL,
            softmax_scale,
            num_decodes,
            num_seqs,
            prompt_counter,
            block_table.stride(0),
            query.stride(0),
            query.stride(1),
            key_cache.stride(-3),
            key_cache.stride(-2),
            value_cache.stride(-3),
            value_cache.stride(-2),
            # Block stride of the buffer this kernel actually reads: contiguous
            # dequant scratch for fp8, interleaved real cache for bf16.
            key_cache_p.stride(0),
            value_cache_p.stride(0),
            k_scale,
            v_scale,
            sink=sink,
            USE_SINKS=(sink is not None),
            window_size_left=window_size_left,
            kv_group_num=kv_group_num,
            num_heads_kv=num_heads_kv,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DV=BLOCK_DV,
            BLOCK_Q=BLOCK_Q,
            PAGE_SIZE=PAGE_SIZE,
            Lk=BLOCK_DMODEL,
            Lv=BLOCK_DV,
            # Prior autotune parameters
            BLOCK_N=32,
            num_warps=BLOCK_Q // 8,
            num_stages=2,
            # NOTE: grf_mode="256" removed — see decode-kernel note above.
        )
    # Reduction
    _fwd_kernel_stage2[(num_seqs, num_head_blocks)](
        FLAT_ATTN_LOGITS,
        FLAT_ATTN_SINGLE,
        output,
        flat_offset,
        num_splits_k,
        NUM_SPLITS_REAL,
        cu_seqlens_q,
        num_heads_q,
        output.stride(0),
        output.stride(1),
        HEADS_PER_BLOCK=HEADS_PER_BLOCK,
        BLOCK_DV=BLOCK_DV,
        Lv=BLOCK_DV,
        # Prior autotune parameters
        num_warps=4,
        num_stages=1,
    )




if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake

if hasattr(torch.ops._xpu_C, "fp8_gemm"):

    @register_fake("_xpu_C::fp8_gemm")
    def _fp8_gemm_fake(
        q_input: torch.Tensor,
        q_weight: torch.Tensor,
        out_dtype: torch.dtype,
        input_scales: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_2d = q_input.view(-1, q_input.shape[-1])
        M = input_2d.size(0)
        N = q_weight.size(1)
        return torch.empty((M, N), dtype=out_dtype, device=q_input.device)


if hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16"):

    @register_fake("_xpu_C::fp8_gemm_w8a16")
    def _fp8_gemm_w8a16_fake(
        input: torch.Tensor,
        q_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_2d = input.view(-1, input.shape[-1])
        M = input_2d.size(0)
        N = q_weight.size(1)
        return torch.empty((M, N), dtype=input.dtype, device=input.device)


if hasattr(torch.ops._xpu_C, "int4_gemm_w4a8"):

    @register_fake("_xpu_C::int4_gemm_w4a8")
    def _int4_gemm_w4a8_fake(
        input: torch.Tensor,
        input_scales: torch.Tensor,
        input_zero_points: torch.Tensor,
        q_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zp: torch.Tensor,
        group_size: int,
        g_idx: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_2d = input.view(-1, input.shape[-1])
        M = input_2d.size(0)
        N = q_weight.size(1)
        return torch.empty((M, N), dtype=torch.float16, device=input.device)


if hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"):

    @register_fake("_xpu_C::int4_gemm_w4a16")
    def _int4_gemm_w4a16_fake(
        input: torch.Tensor,
        q_weight: torch.Tensor,
        bias: torch.Tensor | None,
        weight_scale: torch.Tensor,
        qzeros: torch.Tensor,
        group_size: int,
        group_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_2d = input.view(-1, input.shape[-1])
        M = input_2d.size(0)
        N = q_weight.size(1)
        return torch.empty((M, N), dtype=input.dtype, device=input.device)


def _gdn_attention_core_xpu_impl(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    projected_states_qkvz: torch.Tensor,
    projected_states_ba: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op wrapping the XPU SYCL GDN kernel for torch.compile."""
    from vllm.forward_context import get_forward_context
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    attn_metadata_raw = forward_context.attn_metadata

    if attn_metadata_raw is None:
        return

    assert isinstance(attn_metadata_raw, dict)
    attn_metadata = attn_metadata_raw[self.prefix]
    assert isinstance(attn_metadata, GDNAttentionMetadata)

    num_actual_tokens = attn_metadata.num_actual_tokens
    num_accepted_tokens = attn_metadata.num_accepted_tokens

    num_prefills = attn_metadata.num_prefills
    num_decodes = attn_metadata.num_decodes
    num_spec_decodes = attn_metadata.num_spec_decodes

    has_initial_state = attn_metadata.has_initial_state

    non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
    non_spec_token_indx = attn_metadata.non_spec_token_indx
    non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
    non_spec_state_indices_tensor = (
        non_spec_state_indices_tensor.contiguous()
        if non_spec_state_indices_tensor is not None
        else None
    )

    spec_query_start_loc = attn_metadata.spec_query_start_loc
    spec_token_indx = attn_metadata.spec_token_indx
    spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501

    spec_sequence_masks = attn_metadata.spec_sequence_masks
    if spec_sequence_masks is not None:
        if non_spec_token_indx is not None:
            non_spec_token_indx = non_spec_token_indx.to(torch.int32)
        if spec_token_indx is not None:
            spec_token_indx = spec_token_indx.to(torch.int32)

    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )

    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
        z,
        projected_states_qkvz,
        projected_states_ba,
        self.num_k_heads,
        self.num_v_heads,
        self.head_k_dim,
        self.head_v_dim,
        conv_state=self.kv_cache[0],
        ssm_state=self.kv_cache[1],
        conv_weights=conv_weights,
        conv_bias=self.conv1d.bias,
        activation=self.activation,
        A_log=self.A_log,
        dt_bias=self.dt_bias,
        num_prefills=num_prefills,  # type: ignore[attr-defined]
        num_decodes=num_decodes,  # type: ignore[attr-defined]
        num_spec_decodes=num_spec_decodes,  # type: ignore[attr-defined]
        has_initial_state=has_initial_state,  # type: ignore[attr-defined]
        non_spec_query_start_loc=non_spec_query_start_loc,  # type: ignore[attr-defined]
        non_spec_token_indx=non_spec_token_indx,  # type: ignore[attr-defined]
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,  # type: ignore[attr-defined]
        spec_query_start_loc=spec_query_start_loc,  # type: ignore[attr-defined]
        spec_token_indx=spec_token_indx,  # type: ignore[attr-defined]
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,  # type: ignore[attr-defined]
        num_actual_tokens=num_actual_tokens,  # type: ignore[attr-defined]
        tp_size=self.tp_size,
        reorder_input=not self.gqa_interleaved_layout,
    )


def _gdn_attention_core_xpu_fake(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    projected_states_qkvz: torch.Tensor,
    projected_states_ba: torch.Tensor,
    layer_name: str,
) -> None:
    return


def _xpu_ops_deepseek_scaling_rope_impl(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    offsets: torch.Tensor | None,
    cos_sin_cache: torch.Tensor | None,
    rotary_dim: int,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert key is not None
    return torch.ops._xpu_C.deepseek_scaling_rope(
        positions, query, key, offsets, cos_sin_cache, rotary_dim, is_neox_style
    )


def _xpu_ops_deepseek_scaling_rope_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    offsets: torch.Tensor | None,
    cos_sin_cache: torch.Tensor | None,
    rotary_dim: int,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return query, key


def _xpu_fp8_mqa_logits_impl(
    q: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._xpu_C.fp8_mqa_logits(
        q,
        k_quant,
        k_scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )


def _xpu_fp8_mqa_logits_fake(
    q: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (q.shape[0], k_quant.shape[0]),
        dtype=torch.float32,
        device=q.device,
    )


def _xpu_fp8_paged_mqa_logits_impl(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    return torch.ops._xpu_C.fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
    )


def _xpu_fp8_paged_mqa_logits_fake(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    batch_size, next_n = q.shape[:2]
    return torch.empty(
        (batch_size * next_n, max_model_len),
        dtype=torch.float32,
        device=q.device,
    )


def _topk_topp_sample_impl(
    random_sampled: torch.Tensor,
    logits_to_return: torch.Tensor | None,
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    logprobs_mode: str,
    seeds: torch.Tensor | None,
    lambda_: float = 1.0,
) -> None:
    torch.ops._xpu_C.topk_topp_sampler(
        random_sampled, logits_to_return, logits, k, p, logprobs_mode, seeds, lambda_
    )
    return


def _topk_topp_sample_fake(
    random_sampled: torch.Tensor,
    logits_to_return: torch.Tensor | None,
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    logprobs_mode: str,
    seeds: torch.Tensor | None,
    lambda_: float = 1.0,
) -> None:
    return


def _xpu_mxfp8_quantize_impl(
    x: torch.Tensor, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    MXFP8_BLOCK_SIZE = 32
    assert x.shape[-1] % MXFP8_BLOCK_SIZE == 0
    if dtype is not None:
        assert dtype in (torch.float8_e4m3fn, torch.float8_e5m2), (
            f"Unsupported dtype for xpu_mxfp8_quantize: {dtype}. "
            f"Expected torch.float8_e4m3fn or torch.float8_e5m2."
        )
    else:
        dtype = current_platform.fp8_dtype()

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max
    eps = 1e-10

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    shape = x.shape[:-1] + (x.shape[-1] // MXFP8_BLOCK_SIZE,)
    x_s = torch.empty(shape, device=x.device, dtype=torch.float32)
    torch.ops._C.per_token_group_fp8_quant(
        x,
        x_q,
        x_s,
        MXFP8_BLOCK_SIZE,
        eps,
        fp8_min,
        fp8_max,
        True,
        False,
        False,  # dummy_is_scale_transposed, dummy_is_tma_aligned
    )
    x_s = x_s.to(torch.float8_e8m0fnu)
    return x_q, x_s


def _xpu_mxfp8_quantize_fake(
    x: torch.Tensor, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if dtype is None:
        dtype = current_platform.fp8_dtype()

    MXFP8_BLOCK_SIZE = 32

    shape = x.shape[:-1] + (x.shape[-1] // MXFP8_BLOCK_SIZE,)
    x_s = torch.zeros(shape, device=x.device, dtype=torch.float32)

    return x.to(dtype), x_s.to(torch.float8_e8m0fnu)


def _xpu_mxfp4_quantize_impl(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    MXFP4_BLOCK_SIZE = 32
    eps = 1e-10
    assert x.ndim == 2, "input must be 2-D"
    assert x.shape[-1] % MXFP4_BLOCK_SIZE == 0, (
        f"last dimension {x.shape[-1]} must be divisible by group_size "
        f"{MXFP4_BLOCK_SIZE}"
    )
    assert x.is_contiguous(), "input groups must be contiguous"

    M, N = x.shape

    # Packed FP4 output: two nibbles per byte
    x_q = torch.empty(M, N // 2, device=x.device, dtype=torch.uint8)
    x_s = torch.empty(M, N // MXFP4_BLOCK_SIZE, device=x.device, dtype=torch.float32)

    torch.ops._C.per_token_group_quant_mxfp4(x, x_q, x_s, MXFP4_BLOCK_SIZE, eps)

    x_q = x_q.view(torch.float4_e2m1fn_x2)
    x_s = x_s.to(dtype=torch.float8_e8m0fnu, memory_format=torch.preserve_format)
    return x_q, x_s


def _xpu_mxfp4_quantize_fake(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    MXFP4_BLOCK_SIZE = 32
    M, N = x.shape

    # Packed FP4 output: two nibbles per byte
    x_q = torch.empty(M, N // 2, device=x.device, dtype=torch.uint8)
    x_s = torch.empty(M, N // MXFP4_BLOCK_SIZE, device=x.device, dtype=torch.float32)

    x_q = x_q.view(torch.float4_e2m1fn_x2)
    x_s = x_s.to(dtype=torch.float8_e8m0fnu, memory_format=torch.preserve_format)
    return x_q, x_s


@triton.jit
def _softplus(x):
    return tl.where(x <= 20.0, tl.math.log(tl.math.exp(x) + 1.0), x)


@triton.jit
def _selective_scan_fwd_kernel(
    # Pointers to input tensors
    u_ptr,
    delta_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    delta_bias_ptr,
    # Pointers to output tensors (out aliases delta, out_z aliases z)
    out_ptr,
    out_z_ptr,
    # SSM states
    ssm_states_ptr,
    # Optional pointers
    query_start_loc_ptr,
    cache_indices_ptr,
    has_initial_state_ptr,
    # APC pointers
    block_idx_first_ptr,
    block_idx_last_ptr,
    initial_state_idx_ptr,
    cu_chunk_seqlen_ptr,
    last_chunk_indices_ptr,
    # Dimensions
    batch: tl.int32,
    dim: tl.int32,
    seqlen: tl.int32,
    dstate: tl.int32,
    n_groups: tl.int32,
    dim_ngroups_ratio: tl.int32,
    # Strides for u (and out, since out = delta which has same layout)
    u_batch_stride: tl.int64,
    u_d_stride: tl.int64,
    # Strides for delta
    delta_batch_stride: tl.int64,
    delta_d_stride: tl.int64,
    # Strides for A
    A_d_stride: tl.int64,
    A_dstate_stride: tl.int64,
    # Strides for B
    B_batch_stride: tl.int64,
    B_group_stride: tl.int64,
    B_dstate_stride: tl.int64,
    # Strides for C
    C_batch_stride: tl.int64,
    C_group_stride: tl.int64,
    C_dstate_stride: tl.int64,
    # Strides for z
    z_batch_stride: tl.int64,
    z_d_stride: tl.int64,
    # Strides for out
    out_batch_stride: tl.int64,
    out_d_stride: tl.int64,
    # Strides for out_z
    out_z_batch_stride: tl.int64,
    out_z_d_stride: tl.int64,
    # Strides for ssm_states
    ssm_batch_stride: tl.int64,
    ssm_dim_stride: tl.int64,
    ssm_dstate_stride: tl.int64,
    # Cache strides
    cache_indices_stride: tl.int64,
    # Scalar params
    null_block_id: tl.int64,
    block_size: tl.int32,
    # Compile-time constants
    delta_softplus: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_DELTA_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_CACHE_INDICES: tl.constexpr,
    CACHE_ENABLED: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    dim_idx = tl.program_id(1)
    group_idx = dim_idx // dim_ngroups_ratio

    # Determine sequence boundaries
    if IS_VARLEN:
        seq_start = tl.load(query_start_loc_ptr + batch_idx).to(tl.int32)
        seq_end = tl.load(query_start_loc_ptr + batch_idx + 1).to(tl.int32)
        actual_seqlen = seq_end - seq_start
    else:
        seq_start = 0
        actual_seqlen = seqlen

    # Determine cache index for ssm_states
    if CACHE_ENABLED:
        init_state_idx = tl.load(initial_state_idx_ptr + batch_idx).to(tl.int32)
        load_cache_slot = tl.load(
            cache_indices_ptr + batch_idx * cache_indices_stride + init_state_idx
        ).to(tl.int64)
        if load_cache_slot == null_block_id:
            return
    elif HAS_CACHE_INDICES:
        cache_index = tl.load(cache_indices_ptr + batch_idx).to(tl.int64)
        if cache_index == null_block_id:
            return
        load_cache_slot = cache_index
    else:
        load_cache_slot = batch_idx.to(tl.int64)

    # Load D value
    D_val = 0.0
    if HAS_D:
        D_val = tl.load(D_ptr + dim_idx).to(tl.float32)

    # Load delta_bias value
    delta_bias_val = 0.0
    if HAS_DELTA_BIAS:
        delta_bias_val = tl.load(delta_bias_ptr + dim_idx).to(tl.float32)

    # Load A values for this dim - shape (dstate,)
    dstate_offs = tl.arange(0, BLOCK_DSTATE)
    dstate_mask = dstate_offs < dstate
    A_vals = tl.load(
        A_ptr + dim_idx * A_d_stride + dstate_offs * A_dstate_stride,
        mask=dstate_mask,
        other=0.0,
    ).to(tl.float32)

    # Initialize state vector
    state = tl.zeros((BLOCK_DSTATE,), dtype=tl.float32)

    # Load initial state if available
    has_init = False
    if has_initial_state_ptr is not None:
        has_init = tl.load(has_initial_state_ptr + batch_idx)
    if has_init:
        state = tl.load(
            ssm_states_ptr
            + load_cache_slot * ssm_batch_stride
            + dim_idx * ssm_dim_stride
            + dstate_offs * ssm_dstate_stride,
            mask=dstate_mask,
            other=0.0,
        ).to(tl.float32)

    # Compute base addresses for u and delta
    if IS_VARLEN:
        u_base = u_ptr + dim_idx * u_d_stride + seq_start * u_batch_stride
        delta_base = (
            delta_ptr + dim_idx * delta_d_stride + seq_start * delta_batch_stride
        )
        out_base = out_ptr + dim_idx * out_d_stride + seq_start * out_batch_stride
        B_base = B_ptr + group_idx * B_group_stride + seq_start * B_batch_stride
        C_base = C_ptr + group_idx * C_group_stride + seq_start * C_batch_stride
    else:
        u_base = u_ptr + batch_idx * u_batch_stride + dim_idx * u_d_stride
        delta_base = (
            delta_ptr + batch_idx * delta_batch_stride + dim_idx * delta_d_stride
        )
        out_base = out_ptr + batch_idx * out_batch_stride + dim_idx * out_d_stride
        B_base = B_ptr + batch_idx * B_batch_stride + group_idx * B_group_stride
        C_base = C_ptr + batch_idx * C_batch_stride + group_idx * C_group_stride

    if HAS_Z:
        if IS_VARLEN:
            z_base = z_ptr + dim_idx * z_d_stride + seq_start * z_batch_stride
            out_z_base = (
                out_z_ptr + dim_idx * out_z_d_stride + seq_start * out_z_batch_stride
            )
        else:
            z_base = z_ptr + batch_idx * z_batch_stride + dim_idx * z_d_stride
            out_z_base = (
                out_z_ptr + batch_idx * out_z_batch_stride + dim_idx * out_z_d_stride
            )

    # Determine chunk boundaries for APC mode
    if CACHE_ENABLED:
        last_chunk_idx = tl.load(last_chunk_indices_ptr + batch_idx).to(tl.int32)
        if batch_idx == 0:
            first_chunk_idx = 0
        else:
            first_chunk_idx = (
                tl.load(last_chunk_indices_ptr + batch_idx - 1).to(tl.int32) + 1
            )
        n_chunks = last_chunk_idx - first_chunk_idx + 1
        first_chunk_tokens = tl.load(cu_chunk_seqlen_ptr + first_chunk_idx + 1).to(
            tl.int32
        ) - tl.load(cu_chunk_seqlen_ptr + first_chunk_idx).to(tl.int32)
        block_idx_first = tl.load(block_idx_first_ptr + batch_idx).to(tl.int32)
        chunk_start_offset = 0
        if n_chunks > 1 and first_chunk_tokens < block_size:
            chunk_start_offset = block_size - first_chunk_tokens
        current_position = block_idx_first * block_size + chunk_start_offset
    else:
        n_chunks = 1
        first_chunk_idx = 0

    # Sequential scan over the sequence
    tokens_processed = 0
    for chunk in range(0, n_chunks if CACHE_ENABLED else 1):
        if CACHE_ENABLED:
            chunk_tokens = tl.load(
                cu_chunk_seqlen_ptr + first_chunk_idx + chunk + 1
            ).to(tl.int32) - tl.load(cu_chunk_seqlen_ptr + first_chunk_idx + chunk).to(
                tl.int32
            )
        else:
            chunk_tokens = actual_seqlen

        for local_pos in range(chunk_tokens):
            pos = tokens_processed + local_pos
            # Load u value
            u_val = tl.load(u_base + pos).to(tl.float32)

            # Load delta value
            delta_val = tl.load(delta_base + pos).to(tl.float32)

            # Apply delta bias
            if HAS_DELTA_BIAS:
                delta_val = delta_val + delta_bias_val

            # Apply softplus
            if delta_softplus:
                delta_val = _softplus(delta_val)

            delta_u = delta_val * u_val

            # Compute dA = exp(delta * A) for all dstate elements
            dA = tl.exp(delta_val * A_vals)

            # Load B values for this position
            B_vals = tl.load(
                B_base + dstate_offs * B_dstate_stride + pos,
                mask=dstate_mask,
                other=0.0,
            ).to(tl.float32)

            # Load C values for this position
            C_vals = tl.load(
                C_base + dstate_offs * C_dstate_stride + pos,
                mask=dstate_mask,
                other=0.0,
            ).to(tl.float32)

            # Update state: state = dA * state + delta * u * B
            state = dA * state + delta_u * B_vals

            # Compute output: out = sum(state * C) + D * u
            out_val = tl.sum(state * C_vals, axis=0)
            if HAS_D:
                out_val = out_val + D_val * u_val

            # Store output
            tl.store(out_base + pos, out_val.to(out_ptr.dtype.element_ty))

            if HAS_Z:
                z_val = tl.load(z_base + pos).to(tl.float32)
                out_z_val = out_val * z_val / (1.0 + tl.exp(-z_val))
                tl.store(
                    out_z_base + pos,
                    out_z_val.to(out_z_ptr.dtype.element_ty),
                )

        tokens_processed += chunk_tokens

        # Store intermediate state for APC mode
        if CACHE_ENABLED:
            if chunk == n_chunks - 1:
                store_slot = tl.load(
                    cache_indices_ptr
                    + batch_idx * cache_indices_stride
                    + tl.load(block_idx_last_ptr + batch_idx).to(tl.int32)
                ).to(tl.int64)
            else:
                block_idx_done = (current_position + chunk_tokens - 1) // block_size
                store_slot = tl.load(
                    cache_indices_ptr
                    + batch_idx * cache_indices_stride
                    + block_idx_done
                ).to(tl.int64)

            tl.store(
                ssm_states_ptr
                + store_slot * ssm_batch_stride
                + dim_idx * ssm_dim_stride
                + dstate_offs * ssm_dstate_stride,
                state.to(ssm_states_ptr.dtype.element_ty),
                mask=dstate_mask,
            )
            current_position += chunk_tokens

    # Store final state for non-APC mode
    if not CACHE_ENABLED:
        tl.store(
            ssm_states_ptr
            + load_cache_slot * ssm_batch_stride
            + dim_idx * ssm_dim_stride
            + dstate_offs * ssm_dstate_stride,
            state.to(ssm_states_ptr.dtype.element_ty),
            mask=dstate_mask,
        )


# Global flag to ensure ops are registered only once
_OPS_REGISTERED = False


class xpu_ops:
    @staticmethod
    @torch.compile
    def dynamic_per_token_int8_quant_ref(
        input: torch.Tensor, use_sym_quant: bool, bits: int
    ):
        original_sizes = input.size()
        # view is not safe in torch.compile if input is not contiguous
        input = input.reshape(
            -1, original_sizes[-1]
        )  # Flatten except for the last dimension
        qmin = -(2 ** (bits - 1)) if use_sym_quant else 0
        qmax = 2 ** (bits - 1) - 1 if use_sym_quant else 2**bits - 1
        min_val = torch.min(input, dim=-1)[0].to(dtype=torch.float32).unsqueeze(-1)
        max_val = torch.max(input, dim=-1)[0].to(dtype=torch.float32).unsqueeze(-1)
        if use_sym_quant:
            scale = (
                torch.maximum(torch.abs(min_val), torch.abs(max_val)) / qmax
            ).clamp(min=1e-5)
            zero_point = torch.zeros_like(scale).to(dtype=torch.int32)
        else:
            scale = ((max_val - min_val) / qmax).clamp(min=1e-5)
            zero_point = -1 * torch.round(min_val / scale).to(dtype=torch.int32)
        scale = scale.to(dtype=input.dtype)
        quantized = torch.clamp(
            torch.round(input / scale.to(dtype=torch.float32) + zero_point),
            qmin,
            qmax,
        ).to(dtype=torch.int8 if use_sym_quant else torch.uint8)
        return (
            quantized.view(original_sizes),
            scale.view(original_sizes[:-1] + (1,)),
            zero_point.view(original_sizes[:-1] + (1,)),
        )

    @staticmethod
    def flash_attn_varlen_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None = None,
        causal: bool = False,
        out: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        alibi_slopes: torch.Tensor | None = None,
        window_size: list[int] | None = None,
        softcap: float | None = 0.0,
        seqused_k: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        # passed in qwen vl
        dropout_p: float = 0.0,
        # The following parameters are not used in xpu kernel currently,
        # we keep API compatible to CUDA's.
        scheduler_metadata=None,
        fa_version: int = 2,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        num_splits=0,
        return_softmax_lse: bool | None = False,
        s_aux: torch.Tensor | None = None,
        return_attn_probs: bool | None = False,
        dynamic_causal: torch.Tensor | None = None,
        mask_mod: Callable | None = None,
        aux_tensors: list | None = None,
        **kwargs,
    ):
        assert cu_seqlens_k is not None or seqused_k is not None, (
            "cu_seqlens_k or seqused_k must be provided"
        )
        assert cu_seqlens_k is None or seqused_k is None, (
            "cu_seqlens_k and seqused_k cannot be provided at the same time"
        )
        assert block_table is None or seqused_k is not None, (
            "when enable block_table, seqused_k is needed"
        )
        assert block_table is not None or cu_seqlens_k is not None, (
            "when block_table is disabled, cu_seqlens_k is needed"
        )
        if out is None:
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        real_window_size: tuple[int, int]
        if window_size is None:
            real_window_size = (-1, -1)
        else:
            assert len(window_size) == 2
            real_window_size = (window_size[0], window_size[1])  # noqa: F841

        # In encode attention, k and v maybe not contiguous and current
        # kernel can't handle it
        if block_table is None:
            k = k.contiguous()
            v = v.contiguous()

        if USE_TRITON_XPU_ATTN and block_table is not None:
            if (q.dtype != torch.float16) and (q.dtype != torch.bfloat16):
                q = q.to(out.dtype)
            assert alibi_slopes is None, "Alibi not supported in triton xpu attn"
            return flash_attn_varlen_func_triton(
                out,
                q,
                k,
                v,
                cu_seqlens_q,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale if softmax_scale is not None else q.shape[-1] ** (-0.5),
                causal,
                block_table,
                alibi_slopes,
                sink=s_aux,
                window_size_left=real_window_size[0],
                window_size_right=real_window_size[1],
                k_scale=k_descale,
                v_scale=v_descale,
            )

        return flash_attn_varlen_func(
            out=out,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            block_table=block_table,
            s_aux=s_aux,
            window_size=real_window_size,
            # alibi_slopes = alibi_slopes,
            # softcap=softcap,
            return_softmax_lse=return_softmax_lse,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    @staticmethod
    def get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        cache_seqlens: torch.Tensor,
        qkv_dtype=torch.bfloat16,
        headdim_v=None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k_new: torch.Tensor | None = None,
        cache_leftpad: torch.Tensor | None = None,
        page_size: int | None = None,
        max_seqlen_k_new=0,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        has_softcap=False,
        num_splits=0,  # Can be tuned for speed
        pack_gqa=None,  # Can be tuned for speed
        sm_margin=0,  # Can be tuned if some SMs are used for communication
    ) -> None:
        logger.warning_once(
            "get_scheduler_metadata is not implemented for xpu_ops, returning None."
        )
        return None

    @staticmethod
    def selective_scan_fwd(
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D_: torch.Tensor | None,
        z_: torch.Tensor | None,
        delta_bias_: torch.Tensor | None,
        delta_softplus: bool,
        query_start_loc: torch.Tensor | None,
        cache_indices: torch.Tensor | None,
        has_initial_state: torch.Tensor | None,
        ssm_states: torch.Tensor,
        null_block_id: int,
        block_size: int = 1024,
        block_idx_first_scheduled_token: torch.Tensor | None = None,
        block_idx_last_scheduled_token: torch.Tensor | None = None,
        initial_state_idx: torch.Tensor | None = None,
        cu_chunk_seqlen: torch.Tensor | None = None,
        last_chunk_indices: torch.Tensor | None = None,
    ) -> None:
        varlen = query_start_loc is not None
        batch_size = (
            (query_start_loc.shape[0] - 1)
            if query_start_loc is not None
            else u.shape[0]
        )
        dim = u.shape[0] if varlen else u.shape[1]
        total_seqlen = u.shape[1] if varlen else u.shape[2]
        dstate = A.size(1)
        n_groups = B.size(0) if varlen else B.size(1)
        dim_ngroups_ratio = dim // n_groups

        has_z = z_ is not None
        has_D = D_ is not None
        has_delta_bias = delta_bias_ is not None
        has_cache_indices = cache_indices is not None
        cache_enabled = block_idx_first_scheduled_token is not None

        # out and out_z alias delta and z respectively
        out = delta
        out_z = z_ if z_ is not None else delta  # won't be used if not has_z

        BLOCK_DSTATE = triton.next_power_of_2(dstate)

        # Compute strides
        if varlen:
            u_batch_stride = u.stride(1)
            u_d_stride = u.stride(0)
            delta_batch_stride = delta.stride(1)
            delta_d_stride = delta.stride(0)
            B_batch_stride = B.stride(2)
            B_group_stride = B.stride(0)
            B_dstate_stride = B.stride(1)
            C_batch_stride = C.stride(2)
            C_group_stride = C.stride(0)
            C_dstate_stride = C.stride(1)
            out_batch_stride = out.stride(1)
            out_d_stride = out.stride(0)
            if z_ is not None:
                z_batch_stride = z_.stride(1)
                z_d_stride = z_.stride(0)
                out_z_batch_stride = out_z.stride(1)
                out_z_d_stride = out_z.stride(0)
            else:
                z_batch_stride = 0
                z_d_stride = 0
                out_z_batch_stride = 0
                out_z_d_stride = 0
        else:
            u_batch_stride = u.stride(0)
            u_d_stride = u.stride(1)
            delta_batch_stride = delta.stride(0)
            delta_d_stride = delta.stride(1)
            B_batch_stride = B.stride(0)
            B_group_stride = B.stride(1)
            B_dstate_stride = B.stride(2)
            C_batch_stride = C.stride(0)
            C_group_stride = C.stride(1)
            C_dstate_stride = C.stride(2)
            out_batch_stride = out.stride(0)
            out_d_stride = out.stride(1)
            if z_ is not None:
                z_batch_stride = z_.stride(0)
                z_d_stride = z_.stride(1)
                out_z_batch_stride = out_z.stride(0)
                out_z_d_stride = out_z.stride(1)
            else:
                z_batch_stride = 0
                z_d_stride = 0
                out_z_batch_stride = 0
                out_z_d_stride = 0

        ssm_batch_stride = ssm_states.stride(0)
        ssm_dim_stride = ssm_states.stride(1)
        ssm_dstate_stride = ssm_states.stride(2)
        cache_indices_stride = (
            cache_indices.stride(0) if cache_indices is not None else 0
        )

        grid = (batch_size, dim)
        _selective_scan_fwd_kernel[grid](
            u,
            delta,
            A,
            B,
            C,
            D_ if has_D else u,  # dummy, won't be dereferenced
            z_ if has_z else u,  # dummy
            delta_bias_ if has_delta_bias else u,  # dummy
            out,
            out_z,
            ssm_states,
            query_start_loc if varlen else u,  # dummy
            cache_indices if has_cache_indices else u,  # dummy
            has_initial_state,
            # APC pointers
            block_idx_first_scheduled_token if cache_enabled else u,
            block_idx_last_scheduled_token if cache_enabled else u,
            initial_state_idx if cache_enabled else u,
            cu_chunk_seqlen if cache_enabled else u,
            last_chunk_indices if cache_enabled else u,
            # Dimensions
            batch_size,
            dim,
            total_seqlen,
            dstate,
            n_groups,
            dim_ngroups_ratio,
            # Strides
            u_batch_stride,
            u_d_stride,
            delta_batch_stride,
            delta_d_stride,
            A.stride(0),
            A.stride(1),
            B_batch_stride,
            B_group_stride,
            B_dstate_stride,
            C_batch_stride,
            C_group_stride,
            C_dstate_stride,
            z_batch_stride,
            z_d_stride,
            out_batch_stride,
            out_d_stride,
            out_z_batch_stride,
            out_z_d_stride,
            ssm_batch_stride,
            ssm_dim_stride,
            ssm_dstate_stride,
            cache_indices_stride,
            null_block_id,
            block_size,
            # Compile-time constants
            delta_softplus=delta_softplus,
            HAS_D=has_D,
            HAS_Z=has_z,
            HAS_DELTA_BIAS=has_delta_bias,
            IS_VARLEN=varlen,
            HAS_CACHE_INDICES=has_cache_indices,
            CACHE_ENABLED=cache_enabled,
            BLOCK_DSTATE=BLOCK_DSTATE,
        )

    @staticmethod
    def register_ops_once() -> None:
        global _OPS_REGISTERED
        if not _OPS_REGISTERED:
            # register all the custom ops here
            direct_register_custom_op(
                op_name="xpu_ops_deepseek_scaling_rope",
                op_func=_xpu_ops_deepseek_scaling_rope_impl,
                mutates_args=[],
                fake_impl=_xpu_ops_deepseek_scaling_rope_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="xpu_mxfp8_quantize",
                op_func=_xpu_mxfp8_quantize_impl,
                fake_impl=_xpu_mxfp8_quantize_fake,
            )

            direct_register_custom_op(
                op_name="xpu_mxfp4_quantize",
                op_func=_xpu_mxfp4_quantize_impl,
                fake_impl=_xpu_mxfp4_quantize_fake,
            )

            direct_register_custom_op(
                op_name="xpu_fp8_mqa_logits",
                op_func=_xpu_fp8_mqa_logits_impl,
                fake_impl=_xpu_fp8_mqa_logits_fake,
            )

            direct_register_custom_op(
                op_name="xpu_fp8_paged_mqa_logits",
                op_func=_xpu_fp8_paged_mqa_logits_impl,
                fake_impl=_xpu_fp8_paged_mqa_logits_fake,
            )

            direct_register_custom_op(
                op_name="gdn_attention_core_xpu",
                op_func=_gdn_attention_core_xpu_impl,
                mutates_args=["core_attn_out", "z"],
                fake_impl=_gdn_attention_core_xpu_fake,
            )

            direct_register_custom_op(
                op_name="xpu_topk_topp_sampler",
                op_func=_topk_topp_sample_impl,
                fake_impl=_topk_topp_sample_fake,
            )

            _OPS_REGISTERED = True


xpu_ops.register_ops_once()
