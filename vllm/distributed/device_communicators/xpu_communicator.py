# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

from .base_device_communicator import DeviceCommunicatorBase

logger = init_logger(__name__)

# Opt-in fp8-transport all-reduce. Comm/compute do not overlap on XPU+oneCCL
# over PCIe and the all-reduce is purely bandwidth-bound, so the only lever on
# its cost is moving fewer bytes. This replaces a bf16 all-reduce with a bf16
# reduce-scatter (accurate sum) followed by an fp8 all-gather (compressed
# transport). The sum is computed in bf16; the only precision loss is a single
# fp8 quantization of the already-summed shard.
#
# Microbenched on gpt-oss TP=8 (3072x2880): bf16 all-reduce 1.78ms;
# fp8 transport with a STATIC scale + a single all-gather + a fused Triton
# dequant = 1.38ms (1.29x). A dynamic per-rank scale needs a second all-gather
# whose ~0.34ms collective-launch penalty (for 8 floats!) erases the win
# (1.03x), so we use a fixed VLLM_XPU_FP8_ALLREDUCE_SCALE instead. The eager
# (un-fused) dequant and torch.compile variants only reach 0.83x/1.04x; the
# fused Triton dequant on the full-size gathered tensor is what makes it pay.
VLLM_XPU_FP8_ALLREDUCE = os.environ.get("VLLM_XPU_FP8_ALLREDUCE", "0") == "1"
# Static dequant scale: summed-shard values are divided by this before the fp8
# cast and multiplied back after gather. It MUST be large enough that no
# summed activation saturates e4m3, i.e. scale >= (max activation amax)/448.
# CRITICAL: saturation, not mantissa precision, is what destroys accuracy.
# Measured gpt-oss-120b TP=8 summed-shard amax reaches ~48k-57k, so a scale of
# 0.05 (covers only 22.4) saturated nearly every all-reduce -> coherent-looking
# but systematically-wrong tokens -> <10% MLPerf score. Because e4m3 is a
# floating-point format its ~2.6% relative error is roughly constant across
# magnitudes, so sizing the scale generously costs no precision on small
# values. 128 covers amax up to 57344. Validate amax per model via
# VLLM_XPU_FP8_ALLREDUCE_AMAX_LOG=1 and raise this if it ever saturates.
VLLM_XPU_FP8_ALLREDUCE_SCALE = float(
    os.environ.get("VLLM_XPU_FP8_ALLREDUCE_SCALE", "128.0")
)
# Minimum leading-dim (token count) for the fp8 path to engage. The fp8 path
# trades wire bytes for a fixed overhead (two collectives instead of one, plus
# the quant and the Triton dequant launch), so it only wins once the message is
# big enough to be bandwidth-bound. Microbenched crossover on gpt-oss TP=8
# (H=2880, fused-Triton path, 8-rank PCIe): fp8 loses at <=192 tokens (0.83x),
# breaks even at 256 (1.00x), and wins from 320 up (1.14x -> 1.24x at 3072).
# Default 256 = the measured break-even; raise it for a more conservative gate,
# or set to a small value to engage fp8 on (nearly) every all-reduce.
VLLM_XPU_FP8_ALLREDUCE_MIN_TOKENS = int(
    os.environ.get("VLLM_XPU_FP8_ALLREDUCE_MIN_TOKENS", "256")
)
_FP8_DTYPE = torch.float8_e4m3fn

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _fp8_dequant_kernel(
        gathered_ptr, scale, out_ptr, n_elem, BLOCK: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elem
        v = tl.load(gathered_ptr + offs, mask=mask).to(tl.float32)
        tl.store(out_ptr + offs, (v * scale).to(tl.bfloat16), mask=mask)

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


class XpuCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        self.ca_comm: None = None
        if self.use_all2all:
            if self.all2all_backend in ("naive", "allgather_reducescatter"):
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
                logger.info("Using AgRs manager on XPU device.")

            else:  # type: ignore[has-type]
                logger.warning(
                    "`%s` all2all manager is not supported on XPU. "
                    "Falling back to AgRs manager for XPU, "
                    "which is the Default backend",
                    self.all2all_backend,  # type: ignore[has-type]
                )
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
                logger.info("Using AgRs manager on XPU device.")

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if VLLM_XPU_FP8_ALLREDUCE and self._fp8_allreduce_applicable(input_):
            return self._fp8_all_reduce(input_)
        output = input_.clone()
        dist.all_reduce(output, group=self.device_group)
        return output

    def _fp8_allreduce_applicable(self, input_: torch.Tensor) -> bool:
        # Only worthwhile for the large bf16 activation all-reduces, and only
        # when the leading dim splits evenly across ranks (reduce-scatter
        # requirement). Needs the fused Triton dequant to actually be a win.
        # Small (decode-heavy) or non-divisible tensors fall back to bf16: the
        # fp8 path's fixed overhead only pays off above the token-count crossover
        # (see VLLM_XPU_FP8_ALLREDUCE_MIN_TOKENS). The per-rank shard must also be
        # a multiple of 4 fp8 elements so it reinterprets cleanly as int32 for
        # the all-gather (see _fp8_all_reduce step 3).
        if not (
            _HAS_TRITON
            and input_.dtype == torch.bfloat16
            and input_.is_contiguous()
            and input_.dim() >= 1
            and input_.shape[0] >= VLLM_XPU_FP8_ALLREDUCE_MIN_TOKENS
            and input_.shape[0] % self.world_size == 0
        ):
            return False
        shard_numel = input_.numel() // self.world_size
        return shard_numel % 4 == 0

    def _fp8_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """bf16 reduce-scatter (accurate sum) + fp8 all-gather + fused dequant.

        Uses a single static scale (no per-rank scale collective; the second
        all-gather's launch penalty would erase the wire savings) and a fused
        Triton dequant on the full-size gathered tensor.
        """
        world_size = self.world_size
        x = input_.contiguous()
        scale = VLLM_XPU_FP8_ALLREDUCE_SCALE
        inv_scale = 1.0 / scale

        # 1. Accurate sum in bf16, sharded across ranks.
        shard_shape = (x.shape[0] // world_size,) + tuple(x.shape[1:])
        shard = torch.empty(shard_shape, dtype=torch.bfloat16, device=x.device)
        dist.reduce_scatter_tensor(shard, x, group=self.device_group)

        if os.environ.get("VLLM_XPU_FP8_ALLREDUCE_AMAX_LOG"):
            _am = shard.abs().max().item()
            if _am > getattr(self, "_fp8_amax_seen", 0.0):
                self._fp8_amax_seen = _am
                logger.info("[fp8_allreduce] new max summed-shard amax=%.3f "
                            "(scale=%.4g covers up to %.1f)",
                            _am, scale, scale * 448.0)

        # 2. Quantize the summed shard to fp8 with the static scale (cheap on
        #    the small shard, so eager is fine here).
        shard_fp8 = (shard.float() * inv_scale).to(_FP8_DTYPE).contiguous()

        # 3. All-gather the fp8 shards. oneCCL (>=2021.15) rejects all 8-bit
        #    collective dtypes (fp8/uint8/int8 all raise "unsupported datatype
        #    UINT8"); only >=16-bit types work. Pack 4 fp8 bytes into one int32
        #    so the wire payload stays 1 byte/elem while using an accepted
        #    dtype, then reinterpret back. Requires shard numel %4==0.
        shard_i32 = shard_fp8.view(-1).view(torch.int32)
        gathered_i32 = torch.empty(
            shard_i32.numel() * world_size, dtype=torch.int32, device=x.device
        )
        dist.all_gather_into_tensor(
            gathered_i32, shard_i32, group=self.device_group
        )
        gathered_fp8 = gathered_i32.view(_FP8_DTYPE)

        # 4. Fused Triton dequant over the full-size gathered tensor.
        out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
        n = gathered_fp8.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
        _fp8_dequant_kernel[grid](
            gathered_fp8.view(-1), scale, out.view(-1), n, BLOCK=2048
        )
        return out

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
        world_size = self.world_size

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        dist.reduce_scatter_tensor(output, input_tensor, group=self.device_group)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ):
        world_size = self.world_size

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )
        if sizes is not None and sizes.count(sizes[0]) != len(sizes):
            # if inputs shape in different ranks is not the same using reduce_scatter
            input_splits = list(input_tensor.split(sizes, dim=0))
            dist.reduce_scatter(output, input_splits, group=self.device_group)
        else:
            dist.reduce_scatter_tensor(output, input_tensor, group=self.device_group)
        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size

        # 'sizes' is not needed if all inputs in the same group have the same
        # shape
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        def _all_gather_single(input_: torch.Tensor, sizes: list[int] | None = None):
            input_size = input_.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert input_.shape[dim] == sizes[self.rank_in_group], (
                    f"{input_.shape[dim]} != {sizes[self.rank_in_group]}"
                )
                output_size = (sum(sizes),) + input_size[1:]
            else:
                output_size = (input_size[0] * world_size,) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(
                output_size, dtype=input_.dtype, device=input_.device
            )

            if sizes is not None:
                all_gather_list = []
                for size in sizes:
                    all_gather_list.append(
                        torch.empty(
                            (size,) + input_.shape[1:],
                            dtype=input_.dtype,
                            device=input_.device,
                        )
                    )
                dist.all_gather(all_gather_list, input_, group=self.device_group)
                output_tensor = torch.cat(all_gather_list, dim=0)
            else:
                dist.all_gather([output_tensor], input_, group=self.device_group)
            return output_tensor

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        output_list = []
        for inp in input_:
            output_list.append(_all_gather_single(inp, sizes=sizes))
        return output_list

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        # For xpu path, gather doesn't work properly together with ray
        # cluster so we use all_gather instead for now.
        input_size = input_.size()
        # Allocate output tensor.
        output_tensor = torch.empty(
            (self.world_size,) + input_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather.
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        if self.rank_in_group == dst:
            # Reshape
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )
        else:
            output_tensor = None
        return output_tensor

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> None:
        dist.broadcast(input_, src=src, group=self.device_group)

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and router logits to the appropriate device.
        This is a no-op in the base class.
        """

        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and topk weights/ids to the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Combine the hidden states and router logits from the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.combine(
            hidden_states,
            is_sequence_parallel,
        )
