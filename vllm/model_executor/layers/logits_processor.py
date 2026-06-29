# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that compute logits from hidden_stats."""

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.platforms import current_platform

logger = init_logger(__name__)


# --8<-- [start:logits_processor]
@PluggableLayer.register("logits_processor")
class LogitsProcessor(PluggableLayer):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    # --8<-- [end:logits_processor]

    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: int | None = None,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: float | None = None,
    ) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap
        # Whether to use gather or all-gather to gather the logits.
        self.use_all_gather = current_platform.use_all_gather()

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.logits_as_input:
            logits = hidden_states
        else:
            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale
        return logits

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """gather/all-gather the logits tensor across model parallel group."""
        if self.use_all_gather:
            # Gather is not supported for some devices such as TPUs.
            # Use all-gather instead.
            # NOTE(woosuk): Here, the outputs of every device should not be None
            # because XLA requires strict SPMD among all devices. Every device
            # should execute the same operations after gathering the logits.
            logits = tensor_model_parallel_all_gather(logits)
        else:
            # None may be returned for rank > 0
            logits = tensor_model_parallel_gather(logits)
        return logits

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # Get the logits for the next tokens.
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)

        # Gather logits for TP
        logits = self._gather_logits(logits)

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    def get_top_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vocab-parallel argmax without all-gathering full logits.

        Each TP rank computes local argmax, then only the (value, index) pairs
        are gathered and reduced. Communication: O(batch * 2 * tp_size) vs
        O(batch * vocab_size).
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        tp_size = get_tensor_model_parallel_world_size()

        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale

        # Mask out padding entries beyond org_vocab_size on this shard.
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        local_max_vals, local_max_indices = logits.max(dim=-1)

        # Convert shard-local indices to global vocab indices.
        vocab_start = lm_head.shard_indices.org_vocab_start_index
        global_indices = local_max_indices + vocab_start

        if tp_size == 1:
            return global_indices

        # All-gather (value, index) pairs, then reduce to global argmax.
        # Use float32 to avoid bf16 precision loss on large vocab indices.
        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1
        )
        # [batch, 2] -> [batch, 2 * tp_size]
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        # [batch, tp_size, 2] where [:, :, 0]=values, [:, :, 1]=indices
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)

    # __DIST_SAMPLE__
    @staticmethod
    def _apply_min_tokens_mask(
        logits: torch.Tensor,
        vocab_start: int,
        min_tokens_mask: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> None:
        """Set stop/EOS-token logits to -inf for under-min requests, in place.

        min_tokens_mask is the stock MinTokensLogitsProcessor.logits_slice:
        (batch_row_ids, global_token_ids). Each global token id lives on exactly
        one vocab shard; this rank masks only the ids that fall in its shard
        [vocab_start, vocab_start + shard_width). Reproduces the stock masking
        exactly so the vocab-parallel argmax cannot pick a censored stop token."""
        if min_tokens_mask is None:
            return
        rows, global_tok_ids = min_tokens_mask
        if rows.numel() == 0:
            return
        local_cols = global_tok_ids.to(torch.int64) - vocab_start
        in_shard = (local_cols >= 0) & (local_cols < logits.shape[-1])
        if not bool(in_shard.any()):
            return
        sel_rows = rows.to(torch.int64)[in_shard]
        sel_cols = local_cols[in_shard]
        logits[sel_rows, sel_cols] = -float("inf")

    def gumbel_argmax_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        step_seed: int,
        embedding_bias: torch.Tensor | None = None,
        min_tokens_mask: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Vocab-parallel Gumbel-max sample WITHOUT all-gathering full logits.

        temp=1, top_p=1, top_k=-1 only. Equivalent in distribution to multinomial
        sampling over softmax(full_logits); no normalizer needed.

        Two implementations, selected by env VLLM_XPU_DIST_SAMPLE_FUSED:
          - FUSED (DEFAULT, =1): fused Triton gumbel+argmax kernel (no [B,V] noise
            materialization) + int64 MAX all-reduce. FAST and VALIDATED == stock
            accuracy (mixed subset LCB 62.5% vs stock 61.7%). Pure perf, no accuracy
            impact, so it is on by default whenever dist sampling is enabled.
          - PROTOTYPE (=0): materialize Gumbel noise on the local shard with
            torch.rand_like (fresh global entropy per draw, like the stock sampler's
            q.exponential_()), local argmax, all-gather [B,2] float (value, index)
            pairs, reduce by value-argmax. Also validated == stock; kept as a
            reference / fallback.
        """
        import vllm.envs as envs

        if envs.VLLM_XPU_DIST_SAMPLE_FUSED:
            return self._gumbel_argmax_tokens_fused(
                lm_head, hidden_states, step_seed, embedding_bias, min_tokens_mask
            )
        return self._gumbel_argmax_tokens_proto(
            lm_head, hidden_states, step_seed, embedding_bias, min_tokens_mask
        )

    def _gumbel_argmax_tokens_proto(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        step_seed: int,
        embedding_bias: torch.Tensor | None = None,
        min_tokens_mask: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """PROTOTYPE Gumbel-max (validated == stock). torch.rand_like noise on the
        local shard + [B,2] float (value, global_index) all-gather + value-argmax.

        Communication: O(batch * 2 * tp_size) vs O(batch * vocab_size).
        Noise is fresh per draw (torch.rand_like) and independent per rank — each
        global vocab position lives on exactly one shard, so independent iid Gumbel
        noise composes correctly under the cross-rank value-argmax reduce.
        """
        tp_size = get_tensor_model_parallel_world_size()

        # Local shard logits (NO gather). float32 for stable noise add + compare.
        logits = lm_head.quant_method.apply(
            lm_head, hidden_states, bias=embedding_bias
        ).to(torch.float32)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale

        # Mask padding entries beyond org_vocab_size on this shard.
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        # MinTokens: censor stop/EOS tokens on the owning shard before argmax.
        self._apply_min_tokens_mask(
            logits, lm_head.shard_indices.org_vocab_start_index, min_tokens_mask
        )

        # Gumbel(0,1) = -log(-log(U)), U~Uniform(0,1), fresh entropy each call.
        u = torch.rand_like(logits)
        u.clamp_(min=1.0e-20, max=1.0)
        gumbel = -torch.log(-torch.log(u))
        perturbed = logits + gumbel

        local_max_vals, local_max_indices = perturbed.max(dim=-1)
        vocab_start = lm_head.shard_indices.org_vocab_start_index
        global_indices = local_max_indices + vocab_start

        if tp_size == 1:
            return global_indices.to(torch.int64)

        # All-gather [B,2] (value, global_index) float pairs, reduce by value-argmax.
        local_pair = torch.stack(
            [local_max_vals, global_indices.to(torch.float32)], dim=-1
        )
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)

    def _gumbel_argmax_tokens_fused(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        step_seed: int,
        embedding_bias: torch.Tensor | None = None,
        min_tokens_mask: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Fused Triton gumbel+argmax kernel + cross-rank int64 winner reduce.

        FAST: fused Triton gumbel+argmax kernel (no [B,V] noise materialization)
        packs each rank's shard-local winner into a positive sortable int64
        ((31-bit value key)<<31 | global_index), then a single O(batch) cross-rank
        reduce yields the global winner. VALIDATED == stock accuracy on the mixed
        subset (LCB 62.5% vs stock 61.7%) — the fused kernel is correct.

        The cross-rank reduce defaults to dist.all_reduce(op=MAX). The alternative
        int64 [B,1] all-gather + host max (VLLM_XPU_DIST_SAMPLE_ALLGATHER=1) is
        FASTER but currently BROKEN on XPU: tensor_model_parallel_all_gather on an
        int64 [B,1] tensor mis-pairs/corrupts data across ranks and collapses
        accuracy (livecodebench 23%). Kept behind the flag for debugging only.
        """
        import vllm.envs as envs
        from vllm.v1.worker.gpu.sample.gumbel import dist_gumbel_local_packed
        from vllm.distributed import get_tp_group

        tp_size = get_tensor_model_parallel_world_size()

        # Synced sub-timers (VLLM_STEP_LOG): split dist sample_ms into matmul /
        # gumbel-kernel / reduce to locate the ~4ms. XPU is async so each segment
        # needs torch.xpu.synchronize() to measure real (not enqueue) time.
        import os
        _brk = bool(os.environ.get("VLLM_STEP_LOG"))
        if _brk:
            import time as _t
            torch.xpu.synchronize()
            _t0 = _t.time()

        # Local shard logits (NO gather). float32 for stable noise add + compare.
        logits = lm_head.quant_method.apply(
            lm_head, hidden_states, bias=embedding_bias
        ).to(torch.float32)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale

        # Mask padding entries beyond org_vocab_size on this shard.
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        # MinTokens: censor stop/EOS tokens on the owning shard before the fused
        # gumbel+argmax kernel reads these logits.
        self._apply_min_tokens_mask(
            logits, lm_head.shard_indices.org_vocab_start_index, min_tokens_mask
        )

        if _brk:
            torch.xpu.synchronize()
            _t1 = _t.time()  # matmul + masking done

        vocab_start = lm_head.shard_indices.org_vocab_start_index
        B = logits.shape[0]

        # FP32-REDUCE path: separate (value, global_idx) fp32 + 0-fill SUM all-reduce.
        # int64 XCCL reductions hit a slow elementwise path; fp32 SUM is the tuned
        # collective (~13x faster standalone). Per-rank slots are disjoint so SUM
        # over a 0-filled [B,TP,2] reconstructs the gathered table exactly; idx<2**24
        # is fp32-exact. Mathematically identical winner to the int64 MAX path.
        if tp_size > 1 and envs.VLLM_XPU_DIST_SAMPLE_FP32_REDUCE:
            from vllm.v1.worker.gpu.sample.gumbel import dist_gumbel_local_validx
            import torch.distributed as dist

            value, global_idx = dist_gumbel_local_validx(
                logits, int(step_seed), int(vocab_start), int(self.org_vocab_size)
            )
            if _brk:
                torch.xpu.synchronize()
                _t2 = _t.time()  # gumbel kernel done

            rank = get_tp_group().rank_in_group
            table = logits.new_zeros(B, tp_size, 2)  # fp32
            table[:, rank, 0] = value
            table[:, rank, 1] = global_idx
            dist.all_reduce(
                table, op=dist.ReduceOp.SUM, group=get_tp_group().device_group
            )
            best = table[:, :, 0].argmax(dim=-1, keepdim=True)
            tokens = table[:, :, 1].gather(-1, best).squeeze(-1).to(torch.int64)

            if _brk:
                torch.xpu.synchronize()
                _t3 = _t.time()
                if rank == 0:
                    logger.info(
                        "[DIST_BRK] fp32 B=%d matmul=%.3fms kernel=%.3fms "
                        "reduce=%.3fms total=%.3fms",
                        B, (_t1 - _t0) * 1000, (_t2 - _t1) * 1000,
                        (_t3 - _t2) * 1000, (_t3 - _t0) * 1000,
                    )
            return tokens

        packed = dist_gumbel_local_packed(
            logits, int(step_seed), int(vocab_start), int(self.org_vocab_size)
        )

        if _brk:
            torch.xpu.synchronize()
            _t2 = _t.time()  # gumbel+pack kernel done

        if tp_size == 1:
            return packed & 0x7FFFFFFF

        if envs.VLLM_XPU_DIST_SAMPLE_ALLGATHER:
            # DEBUG-only: int64 [B,1] all-gather + host max. Faster but BROKEN on
            # XPU (cross-rank data corruption). Gather [B,1] (not [B]) so the result
            # is [B,tp] per-row; a flat [B] gather concatenates to [tp*B].
            gathered = tensor_model_parallel_all_gather(
                packed.unsqueeze(-1), dim=-1
            )
            winner = gathered.max(dim=-1).values
            return winner & 0x7FFFFFFF

        # Default: int64 MAX all-reduce. The packed int64 is a positive sortable
        # (31-bit value key)<<31 | global_index, so plain MAX yields the
        # value-then-index winner. vLLM's tensor_model_parallel_all_reduce is
        # SUM-only, so use the raw TP process group for MAX.
        import torch.distributed as dist

        dist.all_reduce(
            packed,
            op=dist.ReduceOp.MAX,
            group=get_tp_group().device_group,
        )

        if _brk:
            torch.xpu.synchronize()
            _t3 = _t.time()  # cross-rank reduce done
            if get_tp_group().rank_in_group == 0:
                logger.info(
                    "[DIST_BRK] i64 B=%d matmul=%.3fms kernel=%.3fms reduce=%.3fms "
                    "total=%.3fms",
                    logits.shape[0],
                    (_t1 - _t0) * 1000,
                    (_t2 - _t1) * 1000,
                    (_t3 - _t2) * 1000,
                    (_t3 - _t0) * 1000,
                )

        return packed & 0x7FFFFFFF

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s
