# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that compute logits from hidden_stats."""

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.platforms import current_platform


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

    @staticmethod
    def _apply_min_tokens_mask(
        logits: torch.Tensor,
        vocab_start: int,
        min_tokens_mask: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> None:
        # Set stop/EOS-token logits to -inf for under-min requests, in place.
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
        """ Vocab-parallel Gumbel-max sample WITHOUT all-gathering full logits.
        temp=1, top_p=1, top_k=-1 only. Equivalent in distribution to multinomial
        sampling over softmax(full_logits); no normalizer needed.
        """
        import vllm.envs as envs
        from vllm.distributed import get_tp_group
        from vllm.v1.worker.gpu.sample.gumbel import dist_gumbel_local_packed

        tp_size = get_tensor_model_parallel_world_size()

        # Local shard logits. float32 for stable noise add + compare.
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

        vocab_start = lm_head.shard_indices.org_vocab_start_index

        # MinTokens: censor stop/EOS tokens on the owning shard before argmax.
        self._apply_min_tokens_mask(logits, vocab_start, min_tokens_mask)

        # fp32-allreduce: separate (value, global_idx) fp32 + 0-fill SUM all-reduce. 
        # Faster than int64 max
        if tp_size > 1 and envs.VLLM_ENABLE_DIST_SAMPLE_FP32_REDUCE:
            from vllm.v1.worker.gpu.sample.gumbel import dist_gumbel_local_validx
            import torch.distributed as dist

            B = logits.shape[0]
            value, global_idx = dist_gumbel_local_validx(
                logits, int(step_seed), int(vocab_start), int(self.org_vocab_size)
            )
            rank = get_tp_group().rank_in_group
            table = logits.new_zeros(B, tp_size, 2)  # fp32
            table[:, rank, 0] = value
            table[:, rank, 1] = global_idx
            dist.all_reduce(
                table, op=dist.ReduceOp.SUM, group=get_tp_group().device_group
            )
            best = table[:, :, 0].argmax(dim=-1, keepdim=True)
            return table[:, :, 1].gather(-1, best).squeeze(-1).to(torch.int64)

        # int64-allreduce: pack (31-bit value key)<<31 | global_index, so a plain MAX
        # yields the value-then-index winner. Current tensor_model_parallel_all_reduce 
        # is SUM-only, so use the raw TP group.
        packed = dist_gumbel_local_packed(
            logits, int(step_seed), int(vocab_start), int(self.org_vocab_size)
        )
        if tp_size == 1:
            return packed & 0x7FFFFFFF

        import torch.distributed as dist

        dist.all_reduce(
            packed, op=dist.ReduceOp.MAX, group=get_tp_group().device_group
        )
        return packed & 0x7FFFFFFF

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s
