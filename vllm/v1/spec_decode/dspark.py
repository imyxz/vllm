# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark block proposer for DeepSeek-V4-Flash-DSpark.

Standard MTP/Eagle drafts one token per sequential step (K steps -> K tokens).
DSpark instead proposes a whole block of ``dspark_block_size`` tokens in a
*single* draft forward, then refines each position with a cheap autoregressive
Markov head and emits a per-position confidence used to decide how much of the
block to keep.  See ``vllm/models/deepseek_v4/nvidia/dspark.py`` for the model
and the reference ``inference/model.py`` (``DSparkBlock.forward_head``).

This subclasses :class:`EagleProposer` to reuse the heavy, framework-coupled
machinery (draft model loading, KV-cache / attention-group setup, CUDA-graph
padding, ``dummy_run``).  Only token generation is replaced: after the single
draft forward produces the block hidden states, :meth:`_refine_block` runs the
Markov + confidence refinement.

HARDWARE-VALIDATION NOTES (this proposer is written without GPU access):
  * The single draft forward must place the ``dspark_block_size`` draft slots
    (``[last_verified_token, noise, noise, ...]``) and condition on the
    projected target-layer hidden state (``main_x``).  The token / position /
    slot-mapping construction below mirrors the K=block_size parallel-drafting
    path; confirm the attention metadata matches the reference sliding-window
    block attention on real weights.
  * Confidence-based dynamic acceptance length (keep the block prefix whose
    confidence exceeds a threshold) is exposed via ``self._last_confidence``;
    integrating it with the verifier/sampler acceptance is the remaining wiring.
"""

import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer

logger = init_logger(__name__)


def _unwrap(model):
    """The draft model may be CUDA-graph-wrapped; DSpark's bespoke helper
    methods (project_main_hidden / markov_refine / ...) live on the inner
    module, so unwrap for those (the wrapper is still used for the forward)."""
    if isinstance(model, BreakableCUDAGraphWrapper):
        return model.unwrap()
    return model


class DSparkProposer(EagleProposer):
    """DeepSeek-V4 DSpark block-parallel speculative proposer."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        super().__init__(vllm_config, device, runner)
        draft_cfg = vllm_config.speculative_config.draft_model_config.hf_config
        self.block_size: int = draft_cfg.dspark_block_size
        self.noise_token_id: int = draft_cfg.dspark_noise_token_id
        self.markov_rank: int = draft_cfg.dspark_markov_rank
        # Per-position confidence of the most recent proposed block, shape
        # (batch_size, block_size). Consumed by the runner to bound acceptance.
        self._last_confidence: torch.Tensor | None = None
        logger.info_once(
            "DSpark proposer initialized (block_size=%d, markov_rank=%d)",
            self.block_size,
            self.markov_rank,
        )

    @torch.inference_mode()
    def propose(
        self,
        num_speculative_tokens,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings=None,
    ) -> torch.Tensor:
        """Propose a block of ``block_size`` draft tokens per request.

        ``target_hidden_states`` here is the DSpark conditioning signal —
        the concatenated mean-pooled hidden states captured at
        ``dspark_target_layer_ids`` (fed by the runner via
        ``get_dspark_target_hidden_states``).
        """
        self.num_speculative_tokens = num_speculative_tokens
        self._last_draft_probs = None
        self._last_confidence = None
        batch_size = common_attn_metadata.batch_size()
        draft = _unwrap(self.model)

        # main_x = main_norm(main_proj(concat target-layer hiddens)).
        main_x = draft.project_main_hidden(target_hidden_states)

        # Reuse the base input-prep so the draft block forward sees the right
        # attention metadata / KV slots (mirrors EagleProposer.propose).
        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=main_x,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )
        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
        )
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )
        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )

        from vllm.forward_context import set_forward_context

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                slot_mapping_size, common_attn_metadata.slot_mapping
            ),
        ):
            block_hidden = self.model(**model_kwargs)

        sample_hidden = block_hidden[token_indices_to_sample]
        draft_token_ids, confidence = self._refine_block(
            draft, sample_hidden, next_token_ids, batch_size
        )
        self._last_confidence = confidence
        return draft_token_ids.view(batch_size, self.block_size)

    def _refine_block(
        self,
        model,
        block_hidden: torch.Tensor,
        next_token_ids: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Markov-refine + confidence over the block (reference
        ``DSparkBlock.forward_head``).

        ``block_hidden`` is the per-position pre-hc_head residual for the block,
        shape ``(batch_size * block_size, hc_mult * hidden_size)``.
        """
        # Base block logits + the hc-collapsed hidden reused by confidence.
        base_logits, collapsed = model.compute_block_outputs(block_hidden)
        vocab = base_logits.shape[-1]
        base_logits = base_logits.view(batch_size, self.block_size, vocab)
        collapsed = collapsed.view(batch_size, self.block_size, -1)

        output_ids = base_logits.new_empty(
            batch_size, self.block_size, dtype=torch.long
        )
        prev = next_token_ids.to(torch.long)
        markov_embeds = []
        for i in range(self.block_size):
            logits_bias, markov_embed = model.markov_refine(prev)
            logits_i = base_logits[:, i] + logits_bias
            prev = logits_i.argmax(dim=-1)
            output_ids[:, i] = prev
            markov_embeds.append(markov_embed)
        markov_embed = torch.stack(markov_embeds, dim=1)
        confidence = model.compute_confidence(collapsed, markov_embed)
        confidence = confidence.view(batch_size, self.block_size)
        return output_ids, confidence

    def get_last_confidence(self) -> torch.Tensor | None:
        """Per-position confidence ``(batch_size, block_size)`` of the last
        proposed block; used by the runner to bound acceptance length."""
        return self._last_confidence
