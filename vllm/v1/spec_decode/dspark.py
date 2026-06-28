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

from copy import copy

import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID
from vllm.v1.worker.utils import AttentionGroup

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
        self.dspark_block_size: int = draft_cfg.dspark_block_size
        self.noise_token_id: int = draft_cfg.dspark_noise_token_id
        self.markov_rank: int = draft_cfg.dspark_markov_rank
        # Per-position confidence of the most recent proposed block, shape
        # (batch_size, block_size). Consumed by the runner to bound acceptance.
        self._last_confidence: torch.Tensor | None = None
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}
        self._per_group_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        logger.info_once(
            "DSpark proposer initialized (block_size=%d, markov_rank=%d)",
            self.dspark_block_size,
            self.markov_rank,
        )

    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self._per_group_block_tables[gid] = block_table
        self._per_group_slot_mappings[gid] = slot_mapping

    def _slot_mapping_buffer_for(self, gid: int) -> torch.Tensor:
        if gid == self.kv_cache_gid:
            return self._slot_mapping_buffer
        buf = self._per_group_slot_mapping_buffers.get(gid)
        if buf is None:
            buf = torch.zeros(self.max_positions, dtype=torch.int64, device=self.device)
            self._per_group_slot_mapping_buffers[gid] = buf
        return buf

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        per_layer: dict[str, torch.Tensor] = {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            buf = self._slot_mapping_buffer_for(gid)
            source = self._per_group_slot_mappings.get(gid, slot_mapping)
            if source is not None and buf.data_ptr() != source.data_ptr():
                n = source.shape[0]
                buf[:n].copy_(source)
                if num_tokens > n:
                    buf[n:num_tokens].fill_(PADDING_SLOT_ID)
            view = buf[:num_tokens]
            for layer_name in attn_group.layer_names:
                per_layer[layer_name] = view
        return per_layer

    def build_per_group_and_layer_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> tuple[list[object], dict[str, object]]:
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            if gid in self._per_group_block_tables:
                cm = copy(common_attn_metadata)
                cm.block_table_tensor = self._per_group_block_tables[gid][:num_reqs]
                if gid in self._per_group_slot_mappings:
                    sm = self._per_group_slot_mappings[gid]
                    if sm.shape[0] >= num_actual_tokens:
                        sm = sm[:num_actual_tokens]
                    cm.slot_mapping = sm
            else:
                cm = common_attn_metadata
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=cm,
                draft_index=draft_index,
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        return

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        layer_to_gid: dict[str, int] = {}
        layer_to_spec: dict[str, KVCacheSpec] = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                layer_to_gid[layer_name] = gid
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    if layer_name in group_spec.kv_cache_specs:
                        layer_to_spec[layer_name] = group_spec.kv_cache_specs[
                            layer_name
                        ]
                    else:
                        target_layer_name = getattr(
                            all_attn_layers.get(layer_name),
                            "kv_sharing_target_layer_name",
                            None,
                        )
                        if (
                            target_layer_name
                            and target_layer_name in group_spec.kv_cache_specs
                        ):
                            layer_to_spec[layer_name] = group_spec.kv_cache_specs[
                                target_layer_name
                            ]
                        else:
                            layer_to_spec[layer_name] = group_spec
                else:
                    layer_to_spec[layer_name] = group_spec

        attention_groups: dict[tuple[tuple[str, str], int], AttentionGroup] = {}
        for layer_name in sorted(self._draft_attn_layer_names):
            if layer_name not in layer_to_spec:
                continue
            attn_layer = all_attn_layers[layer_name]
            attn_backend = attn_layer.get_attn_backend()
            spec = layer_to_spec[layer_name]
            gid = layer_to_gid[layer_name]
            group_key = (attn_backend.full_cls_name(), gid)

            if group_key not in attention_groups:
                kernel_block_size = (
                    kernel_block_sizes[gid]
                    if kernel_block_sizes is not None and gid < len(kernel_block_sizes)
                    else None
                )
                attn_group = AttentionGroup(
                    backend=attn_backend,
                    layer_names=[layer_name],
                    kv_cache_spec=spec,
                    kv_cache_group_id=gid,
                )
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_size=kernel_block_size,
                )
                attention_groups[group_key] = attn_group
            else:
                attention_groups[group_key].layer_names.append(layer_name)

        self.draft_attn_groups = list(attention_groups.values())
        if self.draft_attn_groups:
            self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
            self.block_size = (
                self.draft_attn_groups[0]
                .get_metadata_builder()
                .kv_cache_spec.block_size
            )
        else:
            self.kv_cache_gid = 0
            self.block_size = kv_cache_config.kv_cache_groups[
                0
            ].kv_cache_spec.block_size
        logger.debug("Using block size %d for DSpark drafting layers", self.block_size)

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
        if num_speculative_tokens != self.dspark_block_size:
            raise ValueError(
                "DSpark proposer expects num_speculative_tokens to equal "
                f"dspark_block_size ({self.dspark_block_size}), got "
                f"{num_speculative_tokens}."
            )

        # main_x = main_norm(main_proj(concat target-layer hiddens)).
        main_x = draft.project_main_hidden(target_hidden_states)
        query_end_loc = common_attn_metadata.query_start_loc[1:] - 1
        if num_rejected_tokens_gpu is not None:
            query_end_loc = query_end_loc - num_rejected_tokens_gpu
        request_main_x = main_x[query_end_loc.to(torch.long)]

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
        token_indices_to_sample = token_indices_to_sample.view(
            batch_size, self.dspark_block_size
        )
        flat_sample_indices = token_indices_to_sample.reshape(-1).to(torch.long)
        self.hidden_states[flat_sample_indices] = (
            request_main_x.repeat_interleave(self.dspark_block_size, dim=0)
        )
        token_indices_to_sample = token_indices_to_sample.reshape(-1)
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
        return draft_token_ids.view(batch_size, self.dspark_block_size)

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
        base_logits = base_logits.view(batch_size, self.dspark_block_size, vocab)
        collapsed = collapsed.view(batch_size, self.dspark_block_size, -1)

        output_ids = base_logits.new_empty(
            batch_size, self.dspark_block_size, dtype=torch.long
        )
        prev = next_token_ids.to(torch.long)
        markov_embeds = []
        for i in range(self.dspark_block_size):
            logits_bias, markov_embed = model.markov_refine(prev)
            logits_i = base_logits[:, i] + logits_bias
            prev = logits_i.argmax(dim=-1)
            output_ids[:, i] = prev
            markov_embeds.append(markov_embed)
        markov_embed = torch.stack(markov_embeds, dim=1)
        confidence = model.compute_confidence(collapsed, markov_embed)
        confidence = confidence.view(batch_size, self.dspark_block_size)
        return output_ids, confidence

    def get_last_confidence(self) -> torch.Tensor | None:
        """Per-position confidence ``(batch_size, block_size)`` of the last
        proposed block; used by the runner to bound acceptance length."""
        return self._last_confidence
