# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA Triton sparse MLA fallback for DeepSeek V4 on GB10.

This is a correctness-oriented fallback for platforms where FlashInfer's
DeepSeek V4 sparse MLA runner is present at import time but not executable at
runtime. It reuses the fp8_ds_mla KV layout, CUDA KV insert path, and the
existing Triton BF16 sparse attention kernels.
"""

from typing import TYPE_CHECKING, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLASparseBackend,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata
from vllm.models.deepseek_v4.xpu.xpu_sparse_decode_fp8 import (
    xpu_sparse_decode_fp8 as triton_sparse_decode_fp8,
)
from vllm.platforms import current_platform
from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


class DeepseekV4TritonSparseAttention(DeepseekV4Attention):
    """Slow CUDA/Triton sparse MLA fallback for DeepSeek V4.

    The class intentionally keeps the base CUDA qnorm/RoPE/KV-insert path and
    only replaces the sparse MLA attention call that currently fails on SM121.
    """

    backend_cls = DeepseekV4FlashInferMLASparseBackend
    use_fp8_ds_mla_layout = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            swa_only = self.compress_ratio <= 1
            n_compressed = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            max_kv = n_compressed + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, max_kv, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        num_decode_tokens = swa_metadata.num_decode_tokens
        if swa_metadata.num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if swa_metadata.num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None

        triton_sparse_decode_fp8(
            q=q,
            kv_cache=kv_cache,
            swa_kv_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            attn_sink=self.attn_sink,
            softmax_scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            out=output,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        use_compressed = (
            self.compress_ratio > 1
            and attn_metadata is not None
            and compressed_k_cache is not None
        )

        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if use_compressed:
            if self.compress_ratio == 4 and self.topk_indices_buffer is not None:
                topk_indices = self.topk_indices_buffer[
                    num_decode_tokens : num_decode_tokens + num_prefill_tokens
                ]
            elif self.compress_ratio != 4:
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            else:
                topk_indices = None
            if topk_indices is None:
                use_compressed = False
        if use_compressed:
            top_k = topk_indices.shape[-1]
            n_compressed = (
                self.max_model_len + self.compress_ratio - 1
            ) // self.compress_ratio
        else:
            if self.topk_indices_buffer is not None:
                topk_indices = self.topk_indices_buffer[
                    num_decode_tokens : num_decode_tokens + num_prefill_tokens
                ]
            else:
                # The combine kernel is compiled with arange(0, width).  Keep a
                # dummy non-zero width while TOP_K=0 makes every top-k entry
                # inactive, so the result remains SWA-only.
                topk_indices = torch.empty(
                    (num_prefill_tokens, 1),
                    dtype=torch.int32,
                    device=q.device,
                )
            top_k = 0
            n_compressed = 0

        max_kv = n_compressed + self.window_size + self.max_num_batched_tokens
        num_chunks = (
            swa_metadata.num_prefills + self.PREFILL_CHUNK_SIZE - 1
        ) // self.PREFILL_CHUNK_SIZE

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((self.PREFILL_CHUNK_SIZE, max_kv, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(
                chunk_start + self.PREFILL_CHUNK_SIZE, swa_metadata.num_prefills
            )
            chunk_size = chunk_end - chunk_start
            if use_compressed:
                assert attn_metadata is not None
                assert compressed_k_cache is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                    use_fnuz=False,
                )

            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=n_compressed,
                use_fnuz=current_platform.is_fp8_fnuz(),
            )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, _ = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                max_kv,
                n_compressed,
            )

            kv_ws = kv[:chunk_size].reshape(-1, 1, q.shape[-1])
            out, _, _ = triton_bf16_mla_sparse_interface(
                q=q[query_start:query_end],
                kv=kv_ws,
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                d_v=q.shape[-1],
                block_dpe=0,
            )
            output[query_start:query_end] = out
