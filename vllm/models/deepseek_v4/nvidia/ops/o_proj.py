# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum


def compute_fp8_einsum_recipe() -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128.
    SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def _decode_e8m0_scales(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype in (torch.float8_e8m0fnu, torch.uint8):
        return _upcast_e8m0_to_fp32(scale).contiguous()
    return scale.to(torch.float32)


def _expand_2d_block_scales(
    scale: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    scale = _decode_e8m0_scales(scale)
    row_blocks, col_blocks = scale.shape[-2:]
    row_block = math.ceil(rows / row_blocks)
    col_block = math.ceil(cols / col_blocks)
    scale = torch.repeat_interleave(scale, row_block, dim=-2)[..., :rows, :]
    scale = torch.repeat_interleave(scale, col_block, dim=-1)[..., :, :cols]
    return scale


def _get_cached_wo_a_bf16(
    wo_a: nn.Module,
    n_groups: int,
    o_lora_rank: int,
    hidden_dim: int,
) -> torch.Tensor:
    cached = getattr(wo_a, "_dsv4_wo_a_bf16", None)
    if cached is not None:
        return cached
    if hasattr(wo_a, "weight_scale_inv"):
        wo_a_weight = wo_a.weight.view(n_groups, o_lora_rank, hidden_dim).to(
            torch.float32
        )
        wo_a_scale = _expand_2d_block_scales(
            wo_a.weight_scale_inv.view(n_groups, -1, wo_a.weight_scale_inv.shape[-1]),
            o_lora_rank,
            hidden_dim,
        )
        cached = (wo_a_weight * wo_a_scale).to(torch.bfloat16)
    else:
        cached = wo_a.weight.view(n_groups, o_lora_rank, hidden_dim).to(torch.bfloat16)
    wo_a._dsv4_wo_a_bf16 = cached
    return cached


def _inverse_rope_gptj_torch(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    nope_dim = o.shape[-1] - rope_dim
    out = o.to(torch.bfloat16).clone()
    if o.shape[0] == 0 or rope_dim == 0:
        return out

    half = rope_dim // 2
    cos = cos_sin_cache.index_select(0, positions)[:, :half].to(torch.float32)
    sin = cos_sin_cache.index_select(0, positions)[:, half:].to(torch.float32)
    rope = o[..., nope_dim:].to(torch.float32)
    even = rope[..., 0::2]
    odd = rope[..., 1::2]
    out[..., nope_dim::2] = (even * cos[:, None, :] + odd * sin[:, None, :]).to(
        torch.bfloat16
    )
    out[..., nope_dim + 1 :: 2] = (odd * cos[:, None, :] - even * sin[:, None, :]).to(
        torch.bfloat16
    )
    return out


def _bf16_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    rope_dim: int,
    o_lora_rank: int,
) -> torch.Tensor:
    o_ref = _inverse_rope_gptj_torch(o, positions, cos_sin_cache, rope_dim)
    o_ref = o_ref.view(o.shape[0], n_groups, -1)
    wo_a_weight = _get_cached_wo_a_bf16(
        wo_a, n_groups, o_lora_rank, o_ref.shape[-1]
    )
    z = torch.einsum("tgd,grd->tgr", o_ref, wo_a_weight)
    return wo_b(z.flatten(1))


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + FP8 quant + einsum + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /
    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.
    """
    if current_platform.is_device_capability_family(120):
        return _bf16_o_proj(
            o,
            positions,
            cos_sin_cache,
            wo_a,
            wo_b,
            n_groups=n_groups,
            rope_dim=rope_dim,
            o_lora_rank=o_lora_rank,
        )

    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=tma_aligned_scales,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (wo_a.weight, wo_a.weight_scale_inv),
        z,
        recipe=einsum_recipe,
    )
    return wo_b(z.flatten(1))
