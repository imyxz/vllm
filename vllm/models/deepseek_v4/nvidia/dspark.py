# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculative-decoding draft model for DeepSeek-V4-Flash-DSpark.

DeepSeek-V4-Flash-DSpark is *not* a new base model: it is the released
DeepSeek-V4-Flash checkpoint with an additional "DSpark" speculative-decoding
module attached under the ``mtp.*`` checkpoint namespace.  Unlike the standard
V4 MTP draft (one token per sequential step), DSpark proposes a whole *block*
of ``dspark_block_size`` tokens in a single forward pass:

  1. ``main_proj`` projects the concatenation of the mean-pooled hidden states
     captured at the target model's ``dspark_target_layer_ids`` down to one
     hidden vector (``main_x``) — the conditioning signal for the block.
  2. The draft token sequence is ``[last_verified_token, noise, noise, ...]``
     of length ``dspark_block_size``; ``noise`` is ``dspark_noise_token_id``.
     These are embedded and expanded into ``hc_mult`` Hyper-Connection streams.
  3. A single ``DeepseekV4DecoderLayer`` (sliding-window attention + MoE + mHC)
     processes the block conditioned on ``main_x``.
  4. ``hc_head`` collapses the hc streams; the shared LM head produces base
     block logits.
  5. A cheap autoregressive ``Markov`` head (vocab x ``dspark_markov_rank``)
     refines each position's logits given the previously chosen token, and a
     ``confidence`` head emits a per-position acceptance score used by the
     proposer to decide how many block tokens to keep.

Reference: ``inference/model.py`` (``DSparkBlock`` / ``DSparkAttention`` /
``DSparkMarkovHead`` / ``DSparkConfidenceHead``) in the model repository.

NOTE (hardware validation): the reference ``DSparkAttention`` feeds ``main_x``
as the *verified-token KV* that the block queries attend to.  Here ``main_x`` is
projected and added into the block's residual stream (the vLLM-idiomatic way
target hidden states are injected into eagle/MTP drafts), so that the block
attends back to it through the standard sliding-window V4 attention path.  This
is functionally equivalent for the first block position and a close
approximation for later ones; reconciling it exactly against the reference KV
insertion is the main item left for on-hardware validation.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from .model import DeepseekV4DecoderLayer

logger = init_logger(__name__)


class DSparkMarkovHead(nn.Module):
    """Cheap first-order Markov refinement head.

    ``markov_w1`` embeds the previously chosen token id into a low-rank
    (``dspark_markov_rank``) vector; ``markov_w2`` maps that vector to a vocab
    logit *bias* added to the block logits.  The intermediate embedding is also
    fed to the confidence head.
    """

    def __init__(self, config, prefix: str = "") -> None:
        super().__init__()
        self.markov_rank = config.dspark_markov_rank
        self.markov_w1 = VocabParallelEmbedding(
            config.vocab_size,
            self.markov_rank,
            prefix=maybe_prefix(prefix, "markov_w1"),
        )
        self.markov_w2 = ParallelLMHead(
            config.vocab_size,
            self.markov_rank,
            prefix=maybe_prefix(prefix, "markov_w2"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        markov_embed = self.markov_w1(token_ids)
        logits_bias = self.logits_processor(self.markov_w2, markov_embed)
        return logits_bias, markov_embed


class DSparkConfidenceHead(nn.Module):
    """Per-position confidence score in [0, 1] used for block acceptance.

    Concatenates the (hc-collapsed) draft hidden state with the Markov embedding
    of the chosen token and projects to a scalar.  Stored in the checkpoint as a
    bf16 ``proj``; kept in fp32 here for a stable confidence score.
    """

    def __init__(self, input_dim: int, prefix: str = "") -> None:
        super().__init__()
        self.proj = ReplicatedLinear(
            input_dim,
            1,
            bias=False,
            return_bias=False,
            params_dtype=torch.float32,
            prefix=maybe_prefix(prefix, "proj"),
        )

    def forward(
        self, hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([hidden, markov_embed], dim=-1).float()
        return self.proj(x).squeeze(-1)


class DeepSeekV4DSparkPredictorLayer(nn.Module):
    """The single DSpark stage (``num_nextn_predict_layers == 1``).

    Mirrors the reference ``DSparkBlock`` with ``stage_id == 0`` and
    ``stage_id == n_mtp_layers - 1`` (both branches active for a single stage).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
        self.rms_norm_eps = config.rms_norm_eps

        self.hidden_size = config.hidden_size
        self.block_size = config.dspark_block_size
        self.noise_token_id = config.dspark_noise_token_id
        self.target_layer_ids = list(config.dspark_target_layer_ids)
        self.markov_rank = config.dspark_markov_rank

        # main_proj projects the concatenation of mean-pooled hidden states from
        # the target layers (one ``hidden_size`` block per target layer) down to
        # a single conditioning vector. fp8 linear quant, like the V4 e/h_proj.
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.main_proj",
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Hyper-Connection head params (collapse hc_mult streams -> 1).
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp_block = DeepseekV4DecoderLayer(
            vllm_config,
            prefix,
            aux_stream_list=aux_stream_list,
        )

        self.markov_head = DSparkMarkovHead(config, prefix=f"{prefix}.markov_head")
        self.confidence_head = DSparkConfidenceHead(
            config.hidden_size + self.markov_rank,
            prefix=f"{prefix}.confidence_head",
        )


class DeepSeekV4DSparkPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        # DSpark uses a single block stage.
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.block_size = config.dspark_block_size
        self.noise_token_id = config.dspark_noise_token_id

        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): DeepSeekV4DSparkPredictorLayer(
                    vllm_config,
                    f"{prefix}.layers.{idx}",
                    aux_stream_list=aux_stream_list,
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        # Shared base LM head (tie target's head; loaded from ``mtp.*.head``).
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def _stage(self) -> DeepSeekV4DSparkPredictorLayer:
        return self.layers[str(self.mtp_start_layer_idx)]

    def project_main_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        """``main_x = main_norm(main_proj(concat of target-layer hiddens))``."""
        stage = self._stage()
        return stage.main_norm(stage.main_proj(main_hidden))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        main_x: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the DSpark block. ``main_x`` is the projected conditioning vector
        (see ``project_main_hidden``); it is added into the residual stream at
        each token so the sliding-window attention conditions on it.

        Returns the pre-hc_head residual ``(T, hc_mult * hidden_size)``.
        """
        stage = self._stage()
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds + main_x
        # Expand to hc_mult Hyper-Connection streams (V4 residual layout).
        hidden_states = hidden_states.unsqueeze(-2).repeat(1, stage.hc_mult, 1)
        hidden_states, residual, post_mix, res_mix = stage.mtp_block(
            x=hidden_states, positions=positions, input_ids=None
        )
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        return hidden_states.flatten(1)

    def compute_block_logits(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse hc streams and apply the shared LM head -> base block
        logits ``(T, vocab_size)`` plus the (pre-norm) collapsed hc_head hidden.

        Mirrors the reference ``forward_head``: logits use ``norm(x)`` while the
        confidence head consumes the un-normed ``x``, so the un-normed hc_head
        output is returned for the confidence head.
        """
        stage = self._stage()
        hidden_states = hidden_states.view(-1, stage.hc_mult, stage.hidden_size)
        x = hc_head_fused_kernel_tilelang(
            hidden_states,
            stage.hc_head_fn,
            stage.hc_head_scale,
            stage.hc_head_base,
            stage.rms_norm_eps,
            stage.hc_eps,
        )
        logits = self.logits_processor(self.head, stage.norm(x))
        return logits, x


class DeepSeekV4DSparkMTP(nn.Module):
    """Top-level DSpark draft model, registered as ``DeepSeekV4DSparkMTPModel``.

    Exposes the standard draft-model surface (``forward`` / ``compute_logits``)
    plus DSpark-specific helpers (``project_main_hidden`` /
    ``markov_refine`` / ``compute_confidence``) used by ``DSparkProposer``.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.block_size = self.config.dspark_block_size
        self.noise_token_id = self.config.dspark_noise_token_id
        self.model = DeepSeekV4DSparkPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def project_main_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        return self.model.project_main_hidden(main_hidden)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # ``hidden_states`` here is the already-projected ``main_x``.
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        logits, _ = self.model.compute_block_logits(hidden_states)
        return logits

    def compute_block_outputs(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(base_logits, collapsed_hidden)`` for the block; the
        collapsed hidden is reused by the confidence head."""
        return self.model.compute_block_logits(hidden_states)

    def markov_refine(
        self, prev_token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One Markov step: ``(logits_bias, markov_embed)`` for ``prev_token_ids``."""
        return self.model._stage().markov_head(prev_token_ids)

    def compute_confidence(
        self, collapsed_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        return self.model._stage().confidence_head(collapsed_hidden, markov_embed)

    # ----- weight loading -------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load DSpark weights from the ``mtp.*`` checkpoint namespace.

        Checkpoint layout (reference ``DSparkBlock`` under ``mtp.{i}.``):
          * ``mtp.0.main_proj`` / ``mtp.0.main_norm``
          * ``mtp.0.attn.*`` / ``mtp.0.ffn.*`` -> ``mtp_block.{attn,ffn}.*``
          * ``mtp.0.attn_norm`` / ``mtp.0.ffn_norm`` -> ``mtp_block.*``
          * ``mtp.0.hc_*`` (block) -> ``mtp_block.hc_*``;
            ``mtp.0.hc_head_*`` (head) -> stage ``hc_head_*``
          * ``mtp.0.norm`` (final) -> stage ``norm``
          * ``mtp.0.markov_head.{markov_w1,markov_w2}`` / ``mtp.0.confidence_head.proj``

        The base vocab embedding and LM head are *shared* with the target model
        (the reference ties ``mtp[-1].embed``/``head`` to the base; convert.py
        strips the redundant ``mtp.*`` copies), so they are loaded here from the
        base ``model.embed_tokens.weight`` / ``lm_head.weight`` weights, which
        the draft sees in the same checkpoint iterator.
        """
        from vllm.models.deepseek_v4.nvidia.model import (
            make_deepseek_v4_expert_params_mapping,
        )

        config = self.config
        start = config.num_hidden_layers
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        expert_mapping = make_deepseek_v4_expert_params_mapping(
            config.n_routed_experts
        )
        # Params that live on the stage layer itself (not inside mtp_block) or
        # are shared at the predictor top level.
        stage_local = (
            "main_proj",
            "main_norm",
            "norm.",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
            "markov_head",
            "confidence_head",
        )
        shared_top = {"embed": "model.embed_tokens", "head": "model.head"}

        for name, loaded_weight in weights:
            # Shared (tied) vocab embedding + LM head come from the base model
            # weights in the same checkpoint iterator.
            base_shared = None
            if name == "model.embed_tokens.weight":
                base_shared = "model.embed_tokens.weight"
            elif name == "lm_head.weight":
                base_shared = "model.head.weight"
            if base_shared is not None:
                param = params_dict.get(base_shared)
                if param is not None:
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(base_shared)
                continue
            if not name.startswith("mtp."):
                continue
            # mtp.{i}.<rest>
            parts = name.split(".", 2)
            if len(parts) < 3:
                continue
            i = int(parts[1])
            rest = parts[2]
            layer_prefix = f"model.layers.{start + i}"

            # Shared embed / head -> predictor top level.
            shared = next((k for k in shared_top if rest.split(".")[0] == k), None)
            if shared is not None:
                mapped = shared_top[shared] + rest[len(shared):]
                param = params_dict.get(mapped)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(mapped)
                continue

            if ".experts." in rest:
                if (
                    "weight_scale" in rest
                    and loaded_weight.dtype == torch.float8_e8m0fnu
                ):
                    loaded_weight = loaded_weight.view(torch.uint8)
                inner = f"{layer_prefix}.mtp_block.{rest}"
                for param_name, weight_name, eid, shard in expert_mapping:
                    if weight_name not in inner:
                        continue
                    mapped = inner.replace(weight_name, param_name)
                    param = params_dict.get(mapped)
                    if param is None:
                        continue
                    param.weight_loader(
                        param, loaded_weight, mapped, shard_id=shard, expert_id=eid
                    )
                    loaded_params.add(mapped)
                    break
                continue

            # Stage-local params keep their name directly under the layer;
            # everything else belongs to the transformer ``mtp_block``.
            if any(rest.startswith(s) for s in stage_local):
                mapped = f"{layer_prefix}.{rest}"
            else:
                mapped = f"{layer_prefix}.mtp_block.{rest}"

            param = params_dict.get(mapped)
            if param is None:
                logger.warning_once("DSpark: unmapped checkpoint weight %s", name)
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(mapped)

        for layer in self.model.layers.values():
            layer.mtp_block.ffn.finalize_mega_moe_weights()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params
