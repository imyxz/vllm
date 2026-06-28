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
  3. The ``mtp.0..n`` DSpark stages process the block conditioned on ``main_x``.
  4. The final stage ``hc_head`` collapses the hc streams; the shared LM head
     produces base block logits.
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

import re
from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
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


def _map_deepseek_v4_weight_name(name: str, expert_dtype: str) -> str:
    """Apply the DeepSeek-V4 checkpoint-to-vLLM suffix mapping used by the
    base model loader, without changing the mtp.* prefix handled below."""
    if expert_dtype == "fp4":
        name = re.sub(
            r"(\.experts\.\d+\.w[123])\.scale$",
            r"\1.weight_scale",
            name,
        )
        name = re.sub(r"\.scale$", ".weight_scale_inv", name)
    else:
        name = re.sub(r"\.scale$", ".weight_scale_inv", name)
    if name.endswith(".ffn.gate.bias"):
        name = name.removesuffix(".ffn.gate.bias") + (
            ".ffn.gate.e_score_correction_bias"
        )
    return name


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
    """One DSpark stage from the ``mtp.*`` checkpoint namespace.

    The released DSpark checkpoint stores three stages: ``mtp.0`` owns
    ``main_proj``/``main_norm``, intermediate stages only own a decoder block,
    and the final stage owns ``hc_head``/``norm``/``markov_head``/
    ``confidence_head``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        stage_id: int,
        n_mtp_layers: int,
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
        self.stage_id = stage_id
        self.is_first_stage = stage_id == 0
        self.is_last_stage = stage_id == n_mtp_layers - 1

        if self.is_first_stage:
            # main_proj projects the concatenation of mean-pooled hidden states
            # from the target layers down to one conditioning vector.
            self.main_proj = ReplicatedLinear(
                config.hidden_size * len(self.target_layer_ids),
                config.hidden_size,
                bias=False,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.main_proj",
            )
            self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size

        self.mtp_block = DeepseekV4DecoderLayer(
            vllm_config,
            prefix,
            aux_stream_list=aux_stream_list,
        )

        if self.is_last_stage:
            # Hyper-Connection head params (collapse hc_mult streams -> 1).
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
            self.markov_head = DSparkMarkovHead(
                config, prefix=f"{prefix}.markov_head"
            )
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + self.markov_rank,
                prefix=f"{prefix}.confidence_head",
            )


class DeepSeekV4DSparkPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = (
            getattr(config, "n_mtp_layers", None)
            or len(getattr(config, "dspark_target_layer_ids", []) or [])
            or getattr(config, "num_nextn_predict_layers", 1)
        )
        self.block_size = config.dspark_block_size
        self.noise_token_id = config.dspark_noise_token_id

        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): DeepSeekV4DSparkPredictorLayer(
                    vllm_config,
                    f"{prefix}.layers.{idx}",
                    stage_id=idx - self.mtp_start_layer_idx,
                    n_mtp_layers=self.num_mtp_layers,
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

    def _first_stage(self) -> DeepSeekV4DSparkPredictorLayer:
        return self.layers[str(self.mtp_start_layer_idx)]

    def _last_stage(self) -> DeepSeekV4DSparkPredictorLayer:
        return self.layers[str(self.mtp_start_layer_idx + self.num_mtp_layers - 1)]

    def project_main_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        """``main_x = main_norm(main_proj(concat of target-layer hiddens))``."""
        stage = self._first_stage()
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
        first_stage = self._first_stage()
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds + main_x
        # Expand to hc_mult Hyper-Connection streams (V4 residual layout).
        hidden_states = hidden_states.unsqueeze(-2).repeat(
            1, first_stage.hc_mult, 1
        )
        residual, post_mix, res_mix = None, None, None
        for stage in self.layers.values():
            hidden_states, residual, post_mix, res_mix = stage.mtp_block(
                x=hidden_states,
                positions=positions,
                input_ids=None,
                post_mix=post_mix,
                res_mix=res_mix,
                residual=residual,
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
        stage = self._last_stage()
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
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
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
        return self.model._last_stage().markov_head(prev_token_ids)

    def compute_confidence(
        self, collapsed_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        return self.model._last_stage().confidence_head(collapsed_hidden, markov_embed)

    # ----- weight loading -------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load DSpark weights from the ``mtp.*`` checkpoint namespace.

        Checkpoint layout (reference ``DSparkBlock`` under ``mtp.{i}.``):
          * ``mtp.0.main_proj`` / ``mtp.0.main_norm``
          * ``mtp.{i}.attn.*`` / ``mtp.{i}.ffn.*`` ->
            ``layers.{num_hidden_layers+i}.mtp_block.{attn,ffn}.*``
          * ``mtp.{i}.attn_norm`` / ``mtp.{i}.ffn_norm`` -> ``mtp_block.*``
          * ``mtp.{last}.hc_head_*`` / ``norm`` / ``markov_head`` /
            ``confidence_head`` -> final stage-local params

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
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = config.num_attention_heads // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

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
            orig_name = name
            name = _map_deepseek_v4_weight_name(name, expert_dtype)
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
                inner = f"{layer_prefix}.mtp_block.{rest}"
                # Shared-expert down projection is stored as ``w2``.
                if ".shared_experts.w2" in inner:
                    inner = inner.replace(
                        ".shared_experts.w2", ".shared_experts.down_proj"
                    )
                # Fused projections, mirroring the base DeepseekV4 loader:
                #   attn.wq_a / attn.wkv      -> attn.fused_wqa_wkv (shards 0/1)
                #   shared_experts.w1 / .w3   -> gate_up_proj        (shards 0/1)
                # Without this the checkpoint's separate wq_a/wkv and shared
                # expert w1/w3 tensors stay unmapped and the draft runs with
                # uninitialized attention / shared-expert weights.
                fused_mapping = (
                    ("attn.fused_wqa_wkv", "attn.wq_a", 0),
                    ("attn.fused_wqa_wkv", "attn.wkv", 1),
                    ("gate_up_proj", "w1", 0),
                    ("gate_up_proj", "w3", 1),
                )
                fused = False
                for param_name, weight_name, shard_id in fused_mapping:
                    if weight_name not in inner:
                        continue
                    cand = inner.replace(weight_name, param_name)
                    param = params_dict.get(cand)
                    if param is None:
                        continue
                    param.weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(cand)
                    fused = True
                    break
                if fused:
                    continue
                mapped = inner

            param = params_dict.get(mapped)
            if param is None:
                logger.warning_once(
                    "DSpark: unmapped checkpoint weight %s (mapped from %s)",
                    name,
                    orig_name,
                )
                continue
            if "attn.attn_sink" in mapped:
                narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                n = narrow_weight.shape[0]
                param[:n].copy_(narrow_weight)
                loaded_params.add(mapped)
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(mapped)

        for layer in self.model.layers.values():
            layer.mtp_block.ffn.finalize_mega_moe_weights()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params
