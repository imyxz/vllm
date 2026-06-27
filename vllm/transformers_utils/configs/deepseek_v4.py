# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import PretrainedConfig


class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"

    def __init__(
        self,
        max_position_embeddings: int = 1048576,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        rope_theta: float = 10000.0,
        # DSpark speculative-decoding module (DeepSeek-V4-Flash-DSpark).
        # All default to the "no DSpark" values so plain V4 / V4-Base
        # checkpoints are unaffected; a DSpark checkpoint sets
        # ``dspark_block_size > 0`` and a non-empty ``dspark_target_layer_ids``.
        num_nextn_predict_layers: int = 0,
        dspark_block_size: int = 0,
        dspark_noise_token_id: int = 0,
        dspark_target_layer_ids: list[int] | None = None,
        dspark_markov_rank: int = 256,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.rope_parameters = rope_scaling or rope_parameters
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.dspark_block_size = dspark_block_size
        self.dspark_noise_token_id = dspark_noise_token_id
        self.dspark_target_layer_ids = dspark_target_layer_ids or []
        self.dspark_markov_rank = dspark_markov_rank
        super().__init__(**kwargs)

    @property
    def is_dspark(self) -> bool:
        """True for DeepSeek-V4-Flash-DSpark checkpoints (the variant with a
        DSpark speculative-decoding module attached under the ``mtp.*``
        namespace)."""
        return self.dspark_block_size > 0 and len(self.dspark_target_layer_ids) > 0
