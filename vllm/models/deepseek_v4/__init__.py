# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 model — hardware-isolated entry point.

The actual implementation lives under ``nvidia/`` and ``amd/``; this module
picks the right one for the current platform and re-exports the public
classes used by the model registry and quantization config lookup.
"""

from vllm.platforms import current_platform

from .quant_config import DeepseekV4FP8Config


def _unsupported_dspark(platform: str):
    """Placeholder for the DSpark draft on platforms where it isn't wired yet.

    The DSpark draft (DeepSeek-V4-Flash-DSpark) currently reuses the NVIDIA
    ``DeepseekV4DecoderLayer`` and mHC kernels; the ROCm/XPU paths are a
    follow-up. Importing stays cheap — the error is raised only if a DSpark
    checkpoint is actually instantiated on these platforms.
    """

    class _UnsupportedDSpark:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                f"DeepSeek-V4 DSpark speculative decoding is not yet supported "
                f"on {platform}. Run DeepSeek-V4-Flash-DSpark without speculative "
                f"decoding, or use the NVIDIA backend."
            )

    return _UnsupportedDSpark


# Pick the per-platform implementation. The NVIDIA branch is the static
# default that mypy sees; the ROCm/XPU branches override at runtime and are
# kept type-compatible via ``# type: ignore[assignment]``.
if current_platform.is_rocm():
    from .amd.model import DeepseekV4ForCausalLM
    from .amd.mtp import DeepSeekV4MTP

    DeepSeekV4DSparkMTP = _unsupported_dspark("ROCm")
elif current_platform.is_xpu():
    from .xpu.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .xpu.mtp import DeepSeekV4MTP  # type: ignore[assignment]

    DeepSeekV4DSparkMTP = _unsupported_dspark("XPU")
else:
    from .nvidia.dspark import DeepSeekV4DSparkMTP  # type: ignore[assignment]
    from .nvidia.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .nvidia.mtp import DeepSeekV4MTP  # type: ignore[assignment]

__all__ = [
    "DeepSeekV4MTP",
    "DeepSeekV4DSparkMTP",
    "DeepseekV4FP8Config",
    "DeepseekV4ForCausalLM",
]
