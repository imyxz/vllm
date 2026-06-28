# DeepSeek V4 Flash DSpark on 2x DGX Spark GB10

Validation date: 2026-06-28

## Setup

- Repository branch: `claude/vllm-deepseek-v4-flash-oy0hp7`
- Model: `/home/imyxz/llm-models/DeepSeek-V4-Flash-DSpark`
- Hardware: 2x DGX Spark GB10, one GPU per node, connected over the 100G interface
- Ray address: `192.168.101.1:6379`
- Tensor parallel size: `2`
- Max model length used for validation: `4096`
- KV cache dtype: `fp8_ds_mla`
- MoE backend: `marlin`
- Linear backend: `triton`
- DeepGEMM E8M0/MegaMoE disabled for GB10 validation fallback

## What Works

- vLLM can route `deepseek_v4` DSpark speculative config to `DeepSeekV4DSparkMTPModel`.
- The base DeepSeek V4 Flash DSpark checkpoint loads across both GB10 nodes with TP=2.
- DSpark proposer construction works on both tensor-parallel ranks.
- DSpark drafter loads without crashing.
- End-to-end serving with `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'` starts successfully.
- Short completion, long prefill, non-streaming decode, and streaming decode requests all complete.

## GB10 Fallbacks Added

- Added a CUDA/Triton sparse MLA fallback for GB10 where the FlashInfer DSv4 sparse MLA path is not executable on SM121.
- Disabled or bypassed SM100-only/unsupported paths for:
  - DeepGEMM MHC on SM12x
  - FlashInfer sparse MLA DSv4 target path
  - cooperative/persistent sparse top-k kernels on SM12x
  - sparse `o_proj` DeepGEMM path, falling back to BF16 where needed
- Decoded E8M0 FP8 scales for Triton block-FP8 fallback.
- Added DSpark-specific vLLM routing, proposer dispatch, target hidden-state plumbing, and multi KV-group metadata support.
- Added tensor-parallel slicing for DSpark `attn_sink` weights.
- Fixed the Triton sparse prefill fallback so SWA-only prefill does not compile `tl.arange(0, 0)`.

## Measured Results

Base fallback run, without DSpark:

- Long prefill: 2304 prompt tokens in 2.506s, about 920 prompt tok/s
- Decode: 64 completion tokens in 10.533s, about 6.08 tok/s
- Streaming decode: TTFT about 0.231s, about 6.66 chunks/s after TTFT

DSpark speculative run:

- Long prefill: 2305 prompt tokens in 2.63-2.74s, about 840-878 prompt tok/s
- Decode: 64 completion tokens in 10.28-10.59s, about 6.05-6.23 tok/s
- Streaming decode: TTFT about 0.159s, about 6.05 chunks/s after TTFT

## Current Conclusion

The DSpark path is now runnable on 2x DGX Spark GB10, but it is not yet a useful acceleration path. Decode throughput is roughly the same as the base fallback path, and prefill is slightly slower in the measured runs.

The main blocker is incomplete DSpark draft weight mapping. The drafter reports only 67 loaded params and logs unmapped checkpoint weights under:

- `mtp.{0,1,2}.attn.wkv.*`
- `mtp.{0,1,2}.attn.wq_a.*`
- `mtp.{0,1,2}.ffn.shared_experts.w{1,2,3}.*`

Speculative acceptance confirms this: average draft acceptance was about 1-2%, with mean acceptance length around 1.05-1.12 tokens. This is effectively no speculative speedup.

Output quality is also not reliable in this fallback state. Some short prompts produce malformed text, so correctness should not be considered validated yet.

## Next Fixes

1. Complete DSpark MTP module mapping for `attn.wkv`, `attn.wq_a`, and shared expert weights/scales.
2. Re-check target hidden-state capture shape and values after the draft module is fully loaded.
3. Re-run greedy base-vs-DSpark token equality tests after the drafter is complete.
4. Re-run acceptance and throughput benchmarks. A healthy DSpark run should show materially higher acceptance than the current 1-2%.
5. Add GB10-specific kernel config/autotune data for the default W8A8 block-FP8 shapes to reduce fallback overhead.
