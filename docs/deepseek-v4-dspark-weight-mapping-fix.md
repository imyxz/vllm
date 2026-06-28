# DSpark draft weight-mapping fix (for the GB10 re-validation)

Commit: `4d051fc` on `claude/vllm-deepseek-v4-flash-oy0hp7`
File changed: `vllm/models/deepseek_v4/nvidia/dspark.py` (`DeepSeekV4DSparkMTP.load_weights`)

This addresses **Next Fix #1** in `docs/deepseek-v4-dspark-gb10-validation.md`
(incomplete DSpark draft weight mapping → ~67 loaded params → ~1–2% acceptance →
no speedup).

## Root cause

The DSpark draft reuses `DeepseekV4DecoderLayer` for each `mtp.{i}` stage, so its
parameters use the **same fused layout as the base DeepSeek-V4 model**. The base
(and standard-MTP) loaders fuse several checkpoint tensors into single params:

| Checkpoint tensor(s)                         | Fused vLLM param            | shard_id |
| -------------------------------------------- | --------------------------- | -------- |
| `attn.wq_a`                                  | `attn.fused_wqa_wkv`        | 0        |
| `attn.wkv`                                   | `attn.fused_wqa_wkv`        | 1        |
| `ffn.shared_experts.w1`                      | `...shared_experts.gate_up_proj` | 0   |
| `ffn.shared_experts.w3`                      | `...shared_experts.gate_up_proj` | 1   |
| `ffn.shared_experts.w2`                      | `...shared_experts.down_proj`    | —   |

The old DSpark `load_weights` copied `mtp_block.*` tensors verbatim with
`default_weight_loader`, so `mtp.{i}.attn.wq_a`, `mtp.{i}.attn.wkv`, and
`mtp.{i}.ffn.shared_experts.w{1,2,3}` never matched any parameter name and were
silently skipped (logged as `unmapped checkpoint weight ...`). The draft then ran
with **uninitialized attention and shared-expert weights**, producing garbage
drafts and near-zero acceptance.

## Fix

In `DeepSeekV4DSparkMTP.load_weights`, for the `mtp_block.*` (non-stage-local,
non-routed-expert) branch, apply the same fused mapping the base loader uses
before falling back to `default_weight_loader`:

- rename `.shared_experts.w2` → `.shared_experts.down_proj`;
- map `attn.wq_a`/`attn.wkv` → `attn.fused_wqa_wkv` (shards 0/1);
- map shared-expert `w1`/`w3` → `gate_up_proj` (shards 0/1);
- call the fused param's `weight_loader(param, loaded_weight, shard_id)`.

Routed experts (`.experts.`) are still handled by the expert mapping; FP8/FP4
scale-suffix and the `ffn.gate.bias` → `e_score_correction_bias` renames are
handled by `_map_deepseek_v4_weight_name` and remain unchanged.

## How to re-validate on the 2× DGX Spark GB10

1. Pull the branch (`git pull`), keep the GB10 fallback env from the previous run
   (KV `fp8_ds_mla`, MoE `marlin`, linear `triton`, DeepGEMM/MegaMoE disabled).
2. Serve with DSpark:
   ```
   vllm serve deepseek-ai/DeepSeek-V4-Flash-DSpark \
     --tensor-parallel-size 2 --max-model-len 4096 --enforce-eager \
     --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
   ```
3. Confirm in the startup logs:
   - `DSpark draft model loaded: N params` — **N must be much larger than 67**.
   - **No** remaining `DSpark: unmapped checkpoint weight ...` warnings for
     `attn.wkv`, `attn.wq_a`, or `shared_experts.w{1,2,3}`. If any remain, paste
     the exact names — there may be more suffixes to map.
4. Correctness (Next Fix #3): greedy (`temperature=0`) base-vs-DSpark on the same
   prompts → token sequences should be **identical**.
5. Acceptance / throughput (Next Fix #4): average draft acceptance should be
   **materially higher than 1–2%**, with mean acceptance length well above ~1.1,
   and decode tok/s above the base fallback.

## If acceptance is still low after this

Then the remaining suspects are (in order):
1. **Target hidden-state capture** (Next Fix #2): verify
   `DeepseekV4ForCausalLM.get_dspark_target_hidden_states()` returns a tensor of
   shape `(num_tokens, len(dspark_target_layer_ids) * hidden_size)` and that the
   captured values come from the right layers (`dspark_target_layer_ids`, mean
   over the `hc_mult` streams).
2. **Proposer block attention** (`vllm/v1/spec_decode/dspark.py`): the draft
   block forward must place `[last_verified_token, noise, ...]` and attend to the
   `main_x` conditioning; confirm the attention metadata / KV slots match the
   reference sliding-window block attention.

Report back: the new loaded-param count, any residual unmapped-weight warnings,
acceptance rate / mean acceptance length, and one short base-vs-DSpark greedy
output pair.
