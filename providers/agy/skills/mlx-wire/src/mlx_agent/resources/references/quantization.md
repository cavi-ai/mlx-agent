# MLX quantization guide

How to pick a quant on Apple Silicon. Read this before recommending a specific quantized repo.

## The tradeoff ladder

| Quant | Size vs bf16 | Quality | When to pick |
| --- | --- | --- | --- |
| `bf16` / `fp16` | 1.0× | reference | Fine-tuning source, quality ceiling checks. Rarely for serving. |
| `8bit` | ~0.5× | near-reference | RAM is comfortable and the task is quality-sensitive (reasoning, code review). |
| `4bit` | ~0.25× | good | Default serving choice. Best size/quality point for most models. |
| `MXFP4` | ~0.25× | good | Native format for gpt-oss; use the publisher's own MXFP4 builds. |
| `3bit` and below | ≤0.2× | noticeably worse | Only when nothing else fits; expect degradation on reasoning and tool calling. |

Rules of thumb:

- **Spend RAM on a bigger model at 4bit, not a smaller model at 8bit.** A 32B at 4bit (~18 GB) usually beats a 14B at 8bit (~15 GB) on hard tasks.
- **Mixed MoE quants matter.** MoE models (e.g. Qwen3-A3B) activate few parameters per token, so at the same RAM they decode much faster than dense. On memory-bandwidth-bound Apple Silicon, active parameters set decode speed.
- **Weights are not the whole budget.** KV cache grows linearly with context: `2 × layers × kv_heads × head_dim × context × 2 bytes` (fp16). A model that "fits" can still die at long context. `mlx-agent discover --context N` computes this for you; the candidate's `estimates.kv.max_context_tokens` is the honest ceiling.
- **Publisher quants beat community re-quants** for quality-sensitive roles: mlx-community and the original publisher calibrate with different data and care. When both exist, prefer the publisher or `mlx-community` (trusted), then unsloth.
- **LoRA preserves quant.** `mlx_lm.lora` trains on the quantized base and `mlx_lm.fuse` keeps the quant, so you do not need bf16 weights to fine-tune.

## Reasoning models and quant

Reasoning models emit long hidden chains, so they are more sensitive to aggressive quant than chat models (errors compound over thousands of generated tokens). Prefer 8bit for a reasoning role when RAM allows; 4bit is acceptable for MXFP4-native families.
