# Model family quirks (chat templates, tool calling, reasoning)

Family-specific behaviors that change how you wire, prompt, and verify. Read this before debugging a "broken" model — it is usually a template mismatch.

## Qwen (Qwen3 and later)

- Ships **hybrid reasoning**: `enable_thinking` in the chat template toggles hidden `<think>` blocks. `mlx-agent` detects this from the template and tags reasoning models so they stay out of fast/utility roles.
- Tool calling uses the Hermes-style `<tool_call>` JSON blocks; works well through `mlx_lm.server` and LM Studio. Verified tool-use is still per-model — template support is not behavior proof.
- MoE variants (`A3B`, `A35B`) are the speed pick on Apple Silicon; dense variants are the consistency pick.

## Gemma

- **No system role.** A system prompt must be folded into the first user turn; runtimes that auto-inject a system role produce degraded or refused outputs.
- Google ships **QAT (quantization-aware trained)** builds — their 4bit is unusually strong; prefer publisher QAT over generic 4bit re-quants.
- Tool calling is template-level and inconsistent across sizes; verify before assigning a tool-use role.

## gpt-oss (OpenAI open weights)

- **MXFP4 native** — use the MXFP4 builds; do not re-quantize to generic 4bit.
- Uses the **harmony** response format with `reasoning_effort` levels (low/medium/high). These are reasoning models by design: expect hidden analysis channels; never put them in a fast/cheap slot.
- Strong tool calling when served through runtimes that parse harmony (recent `mlx-lm`, LM Studio).

## Llama (Meta, MLX ports)

- Solid generic instruction following; tool calling depends heavily on the exact instruct template version — verify, don't assume.
- Community MLX ports vary in template fidelity; check the repo's `chat_template` presence before wiring (missing template → runtime default, often wrong).

## Vision (Qwen-VL and friends)

- Vision models need `mlx-vlm`; Ollama's engine does not run them. Wire them on their own port (`:8083` by convention) so they never collide with the text server.
- OCR quality varies more with input resolution handling than with quant; keep 4bit unless the documents are dense.

## Embedding models

- Served via `/v1/embeddings` (OpenAI-compatible) or `/api/embed` (Ollama). They cannot chat; a "ready" generation probe is not meaningful for them — rely on the embedding probe, not text generation.
