# MLX runtimes on Apple Silicon

Which runtime actually executes a model, and how to expose it to an OpenAI-compatible agent.

## Ollama (curated MLX)

- Ollama v0.30+ uses MLX as its compute engine on Macs with ≥32 GB unified memory (else falls back to llama.cpp/Metal). Confirm with `ollama ps` → look for `100% GPU`.
- Runs Ollama's **curated** GGUF/quant weights on the MLX runtime — not arbitrary `mlx-community` repos.
- Zero extra setup. Best for tags already in the Ollama library (Gemma QAT, Qwen dense).
- Agent ref: `ollama/<tag>`. Endpoint: `http://127.0.0.1:11434`.

## LM Studio (native MLX) — lowest friction for new models

- Native MLX runtime with a GUI. Download any `mlx-community` / `lmstudio-community` repo, then `lms server start` (OpenAI-compatible at `http://localhost:1234/v1`).
- First-class in OpenClaw: use the built-in `lmstudio` provider, ref `lmstudio/<model>`.

## mlx_lm (native, headless) — text + LoRA

- `pip install mlx-lm`, then:
  `mlx_lm.server --model <hf-repo-or-local> --port 8080 --max-tokens 8192`
- OpenAI-compatible at `http://127.0.0.1:8080/v1`; supports tool calling and `--adapter-path` for serving a LoRA.
- Wire as a custom OpenAI-compatible provider (verify exact key names against your build's provider schema):

  ```jsonc
  // providers[]
  { "id": "mlxlm", "type": "openai", "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "local" }
  ```
  Ref: `mlxlm/<model>`.

## mlx-vlm (native, vision/OCR) — required for VLMs

- Ollama's engine does not run vision models. `pip install mlx-vlm`, then:
  `mlx_vlm.server --model mlx-community/Qwen3-VL-8B-Instruct-4bit --port 8083`
- Custom provider on its own port; ref `mlxvlm/<model>`.

## Native vs Ollama speed

Native `mlx_lm` on native 4-bit MLX weights is meaningfully faster than Ollama on MoE models (~2× tok/s, 3–5× prompt processing); at dense 27B+ both converge on the memory-bandwidth ceiling. Use Ollama for curated tags; add a native runtime for MoE speed, arbitrary HF repos, vision, and LoRA.

## LoRA

`mlx_lm` supports LoRA/QLoRA: `mlx_lm.lora` (train on quantized base) → `mlx_lm.fuse` (fuse, preserves quant) → serve the fused model or `mlx_lm.server --adapter-path ./adapters`. Only worth it for a narrow, stable, high-volume task with proprietary data; otherwise pick a stronger base model.
