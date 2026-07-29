# Runtimes

Which runtime executes a model, and how mlx-agent treats each.

## Ollama (curated MLX)

Ollama 0.30+ uses MLX as its compute engine on Macs with ≥32 GB unified memory. Zero setup for curated tags (`ollama/<tag>`, `:11434`). It cannot run vision models, and it runs Ollama's own curated weights, not arbitrary `mlx-community` repos.

## LM Studio (native MLX)

Native MLX runtime with a GUI. Download any `mlx-community` repo, then `lms server start` for an OpenAI-compatible endpoint at `:1234`. Lowest friction for trying a new model.

## mlx_lm (native, headless)

`pip install mlx-lm`, then `mlx_lm.server --model <repo> --port 8080`. OpenAI-compatible, tool calling, `--adapter-path` for serving a LoRA. Meaningfully faster than Ollama on MoE models. The default text runtime for serve, convert, lora, and fuse.

## mlx-vlm (native, vision)

Required for vision/OCR models (`:8083` by convention). Serves VLMs the other runtimes cannot.

## LiteLLM (router)

One OpenAI-compatible front for many backends (`:4000`). Fleet renders its bounded router YAML subset. A loopback LiteLLM route may point at remote paid backends — mlx-agent verifies route locality before treating it as local.

## Port conventions

| Runtime | Port |
| --- | --- |
| mlx_lm | 8080 |
| mlx-vlm | 8083 |
| LiteLLM | 4000 |
| Ollama | 11434 |
| LM Studio | 1234 |

Bundled reference packs go deeper: `references/quantization.md` (quant tradeoffs), `references/model-families.md` (chat-template and tool-calling quirks), `references/troubleshooting.md` (symptom-first playbook) — shipped inside every generated skill.
