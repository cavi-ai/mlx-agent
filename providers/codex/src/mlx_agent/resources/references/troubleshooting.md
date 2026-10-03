# Troubleshooting local MLX serving

Symptom-first playbook. Each entry: cause, then the fix. Run `mlx-agent doctor models` first — half of these are visible there.

## "It was working yesterday"

1. `mlx-agent doctor models` — look for `drift_missing_model` (deleted weights), `drift_hash_mismatch` (hand-edited config), `endpoint_down`.
2. `mlx-agent serve status` — a serve receipt whose pid died, or an argv mismatch after a manual restart.

## Metal / memory allocation failures at load or mid-generation

- Cause: weights + KV cache exceed unified memory under pressure. macOS will swap-thrash before it errors cleanly.
- Fix: drop context length (KV is linear in context — check `estimates.kv.max_context_tokens` from discovery), close other GPU consumers (browsers, other servers), or drop a quant tier. Two loaded 30B+ models do not coexist on ≤64 GB.

## Slow tokens/sec

- Expected ceiling: decode tok/s ≈ memory bandwidth ÷ active bytes per token. Dense 32B 4bit on a high-end M-series is tens of tok/s, not hundreds.
- Checklist: MoE instead of dense (2–5× at the same RAM); native `mlx_lm` on native MLX weights instead of Ollama for MoE; shorter context (KV reads grow); don't bench while another server is loaded — measurements contend.
- Prompt processing (prefill) is compute-bound, not bandwidth-bound; long prompts are slow on every runtime. Measure with `mlx-agent bench run` before blaming the model.

## Server answers but agent calls fail

- Port collision: two configs on one port (`doctor models` → `drift_endpoint_conflict`). Convention: mlx_lm `:8080`, mlx-vlm `:8083`, LiteLLM `:4000`, Ollama `:11434`, LM Studio `:1234`.
- Auth header: wired configs reference `MLX_AGENT_LOCAL_API_KEY` by design; export any value, but the variable must exist for OpenAI-compatible clients that send a key.
- Chat template missing/wrong: outputs look like raw completion text, tool calls never parse. Check the repo ships a `chat_template`; see `references/model-families.md` for family quirks (Gemma has no system role).

## Tool calling fails verification

- Metadata ≠ behavior. Only a schema-valid synthetic call counts. If the probe fails with `malformed_arguments` or `missing_tool_call`, the model (or its template at that quant) does not do reliable tool use — pick a different model for the tool-use role; do not lower the bar.

## Gated repositories

- `gated: true` means HF auth is required before any runtime can pull. Accept the terms on the Hub page with the account that owns the local `HF_TOKEN`; nothing in this pack downloads, so nothing here can accept terms for you.
