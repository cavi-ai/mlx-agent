# Scout: discover with evidence

Scout is read-only. It ranks MLX candidates for the host and records whether each field is a fetched fact, estimate, or heuristic.

```bash
python3 scripts/mlx-agent discover --role coding --limit 5 --json
python3 scripts/mlx-agent discover --role coding --state-dir .mlx-agent/scout-cache --json
python3 scripts/mlx-agent discover --role coding --state-dir .mlx-agent/scout-cache --offline --json
```

`--offline` never contacts Hugging Face. It needs a cache created by an earlier online run with the same request policy; otherwise Scout returns `offline_cache_missing`. Use `MLX_AGENT_STATE_DIR` instead of `--state-dir` when a host should supply one durable cache location.

In native hosts, ask for `/mlx-scout` except Codex, where the installed skill is `$mlx-agent:mlx-scout`. Fixture-backed output carries a synthetic warning and is not live model-catalog evidence.

## Roles and tool-use evidence

Scout roles are `general`, `coding`, `reasoning`, `vision`, `embedding`, and `tool-use`. A model retains its primary role and can also have `tool-use` membership. Tool-use membership may come from explicit Hugging Face chat-template or tag metadata, with a narrow model-name heuristic as a fallback, but metadata is not verification.

Verification status and evidence strength answer different questions. A `metadata-only` status means no runtime tool call was verified; it also covers heuristic-only, no-runtime evidence. The strength field distinguishes `metadata_only` from the weaker `heuristic_only` source, while `runtime_inventory` and `runtime_tested` record local evidence.
