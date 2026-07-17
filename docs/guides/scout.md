# Scout: discover with evidence

Scout is read-only. It ranks MLX candidates for the host and records whether each field is a fetched fact, estimate, or heuristic.

```bash
python3 scripts/mlx-agent discover --role coding --limit 5 --json
python3 scripts/mlx-agent discover --role coding --state-dir .mlx-agent/scout-cache --json
python3 scripts/mlx-agent discover --role coding --state-dir .mlx-agent/scout-cache --offline --json
```

`--offline` never contacts Hugging Face. It needs a cache created by an earlier online run with the same request policy; otherwise Scout returns `offline_cache_missing`. Use `MLX_AGENT_STATE_DIR` instead of `--state-dir` when a host should supply one durable cache location.

In native hosts, ask for `/mlx-scout` except Codex, where the installed skill is `$mlx-agent:mlx-scout`. Fixture-backed output carries a synthetic warning and is not live model-catalog evidence.
