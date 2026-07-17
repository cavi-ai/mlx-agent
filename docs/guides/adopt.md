# Adopt: preserve evidence before recommending

Adopt creates a resumable, explicit state file. It may inspect a local runtime, but it does not download a model or change a provider configuration.

```bash
python3 scripts/mlx-agent adopt start --state .mlx-agent/adoption.json --role coding --json
python3 scripts/mlx-agent adopt status --state .mlx-agent/adoption.json --json
python3 scripts/mlx-agent adopt resume --state .mlx-agent/adoption.json --json
```

The state records the discovery facts, verification evidence level, shortlist, decisions, and next phase. Treat `runtime_tested` as stronger than `runtime_inventory`, `metadata_only`, or `heuristic_only`; a recommendation is not proof that a model is available, licensed, or suitable for production.

Use `/mlx-adopt` in command-based hosts or `$mlx-agent:mlx-adopt` in Codex. Keep the state path visible to the user and resume it instead of recreating policy after interruption.
