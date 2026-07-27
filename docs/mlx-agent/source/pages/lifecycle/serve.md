# Serve

Serve launches local MLX servers through the same preview-confirm discipline as wire.

```bash
python3 scripts/mlx-agent serve start --repo <repo> --runtime mlx_lm
python3 scripts/mlx-agent serve start --repo <repo> --runtime mlx_lm --confirm --preview-hash <hash>
python3 scripts/mlx-agent serve status
python3 scripts/mlx-agent serve stop --port 8080
```

## Hard gates

- Model present in the Hugging Face cache (never downloads).
- Runtime executable already installed (never installs).
- Port free and unclaimed by wired configs; loopback-only bind.
- Every start writes a receipt: pid, argv, port, log path.

`serve stop` signals only receipt-owned processes, and only after verifying the live pid's argv matches the receipt — never a port-scan kill. `serve status` cross-checks receipts against live processes.

## Recipes and adapters

`mlx_lm` (default :8080) and `mlx-vlm` (default :8083) launch directly; Ollama and LM Studio manage their own servers. `--adapter-path` serves a trained LoRA alongside the base on `mlx_lm`.

## launchd

`serve start --launchd` installs the same plan as a launchd agent: a deterministic plist (managed `com.mlx-agent.serve.<port>` label) applied through the transaction flow. It prints the exact `launchctl bootstrap` command — loading stays with you.
