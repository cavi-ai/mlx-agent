# Your first local model

Run these commands from a checkout after [installation](installation.md).
Have Python 3.11+ and a text-model runtime available. Replace `<repo>` with a
selected MLX repository. Duration depends on model size and cached weights.

## 1. Discover

Inspect the host, then discover candidates:

```bash
python3 scripts/mlx-agent inspect-host --json
python3 scripts/mlx-agent discover --role coding --limit 3 --json
```

Review sizes, license, gating, runtime compatibility, and evidence. Add
`--context 32768` to check estimated weights plus KV-cache memory at that context.
Fit is an estimate; leave room for runtime overhead and other applications.
Discovery does not download weights.

## 2. Make weights available and serve them

If weights are not already cached, use your runtime's download workflow or
[Intake](../lifecycle/intake.md). Intake fetching requires `huggingface_hub`
in the selected environment:

```bash
python3 scripts/mlx-agent intake resolve <repo> --json
python3 scripts/mlx-agent intake fetch <repo> --json
# Review the download plan; copy data.plan.preview_hash.
python3 scripts/mlx-agent intake fetch <repo> --confirm --preview-hash <fetch-hash> --json
python3 scripts/mlx-agent intake status --json
```

Inspect the finished download before using it. Original checkpoints may need
[conversion](../lifecycle/convert.md); already-usable MLX weights can be served
directly. With `mlx_lm` already installed:

```bash
python3 scripts/mlx-agent serve start --repo <repo> --runtime mlx_lm --port 8080 --json
# Review the launch plan; copy data.plan.preview_hash.
python3 scripts/mlx-agent serve start --repo <repo> --runtime mlx_lm --port 8080 --confirm --preview-hash <serve-hash> --json
python3 scripts/mlx-agent serve status --json
```

Use `--path` instead of `--repo` for weights outside the cache. Inspect the
receipt's log path if startup fails. Ollama and LM Studio manage their own
servers and loading; use their normal workflow instead of `serve start`.

## 3. Verify

Run the resumable adoption workflow:

```bash
python3 scripts/mlx-agent adopt start --state ./adopt.json --role coding --json
python3 scripts/mlx-agent adopt status --state ./adopt.json --json
```

Adopt discovers and checks a shortlist, then retains evidence and recommendations.
Inspect the result: metadata evidence is not a runtime test. If interrupted,
use `adopt resume --state ./adopt.json`. Add `--measure` on start to measure
verified candidates before comparison. See [Adopt](../capabilities/adopt.md).

## 4. Measure

```bash
python3 scripts/mlx-agent bench run --repo <served-model-id> --runtime mlx_lm --json
```

Use the exact model identifier advertised by the running runtime. Bench reports
timings and throughput for an already-served model; it does not start a server
or download weights. See [Bench](../capabilities/bench.md).

## 5. Wire

Choose a dedicated configuration path. Wire emits a runtime/provider block;
integrate it with the consumer that reads that format.

```bash
python3 scripts/mlx-agent wire render <repo> --target mlx_lm --path ./provider.json --json
python3 scripts/mlx-agent wire apply <repo> --target mlx_lm --path ./provider.json --json
# Review the diff; copy data.preview.preview_hash.
python3 scripts/mlx-agent wire apply <repo> --target mlx_lm --path ./provider.json --confirm --preview-hash <wire-hash> --json
```

Retain the returned receipt path. The server must remain available for your
agent to use its connection configuration.

## Inspect and recover

```bash
python3 scripts/mlx-agent wire status <receipt-path> --json
python3 scripts/mlx-agent wire rollback <receipt-path> --json
# Review the rollback preview and use its new hash.
python3 scripts/mlx-agent wire rollback <receipt-path> --confirm --preview-hash <rollback-hash> --json
python3 scripts/mlx-agent serve stop --port 8080 --json
```

Rollback restores the configuration backup; stop targets the server owned by a
serve receipt. Keep the same receipt root if you supplied `--receipts-dir`.
See [Troubleshooting](../guides/troubleshooting.md) for failed previews and jobs.
