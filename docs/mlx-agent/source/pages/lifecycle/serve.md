# Serve

Serve launches local MLX servers from a reviewed plan. Use `mlx_lm` for text
or `mlx-vlm` for vision/language models. Ollama and LM Studio manage their
own servers.

## Start and inspect

```bash
python3 scripts/mlx-agent serve start --repo <repo> --runtime mlx_lm --port 8080 --json
python3 scripts/mlx-agent serve start --repo <repo> --runtime mlx_lm --port 8080 --confirm --preview-hash <hash> --json
python3 scripts/mlx-agent serve status --json
python3 scripts/mlx-agent serve stop --port 8080 --json
```

Review `data.plan` and its `preview_hash` before confirming with identical
arguments. The source must be cached; use `--path /absolute/local/model`
instead of `--repo` for another local directory. The runtime executable must
already be installed, and the port must be free and unclaimed by wired configs.
Binding is loopback-only.

A receipt records the PID, arguments, port, and log. Status cross-checks receipts
against live processes; an accepted launch is not proof the model finished loading.
Stop verifies process ownership against the receipt before signaling it.
If you supplied `--receipts-dir`, use the same base for status and stop.

For text LoRA serving, add `--adapter-path /absolute/local/adapter` to both
preview and confirmation. The directory must contain valid `adapter_config.json`
and `adapters.safetensors` files without symbolic links. The preview binds both
files' content hashes and sizes. Supervised eager and JIT startup recheck them; a changed
adapter requires a fresh preview. Receipts and status retain the adapter path and
reviewed identity separately from the base model. Historical receipts can expose
an adapter path without a content identity; unavailable historical JIT configuration
is reported as `adapter_status: unknown`. Direct `--launchd` plists retain their
existing deferred-launch behavior; these worker-load checks apply to supervised
serving.

## Load on request

JIT keeps an endpoint reachable without retaining a loaded model worker:

```bash
python3 scripts/mlx-agent serve start --path /absolute/local/model --runtime mlx_lm --port 8080 --jit --json
python3 scripts/mlx-agent serve start --path /absolute/local/model --runtime mlx_lm --port 8080 --jit --confirm --preview-hash <hash> --json
python3 scripts/mlx-agent serve status --json
python3 scripts/mlx-agent serve unload --port 8080 --expected-pid <gateway-pid> --json
```

`/v1/models` stays available while unloaded. Chat/completion requests start one
owned worker from the confirmed local files using Hugging Face offline settings.
Concurrent cold requests share the load. Unload releases the worker and keeps
the gateway reachable; active requests, including streams, block it. Changed
model files need a new serve plan.

Status reports gateway liveness and `model_state` separately. Stop terminates
the owned gateway/worker process group. The gateway serves its selected model
and rejects request-time adapter/draft-model overrides.

JIT rechecks the adapter before and after each worker load, including after
unloading. Older adapter gateways without a recorded content identity require
a fresh serve preview before loading another worker.

## Memory policy

Include policy flags in the initial JIT preview and confirmation:

```bash
python3 scripts/mlx-agent serve start --path /absolute/local/model --runtime mlx_lm --jit --idle-timeout 600 --min-headroom-gb 2 --json
python3 scripts/mlx-agent serve start --path /absolute/local/model --runtime mlx_lm --jit --idle-timeout 600 --min-headroom-gb 2 --confirm --preview-hash <hash> --json
```

Use the gateway PID from status to replace the whole policy without restarting:

```bash
python3 scripts/mlx-agent serve policy --port 8080 --expected-pid <gateway-pid> --idle-timeout 600 --min-headroom-gb 2 --json
python3 scripts/mlx-agent serve policy --port 8080 --expected-pid <gateway-pid> --idle-timeout 600 --keep-loaded --min-headroom-gb 2 --json
```

Idle time starts after the last request or stream finishes. `--idle-timeout 0`
disables automatic unload. `--keep-loaded` prevents automatic unload after use;
it does not preload weights, and manual unload remains available. Omitting
`--keep-loaded` in a policy replacement enables idle unload; omitting
`--min-headroom-gb` disables admission checking. Policy is persisted atomically
in private gateway configuration. Start without policy flags retains manual
unload behavior. Policy flags require JIT.

Before a guarded cold load, fresh OS headroom must cover estimated local weight
bytes × 1.10, a 1.5 GB runtime allowance, and the requested reserve in decimal GB.
Unknown headroom or weight size returns HTTP 503 `memory_unknown`; insufficient
space returns HTTP 503 `insufficient_headroom`. The endpoint remains reachable.
Status includes policy, the dated estimate in `memory_check`, and
`load_blocked_reason`.

The check does not reserve memory across endpoints. Other applications,
concurrent cold loads, and later context growth can change usage. See
[Troubleshooting](../guides/troubleshooting.md).

## Adapters and launchd

`--adapter-path` serves a local LoRA with `mlx_lm`. Default ports are 8080
for `mlx_lm` and 8083 for `mlx-vlm`.

`--runtime auto` reads local `config.json` and selects a declared MLX-LM or
MLX-VLM loader. Optional backends use their isolated executable, bound into
the eager or JIT preview, rather than global PATH. Specialized preconverted
packs use declared backend adapters without importing runtime code from the
model repository. Review the resolved runtime and executable before confirmation.

`serve start --launchd` installs a receipt-owned plist under a managed
`com.mlx-agent.serve.<port>` label. Review and confirm its plan, then run the
printed `launchctl bootstrap` command yourself. JIT does not support direct
`--launchd`; supervise the gateway through an existing process manager instead.
