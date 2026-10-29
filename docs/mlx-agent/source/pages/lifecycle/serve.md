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

## Serve a trained adapter

Use a completed LoRA receipt to select the base snapshot and adapter together:

```bash
python3 scripts/mlx-agent serve start --training-receipt /absolute/path/training.json --runtime auto --port 8080 --json
python3 scripts/mlx-agent serve start --training-receipt /absolute/path/training.json --runtime auto --port 8080 --confirm --preview-hash <hash> --json
```

The receipt must record completed training, the pinned base file identity and
matching adapter content hashes. The preview captures the receipt bytes as well
as both artifacts; changes require a new preview. Do not combine the receipt with
`--repo`, `--path` or `--adapter-path`. This text LoRA flow uses MLX-LM and supervised
startup; direct `--launchd` is refused. JIT rechecks the same binding on each load
and includes adapter tensors in its memory estimate. Completion is not a model
quality verdict. New training previews record absolute adapter destinations;
relative paths in older receipts resolve from the invocation's working directory
and still have to match the recorded content hashes.

## M5 prompt acceleration

Add `--m5-prefill` to both the preview and confirmation to try the experimental
Q4 prompt kernel with MLX-LM. The default remains native MLX. The kernel applies
to inference-only Qwen2 MLP projections with FP16 activations, affine 4-bit weights,
group size 64 and dimensions divisible by 64, when a batch has at least 128 rows.
It retains the original quantized weight storage. Inputs with fewer than 128
rows use native MLX.

This path requires an M5 GPU (`applegpu_g17s`), macOS 26, MLX and mlx-metal 0.31.1,
and mlx-lm 0.29.1. Unsupported hardware, dependencies, models and input layouts
retain native execution. A compiler failure during model preparation also retains
the original model. LoRA adapters are excluded. The first load compiles the selected
matrix shapes; it can take longer. Performance depends on prompt and model shape.

The preview and receipt record `prefill_kernel`: the requested profile, dependency
versions and bundled source hashes. This records configuration, not proof that a
particular request used the kernel. The serve log reports the adapted layer count.
The adapter wraps only an MLX-LM entrypoint installed alongside the current Python
with a matching interpreter. Other or unavailable entrypoints retain their exact
native launch arguments and record `adapter_enabled: false`; no alternate Python
environment is selected. Other serving runtimes reject this MLX-LM-only option.
Changed source or dependencies require a fresh preview; each JIT worker checks the
same fingerprint before loading. No runtime source or model code is downloaded.

## Code-edit acceleration

Add `--prompt-lookup` to both preview and confirmation for code edits that reuse
source text from the prompt. With client temperature zero, the server proposes up
to eight tokens from repeated prompt sequences and verifies them with the target
model before emitting them. It indexes only the last 8192 prompt tokens. No draft
model or new weights are required. Native batching remains the default when the
option is omitted.

In this mode, plain greedy requests run one at a time. Sampling, logit penalties or
biases, seeds, logprobs, adapters and draft models retain native generation. Only
ordinary KVCache states use lookup; rotating, quantized and hybrid caches retain
native generation. Prompt cache reuse, stop tokens and cancellation retain the
server's token alignment. Performance depends on copied text; ordinary chat may
be slower. Keep this option off for concurrent chat workloads.

The supported runtime is macOS with MLX and mlx-metal 0.31.1 and mlx-lm 0.29.1.
Other dependency versions use native MLX. Like `--m5-prefill`, this option wraps
only the current interpreter's own entrypoint, records source and dependency
hashes (`decode_profile`) and requires a fresh preview if they change. A foreign
entrypoint retains its native arguments and records `adapter_enabled: false`.
Other serving runtimes reject the option. It can be combined with `--m5-prefill`.

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

### Anthropic Messages API

The JIT gateway also answers `POST /v1/messages` in the Anthropic Messages
format, streamed and unstreamed, by translating to the worker's
`/v1/chat/completions`. Text, system prompts, custom tools (`tool_use` and
`tool_result`), `max_tokens`, `temperature`, `top_p`, `top_k` and
`stop_sequences` are carried over; usage reports input, output and cached
prompt tokens when the worker provides them. Requests the local runtime cannot
honor are refused with an `invalid_request_error`: image or document content,
non-custom tools, forced `tool_choice` (`any`, `tool`) and other parameters.
`tool_choice: none` sends no tools. Tool-use ids come from the worker, or are
generated when it gives none. The model name must be the gateway's model.

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

`--thinking off` asks the chat template to skip reasoning: `mlx_lm` receives
`--chat-template-args {"enable_thinking":false}`, and the reviewed plan records
`thinking: off`. Templates without that switch ignore it. Runtimes whose recipe
has no such argument record `thinking: unsupported` and start unchanged.

`--runtime auto` reads local `config.json` and selects a declared MLX-LM or
MLX-VLM loader. Optional backends use their isolated executable, bound into
the eager or JIT preview, rather than global PATH. Specialized preconverted
packs use declared backend adapters without importing runtime code from the
model repository. Review the resolved runtime and executable before confirmation.

`serve start --launchd` installs a receipt-owned plist under a managed
`com.mlx-agent.serve.<port>` label. Review and confirm its plan, then run the
printed `launchctl bootstrap` command yourself. JIT does not support direct
`--launchd`; supervise the gateway through an existing process manager instead.
