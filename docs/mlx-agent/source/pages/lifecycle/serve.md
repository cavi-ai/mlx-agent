# Serve

## Shared agent memory budget

`serve budget --json` reports the engine's shared memory policy. Configure a
64 GB limit with an 8 GB transient reserve using:

```sh
mlx-agent serve budget --limit-gb 64 --reserve-gb 8 --json
```

The reserve stays inside the limit, leaving 56 GB for admission. Reservations
count pending loads, resident model workers, active workspace and process
overhead. `--include-pid PID` includes an existing agent process and its family,
bound to that process's start identity; changing only the limits preserves those
bindings. All values use decimal GB. Unknown accounting blocks new allocations.

### Per-route task job caps

Set `--memory-budget-gb` on a typed task route to cap each job's total memory,
including resident weights and workspace. The value is decimal GB (1–1024).
For example, preview a route with a 12 GB cap, then confirm that preview:

```sh
mlx-agent serve start --path /absolute/local/model --runtime tasks --port 8085 --memory-budget-gb 12 --json
mlx-agent serve policy --port 8085 --expected-pid <gateway-pid> --memory-budget-gb 16 --json
```

`serve policy` saves changes for subsequent jobs; an active job keeps its admitted
cap. Omission preserves the cap; `--clear-memory-budget` removes it. Saved
profiles accept `--memory-budget-gb` without Strata context or decoding settings
for task routes. Status exposes the cap as `memory_policy.model_memory_budget_gb`.
For Strata, the same option retains its model-allocation meaning.

With a shared agent budget configured, capped task routes and served text routes
can execute together. The sum of resident memory, job reservations, external
agents and process overhead must fit admission. A job reserves its full cap or
fails with `agent_memory_budget_exceeded`; it never silently receives a smaller
cap. A cap below the model and workspace estimate fails with
`job_memory_cap_exceeded`. Increase the route cap or retry after another job
finishes. System-headroom checks still apply. `serve budget --json` reports the
accounted totals and each active allocation's `execution_bytes` ceiling.
Observed allocations consume that reservation, avoiding a second charge as a
worker grows. Completion releases workspace while retaining any idle weights.

An uncapped task keeps exclusive host execution and the existing remaining-pool
ceiling. Without a shared budget, all execution remains exclusive; a configured
task cap still sets its MLX allocation ceiling. Conversion, training, cache
maintenance and other non-serving operations always require exclusive ownership.
Each shared execution persists its process identity and worker group separately;
orphaned groups retain ownership until confirmed stopped. Legacy ownership
records continue to block admission while their owner or group may be alive.

Typed workers receive an MLX allocation ceiling before executing a job, leaving
512 MB of the admitted capacity for process overhead. Allocation failures are
returned as job errors; increase the route cap when its workspace needs more. The
gateway also observes process-family memory and stops only its own model worker
on a measured breach. This policy does not control unrelated applications or
replace system-headroom checks. Existing servers must be restarted on a supporting
engine to participate; installing newer source does not update a running process.

Serve launches local MLX servers from a reviewed plan. Use `mlx_lm` for text,
`mlx-vlm` for vision/language models, or `strata` for prepared compatible GGUF
profiles. `auto` selects typed job serving for
supported classifiers, text embeddings, speech, music, image, video and 3D models. Ollama and LM Studio manage their
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

## Discover running services

`serve discover --json` reports services across registered receipt locations and
live MLX Agent gateways owned by the current user. New launches register their
receipt location in the per-user service index. Repeat `--receipts-dir PATH` to
include another known location. Discovery is read-only; each result includes its
receipt location, process identity and reported model residency. An unloaded JIT
gateway can remain available to clients.

Use the result's `receipts_dir` for controls. `--expected-pid` and
`--expected-start` bind stop, unload and policy changes to the selected process;
a changed process is refused. Errors from an unavailable location are reported
separately from services found elsewhere.

## Front door

`serve front-door` puts one authenticated loopback port in front of every
discovered route that verifies the same API key:

```bash
mlx-agent serve front-door --port 8090 --api-key-file /absolute/private/model.key
```

Start each route with the same key, through `--api-key-file` or
`--api-key-sha256`. The front door lists only live JIT gateways and typed
endpoints whose receipts record that key digest. Routes without a key, with
another key or with a changed process are never listed or forwarded to.
Repeat `--receipts-dir PATH` to include locations outside the service index.

`GET /v1/models` names each route by its model directory, or by its Hub
repository for a cached snapshot, without local paths. A name served by more
than one route is withheld and refused with `ambiguous_model`. Requests to
`/v1/chat/completions`, `/v1/completions`, `/v1/messages` and `/v1/responses`
go to the JIT gateway serving `model`; `/v1/embeddings` goes to a typed
endpoint that advertises `embedding`. The request body and the client's
credential are forwarded unchanged and the route checks the key again.
Responses, including event streams, are relayed as they arrive.

The front door applies the same key, Host and Origin checks as a gateway;
`GET /health` stays open. It runs in the foreground and loads no model; the
routes keep their own loading, memory admission and lifecycle. Stored
response reads and `/v1/tasks` stay on each route's own port.

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
group size 64 and dimensions divisible by 64, when a batch has at least 64 rows.
It retains the original quantized weight storage. Inputs with fewer than 64
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

## Repeated-image acceleration

Add `--vision-cache` to both preview and confirmation to reuse Qwen2.5-VL and
Qwen3-VL image features across questions about the same image, including dense
and MoE Qwen3-VL models.
It preserves all DeepStack features and uses the installed MLX-VLM
0.7.4 backend with MLX/MLX-metal 0.32.3, MLX-LM 0.32.2, NumPy 2.5.3 and
Transformers 5.18.0.
The image encoder and model processor keep their native calculations and
resolution; text embeddings, position data, KV caches and generation remain
request-specific.

The cache holds at most 20 entries and 128 MiB of feature tensor data, including
DeepStack tensors, in worker memory. Unloading the worker releases it; no image
or cache files are written.
Keys include processed pixel content, dtype, shape and image layout, so changing
an image at the same URL cannot reuse stale features. Unsupported models,
training and adapters retain native processing.

The preview, receipt and status record `vision_profile`. Its fingerprint binds
the adapter and installed runtime sources. Changed source or dependencies require
a fresh preview. The wrapper uses the backend's own Python in isolated mode;
unavailable or incompatible backends keep their native launch arguments. Logs
report cache hits, retained tensor bytes and encoder calls. This option requires
`mlx-vlm`; other runtimes reject it.

## Code-edit acceleration

Add `--prompt-lookup` to both preview and confirmation for code edits that reuse
source text from the prompt. With client temperature zero, the server proposes
continuations from repeated prompt sequences and verifies them with the target
model before emitting them. It indexes only the last 8192 prompt tokens. No draft
model or new weights are required. Native batching remains the default when the
option is omitted.

In this mode, plain greedy requests run one at a time. Sampling, logit penalties or
biases, logprobs, adapters and draft models retain native generation. Seeds retain
the pinned runtime's temperature-zero argmax behavior. Only
ordinary KVCache states use lookup; rotating, quantized and hybrid caches retain
native generation. Prompt cache reuse, stop tokens and cancellation retain the
server's token alignment. Each request first measures two target single-token
verification calls. M5 GPUs (`applegpu_g17s`) then start with up to 32 proposed
tokens; other GPUs start with four and retain an eight-token limit. Partial
acceptance shrinks the window; full acceptance grows it within the device limit.
Two consecutive rejected proposals, four unprofitable verification blocks, a
single block costing over twice the calibrated per-token target time, or an
absent prompt match hand the remaining generation to native decoding. M5 lookup
searches up to eight initial output tokens for a copied prefix, allowing short
output wrappers before code. Other GPUs retain a three-token search. Cost
decisions use target verification time per emitted token; they do not establish
a comparative speedup over native generation. Logs retain proposed/accepted
tokens, native tokens and the fallback reason. Performance depends on copied
text; keep this option off for concurrent chat workloads.

The request loop waits briefly on an empty queue to leave CPU time for response
delivery and other applications. New arrivals wake the loop immediately.

The supported runtime is macOS with MLX and mlx-metal 0.31.1 and mlx-lm 0.29.1.
Other dependency versions use native MLX. Like `--m5-prefill`, this option wraps
only the current interpreter's own entrypoint, records source and dependency
hashes (`decode_profile`) and requires a fresh preview if they change. A foreign
entrypoint retains its native arguments and records `adapter_enabled: false`.
Other serving runtimes reject the option. It can be combined with `--m5-prefill`.

## Splash

`--runtime splash` serves local weights through an installed Splash 1.4.0 or
later launcher (`brew install incoai/tap/splash`; Apple M3 or newer, macOS 26.4
or later). Serve finds `splash` through `MLX_AGENT_SPLASH`, `PATH`, Homebrew and
`~/.local/bin`, and never installs it.

```bash
python3 scripts/mlx-agent serve start --path /absolute/strata/profile --runtime splash --json
python3 scripts/mlx-agent serve start --path /absolute/mlx/model --runtime splash --json
python3 scripts/mlx-agent serve start --repo mlx-community/Qwen3.8-27B-4bit --runtime splash --json
python3 scripts/mlx-agent serve start --path /absolute/model.gguf --runtime splash --splash-draft-model /absolute/draft --splash-max-memory-gb 48 --splash-max-context 65536 --json
```

`--path` selects a GGUF file or any part of a split GGUF, a directory holding
one GGUF model, a quantized MLX directory, or a prepared Strata profile, whose
GGUF is served unchanged. Every split part must be present. `--repo` serves the
cached Hugging Face snapshot of an MLX or GGUF repository. Weights are linked
in place, never copied or converted. The server runs with `--offline` on
`127.0.0.1` and lists the model under its `--repo` ID, or as `local/<name>`
for `--path`.

Splash 1.6 serves the Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3.5-122B-A10B and
Qwen3.8-Flash-Next families and their fine-tunes. MLX weights must be
quantized (affine 2–8 bits, mxfp4 or nvfp4). Every model decodes with its
family's DFlash2 draft: pass `--splash-draft-model`, or keep the family's
draft in the Hugging Face cache. Splash refuses to start without one.
`convert scan` reports each quantized MLX output's Splash family, as Splash's
engine checks its configuration.

- `--splash-draft-model` selects a local DFlash2 draft directory
  (`config.json` with `dflash_config`, and safetensors weights).
- `--splash-max-memory-gb` caps Splash memory in decimal GB (1–1024).
- `--splash-max-context` caps context in tokens (512–1,010,000).
- `--thinking off` sets the default reasoning effort to `none`.

The preview binds the launcher, its reported version, and the content of the
target and draft files; a change requires a fresh preview. Splash rejects
adapters and releases idle weights itself. Stop signals
the receipt's process group and waits 20 seconds before a forced kill, since
Splash gives its engine 15 seconds to exit.

`--jit` serves a Splash model directory or Strata profile through the
load-on-request gateway, so `--api-key-sha256` and the front door work as
for the other runtimes:

```bash
python3 scripts/mlx-agent serve start --path /absolute/strata/profile --runtime splash --jit --api-key-sha256 <digest> --splash-max-memory-gb 54 --json
```

The gateway starts Splash's own Python launcher with a fresh worker key in
`SPLASH_API_KEY` and its process ID in `SPLASH_APP_PID`, so Splash stops when
the gateway exits. Requests reach Splash under its served name. Memory
admission reserves `--splash-max-memory-gb` when it is set, otherwise the
target and draft weights plus 10% and 1.5 GB.

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

The gateway waits for the worker's reply to start without a deadline, including
unstreamed answers that the worker sends only when generation finishes. A
client that disconnects before the reply starts ends its request. Streamed
Messages answers turn the worker's prompt-progress keepalives into `ping`
events, and streamed Responses answers pass them on as SSE comments.

Status reports gateway liveness and `model_state` separately, and the loaded
worker's `worker_port`. Stop terminates
the owned gateway/worker process group. The gateway serves its selected model
and rejects request-time adapter/draft-model overrides.

JIT rechecks the adapter before and after each worker load, including after
unloading. Older adapter gateways without a recorded content identity require
a fresh serve preview before loading another worker.

### OpenAI Chat Completions

The JIT gateway forwards `POST /v1/chat/completions` and `POST /v1/completions`
to the worker and returns answers in the OpenAI shape: every tool call has an
id, and an `index` when streamed; a turn that calls tools ends with the
`tool_calls` finish reason; and unstreamed answers do not end with the model's
end-of-sequence token, which `mlx_lm` 0.29.1 leaves in place. Worker errors
pass through unchanged.

### Request defaults and usage

Managed chat gateways accept `--system-prompt`, `--default-temperature` (0–2)
and `--default-top-p` (0–1) with `serve start --jit`. The system prompt is bounded
to 8,192 UTF-8 bytes. These settings participate in serve preview and confirmation.
Sampling defaults fill omitted fields. A default system prompt is added only to
chat requests without a system or developer message; explicit client settings take
precedence. Typed task endpoints do not accept these defaults.

`serve status` includes `request_defaults` for a gateway, with a system-prompt
SHA-256 instead of its text. Its `inference_telemetry` schema 1 reports counters
for that gateway process: accepted, completed, failed and measured requests;
reported input/output tokens; and the latest response duration and output rate.
The duration includes admission, loading and response time, so this rate is
end-to-end response throughput rather than the model's decode speed.

Token totals include only completed replies with valid worker-reported usage.
`measured_requests` identifies that subset; missing usage is `null`, not an estimate
or a zero. Stream observations do not request extra usage or change forwarded
responses. Counters reset with the gateway process and retain numeric measurements
only. Status uses the existing protected control endpoint. Direct workers and
external hosts that do not report these metrics remain unavailable.

### Anthropic Messages API

The JIT gateway also answers `POST /v1/messages` in the Anthropic Messages
format, streamed and unstreamed, by translating to the worker's
`/v1/chat/completions`. Text, system prompts, custom tools (`tool_use` and
`tool_result`), `max_tokens`, `temperature`, `top_p`, `top_k` and
`stop_sequences` are carried over; usage reports input, output and cached
prompt tokens when the worker provides them. With `mlx-vlm`, user messages can
include image blocks with a base64 source (JPEG, PNG, GIF or WebP) or an HTTP(S)
URL. Text and image order is preserved. Image resolution follows the selected
model's processor. Requests the local runtime cannot
honor are refused with an `invalid_request_error`: images on text runtimes,
document content, image file IDs, images in system, assistant or tool-result content,
non-custom tools, forced `tool_choice` (`any`, `tool`) and other parameters.
`tool_choice: none` sends no tools. Tool-use ids come from the worker, or are
generated when it gives none. The model name must be the gateway's model.

### OpenAI Responses API

The JIT gateway also answers `POST /v1/responses`, streamed and unstreamed, by
translating to the worker's `/v1/chat/completions`. `instructions`, text
message items (`developer` messages become system messages), `function` tools,
`function_call` and `function_call_output` items, `max_output_tokens`,
`temperature` and `top_p` are carried over. Streams emit `response.created`,
output item, content part, text and function-call argument events, then
`response.completed`, or `response.incomplete` when the token limit stops the
answer. `parallel_tool_calls`, `prompt_cache_key` and `client_metadata` are
accepted without changing generation; an empty `reasoning` object and
`include: ["reasoning.encrypted_content"]` are accepted because no reasoning
items are produced. With `mlx-vlm`, user message content can include
`input_image` items with an `image_url` containing an HTTP(S) URL or a base64
image data URL. Text and image order is preserved; image detail hints are
forwarded to the worker, whose model processor controls resolution.
Images on text runtimes, image file IDs, images outside user messages, file
content, other tool types, `strict`
function schemas, reasoning controls, structured text formats, forced
`tool_choice` and other parameters, including `previous_response_id`, are
refused with an `invalid_request_error`.

Background requests complete before the response returns. Responses created
with `store` (the default) are kept in gateway memory, the most recent 64, and
`GET /v1/responses/<id>` returns them until the gateway stops.

## Memory policy

### Strata model budgets

For a prepared Strata GGUF profile, add `--memory-budget-gb` to its JIT preview
and confirmation. The estimated total includes dense weights, expert storage,
context and runtime working memory. Choose `--strata-decoding plain` (the default),
`lookup` (verified guesses copied from earlier prompt text), or `mtp` (verified
guesses from prepared draft-head weights). Lookup loads no draft-head weights;
MTP is loaded and budgeted only when selected. The legacy
`--strata-speculation on|off` switch remains available.

Paged MTP budgets use the supported draft head's packed weights and float16
scales to size the expert cache. The minimum startup allowance still covers
head loading before the cache is populated. Resident loading retains its full
draft-file allowance; unrecognized head layouts use that allowance for paging too.

```bash
python3 scripts/mlx-agent serve start --path /absolute/strata/profile --runtime strata --jit --memory-budget-gb 24 --strata-context-tokens 16384 --min-headroom-gb 24 --unload-under-pressure --json
```

Use different budgets for shared work or dedicated inference. The planner keeps
the faster resident path when it fits; otherwise it assigns a fixed expert cache
and reads other experts from disk. Workers disable separate conversation parking,
GPU keepalive and array wiring. A budget is an allocation estimate, not a hard process-RAM
limit. OS headroom remains a separate admission and pressure guard.

### Saved serving settings

Save named settings for shared work or dedicated inference. Profiles
contain headroom, residency and execution-wait settings; choose the model
and endpoint when starting. Strata profiles can also include a model budget,
context and decoding settings. Settings stay under the selected receipts directory.

```bash
python3 scripts/mlx-agent serve profile save shared --memory-budget-gb 24 --min-headroom-gb 24 --strata-decoding lookup --json
python3 scripts/mlx-agent serve profile save working --min-headroom-gb 24 --idle-timeout 600 --execution-wait 30 --json
python3 scripts/mlx-agent serve profile list --json
python3 scripts/mlx-agent serve start --path /absolute/strata/profile --runtime strata --jit --profile shared --json
```

Launch uses the usual preview and confirmation. Explicit launch options override
saved settings. Use `serve profile show shared`, `save shared --replace` with the
updated values, or `delete shared` to manage them. Saved changes apply to future
launches; use `serve policy` to change a running endpoint's memory budget.
Add `--no-unload-under-pressure` when saving a profile that uses `--keep-loaded`.
Each launch still checks available memory; a saved profile does not reserve RAM.

Change the budget on the same endpoint using its gateway PID:

```bash
python3 scripts/mlx-agent serve policy --port 8080 --expected-pid <gateway-pid> --memory-budget-gb 48 --min-headroom-gb 32 --unload-under-pressure --json
```

Active requests finish before a budget change unloads the worker. New requests
receive `model_reconfiguring` until it unloads; the next request loads the new
allocation. Loading or unloading workers reject budget changes with `model_busy`.
Status exposes the requested `memory_allocation` and `memory_budget_pending`.
Omitting the budget in a later policy update preserves it; `serve policy
--clear-memory-budget` explicitly returns to the original allocation settings.
A budget defaults to
a 2 GB reserve when no headroom is supplied and cannot be combined with an
explicit expert-memory limit at startup. These allocation settings require Strata JIT; typed task routes use the option
as a per-job ceiling instead.

### Idle and pressure controls

Include policy flags in the initial JIT preview and confirmation:

```bash
python3 scripts/mlx-agent serve start --path /absolute/local/model --runtime mlx_lm --jit --idle-timeout 600 --min-headroom-gb 2 --json
python3 scripts/mlx-agent serve start --path /absolute/local/model --runtime mlx_lm --jit --idle-timeout 600 --min-headroom-gb 2 --confirm --preview-hash <hash> --json
```

Use the gateway PID from status to replace idle and pressure controls without restarting:

```bash
python3 scripts/mlx-agent serve policy --port 8080 --expected-pid <gateway-pid> --idle-timeout 600 --min-headroom-gb 2 --json
python3 scripts/mlx-agent serve policy --port 8080 --expected-pid <gateway-pid> --idle-timeout 600 --keep-loaded --min-headroom-gb 2 --json
```

Idle time starts after the last request or stream finishes. `--idle-timeout 0`
disables automatic idle unload. `--keep-loaded` prevents automatic idle unload;
it does not preload weights, and manual unload remains available. Omitting
`--keep-loaded` in a policy replacement allows configured idle unloading; omitting
`--min-headroom-gb` disables admission checking when no model budget is active,
and omitting `--context-tokens`
removes the KV cache term. Policy is persisted atomically
in private gateway configuration. Start without policy flags retains manual
unload behavior. Policy flags require JIT.

Add `--unload-under-pressure` with `--min-headroom-gb` to protect the reserve
after loading. New requests return `insufficient_headroom` when OS headroom is
below the reserve; an idle worker unloads while its endpoint stays reachable.
Active requests and streams finish before automatic reclamation. Unknown
headroom returns `memory_unknown` for new requests and does not trigger pressure
unloading.
This option cannot be combined with `--keep-loaded`.

The loaded-request check counts only the reserve. Status records the latest
observation in `resident_memory_check` and any refusal in
`request_blocked_reason`. The cold-load estimate below still applies after
reclamation. Omitting `--unload-under-pressure` in a policy replacement disables
both the loaded-request guard and pressure unloading.

Before a guarded cold load, fresh OS headroom must cover estimated local weight
bytes × 1.10, a 1.5 GB runtime allowance, and the requested reserve in decimal GB.

Add `--context-tokens N` (1024–1048576, with `--min-headroom-gb`) to also cover
the KV cache for N tokens: 2 × layers × KV heads × head dimension × 2 bytes per
token, read from the local `config.json` or its `text_config`. The estimate
assumes every layer keeps a full-attention cache; models with sliding-window or
linear-attention layers need less. A configuration without these values returns
`memory_unknown`. Strata endpoints refuse the option because their estimate
already includes the cache at `--strata-context-tokens`.

Unknown headroom or weight size returns HTTP 503 `memory_unknown`; insufficient
space returns HTTP 503 `insufficient_headroom`. The endpoint remains reachable.
Status includes policy, the dated estimate in `memory_check`, and
`load_blocked_reason`.

Endpoints coordinate pending loads and configured workspace allowances through
a shared memory registry. Other applications and growth beyond the configured
estimates can still change usage. See
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

## API key authentication

Add `--api-key-file /absolute/private/key` to both preview and confirmation
for a supervised loopback endpoint. The file must be an owned regular file
readable only by its owner, containing a 32–256 character URL-safe API key.
Keep it outside the model directory. Clients send `Authorization: Bearer KEY`
or `x-api-key: KEY`; never put the key in URLs or command arguments.

The native app instead keeps generated keys in Keychain and passes the
verification digest with `--api-key-sha256`. Select one credential source.
Plans, receipts and process arguments contain the digest, never the API key.
Authentication requires JIT for chat runtimes and is also available for typed
model endpoints. Omitting it preserves existing local access.

The gateway authenticates model discovery, chat, Messages, Responses, stored
responses, typed jobs, cancellation, results and artifacts before inference.
Missing, wrong or ambiguous credentials receive HTTP 401. The small
`GET /health` response remains available without a key. Host and Origin checks
also apply. Control uses a separate generated credential; an inference key
cannot unload a model or change memory policy.

The reviewed plan binds the credential digest and authentication source.
Startup rechecks both; changed inputs require a fresh preview. File credentials
are rechecked on requests. Stop and review a new key to revoke the previous one.
A key authorizes its holder; it does not identify an executable or provide a
process allowlist.

Plain MLX-LM workers use an inherited Unix listener with a separate private key.
That path accepts fixed-model text chat within a 1 MiB request limit. Vision,
Strata and accelerated chat workers use a loopback port authenticated with a
separate private key. A public endpoint key cannot access these workers.
Typed models run through the supervised resident-task transport and open no
private inference HTTP port. Existing servers retain their configured access
controls until explicitly changed.

For a supervisor such as Secure Agent, `mlx-agent serve private-worker`
accepts `--model` and an inherited `--socket-fd` instead of opening a public
listener. `--describe` reports its protocol without importing MLX or starting
a model. The supervisor owns lifecycle and socket cleanup. This path requires
an installed MLX-LM virtual environment and existing local weights; it never
downloads models or installs a runtime. The Unix directory controls access
by user and does not isolate other processes already running as that user.

## Foreground protected service

Existing service managers can run an authenticated gateway in the foreground:

```sh
mlx-agent serve protected-service --model /absolute/local/model --port 8080 --api-key-file /absolute/private/model.key
```

The key must be an owned, owner-only file. Weights must already exist locally.
The gateway uses a private Unix worker, refuses unrelated browser Host/Origin
headers and checks credentials before inference. SIGTERM closes its worker and
private temporary state. The existing service manager owns restart and stop;
this command does not create a launchd job or a serve receipt. Retire any old
raw model listener before claiming backend isolation. `--describe` reports the
`protected-service-v1` capability and source identity without loading a model.
## Typed model endpoints

For a supported local non-chat model, preview with `--runtime auto` or `tasks`.
The selected backend must already be installed. The endpoint binds one local
model and advertises its supported `task_kinds` at `GET /v1/models`. It never
executes code from a model repository or downloads weights.

```bash
python3 scripts/mlx-agent serve start --path /absolute/local/model --runtime auto --release-after-request --port 8085 --json
```

Confirm with the same arguments and the returned preview hash. Use
`--keep-loaded --no-release-after-request` to retain the loaded model between
jobs instead. The default for typed endpoints releases weights after each job.
By default, a job that cannot acquire host compute fails with `compute_busy`;
any retained idle model stays loaded for retry. Set `--execution-wait 30` to
wait up to 30 seconds instead. Cancelling a waiting job preserves its idle model.
`serve policy` can change either mode on an owned endpoint. Manual `serve unload`
releases idle weights without closing its port; active jobs must finish or be
cancelled first. Status reports residency, worker PID, active jobs and memory policy.
Headroom admission and optional idle/pressure handling use the same controls as
chat JIT endpoints. Task routes also accept the per-job cap described above.
Text KV estimates and Strata allocation budgets apply to their own runtimes.

JIT and typed task endpoints coordinate estimated memory admission on the host.
Pending loads and task workspace share the largest headroom reserve configured
by their registered endpoints. Allocated resident weights are already reflected
in OS headroom and are not deducted again. Coding workers retain their runtime
and configured KV/expert-cache growth allowances while loaded; task workspace
is released after job completion.
These admission estimates are not OS-enforced process memory limits.

Status includes `memory_reservation` with the endpoint allocation and a shared
registry snapshot from its latest admission or completion. Typed workers also
report timestamped, versioned `worker_memory` load and
cache-hit counts, load time, and MLX active, cache and peak bytes. Unavailable
allocator measurements are `null`. Cold coding allocations remain reserved
until inference output is observed; HTTP readiness alone does not release them.

Submit `POST /v1/tasks` with `{"kind":"...","parameters":{...}}`, using a kind
advertised by that endpoint and the parameters in the [task contract](tasks.md).
The model path is bound by the endpoint, not selectable in the request. One job
runs at a time; another submission while busy is refused. Accepted jobs have
private durable input snapshots and validated, hashed results.

Read `GET /v1/tasks/ID` and `GET /v1/tasks/ID/result`; cancel with
`POST /v1/tasks/ID/cancel` and `{}`. Download generated WAV, PNG, MP4 or GLB files
from `GET /v1/tasks/ID/artifacts/output`. The complete result JSON is available
at `GET /v1/tasks/ID/artifacts/result`, including results too large to return
inline. The server retains accepted jobs if the client disconnects. Endpoint
interruption is recorded as interrupted when its job is next inspected.

An endpoint that advertises `embedding` also answers OpenAI
`POST /v1/embeddings` with `{"model":"...","input":...}`. `model` is the `id`
from `GET /v1/models` or the model directory name, as a gateway accepts, and is
echoed in the response; `input` is one string or a list of strings. Token
arrays and `input_type` are refused. `encoding_format` is `float` (default) or
`base64` little-endian float32. `dimensions`, when sent, must equal the model's
native width. Inputs run as embedding jobs of up to 128 texts with a maximum
length of 8192 tokens, and the response returns once every job has finished.
Embedding requests queue behind each other; a request made while a
`/v1/tasks` job runs is refused with `model_busy`. The response has no `usage`.

Typed endpoint history lives in `.mlx-agent-receipts/tasks/PORT` under the chosen
`--receipts-dir`. CLI inspection uses that exact path as `task --state-dir`.
These endpoints do not implement chat completions, Messages or Responses.
Embedding endpoints also serve `POST /v1/embeddings`.


### Shared execution admission

Resident coding, vision and typed-model endpoints coordinate host execution
ownership with other guarded model operations. Idle endpoints retain their weights without
holding execution ownership. Requests to one text backend can use its native
batching. Without a shared memory budget, another model waits or receives
HTTP 503 `compute_busy`. With that budget, served text routes and capped task
routes share execution subject to memory admission. Uncapped task routes and
non-serving operations remain exclusive.

`--execution-wait` accepts 0–3600 seconds and defaults to 0. It is available on
`serve start`, `serve policy` and saved profiles. Waiting requests count toward
the endpoint request limit. Disconnecting while waiting cancels admission before
loading. A disconnected active request retains ownership while its upstream reply
drains; a reply that cannot drain within 30 seconds stops the owned backend before
releasing execution. Automatic idle and pressure unloads defer while another
model owns execution. This is bounded waiting, without FIFO ordering or preemption.

Common profiles work with coding, vision, classification and generation endpoints.
Profiles containing Strata context or decoding settings require Strata. A budget
without those settings can also cap typed task jobs.
Explicit launch flags override saved values. For retained typed models, include
`--keep-loaded --no-release-after-request --no-unload-under-pressure` when saving
the profile. Status reports active and waiting text requests in `execution`.


### Allocator cache controls

Resident task workers support a cap on free MLX allocator buffers and automatic
reclamation while retaining loaded weights. These settings cover supported task
backends, including classification, audio and 3D generation. They are separate
from model, expert, KV and denoiser caches; they do not impose a process RAM limit.

| Policy field | Values | Behavior |
| --- | --- | --- |
| `allocator_cache_limit_mib` | `null` or integer 0–1048576 | Cap free allocator buffers in MiB; 0 disables caching; null restores the worker's original limit. |
| `cache_reclaim` | `off`, `idle`, `pressure` | Keep the allocator cache, clear it after completed work, or clear it while idle below the headroom reserve. Default: off. |

Both settings are optional. Pressure reclamation requires
`minimum_headroom_gb` and works with `keep_loaded`: it clears free buffers while
preserving active model tensors. A smaller cache can require more allocations
on the next request. Existing behavior is preserved when these settings are absent.

```bash
python3 scripts/mlx-agent serve start --path /absolute/task/model --runtime tasks --keep-loaded --no-release-after-request --allocator-cache-limit-mib 1024 --cache-reclaim idle --json
python3 scripts/mlx-agent serve policy --port 8085 --expected-pid <gateway-pid> --keep-loaded --no-release-after-request --no-unload-under-pressure --min-headroom-gb 24 --cache-reclaim pressure --json
python3 scripts/mlx-agent serve reclaim-cache --port 8085 --expected-pid <gateway-pid> --json
```

Use `--clear-allocator-cache-limit` to restore the worker's original limit.
The cache flags also work with `serve profile save`; explicit launch flags override
saved values. A profile that retains weights should use `--keep-loaded
--no-release-after-request --no-unload-under-pressure`.

Workbench and other clients can inspect `allocator_cache` in endpoint/service
status. `supported`, `policy_fields`, `reclaim_modes` and `control_path` describe
the interface. `available` is null until the worker reports runtime capability.
`configured_policy`, `applied_policy` and `pending` distinguish saved settings
from settings applied by the worker. `active` identifies a control operation;
`last_reclaim` reports the reason, cached bytes before/after, released cached bytes
and timestamp. Allocator measurements do not guarantee equivalent OS headroom.

Apply settings through the protected `POST /_mlx/policy`, for example:

```json
{"allocator_cache_limit_mib":1024,"cache_reclaim":"pressure","minimum_headroom_gb":24}
```

`POST /_mlx/cache` with an empty JSON object reclaims an idle loaded worker's
free buffers immediately. It requires the endpoint control token. Active jobs
return `model_busy`; host contention returns `compute_busy`. Automatic maintenance
defers during contention. Saved changes during a job apply at its next safe
boundary. Reclamation never starts an unloaded model.

Chat runtimes advertise `supported: false` for these controls and reject active
allocator settings. Their existing context and Strata expert-memory controls
remain available through their own policies.
