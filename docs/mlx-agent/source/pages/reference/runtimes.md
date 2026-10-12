# Runtimes

A provider is the coding-agent host that invokes mlx-agent. A runtime loads and
executes model weights. Installing a provider adapter does not install a runtime.
Choose one for the model format and task you intend to run.

| Runtime | Role in mlx-agent | Conventional port |
| --- | --- | --- |
| `mlx_lm` | Native text serving; default text conversion, LoRA, and fusion path | 8080 |
| `mlx-vlm` | Native vision/language serving and media tasks | 8083 |
| LM Studio | Desktop-managed models with a local OpenAI-compatible server | 1234 |
| Ollama | Runtime-managed models and tags through its local API | 11434 |
| LiteLLM | Router over explicitly configured backends | 4000 |

Ports are defaults or conventions, not proof that a server is running.
Use `inspect-host`, runtime inventory, and `serve status` as appropriate.
Serve's `--port` and Fleet's `--port-map role=port` allow explicit choices.

## Native MLX servers

`serve start` launches `mlx_lm` or `mlx-vlm` with reviewed arguments and
local weights. The executable must already be installed. `--path` selects a
local directory; `--repo` selects a cached repository. Text adapters use
`--adapter-path`. See [Serve](../lifecycle/serve.md) for JIT, launchd, memory
policy, and process ownership.

## Managed desktop and tag runtimes

LM Studio and Ollama manage loading and downloads. Select a compatible model in
the runtime first. Use its advertised identifier for benchmarks and verification;
a Hub repository ID and an Ollama tag are not interchangeable.

Detection and configuration rendering do not establish runtime support for
arbitrary Hub repositories or every modality. Review the runtime's model support
and your actual inventory.

## Local routers

A loopback LiteLLM endpoint can forward to a remote or paid provider.
mlx-agent checks route locality before treating it as local for verification.
Review the router configuration and selected model mapping. A local socket
alone does not prove local inference.

[Fleet](../lifecycle/fleet.md) emits a bounded router configuration; it does not
start LiteLLM or load the referenced model servers.

## Optional task backends

[Backends](../lifecycle/backends.md) manage locked converter and task-runner
environments, separate from an always-running HTTP server.
[Local inference](../lifecycle/local-inference.md) covers speech, classification,
media description, image generation, and video generation.

Reference packs shipped in `src/mlx_agent/resources/references/` cover
quantization, model families, and serving troubleshooting. Compare performance
on your actual host with [Bench](../capabilities/bench.md).
