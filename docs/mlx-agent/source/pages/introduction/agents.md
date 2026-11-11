# Agent integration

mlx-agent gives coding agents three entry points into the same Python core:
host commands or skills, an advisor agent, and an MCP server. All three use the
same validated argument grammar, so an agent never builds a shell command.

## Commands and skills

Each host gets eight capabilities:

| Capability | Purpose |
| --- | --- |
| `mlx-scout` | Discover Hub models that fit this Mac, by role. |
| `mlx-adopt` | Verify candidates with local role probes and recommend one. |
| `mlx-wire` | Preview and confirm a client configuration change. |
| `mlx-bench` | Measure a model that a local runtime already serves. |
| `mlx-doctor` | Report model inventories, wiring drift and endpoint health. |
| `mlx-watch` | Record and compare upstream changes to owned models. |
| `mlx-fleet` | Preview and confirm a per-role router configuration. |
| `mlx-task` | Run, track and collect durable local inference tasks. |

Claude Code, OpenCode, OpenClaw and Hermes Agent invoke them as `/mlx-*`
commands. Codex uses `$mlx-agent:mlx-*` skills. Agy and portable AgentSkills
hosts load the named `mlx-*` skills. [Installation](installation.md) covers each
host.

Command arguments are treated as opaque data. Claude Code passes them to the
bundled MCP tool, OpenCode passes them on stdin to a narrow custom tool, and
skill hosts call the bundled launcher. In every case the core parses them
against an allowlist and refuses shell operators.

## Advisor agent

Claude Code and OpenCode also install `mlx-advisor`, a subagent that covers the
whole lifecycle: choosing a model, checking what is installed, measuring a
served model, running local inference tasks and connecting a client. It follows
the same boundaries as the commands. It does not download weights, install
runtimes or write configuration files directly, and it applies Wire and Fleet
changes only after you confirm the exact preview hash.

## MCP server

`scripts/mlx-agent-mcp` is a stdio MCP server. The Claude Code plugin registers
it automatically. For any other MCP client, register the launcher from a
checkout or an installed package:

```json
{
  "mcpServers": {
    "mlx-agent": {
      "command": "python3",
      "args": ["/absolute/path/to/mlx-agent/scripts/mlx-agent-mcp"]
    }
  }
}
```

The server offers two tools:

- `mlx_agent_capabilities` takes no arguments and returns the supported
  [inference task](../lifecycle/tasks.md) kinds and request contracts. It does
  not load models, read task state or contact registries.
- `mlx_agent_execute` runs one validated action. Set `capability` to `scout`,
  `adopt`, `wire`, `bench`, `doctor`, `watch`, `fleet`, `lab` or `task`, and
  `arguments` to the action text, at most 4,096 UTF-8 bytes.

```json
{
  "name": "mlx_agent_execute",
  "arguments": {"capability": "doctor", "arguments": "models --json"}
}
```

Results carry `status`, `exit_code`, `stdout` and `stderr`, each stream capped
at 16,384 characters with a `*_truncated` flag. Invalid arguments are reported as
non-retryable errors. `mlx_agent_execute` is marked as able to change local
files, start jobs and contact model registries, so clients that honor tool
annotations ask before calling it.

## Confirmation boundaries

- Discovery, adoption, bench, doctor and watch never download weights or change
  configuration.
- Wire and Fleet return a diff and a preview hash first. The change applies only
  when the same command runs again with `--confirm --preview-hash <hash>`, and
  it records a receipt that `wire rollback` can undo.
- Tasks run only against an existing local model and its installed backend.
- `intake fetch`, `backend install`, `convert start` and `serve start` remain
  CLI commands with their own reviewed previews; agents propose them and you
  confirm.
