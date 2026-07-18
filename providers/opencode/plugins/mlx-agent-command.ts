import { tool } from "@opencode-ai/plugin"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

const MAX_ARGUMENT_BYTES = 4096
const MAX_OUTPUT_BYTES = 16384
const pluginDirectory = dirname(fileURLToPath(import.meta.url))
const runtimeRoot = join(pluginDirectory, "..", "src")
const encoder = new TextEncoder()

async function readBounded(stream: ReadableStream<Uint8Array>) {
  const decoder = new TextDecoder()
  const reader = stream.getReader()
  let text = ""
  let size = 0
  let truncated = false
  try {
    while (size <= MAX_OUTPUT_BYTES) {
      const next = await reader.read()
      if (next.done) break
      const remaining = MAX_OUTPUT_BYTES - size
      if (next.value.byteLength > remaining) {
        text += decoder.decode(next.value.slice(0, Math.max(0, remaining)), { stream: true })
        truncated = true
        await reader.cancel()
        break
      }
      text += decoder.decode(next.value, { stream: true })
      size += next.value.byteLength
    }
  } finally {
    reader.releaseLock()
  }
  return { text: text + decoder.decode(), truncated }
}

export const MLXAgentCommandPlugin = async () => ({
  tool: {
    mlx_agent_command: tool({
      description: "Run one validated MLX Scout, Adopt, or Wire command without shell interpolation.",
      args: {
        capability: tool.schema.enum(["scout", "adopt", "wire"]),
        arguments: tool.schema.string().max(MAX_ARGUMENT_BYTES),
      },
      async execute(args) {
        const child = Bun.spawn({
          cmd: ["python3", "-m", "mlx_agent.command_executor", "--provider", "opencode", "--capability", args.capability],
          cwd: runtimeRoot,
          env: { ...globalThis.process.env, PYTHONPATH: runtimeRoot },
          stdin: "pipe",
          stdout: "pipe",
          stderr: "pipe",
        })
        const writer = child.stdin.getWriter()
        await writer.write(encoder.encode(args.arguments))
        await writer.close()
        const [stdout, stderr, exitCode] = await Promise.all([
          readBounded(child.stdout),
          readBounded(child.stderr),
          child.exited,
        ])
        return JSON.stringify({
          status: exitCode === 0 ? "ok" : "error",
          capability: args.capability,
          exit_code: exitCode,
          stdout: stdout.text,
          stderr: stderr.text,
          stdout_truncated: stdout.truncated,
          stderr_truncated: stderr.truncated,
        })
      },
    }),
  },
})
