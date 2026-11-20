import { Plugin } from "@opencode/plugin"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

const PROVIDER_ID = "mlx-agent"
const DISCOVERY_TIMEOUT_MS = 15000
const runtimeRoot = join(dirname(fileURLToPath(import.meta.url)), "..", "src")

type Served = { id: string; runtime: string; settings: { baseURL: string } & Record<string, string> }

function isServed(value: unknown): value is Served {
  const row = value as Served
  return typeof row?.id === "string" && row.id.length > 0 && typeof row.runtime === "string"
    && typeof row.settings === "object" && row.settings !== null
    && Object.values(row.settings).every((item) => typeof item === "string")
    && /^http:\/\/127\.0\.0\.1:\d{1,5}\/v1$/.test(row.settings.baseURL)
}

async function discover(): Promise<Served[]> {
  const child = Bun.spawn({
    cmd: ["python3", "-m", "mlx_agent.opencode_provider"],
    cwd: runtimeRoot,
    env: { ...globalThis.process.env, PYTHONPATH: runtimeRoot },
    stdin: "ignore",
    stdout: "pipe",
    stderr: "pipe",
  })
  const timer = setTimeout(() => child.kill(), DISCOVERY_TIMEOUT_MS)
  try {
    const [stdout, exitCode] = await Promise.all([new Response(child.stdout).text(), child.exited])
    if (exitCode !== 0) return []
    const models = JSON.parse(stdout)?.models
    return Array.isArray(models) ? models.filter(isServed) : []
  } catch {
    return []
  } finally {
    clearTimeout(timer)
  }
}

export default Plugin.define({
  id: "mlx-agent-provider",
  setup: async (ctx) => {
    const served = await discover()
    await ctx.provider.transform((providers) => {
      if (served.length === 0) return
      providers.update(PROVIDER_ID, (provider) => {
        provider.activation = "enabled"
        provider.name = "mlx-agent (local)"
        provider.package = "aisdk:@ai-sdk/openai-compatible"
        provider.settings = { ...provider.settings, timeout: false }
      })
      for (const row of served) {
        providers.models.update(PROVIDER_ID, row.id, (model) => {
          model.name = row.id
          model.settings = { ...model.settings, ...row.settings }
          model.capabilities = {
            ...model.capabilities,
            tools: true,
            input: row.runtime === "mlx-vlm" ? ["text", "image"] : ["text"],
            output: ["text"],
          }
        })
      }
    })
  },
})
