#!/usr/bin/env node
import { spawn } from "node:child_process"

const [runtimeRoot, capability, rawArguments] = process.argv.slice(2)
if (!runtimeRoot || !["scout", "adopt", "wire"].includes(capability) || rawArguments === undefined) {
  process.stderr.write("invalid fixture invocation\n")
  process.exit(2)
}

const MAX_OUTPUT_BYTES = 16384
const child = spawn("python3", ["-m", "mlx_agent.command_executor", "--provider", "opencode", "--capability", capability], {
  cwd: runtimeRoot,
  env: { ...process.env, PYTHONPATH: runtimeRoot },
  shell: false,
  stdio: ["pipe", "pipe", "pipe"],
})
const collect = (stream) => new Promise((resolve) => {
  const chunks = []
  let size = 0
  let truncated = false
  stream.on("data", (chunk) => {
    const value = Buffer.from(chunk)
    const remaining = MAX_OUTPUT_BYTES - size
    if (remaining <= 0) {
      truncated = true
      return
    }
    chunks.push(value.subarray(0, remaining))
    size += Math.min(value.length, remaining)
    truncated ||= value.length > remaining
  })
  stream.on("end", () => resolve({ text: Buffer.concat(chunks).toString("utf8"), truncated }))
})
const stdout = collect(child.stdout)
const stderr = collect(child.stderr)
child.stdin.end(rawArguments, "utf8")
const exitCode = await new Promise((resolve, reject) => {
  child.once("error", reject)
  child.once("close", resolve)
})
process.stdout.write(JSON.stringify({
  status: exitCode === 0 ? "ok" : "error",
  capability,
  exit_code: exitCode,
  stdout: await stdout,
  stderr: await stderr,
}) + "\n")
