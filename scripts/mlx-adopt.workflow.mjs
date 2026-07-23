export const meta = {
  name: 'mlx-adopt',
  description: 'Compatibility wrapper for durable MLX adoption state.',
}

const pluginRoot = (args && args.pluginRoot) || '.'
const statePath = (args && args.statePath) || '.mlx-agent-adoption.json'
const shellQuote = (value) => `'${String(value).replace(/'/g, "'\\''")}'`
const allowedRoles = new Set(["general","coding","reasoning","vision","embedding","tool-use"])
const requestedRoles = (args && Array.isArray(args.roles)) ? args.roles : []
const roles = requestedRoles.filter((role) => allowedRoles.has(role))
const selectedRoles = roles.length ? roles : ['general']
const executable = shellQuote(`${pluginRoot}/scripts/mlx-agent`)
const state = shellQuote(statePath)
const roleArguments = selectedRoles.map((role) => `--role ${role}`).join(' ')
const command = `python3 ${executable} adopt start --state ${state} ${roleArguments} --json`

return agent(
  `Run ${command}. If the state already exists or the run was interrupted, run python3 ${executable} adopt resume --state ${state} --json instead. Report the durable adoption state exactly as returned. Do not download model weights or mutate configuration.`,
  { label: 'adopt-state' },
)
