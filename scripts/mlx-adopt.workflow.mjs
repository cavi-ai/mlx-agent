export const meta = {
  name: 'mlx-adopt',
  description: 'Discover, verify, and recommend local MLX model routing for this Apple Silicon host.',
  phases: [
    { title: 'Discover', detail: 'Scan HuggingFace for MLX models per role' },
    { title: 'Verify', detail: 'Test-generate or model-card check top candidates' },
    { title: 'Recommend', detail: 'Synthesize per-role routing with wiring' },
  ],
}

// args.pluginRoot is passed by the /mlx-adopt command (resolved ${CLAUDE_PLUGIN_ROOT}).
const pluginRoot = (args && args.pluginRoot) || '.'
const roles = (args && Array.isArray(args.roles) && args.roles.length)
  ? args.roles
  : ['general', 'coding', 'reasoning', 'vision', 'embedding']
const scout = `${pluginRoot}/skills/mlx-scout/scripts/scout.py`

// ---- Phase 1: Discover ----
phase('Discover')
const DISCOVERY_SCHEMA = {
  type: 'object',
  additionalProperties: true,
  properties: {
    host: { type: 'object', additionalProperties: true },
    candidates: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: true,
        properties: {
          role: { type: 'string' },
          repo: { type: 'string' },
          est_ram_gb: { type: ['number', 'null'] },
          reasoning: { type: 'boolean' },
          fits: { type: 'boolean' },
          trusted: { type: 'boolean' },
        },
        required: ['role', 'repo'],
      },
    },
  },
  required: ['candidates'],
}
const discovery = await agent(
  [
    'Run this command and read its JSON output:',
    `\`python3 ${scout} --json --limit 4\``,
    'It lists MLX models on HuggingFace bucketed by role for this host.',
    `Return the "host" object and a flat "candidates" array covering roles [${roles.join(', ')}],`,
    'each item {role, repo, est_ram_gb, reasoning, fits, trusted}.',
  ].join('\n'),
  { label: 'discover', phase: 'Discover', schema: DISCOVERY_SCHEMA },
)

const candidates = ((discovery && discovery.candidates) || []).filter((c) => roles.includes(c.role))
if (candidates.length === 0) {
  log('No candidates discovered — is python3 available and HuggingFace reachable?')
  return { host: discovery && discovery.host, error: 'no candidates', recommendation: null }
}

// ---- Phase 2: Verify (one agent per candidate, concurrent) ----
const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: true,
  properties: {
    repo: { type: 'string' },
    role: { type: 'string' },
    available_locally: { type: 'boolean' },
    reasoning_confirmed: { type: ['boolean', 'null'] },
    loads: { type: ['boolean', 'null'] },
    note: { type: 'string' },
  },
  required: ['repo', 'role', 'note'],
}
const verified = (await parallel(
  candidates.map((c) => () =>
    agent(
      [
        `Verify local-model candidate "${c.repo}" for the "${c.role}" role on this Apple Silicon host. Do NOT download anything.`,
        '1. Is it already in the local runtime? Query Ollama (`curl -s http://127.0.0.1:11434/api/tags`) and LM Studio (`curl -s http://localhost:1234/v1/models`).',
        '2. If available locally: send ONE short chat generation with a small token budget and inspect the reply. A reasoning model fills the budget with hidden "thinking" and returns empty visible content → reasoning_confirmed=true; set loads=true.',
        `3. If NOT available locally: fetch its model card (\`curl -s https://huggingface.co/api/models/${c.repo}\`) and infer reasoning from tags/name; set available_locally=false, loads=null.`,
        'Return {repo, role, available_locally, reasoning_confirmed, loads, note} — note is one concise sentence.',
      ].join('\n'),
      { label: `verify:${c.repo}`, phase: 'Verify', schema: VERDICT_SCHEMA },
    ).then((v) => ({ ...c, verdict: v })),
  ),
)).filter(Boolean)

// ---- Phase 3: Recommend ----
phase('Recommend')
const recommendation = await agent(
  [
    'Recommend a local MLX model-routing config for this Apple Silicon host.',
    'Host + verified candidates (JSON):',
    JSON.stringify({ host: discovery && discovery.host, verified }, null, 2),
    '',
    `For each role in [${roles.join(', ')}], pick the single best VERIFIED model, preferring reputable`,
    'publishers and models that fit in RAM. For fast/cheap roles (bulk general, embeddings) require',
    'NON-reasoning models — never route a reasoning-confirmed model to a fast/cheap role.',
    'Give exact wiring for each pick (ollama/<tag>, lmstudio/<model>, or a native mlx_lm/mlx-vlm provider).',
    'Return concise markdown: a per-role table (role | model | ~RAM | reasoning | wiring), then a short "how to apply" note.',
  ].join('\n'),
  { label: 'recommend', phase: 'Recommend' },
)

return recommendation
