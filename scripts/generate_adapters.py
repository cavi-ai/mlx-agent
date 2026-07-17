#!/usr/bin/env python3
"""Generate deterministic provider adapters from the canonical plugin manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_PROVIDERS = ("claude", "agentskills")


def _capability_id(manifest: Mapping[str, object], capability: str) -> str:
    return "{0}.{1}".format(manifest["identity"], capability)


def _command_markdown(manifest: Mapping[str, object], capability: str, root: str) -> str:
    identifier = _capability_id(manifest, capability)
    descriptions = manifest["capabilities"]
    description = descriptions[capability]["description"]
    invocation = "python3 {0}/scripts/mlx-agent".format(root)
    front_matter = "---\nname: mlx-{0}\ndescription: {1}\n---\n\n".format(capability, description)
    if capability == "scout":
        body = """# MLX Scout

canonical capability ID: {identifier}

Run the provider-neutral discovery command:

`{invocation} discover $ARGUMENTS`

Present its evidence and recommendations as returned. Discovery must not download model weights or change configuration. If a later download or configuration mutation would help, describe the exact CLI preview first and obtain explicit user confirmation before it.
""".format(identifier=identifier, invocation=invocation)
    elif capability == "adopt":
        body = """# MLX Adopt

canonical capability ID: {identifier}

Use the durable adoption state owned by the structured CLI. Start with a user-visible state path and requested roles:

`{invocation} adopt start --state <state-path> --role <role> --json`

If the state already exists or an earlier run was interrupted, continue it with:

`{invocation} adopt resume --state <state-path> --json`

Report the CLI state and recommendations. Do not recreate adoption policy in this adapter. This operation must not download model weights or change configuration; any later download or mutation requires explicit user confirmation and the reviewed CLI preview.
""".format(identifier=identifier, invocation=invocation)
    else:
        body = """# MLX Wire

canonical capability ID: {identifier}

Use the structured CLI to render a non-mutating configuration preview:

`{invocation} wire render <model> --target <target> --path <config-path> --json`

Show the returned preview and its hash. Do not write configuration files directly. Only after the user explicitly confirms that exact preview, run:

`{invocation} wire apply <model> --target <target> --path <config-path> --confirm --preview-hash <preview-hash> --json`

Never download model weights without an explicit confirmation. Report the transaction receipt returned by the CLI.
""".format(identifier=identifier, invocation=invocation)
    return front_matter + body


def _advisor_markdown(manifest: Mapping[str, object]) -> str:
    scout = _capability_id(manifest, "scout")
    adopt = _capability_id(manifest, "adopt")
    wire = _capability_id(manifest, "wire")
    return """---
description: Provider adapter for the structured MLX agent CLI.
---

# MLX Advisor

canonical capability ID: {scout}
canonical capability ID: {adopt}
canonical capability ID: {wire}

Use only the structured CLI beneath `${{CLAUDE_PLUGIN_ROOT}}/scripts/mlx-agent`. Run `discover` for evidence, `adopt start --state <state-path>` or `adopt resume --state <state-path>` for durable recommendations, and `wire render` before any requested wiring. Do not duplicate adoption policy, download model weights, or write configuration files. A download or configuration mutation is permitted only after explicit user confirmation of the CLI preview; use `wire apply --confirm --preview-hash <preview-hash>` for the reviewed change.
""".format(scout=scout, adopt=adopt, wire=wire)


def _generic_skill_markdown(manifest: Mapping[str, object], capability: str) -> str:
    content = _command_markdown(manifest, capability, "../../..")
    return content.replace("$ARGUMENTS", "<arguments>")


def _plugin_metadata(manifest: Mapping[str, object]) -> str:
    capabilities = manifest["capabilities"]
    payload = {
        "name": manifest["identity"],
        "version": "0.1.0",
        "description": "Structured local MLX discovery, adoption, and wiring for Apple Silicon agents.",
        "author": {"name": "Sasan Sotoodehfar"},
        "homepage": "https://github.com/sasan1200/mlx-agent",
        "repository": "https://github.com/sasan1200/mlx-agent",
        "license": "MIT",
        "keywords": ["mlx", "apple-silicon", "local-llm", "agent-adapter"],
        "capabilities": [
            {"id": _capability_id(manifest, name), "command": capabilities[name]["command"]}
            for name in sorted(capabilities)
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _marketplace_metadata(manifest: Mapping[str, object]) -> str:
    payload = {
        "name": manifest["identity"],
        "description": "Structured local MLX discovery, adoption, and wiring for Apple Silicon agents.",
        "owner": {"name": "Sasan Sotoodehfar"},
        "plugins": [{
            "name": manifest["identity"],
            "description": "Structured local MLX discovery, adoption, and wiring for Apple Silicon.",
            "version": "0.1.0",
            "author": {"name": "Sasan Sotoodehfar"},
            "source": ".",
            "category": "development",
        }],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _workflow() -> str:
    return """export const meta = {
  name: 'mlx-adopt',
  description: 'Compatibility wrapper for durable MLX adoption state.',
}

const pluginRoot = (args && args.pluginRoot) || '.'
const statePath = (args && args.statePath) || '.mlx-agent-adoption.json'
const shellQuote = (value) => `'${String(value).replace(/'/g, "'\\\\''")}'`
const allowedRoles = new Set(['general', 'coding', 'reasoning', 'vision', 'embedding'])
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
"""


def _render(manifest: Mapping[str, object], provider_ids: Sequence[str]) -> Dict[Path, str]:
    selected = tuple(provider_ids)
    unknown = sorted(set(selected) - set(SUPPORTED_PROVIDERS))
    if unknown:
        raise ValueError("unsupported provider IDs: {0}".format(", ".join(unknown)))
    rendered: Dict[Path, str] = {}
    if "claude" in selected:
        claude_paths = {
            Path(".claude-plugin/plugin.json"): _plugin_metadata(manifest),
            Path(".claude-plugin/marketplace.json"): _marketplace_metadata(manifest),
            Path("commands/mlx-scout.md"): _command_markdown(manifest, "scout", "${CLAUDE_PLUGIN_ROOT}"),
            Path("commands/mlx-adopt.md"): _command_markdown(manifest, "adopt", "${CLAUDE_PLUGIN_ROOT}"),
            Path("commands/mlx-wire.md"): _command_markdown(manifest, "wire", "${CLAUDE_PLUGIN_ROOT}"),
            Path("agents/mlx-advisor.md"): _advisor_markdown(manifest),
            Path("scripts/mlx-adopt.workflow.mjs"): _workflow(),
        }
        rendered.update(claude_paths)
        for path, content in claude_paths.items():
            if path.parts[0] in {".claude-plugin", "commands", "agents"}:
                rendered[Path("providers/claude") / path] = content
    if "agentskills" in selected:
        for capability in ("scout", "adopt", "wire"):
            rendered[Path("providers/agentskills/mlx-{0}/SKILL.md".format(capability))] = _generic_skill_markdown(manifest, capability)
    return dict(sorted(rendered.items(), key=lambda item: str(item[0])))


def generate(provider_ids: Iterable[str], output_root: Path) -> List[Path]:
    """Write selected provider adapters as UTF-8 LF files and return sorted paths."""

    manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
    root = Path(output_root)
    written = []
    for relative_path, content in _render(manifest, tuple(provider_ids)).items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))
        written.append(path)
    return written


def _check(provider_ids: Sequence[str]) -> List[Path]:
    manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
    drift = []
    for relative_path, content in _render(manifest, provider_ids).items():
        path = ROOT / relative_path
        expected = content.encode("utf-8")
        if not path.is_file() or path.read_bytes() != expected:
            drift.append(relative_path)
    return drift


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated adapters drift")
    parser.add_argument("--provider", dest="providers", action="append", choices=SUPPORTED_PROVIDERS)
    arguments = parser.parse_args(argv)
    provider_ids = tuple(arguments.providers or SUPPORTED_PROVIDERS)
    if arguments.check:
        drift = _check(provider_ids)
        if drift:
            print("generated adapters drift: {0}".format(", ".join(str(path) for path in drift)), file=sys.stderr)
            return 1
        print("generated adapters are current")
        return 0
    generate(provider_ids, ROOT)
    print("generated adapters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
