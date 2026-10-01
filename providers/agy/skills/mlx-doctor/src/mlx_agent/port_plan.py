"""Draft a porting plan from a port analysis using a local loopback model."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .intake import TRANSPORT_ERRORS
from .port_analysis import PortAnalysisError, analyze_with_sources
from .transactions import _atomic_in_directory
from .verification import _http_json_request, _validated_loopback_url

PORT_PLAN_SCHEMA = "port-plan/1"
DRAFT_TIMEOUT_SECONDS = 900.0
DRAFT_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
SYSTEM_PROMPT = "You write precise engineering plans. You only state facts present in the provided material."
INSTRUCTIONS = (
    "Write an implementation plan for porting the Hugging Face model below to MLX (Apple's array "
    "framework), reusing the existing MLX modules listed. Sections: 1. Summary. 2. Reused modules "
    "(exact module paths). 3. Modules to write, each naming the PyTorch class it mirrors and the "
    "weight prefixes it loads. 4. Weight conversion (prefix renames, dtypes, sharding). "
    "5. Pre- and post-processing (feature extractor, tokenizer). 6. Validation against the "
    "transformers reference (same inputs, compare outputs). 7. Risks and unknowns. Only claim facts "
    "present in the analysis or the source; label anything else as an assumption."
)


def port_plans_root(env=None):
    env = os.environ if env is None else env
    base = env.get("XDG_STATE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".local" / "state"
    return root / "mlx-workbench" / "port-plans"


def build_prompt(payload, sources, budget_chars):
    analysis_json = json.dumps(payload, indent=1, sort_keys=True)
    lines = [INSTRUCTIONS, "", "## Deterministic analysis (JSON)", analysis_json, "", "## Existing MLX modules to reuse"]
    for component in payload["components"]:
        for match in component["matches"]:
            lines.append("- {0} `{1}` -> {2} ({3}, {4})".format(
                component["role"], component["model_type"], match["module"], match["backend"], match["match"]
            ))
    lines += ["", "## Source of the custom code"]
    prompt = "\n".join(lines)
    truncated = False
    for name in sorted(sources, key=lambda item: (not item.startswith("modeling_"), item)):
        block = "\n\n### {0}\n```python\n{1}\n```".format(name, sources[name])
        room = budget_chars - len(prompt)
        if room <= 0:
            truncated = True
            break
        if len(block) > room:
            prompt += block[:room] + "\n[truncated]"
            truncated = True
            break
        prompt += block
    return prompt, truncated


def draft_port_plan(text, endpoint, model, revision=None, context_tokens=32768, max_tokens=4096,
                    out_dir=None, client=None, manifests=None, registries=None, post=None, now=None):
    try:
        parsed = _validated_loopback_url(endpoint)
    except ValueError as error:
        raise PortAnalysisError("endpoint_not_loopback", str(error),
                                "Point --endpoint at a model served on 127.0.0.1 or localhost.") from error
    origin = "{0}://{1}".format(parsed.scheme, parsed.netloc)
    payload, sources = analyze_with_sources(text, revision, client, manifests, registries)
    analysis_sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    budget = max(4000, (int(context_tokens) - int(max_tokens) - 512) * 3)
    prompt, truncated = build_prompt(payload, sources, budget)
    post = post or (lambda url, body, timeout: _http_json_request(
        url, "POST", payload=body, timeout=timeout, max_response_bytes=DRAFT_MAX_RESPONSE_BYTES))
    try:
        response = post(
            "{0}/v1/chat/completions".format(origin),
            {"model": model, "temperature": 0, "max_tokens": int(max_tokens),
             "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]},
            DRAFT_TIMEOUT_SECONDS,
        )
    except TRANSPORT_ERRORS as error:
        raise PortAnalysisError("draft_failed", "The local model request failed: {0}".format(error),
                                "Check that {0} is serving {1}, then retry.".format(origin, model)) from error
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        content = None
    if not isinstance(content, str) or not content.strip():
        raise PortAnalysisError("draft_empty", "The local model returned no plan text.",
                                "Retry with a larger --max-tokens or a stronger local model.")
    stamp = (now or (lambda: datetime.now(timezone.utc)))()
    repo = payload["source"]["repo"]
    document = (
        "# Port plan draft: {0}\n\n"
        "> Draft by {1} via {2}; analysis sha256 {3}; generated {4}. "
        "The analysis is the evidence; this prose is an unverified model draft.\n\n{5}\n"
    ).format(repo, model, origin, analysis_sha, stamp.isoformat(), content.strip())
    directory = Path(out_dir) if out_dir is not None else port_plans_root()
    if directory.is_symlink():
        raise PortAnalysisError("symlink_refused", "{0} is a symbolic link.".format(directory),
                                "Use a real directory for port plans.")
    directory.mkdir(parents=True, exist_ok=True)
    filename = "{0}-{1}.md".format(repo.replace("/", "--"), stamp.strftime("%Y%m%dT%H%M%SZ"))
    encoded = document.encode("utf-8")
    _atomic_in_directory(directory, filename, encoded, 0o600)
    return {
        "schema": PORT_PLAN_SCHEMA, "path": str(directory / filename), "model": model,
        "endpoint": origin, "analysis_sha256": analysis_sha, "prompt_chars": len(prompt),
        "truncated": truncated, "bytes": len(encoded),
    }
