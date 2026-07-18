#!/usr/bin/env python3
"""Validate committed schemas and runtime handoff parity with Draft 2020-12."""

import copy
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx_agent.adoption import AdoptionState, PHASES  # noqa: E402


def _load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _base_adoption_state():
    return {
        "schema_version": "1.1",
        "workflow_id": "schema-parity-fixture",
        "revision": 1,
        "phase": "inspect",
        "status": "running",
        "request": {
            "roles": ["general"],
            "shortlist_limit": 1,
            "allow_network": False,
            "offline": True,
            "refresh": False,
            "memory_gb": None,
            "quantization": None,
            "licenses": [],
            "include_gated": True,
            "publishers": [],
            "runtime": None,
            "fast": True,
        },
        "completed_phases": [],
        "host": {},
        "discovery": {},
        "shortlist": [],
        "evidence": [],
        "comparisons": [],
        "recommendations": [],
        "warnings": [],
        "errors": [],
        "created_at": "2026-07-17T12:00:00+00:00",
        "updated_at": "2026-07-17T12:00:00+00:00",
    }


def _runtime_accepts(value):
    try:
        AdoptionState.from_dict(value, ROOT / ".schema-parity-state.json")
        return True
    except (TypeError, ValueError):
        return False


def main():
    schema_paths = sorted((ROOT / "schemas").glob("*.schema.json"))
    schemas = {}
    for path in schema_paths:
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        schemas[path.name] = schema

    plugin_validator = Draft202012Validator(schemas["plugin.schema.json"])
    plugin = _load("plugin.json")
    plugin_validator.validate(plugin)
    invalid_plugins = []
    invalid_invocation = copy.deepcopy(plugin)
    invalid_invocation["providers"]["claude"]["invocation"]["prefix"] = "$"
    invalid_plugins.append(invalid_invocation)
    missing_native_minimum = copy.deepcopy(plugin)
    missing_native_minimum["providers"]["gemini"]["minimum_version"] = None
    invalid_plugins.append(missing_native_minimum)
    native_portable = copy.deepcopy(plugin)
    native_portable["providers"]["gemini"]["install_mode"] = "portable"
    invalid_plugins.append(native_portable)
    for index, value in enumerate(invalid_plugins):
        if plugin_validator.is_valid(value):
            raise ValueError("plugin invalid fixture {0} passed schema validation".format(index))

    adoption_validator = Draft202012Validator(
        schemas["adoption-state.schema.json"], format_checker=FormatChecker()
    )
    base = _base_adoption_state()
    for completed_count in range(len(PHASES)):
        valid = copy.deepcopy(base)
        valid["completed_phases"] = list(PHASES[:completed_count])
        valid["phase"] = PHASES[completed_count]
        valid["status"] = "complete" if valid["phase"] == "complete" else "running"
        adoption_validator.validate(valid)
        if not _runtime_accepts(valid):
            raise ValueError("runtime rejected schema-valid adoption prefix {0}".format(completed_count))

    evidence = {
        "repo": "local/model", "role": "general", "strength": "heuristic_only",
        "available_locally": False, "loads": None, "reasoning_confirmed": None,
        "runtime": None, "note": "bounded evidence", "details": {},
    }
    invalid = []
    unsupported_role = copy.deepcopy(base)
    unsupported_role["request"]["roles"] = ["unsupported"]
    invalid.append(unsupported_role)
    noncontiguous = copy.deepcopy(base)
    noncontiguous.update(completed_phases=["inspect", "shortlist"], phase="verify")
    invalid.append(noncontiguous)
    wrong_phase = copy.deepcopy(base)
    wrong_phase.update(completed_phases=["inspect"], phase="verify")
    invalid.append(wrong_phase)
    wrong_status = copy.deepcopy(base)
    wrong_status["status"] = "complete"
    invalid.append(wrong_status)
    early_evidence = copy.deepcopy(base)
    early_evidence["evidence"] = [evidence]
    invalid.append(early_evidence)
    early_comparison = copy.deepcopy(base)
    early_comparison["comparisons"] = [{"repo": "local/model", "role": "general"}]
    invalid.append(early_comparison)
    early_recommendation = copy.deepcopy(base)
    early_recommendation["recommendations"] = [{"repo": "local/model", "role": "general"}]
    invalid.append(early_recommendation)
    missing_verified_evidence = copy.deepcopy(base)
    missing_verified_evidence.update(
        completed_phases=["inspect", "discover", "shortlist", "verify"],
        phase="compare",
        shortlist=[{"repo": "local/model", "role": "general"}],
    )
    invalid.append(missing_verified_evidence)

    for index, value in enumerate(invalid):
        if adoption_validator.is_valid(value) or _runtime_accepts(value):
            raise ValueError("adoption invalid fixture {0} passed schema or runtime validation".format(index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
