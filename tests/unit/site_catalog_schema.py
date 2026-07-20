"""Small fail-closed JSON Schema validator used by the catalog contract tests.

The repository has no runtime dependency manifest, so this test-only validator
implements the Draft 2020-12 keywords used by ``site-catalog.schema.json``.
It deliberately raises for schema keywords it does not understand.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit


class SchemaValidationError(AssertionError):
    """Raised when a catalog instance does not satisfy its schema."""


_ANNOTATION_KEYWORDS = {"$schema", "$id", "title"}
_VALIDATION_KEYWORDS = {
    "$defs",
    "$ref",
    "additionalProperties",
    "allOf",
    "const",
    "contains",
    "enum",
    "format",
    "items",
    "maxContains",
    "maxItems",
    "minContains",
    "minItems",
    "minLength",
    "properties",
    "pattern",
    "required",
    "type",
    "uniqueItems",
}


def validate_site_catalog(instance: object, schema: Mapping[str, object]) -> None:
    """Validate an instance using every JSON Schema keyword used by this schema."""
    _validate(instance, schema, schema, "$")


def _validate(instance: object, schema: Mapping[str, object], root: Mapping[str, object], path: str) -> None:
    unknown = set(schema) - _ANNOTATION_KEYWORDS - _VALIDATION_KEYWORDS
    if unknown:
        raise SchemaValidationError(f"{path}: unsupported schema keyword(s): {sorted(unknown)}")

    if "$ref" in schema:
        _validate(instance, _resolve_reference(schema["$ref"], root, path), root, path)
        return

    if "type" in schema and not _matches_type(instance, schema["type"]):
        raise SchemaValidationError(f"{path}: expected {schema['type']!r}, got {type(instance).__name__}")
    if "const" in schema and not _json_equal(instance, schema["const"]):
        raise SchemaValidationError(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and not any(_json_equal(instance, value) for value in schema["enum"]):
        raise SchemaValidationError(f"{path}: must be one of {schema['enum']!r}")
    if "minLength" in schema and isinstance(instance, str) and len(instance) < schema["minLength"]:
        raise SchemaValidationError(f"{path}: must have at least {schema['minLength']} characters")
    if "pattern" in schema and isinstance(instance, str) and re.search(schema["pattern"], instance) is None:
        raise SchemaValidationError(f"{path}: must match {schema['pattern']!r}")
    if "format" in schema and isinstance(instance, str):
        _validate_format(instance, schema["format"], path)

    if isinstance(instance, Mapping):
        required = schema.get("required", [])
        missing = [key for key in required if key not in instance]
        if missing:
            raise SchemaValidationError(f"{path}: missing required properties {missing}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unexpected = set(instance) - set(properties)
            if unexpected:
                raise SchemaValidationError(f"{path}: unexpected properties {sorted(unexpected)}")
        for key, value_schema in properties.items():
            if key in instance:
                _validate(instance[key], value_schema, root, f"{path}.{key}")

    if _is_json_array(instance):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            raise SchemaValidationError(f"{path}: must contain at least {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise SchemaValidationError(f"{path}: must contain at most {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(value, sort_keys=True, separators=(",", ":")) for value in instance]
            if len(encoded) != len(set(encoded)):
                raise SchemaValidationError(f"{path}: items must be unique")
        if "items" in schema:
            for index, value in enumerate(instance):
                _validate(value, schema["items"], root, f"{path}[{index}]")
        if "contains" in schema:
            matches = sum(
                _is_valid(value, schema["contains"], root, f"{path}[{index}]")
                for index, value in enumerate(instance)
            )
            minimum = schema.get("minContains", 1)
            maximum = schema.get("maxContains")
            if matches < minimum or maximum is not None and matches > maximum:
                raise SchemaValidationError(f"{path}: contains constraint matched {matches} items")

    for index, branch in enumerate(schema.get("allOf", [])):
        _validate(instance, branch, root, f"{path}.allOf[{index}]")


def _resolve_reference(reference: object, root: Mapping[str, object], path: str) -> Mapping[str, object]:
    if not isinstance(reference, str) or not reference.startswith("#/"):
        raise SchemaValidationError(f"{path}: unsupported reference {reference!r}")
    target: object = root
    for token in reference[2:].split("/"):
        if not isinstance(target, Mapping) or token not in target:
            raise SchemaValidationError(f"{path}: unresolved reference {reference!r}")
        target = target[token]
    if not isinstance(target, Mapping):
        raise SchemaValidationError(f"{path}: reference {reference!r} does not resolve to a schema")
    return target


def _is_valid(instance: object, schema: Mapping[str, object], root: Mapping[str, object], path: str) -> bool:
    try:
        _validate(instance, schema, root, path)
    except SchemaValidationError:
        return False
    return True


def _matches_type(value: object, expected: object) -> bool:
    expected_types = expected if isinstance(expected, list) else [expected]
    type_checks = {
        "array": _is_json_array,
        "boolean": lambda item: isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "null": lambda item: item is None,
        "object": lambda item: isinstance(item, Mapping),
        "string": lambda item: isinstance(item, str),
    }
    return any(type_name in type_checks and type_checks[type_name](value) for type_name in expected_types)


def _is_json_array(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _json_equal(left: object, right: object) -> bool:
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    return left == right


def _validate_format(value: str, format_name: object, path: str) -> None:
    if format_name != "uri":
        raise SchemaValidationError(f"{path}: unsupported format {format_name!r}")
    parsed = urlsplit(value)
    if not parsed.scheme:
        raise SchemaValidationError(f"{path}: must be a URI")
