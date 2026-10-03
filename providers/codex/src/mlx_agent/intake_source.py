"""Parse a pasted Hugging Face model link or bare org/name id."""

from __future__ import annotations

import re
import urllib.parse

_HOSTS = frozenset({"huggingface.co", "www.huggingface.co", "hf.co"})
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_FILE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")
_NON_MODEL_OWNERS = frozenset({
    "api", "blog", "collections", "datasets", "docs", "models",
    "organizations", "papers", "settings", "spaces",
})
_MAX_INPUT = 2048


class IntakeSourceError(ValueError):
    """A pasted source that is not a Hugging Face model reference."""

    code = "invalid_source"
    remediation = (
        "Paste a model link such as https://huggingface.co/org/name "
        "(tree, blob, and resolve links work) or a bare org/name id; "
        "org/name/folder or a tree link names a checkpoint in a subfolder."
    )


def validate_revision(text):
    if not isinstance(text, str) or not _REVISION.fullmatch(text):
        raise IntakeSourceError("Unsupported revision: {0!r}".format(text))
    return text


def validate_file(text):
    if not isinstance(text, str) or not text.lower().endswith(".gguf"):
        raise IntakeSourceError("Only .gguf files can be picked: {0!r}".format(text))
    parts = text.split("/")
    if not all(_FILE_SEGMENT.fullmatch(part) for part in parts):
        raise IntakeSourceError("Invalid file path: {0!r}".format(text))
    return text


def validate_subfolder(text):
    parts = text.split("/") if isinstance(text, str) else []
    if not parts or not all(_FILE_SEGMENT.fullmatch(part) for part in parts):
        raise IntakeSourceError("Invalid subfolder: {0!r}".format(text))
    return text


def parse_hf_source(text):
    """Return {"repo", "revision", "file", "subfolder"}; file is set only for a .gguf link, subfolder for
    a checkpoint below the repository root (bare org/name/folder, or a tree link into a folder)."""
    if not isinstance(text, str):
        raise IntakeSourceError("The source must be text.")
    value = text.strip()
    if (
        not value
        or len(value) > _MAX_INPUT
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise IntakeSourceError("The source is empty, too long, or contains control characters.")
    lowered = value.lower()
    host_prefixes = tuple(host + "/" for host in _HOSTS)
    if "://" not in value and not lowered.startswith(host_prefixes):
        return _from_segments(value.split("/"), bare=True, original=value)
    if "://" not in value:
        value = "https://" + value
    parsed = urllib.parse.urlsplit(value)
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() not in ("https", "http") or host not in _HOSTS:
        raise IntakeSourceError("Not a Hugging Face model link: {0}".format(text.strip()))
    if parsed.username is not None or parsed.password is not None:
        raise IntakeSourceError("Links with credentials are refused.")
    try:
        port = parsed.port
    except ValueError as error:
        raise IntakeSourceError("The link has an invalid port.") from error
    if port not in (None, 80, 443):
        raise IntakeSourceError("The link has an unexpected port.")
    segments = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    return _from_segments(segments, bare=False, original=text.strip())


def _from_segments(segments, bare, original):
    if bare and len(segments) < 2:
        raise IntakeSourceError("A bare id must be org/name or org/name/folder: {0}".format(original))
    if len(segments) < 2:
        raise IntakeSourceError("The link does not name a model repository: {0}".format(original))
    owner, name = segments[0], segments[1]
    if owner.lower() in _NON_MODEL_OWNERS:
        raise IntakeSourceError("Datasets, Spaces, and site pages are not models: {0}".format(original))
    if not _SEGMENT.fullmatch(owner) or not _SEGMENT.fullmatch(name):
        raise IntakeSourceError("Invalid repository id: {0}/{1}".format(owner, name))
    result = {"repo": "{0}/{1}".format(owner, name), "revision": "main", "file": None, "subfolder": None}
    if len(segments) >= 4 and segments[2] in ("tree", "blob", "resolve"):
        result["revision"] = validate_revision(segments[3])
        rest = segments[4:]
        if segments[2] != "tree" and rest and rest[-1].lower().endswith(".gguf"):
            result["file"] = validate_file("/".join(rest))
        elif segments[2] == "tree" and rest:
            result["subfolder"] = validate_subfolder("/".join(rest))
    elif bare and len(segments) > 2:
        result["subfolder"] = validate_subfolder("/".join(segments[2:]))
    return result
