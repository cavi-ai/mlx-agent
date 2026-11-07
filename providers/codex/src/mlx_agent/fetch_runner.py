"""Download a Hugging Face snapshot (or one file) and record the outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _snapshot_download(**kwargs):
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def _urlopen(url):
    import urllib.request

    return urllib.request.urlopen(url, timeout=60)  # noqa: S310 - https URLs validated by the backend manifest


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_file(url, sha256, size, target, opener=None):
    """Download ``url`` to ``target`` once, keeping it only when its size and sha256 match."""
    target = Path(target)
    if target.is_file() and target.stat().st_size == size and _sha256(target) == sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest, written = hashlib.sha256(), 0
    with (opener or _urlopen)(url) as response, open(partial, "wb") as handle:
        for block in iter(lambda: response.read(1 << 20), b""):
            written += len(block)
            if written > size:
                break
            digest.update(block)
            handle.write(block)
    if written != size or digest.hexdigest() != sha256:
        partial.unlink()
        raise ValueError("{0} did not match its recorded size and sha256.".format(url))
    partial.replace(target)
    return target


def main(argv=None, download=None, opener=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--file", default=None)
    parser.add_argument("--ignore", action="append", default=None)
    parser.add_argument("--allow", action="append", default=None)
    parser.add_argument("--base-repo", default=None, help="also download this pinned base snapshot first")
    parser.add_argument("--base-revision", default=None)
    parser.add_argument("--base-ignore", action="append", default=None)
    parser.add_argument("--require", action="append", default=[],
                        help="also download REPO=FILE from another repository first")
    parser.add_argument("--download", action="append", default=[],
                        help="also download SHA256=SIZE=URL into --downloads-dir first")
    parser.add_argument("--downloads-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--marker", required=True)
    arguments = parser.parse_args(argv)
    kwargs = {"repo_id": arguments.repo, "revision": arguments.revision}
    if arguments.file:
        kwargs["allow_patterns"] = [arguments.file]
    if arguments.allow:
        kwargs["allow_patterns"] = arguments.allow
    if arguments.ignore:
        kwargs["ignore_patterns"] = arguments.ignore
    if arguments.cache_dir:
        kwargs["cache_dir"] = arguments.cache_dir
    if arguments.local_dir:
        kwargs["local_dir"] = arguments.local_dir
    try:
        if arguments.base_repo:
            base = {"repo_id": arguments.base_repo, "revision": arguments.base_revision or "main"}
            if arguments.base_ignore:
                base["ignore_patterns"] = arguments.base_ignore
            if arguments.cache_dir:
                base["cache_dir"] = arguments.cache_dir
            print("downloading base {0}".format(arguments.base_repo), flush=True)
            (download or _snapshot_download)(**base)
        required = {}
        for item in arguments.require:
            repo, _, name = item.partition("=")
            required.setdefault(repo, []).append(name)
        for repo, names in sorted(required.items()):
            extra = {"repo_id": repo, "allow_patterns": names}
            if arguments.cache_dir:
                extra["cache_dir"] = arguments.cache_dir
            print("downloading required {0}".format(repo), flush=True)
            (download or _snapshot_download)(**extra)
        for item in arguments.download:
            sha256, size, url = item.split("=", 2)
            name = url.rsplit("/", 1)[-1]
            print("downloading {0}".format(url), flush=True)
            fetch_file(url, sha256, int(size), Path(arguments.downloads_dir) / sha256 / name, opener)
        path = (download or _snapshot_download)(**kwargs)
        result, code = {"exit_status": "done", "path": str(path)}, 0
        print("downloaded {0} to {1}".format(arguments.repo, path), flush=True)
    except Exception as error:  # noqa: BLE001 - recorded in the marker and the log
        result, code = {"exit_status": "failed", "error": str(error)}, 1
        print("download failed: {0}".format(error), flush=True)
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    Path(arguments.marker).write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
