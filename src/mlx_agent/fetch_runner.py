"""Download a Hugging Face snapshot (or one file) and record the outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .source_evidence import identities, safe_path
else:
    from source_evidence import identities, safe_path


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
    parser.add_argument("--additional-file", action="append", default=[])
    parser.add_argument("--source-subfolder")
    parser.add_argument("--fresh-destination", action="store_true")
    parser.add_argument("--capture-identities", action="store_true")
    parser.add_argument("--proof-token")
    parser.add_argument("--preview-hash")
    parser.add_argument("--started-at")
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
        kwargs["allow_patterns"] = [arguments.file] + sorted(arguments.additional_file)
    if arguments.allow:
        kwargs["allow_patterns"] = arguments.allow
    if arguments.ignore:
        kwargs["ignore_patterns"] = arguments.ignore
    if arguments.cache_dir:
        kwargs["cache_dir"] = arguments.cache_dir
    if arguments.local_dir:
        kwargs["local_dir"] = arguments.local_dir
    try:
        if arguments.fresh_destination:
            if not arguments.local_dir or arguments.cache_dir or not arguments.proof_token or arguments.base_repo or arguments.require:
                raise ValueError("A managed download needs a new local destination and receipt identity.")
            target = safe_path(arguments.local_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            safe_path(target)
            target.mkdir(mode=0o700)
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
        def download_support():
            for item in arguments.download:
                sha256, size, url = item.split("=", 2)
                name = url.rsplit("/", 1)[-1]
                print("downloading {0}".format(url), flush=True)
                fetch_file(url, sha256, int(size), Path(arguments.downloads_dir) / sha256 / name, opener)
        if not arguments.fresh_destination:
            download_support()
        path = (download or _snapshot_download)(**kwargs)
        if arguments.fresh_destination and arguments.download:
            support = safe_path(arguments.downloads_dir)
            if support == target or not support.is_relative_to(target) or support.exists():
                raise ValueError("The managed support destination is outside the reviewed source or already occupied.")
            download_support()
        result, code = {"exit_status": "done", "path": str(path)}, 0
        if arguments.capture_identities:
            result["files"] = identities(path)
            selected = [arguments.file] + sorted(arguments.additional_file) if arguments.file else []
            names = {item["path"] for item in result["files"]}
            if selected and not set(selected) <= names:
                raise ValueError("The managed download did not contain every reviewed file.")
        print("downloaded {0} to {1}".format(arguments.repo, path), flush=True)
    except Exception as error:  # noqa: BLE001 - recorded in the marker and the log
        result, code = {"exit_status": "failed", "error": str(error)}, 1
        print("download failed: {0}".format(error), flush=True)
    if arguments.proof_token:
        result.update(repo=arguments.repo, revision=arguments.revision, file=arguments.file,
                      subfolder=arguments.source_subfolder, additional_files=sorted(arguments.additional_file), local_dir=arguments.local_dir,
                      preview_hash=arguments.preview_hash, proof_token=arguments.proof_token, started_at=arguments.started_at,
                      downloads_dir=arguments.downloads_dir, downloads=[{"sha256": item.split("=", 2)[0], "size": int(item.split("=", 2)[1]), "url": item.split("=", 2)[2]} for item in arguments.download])
        # --allow PREFIX/* represents an explicitly selected checkpoint.
        if arguments.allow and len(arguments.allow) == 1 and arguments.allow[0].endswith("/*"):
            result["subfolder"] = arguments.allow[0][:-2]
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    Path(arguments.marker).write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
