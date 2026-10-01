"""Download a Hugging Face snapshot (or one file) and record the outcome."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _snapshot_download(**kwargs):
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def main(argv=None, download=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--file", default=None)
    parser.add_argument("--ignore", action="append", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--marker", required=True)
    arguments = parser.parse_args(argv)
    kwargs = {"repo_id": arguments.repo, "revision": arguments.revision}
    if arguments.file:
        kwargs["allow_patterns"] = [arguments.file]
    if arguments.ignore:
        kwargs["ignore_patterns"] = arguments.ignore
    if arguments.cache_dir:
        kwargs["cache_dir"] = arguments.cache_dir
    if arguments.local_dir:
        kwargs["local_dir"] = arguments.local_dir
    try:
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
