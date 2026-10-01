"""Fill a backend manifest's registry snapshot from its pinned wheel (maintenance)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx_agent.backends import BACKENDS_DIR, probe_sources_from_wheel, registry_from_sources  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", help="manifest id, e.g. mlx-audio")
    parser.add_argument("--wheel", required=True, help="path to the pinned wheel")
    arguments = parser.parse_args(argv)
    path = BACKENDS_DIR / "{0}.json".format(arguments.backend)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if "-{0}-".format(manifest["version"]) not in Path(arguments.wheel).name:
        print("wheel {0} does not match pinned version {1}".format(arguments.wheel, manifest["version"]))
        return 2
    manifest["registry"] = registry_from_sources(probe_sources_from_wheel(arguments.wheel, manifest), manifest)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    counts = {category: len(entry["model_types"]) for category, entry in manifest["registry"].items()}
    print(json.dumps({"backend": arguments.backend, "version": manifest["version"], "counts": counts}))
    return 0 if all(counts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
