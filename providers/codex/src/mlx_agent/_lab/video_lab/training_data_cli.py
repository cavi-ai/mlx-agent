"""Read-only CLI for generic training dataset metadata and snapshots."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from video_lab.dataset_cli import DatasetCliError, DatasetExit, load_request
from video_lab.domain import canonical_json
from video_lab.training_data import (
    DatasetNotReady,
    FrozenDataset,
    TrainingDataset,
    freeze_dataset,
)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise DatasetCliError("usage_error", message, DatasetExit.USAGE)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _Parser(description="Validate and freeze generic training dataset metadata.")
    # Shared lab options are accepted; these read-only commands do not open a store.
    parser.add_argument("--database")
    parser.add_argument("--artifacts")
    parser.add_argument("operation", choices=("training-data",))
    parser.add_argument("action", choices=("validate", "freeze", "inspect"))
    parser.add_argument("--request-root", type=Path, required=True)
    parser.add_argument("--request-file", required=True)
    parser.add_argument("--json", action="store_true")
    try:
        args = parser.parse_args(argv)
        model = FrozenDataset if args.action == "inspect" else TrainingDataset
        dataset = load_request(args.request_root, args.request_file, model)
        frozen = dataset if isinstance(dataset, FrozenDataset) else freeze_dataset(dataset)
        if args.action == "freeze":
            data = frozen.model_dump(mode="json")
            data["samples"] = sorted(data["samples"], key=lambda sample: sample["sample_id"])
        else:
            data = {
                "dataset_digest": frozen.dataset_digest,
                "sample_count": len(frozen.samples),
                "split_counts": dict(sorted(Counter(sample.split for sample in frozen.samples).items())),
            }
        envelope = {"schema_version": "training-data-cli.v1", "ok": True, "data": data}
        code = 0
    except DatasetNotReady as error:
        envelope = {"schema_version": "training-data-cli.v1", "ok": False,
                    "error": {"code": "dataset_not_ready", "message": str(error),
                              "reason_codes": list(error.reason_codes)}}
        code = int(DatasetExit.REJECTED)
    except DatasetCliError as error:
        envelope = {"schema_version": "training-data-cli.v1", "ok": False,
                    "error": {"code": error.code, "message": error.safe_message}}
        code = int(error.exit_code)
    print(canonical_json(envelope))
    return code


if __name__ == "__main__":
    sys.exit(main())
