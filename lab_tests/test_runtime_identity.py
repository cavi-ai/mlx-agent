from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_lab.runtime_identity import (
    animegen_runtime_versions,
    runtime_fingerprint,
    source_tree_digest,
)


def test_source_tree_digest_changes_when_package_source_changes(tmp_path: Path) -> None:
    package = tmp_path / "video_lab"
    package.mkdir()
    source = package / "worker.py"
    source.write_text("VALUE = 1\n")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "worker.pyc").write_bytes(b"ignored")

    first = source_tree_digest(package)
    source.write_text("VALUE = 2\n")
    second = source_tree_digest(package)

    assert first.startswith("source-sha256:")
    assert second.startswith("source-sha256:")
    assert first != second


def test_runtime_identity_includes_sorted_normalized_installed_distributions(
    tmp_path: Path,
) -> None:
    package = tmp_path / "video_lab"
    package.mkdir()
    (package / "__init__.py").write_text("")
    versions = animegen_runtime_versions(
        package_root=package,
        distribution_entries=(
            ("Tokenizers", "0.22.2"),
            ("Otherwise_Uncurated.Plugin", "9.1"),
            ("tokenizers", "0.22.2"),
        ),
        python_version="3.12.10",
    )

    assert versions == {
        "python": "3.12.10",
        "dist:otherwise-uncurated-plugin": "9.1",
        "dist:tokenizers": "0.22.2",
        "video-lab": source_tree_digest(package),
    }


def test_otherwise_uncurated_distribution_changes_runtime_fingerprint(tmp_path: Path) -> None:
    package = tmp_path / "video_lab"
    package.mkdir()
    (package / "__init__.py").write_text("")
    first = animegen_runtime_versions(
        package_root=package,
        distribution_entries=(("tokenizers", "0.22.2"), ("some-plugin", "1.0")),
        python_version="3.12.10",
    )
    second = animegen_runtime_versions(
        package_root=package,
        distribution_entries=(("tokenizers", "0.22.2"), ("some-plugin", "2.0")),
        python_version="3.12.10",
    )

    assert runtime_fingerprint(first) != runtime_fingerprint(second)
    assert runtime_fingerprint(first) == runtime_fingerprint(dict(reversed(first.items())))


def test_conflicting_duplicate_normalized_distribution_versions_fail_closed(
    tmp_path: Path,
) -> None:
    package = tmp_path / "video_lab"
    package.mkdir()
    (package / "__init__.py").write_text("")

    with pytest.raises(ValueError, match="conflicting installed versions"):
        animegen_runtime_versions(
            package_root=package,
            distribution_entries=(("Some_Plugin", "1.0"), ("some-plugin", "2.0")),
            python_version="3.12.10",
        )


def test_runtime_identity_never_serializes_package_paths(tmp_path: Path) -> None:
    package = tmp_path / "private" / "video_lab"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")

    versions = animegen_runtime_versions(
        package_root=package,
        distribution_entries=(("example", "1.0"),),
        python_version="3.12.10",
    )

    assert str(tmp_path) not in json.dumps(versions)
