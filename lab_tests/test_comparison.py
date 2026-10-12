import pytest

from lab_tests.test_capabilities import manifest
from video_lab.comparison import ComparisonError, build_controlled_pair, changed_paths
from video_lab.domain import ExperimentDraft


def test_builds_pair_with_exactly_declared_change() -> None:
    pair = build_controlled_pair(
        ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=8)
    )
    assert pair.changed_paths == ("inference_steps",)
    assert pair.variant_b.inference_steps == 8


def test_rejects_unknown_or_invariant_path() -> None:
    with pytest.raises(ComparisonError, match="unknown variable path"):
        build_controlled_pair(
            ExperimentDraft(baseline=manifest(), variable_path="missing", alternate_value=8)
        )
    with pytest.raises(ComparisonError, match="must change exactly one field"):
        build_controlled_pair(
            ExperimentDraft(baseline=manifest(), variable_path="inference_steps", alternate_value=4)
        )


def test_changed_paths_distinguishes_missing_values_from_none() -> None:
    assert changed_paths({}, {"x": None}) == ("x",)
    assert changed_paths({"x": None}, {}) == ("x",)
    assert changed_paths([], [None]) == (".0",)


def test_changed_paths_reports_nested_missing_none_path() -> None:
    assert changed_paths({"outer": {}}, {"outer": {"inner": None}}) == ("outer.inner",)
