from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace


def test_launcher_supports_direct_python_module_execution() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "video_lab.launcher", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Usage:" in completed.stdout


def test_feasibility_resume_keeps_launcher_torch_lazy(monkeypatch) -> None:
    import video_lab.launcher as launcher

    imported: list[str] = []
    app_calls: list[bool] = []

    def load(name: str, **_kwargs: object):
        imported.append(name)
        return SimpleNamespace(app=lambda: app_calls.append(True))

    monkeypatch.setattr(launcher, "_import", load)
    launcher.main()

    assert imported == ["video_lab.cli"]
    assert app_calls == [True]


def test_help_start_and_status_remain_torch_lazy(monkeypatch) -> None:
    import video_lab.launcher as launcher

    for _arguments in (
        ["adapter", "feasibility", "--help"],
        ["adapter", "feasibility", "start"],
        ["adapter", "feasibility", "status", "run-id"],
    ):
        imported: list[str] = []

        def load(name: str, target: list[str] = imported, **_kwargs: object):
            target.append(name)
            return SimpleNamespace(app=lambda: None)

        monkeypatch.setattr(launcher, "_import", load)
        launcher.main()

        assert imported == ["video_lab.cli"]


def test_source_resume_does_not_probe_mps_in_launcher(monkeypatch) -> None:
    import video_lab.launcher as launcher

    app_calls: list[bool] = []

    def load(name: str, **_kwargs: object):
        return SimpleNamespace(app=lambda: app_calls.append(True))

    monkeypatch.setattr(launcher, "_import", load)
    launcher.main()

    assert app_calls == [True]
