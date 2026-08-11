import unittest
import os
import subprocess
from pathlib import Path
from textwrap import dedent
from tempfile import TemporaryDirectory

import sys

from mlx_agent.gguf_runner import _paths_without_script_dir


class PathsWithoutScriptDirTests(unittest.TestCase):
    def test_removes_script_directory(self):
        with TemporaryDirectory() as raw:
            script_dir = Path(raw)
            other = script_dir.parent / "site-packages"
            entries = [str(script_dir), str(other), "/usr/lib/python3"]
            cleaned = _paths_without_script_dir(entries, script_dir, script_dir.parent)
            self.assertEqual(cleaned, [str(other), "/usr/lib/python3"])

    def test_removes_cwd_placeholder_when_cwd_is_script_dir(self):
        with TemporaryDirectory() as raw:
            script_dir = Path(raw)
            cleaned = _paths_without_script_dir(
                ["", ".", str(script_dir), "/safe"],
                script_dir,
                script_dir,
            )
            self.assertEqual(cleaned, ["/safe"])

    def test_keeps_cwd_placeholder_when_cwd_differs(self):
        with TemporaryDirectory() as raw:
            script_dir = Path(raw) / "runner"
            script_dir.mkdir()
            cwd = Path(raw) / "elsewhere"
            cwd.mkdir()
            cleaned = _paths_without_script_dir(["", "."], script_dir, cwd)
            self.assertEqual(cleaned, ["", "."])

    def test_removes_nested_script_directory_entry(self):
        with TemporaryDirectory() as raw:
            script_dir = Path(raw) / "runner"
            nested = script_dir / "nested"
            nested.mkdir(parents=True)
            cleaned = _paths_without_script_dir([str(nested)], script_dir, script_dir)
            self.assertEqual([], cleaned)


class GGUFRunnerScriptTests(unittest.TestCase):
    def test_runner_uses_shadow_resistant_gguf_import(self):
        all_runners, _, _ = self._collect_gguf_runners()
        self._assert_shadow_free_execution([all_runners[0]], "root")

    def test_provider_adapter_runners_avoid_shadowed_gguf(self):
        _, provider_runners, skill_runners = self._collect_gguf_runners()
        self._assert_shadow_free_execution(
            provider_runners + skill_runners,
            "provider-copy",
        )

    @unittest.skipUnless(os.environ.get("CI"), "adapter discovery ordering is CI-only")
    def test_adapter_runner_discovery_order_is_stable(self):
        _, provider_runners, skill_runners = self._collect_gguf_runners()
        self.assertEqual(provider_runners, sorted(provider_runners, key=str))
        self.assertEqual(skill_runners, sorted(skill_runners, key=str))
        if provider_runners and skill_runners:
            combined = provider_runners + skill_runners
            split = len(provider_runners)
            self.assertEqual(combined[:split], sorted(provider_runners, key=str))
            self.assertEqual(combined[split:], sorted(skill_runners, key=str))

        if not provider_runners and not skill_runners:
            self.fail("no gguf runner adapter copies discovered in this checkout")

    def _collect_gguf_runners(self):
        repository_root = Path(__file__).resolve().parents[2]
        canonical_runner = str(repository_root / "src/mlx_agent/gguf_runner.py")
        provider_runners = sorted(
            str(path)
            for path in repository_root.glob(
                "providers/*/src/mlx_agent/gguf_runner.py"
            )
        )
        skill_runners = sorted(
            str(path)
            for path in repository_root.glob(
                "providers/*/skills/*/src/mlx_agent/gguf_runner.py"
            )
        )
        return [canonical_runner] + provider_runners + skill_runners, provider_runners, skill_runners

    def _assert_shadow_free_execution(self, runners, marker_prefix):
        with TemporaryDirectory() as raw:
            environment_root = Path(raw)
            site_packages = environment_root / "site-packages"
            self._install_fake_gguf_site_packages(site_packages)
            gguf_path = environment_root / "model.gguf"
            gguf_path.write_text("fake gguf fixture", encoding="utf-8")

            bin_dir = environment_root / "bin"
            bin_dir.mkdir()
            converter = bin_dir / "mlx_lm.convert"
            converter.write_text(
                dedent(
                    """
                    #!/usr/bin/env python3
                    import os
                    import sys
                    from pathlib import Path

                    marker = os.environ.get("GGUF_RUNNER_TEST_MARKER")
                    args = list(sys.argv[1:])
                    if "--mlx-path" in args:
                        output_index = args.index("--mlx-path") + 1
                        if output_index < len(args):
                            Path(args[output_index]).mkdir(parents=True, exist_ok=True)
                    if marker:
                        with Path(marker).open("a", encoding="utf-8") as handle:
                            handle.write("convert\\n")
                    """
                ).lstrip(),
                encoding="utf-8",
            )
            converter.chmod(0o755)

            for index, runner in enumerate(runners):
                runner_out = environment_root / ("out-" + str(index))
                marker = environment_root / ("{0}-runner-{1}.marker".format(marker_prefix, index))
                result = self._run_script(
                    Path(runner),
                    gguf_path,
                    runner_out,
                    site_packages,
                    bin_dir,
                    marker,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    msg="runner failed: {0}\nstdout:\n{1}\nstderr:\n{2}".format(
                        runner,
                        result.stdout,
                        result.stderr,
                    ),
                )
                marker_contents = marker.read_text(encoding="utf-8").splitlines()
                self.assertEqual(
                    sorted(marker_contents),
                    [
                        "convert",
                        "from_pretrained_model",
                        "from_pretrained_tokenizer",
                    ],
                )
                self.assertTrue(runner_out.is_dir())

            if not runners:
                self.fail("no gguf_runner paths discovered in this checkout")

    def _install_fake_gguf_site_packages(self, site_root):
        gguf_root = site_root / "gguf"
        gguf_root.mkdir(parents=True)
        (gguf_root / "__init__.py").write_text(
            'GGUFReader = object()\n', encoding="utf-8"
        )

        torch_root = site_root / "torch"
        torch_root.mkdir()
        (torch_root / "__init__.py").write_text(
            "# fake torch module for gguf runner tests\n", encoding="utf-8"
        )

        transformers_root = site_root / "transformers"
        transformers_root.mkdir()
        (transformers_root / "__init__.py").write_text(
            dedent(
                """
                import os
                from pathlib import Path

                import gguf


                def _touch(marker_path, value):
                    if marker_path:
                        previous = ""
                        if Path(marker_path).exists():
                            previous = Path(marker_path).read_text(encoding="utf-8")
                        Path(marker_path).write_text(previous + value + "\\n", encoding="utf-8")


                class AutoModelForCausalLM:
                    @classmethod
                    def from_pretrained(cls, directory, gguf_file=None):
                        if not hasattr(gguf, "GGUFReader"):
                            raise RuntimeError("shadowed gguf package import")
                        marker_path = os.environ.get("GGUF_RUNNER_TEST_MARKER")
                        _touch(marker_path, "from_pretrained_model")
                        return cls()

                    def save_pretrained(self, directory):
                        Path(directory).mkdir(parents=True, exist_ok=True)


                class AutoTokenizer:
                    @classmethod
                    def from_pretrained(cls, directory, gguf_file=None):
                        if not hasattr(gguf, "GGUFReader"):
                            raise RuntimeError("shadowed gguf package import")
                        marker_path = os.environ.get("GGUF_RUNNER_TEST_MARKER")
                        _touch(marker_path, "from_pretrained_tokenizer")
                        return cls()

                    def save_pretrained(self, directory):
                        Path(directory).mkdir(parents=True, exist_ok=True)
                """
            ).lstrip(),
            encoding="utf-8",
        )

    def _run_script(self, runner, gguf_path, out_dir, site_packages, bin_dir, marker):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(site_packages)
        environment["PATH"] = str(bin_dir) + os.pathsep + environment.get("PATH", "")
        environment["GGUF_RUNNER_TEST_MARKER"] = str(marker)

        return subprocess.run(
            [
                sys.executable,
                str(runner),
                "--gguf",
                str(gguf_path),
                "--out",
                str(out_dir),
                "--q-bits",
                "4",
            ],
            cwd=site_packages.parent,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
