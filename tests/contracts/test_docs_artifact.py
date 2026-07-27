import json
import re
import unittest
from pathlib import Path

from mlx_agent import __version__


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs" / "mlx-agent" / "source"
ARTIFACT = ROOT / "docs" / "mlx-agent" / "v{0}".format(__version__)
_RELATIVE_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


class DocsArtifactTests(unittest.TestCase):
    def test_source_navigation_matches_version_and_files(self):
        navigation = json.loads((SOURCE / "navigation.json").read_text(encoding="utf-8"))
        self.assertEqual(navigation["version"], __version__)
        self.assertEqual(navigation["title"], "mlx-agent")
        pages = [
            page["path"]
            for section in navigation["sections"]
            for page in section["pages"]
        ]
        self.assertEqual(len(pages), len(set(pages)))
        for relative in pages:
            self.assertTrue((SOURCE / "pages" / relative).is_file(), relative)

    def test_artifact_is_current(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_docs.py"), "--check"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_artifact_manifest_shape(self):
        manifest = json.loads((ARTIFACT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["package"], "mlx-agent")
        self.assertEqual(manifest["product"], "mlx-agent")
        self.assertEqual(manifest["version"], __version__)
        self.assertRegex(manifest["contentSha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(manifest["publicBasePath"], "/docs/mlx-agent/v{0}".format(__version__))
        self.assertEqual(manifest["stableAlias"], "/docs/mlx-agent")

    def test_pages_have_titles_and_no_broken_relative_links(self):
        for page in SOURCE.rglob("*.md"):
            content = page.read_text(encoding="utf-8")
            self.assertTrue(content.startswith("# "), str(page))
            for target in _RELATIVE_LINK.findall(content):
                if "://" in target or target.startswith("#") or target.startswith("mailto:"):
                    continue
                resolved = (page.parent / target.split("#", 1)[0]).resolve()
                self.assertTrue(resolved.is_file(), "{0} -> {1}".format(page, target))

    def test_no_absolute_machine_paths_in_pages(self):
        for page in SOURCE.rglob("*.md"):
            content = page.read_text(encoding="utf-8")
            self.assertNotIn("/Users/", content, str(page))
            self.assertNotIn("/Volumes/", content, str(page))


if __name__ == "__main__":
    unittest.main()
