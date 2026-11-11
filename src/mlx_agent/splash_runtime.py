"""Optional Splash serving for local GGUF and MLX targets, including prepared Strata profiles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

MIN_VERSION = (1, 4, 0)
_VERSION = re.compile(r"Splash (\d+)\.(\d+)\.(\d+)")
_SPLIT = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf")
# Splash's --model rule: an OWNER/REPO repository ID without "--" or "..".
_SERVED_NAME = re.compile(r"[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?")
_LAUNCHER_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "~/.local/bin")
_INSTALL = ("Install Splash 1.4.0 or later on an Apple M3 or newer with macOS 26.4 or later "
            "(brew install incoai/tap/splash); serve never installs runtimes.")


class SplashError(RuntimeError):
    def __init__(self, code, message, remediation=_INSTALL):
        super().__init__(message)
        self.code = code
        self.remediation = remediation


def find_launcher():
    """The `splash` launcher: MLX_AGENT_SPLASH, PATH, then the Homebrew and ~/.local/bin locations."""
    override = os.environ.get("MLX_AGENT_SPLASH")
    candidates = [override] if override else [shutil.which("splash")] + [
        str(Path(directory).expanduser() / "splash") for directory in _LAUNCHER_DIRS]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return Path(candidate).absolute()
    raise SplashError("splash_not_installed", "No executable Splash launcher was found.")


def installed_runtime(launcher=None, run=subprocess.run):
    """Version and identity of a released Splash launcher; a source checkout reports no version."""
    path = Path(launcher) if launcher is not None else find_launcher()
    try:
        result = run([str(path), "--version"], capture_output=True, text=True, timeout=30,
                     stdin=subprocess.DEVNULL, env={**os.environ, "HF_HUB_OFFLINE": "1"})
        resolved = path.resolve()
        content = resolved.read_bytes()
    except (OSError, subprocess.SubprocessError) as error:
        raise SplashError("splash_not_installed", "The Splash launcher could not run: " + str(error)) from error
    match = _VERSION.fullmatch((result.stdout or "").strip())
    if result.returncode != 0 or match is None:
        raise SplashError("splash_not_installed", "The Splash launcher did not report a release version.")
    version = tuple(int(part) for part in match.groups())
    text = ".".join(str(part) for part in version)
    if version < MIN_VERSION:
        raise SplashError("splash_version_unsupported",
                          "Splash {0} predates 1.4.0, the first release serving split GGUFs.".format(text),
                          "Upgrade Splash (brew upgrade incoai/tap/splash).")
    digest = hashlib.sha256()
    for part in (str(path), str(resolved), text):
        digest.update(part.encode() + b"\0")
    digest.update(content)
    return {"launcher": str(path), "resolved": str(resolved), "version": text, "fingerprint": digest.hexdigest()}


def _identity(paths):
    from .strata_runtime import file_identity
    return {str(path): file_identity(path) for path in paths}


def _fingerprint(identities):
    return hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()


def _gguf_parts(path):
    if path.suffix != ".gguf" or not path.is_file():
        raise ValueError("A Splash GGUF target is a file ending in .gguf: " + str(path))
    match = _SPLIT.fullmatch(path.name)
    if match is None:
        return [path]
    stem, _, count = match.groups()
    parts = [path.with_name("{0}-{1:05d}-of-{2}.gguf".format(stem, index, count)) for index in range(1, int(count) + 1)]
    missing = [part.name for part in parts if not part.is_file()]
    if not parts or missing:
        raise FileNotFoundError("Missing split GGUF part: " + (missing[0] if missing else path.name))
    return parts


def _served_model(name):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    cleaned = re.sub(r"-{2,}", "-", re.sub(r"\.{2,}", ".", cleaned))[:96].strip("-.") or "model"
    if cleaned.endswith(".git") or not _SERVED_NAME.fullmatch(cleaned):
        cleaned = "model"
    return "local/" + cleaned


def read_target(path):
    """Resolve a local target without loading weights: a GGUF file or part, a GGUF or MLX directory, or a Strata profile."""
    root = Path(path).expanduser().absolute()
    try:
        source = None
        if root.is_dir() and (root / "strata-model.json").is_file():
            from .gguf import GGUFError
            from .strata_runtime import StrataError, read_model
            try:
                target = Path(read_model(root)["gguf"])
            except (StrataError, GGUFError) as error:
                code = error.code if isinstance(error, StrataError) else "splash_model_invalid"
                raise SplashError(code, str(error), "Select a valid prepared Strata profile or its GGUF file.") from error
            files, weight_format, source, name = _gguf_parts(target), "gguf", "strata_profile", root.name
        elif root.is_file():
            target, files, weight_format = root, _gguf_parts(root), "gguf"
            match = _SPLIT.fullmatch(root.name)
            name = match.group(1) if match else root.stem
        elif root.is_dir() and (root / "config.json").is_file():
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            if not isinstance(config, dict) or not isinstance(config.get("quantization"), dict):
                raise ValueError("Splash serves quantized MLX checkpoints; config.json declares no quantization.")
            weights = sorted(root.glob("*.safetensors"))
            if not weights:
                raise FileNotFoundError("The MLX directory holds no .safetensors weights.")
            target, files, weight_format, name = root, [root / "config.json", *weights], "mlx", root.name
        elif root.is_dir():
            ggufs = sorted(item for item in root.glob("*.gguf") if "mmproj" not in item.name.lower())
            groups = {(_SPLIT.fullmatch(item.name).group(1) if _SPLIT.fullmatch(item.name) else item.stem) for item in ggufs}
            if len(groups) != 1:
                raise ValueError("A Splash GGUF directory must hold the GGUF targets of exactly one name; select the file instead.")
            target, files, weight_format, name = root, _gguf_parts(ggufs[0]), "gguf", groups.pop()
        else:
            raise FileNotFoundError("The Splash target is not present: " + str(root))
        if weight_format == "gguf":
            from .gguf import GGUFError, read_gguf_header
            try:
                architecture = read_gguf_header(files[0])["metadata"].get("general.architecture")
            except GGUFError as error:
                raise ValueError("Not a readable GGUF file: " + str(error)) from error
        else:
            architecture = config.get("model_type")
        identities = _identity(files)
        return {"path": str(root), "target": str(target), "source": source, "format": weight_format,
                "architecture": architecture if isinstance(architecture, str) else None,
                "files": [str(item) for item in files], "weight_bytes": sum(item.stat().st_size for item in files),
                "served_model": _served_model(name), "fingerprint": _fingerprint(identities)}
    except SplashError:
        raise
    except FileNotFoundError as error:
        raise SplashError("model_not_local", str(error), "Point --path at local model files; serve never downloads models.") from error
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise SplashError("splash_model_invalid", str(error), "Select a quantized GGUF or MLX target that Splash supports.") from error


def read_draft(path):
    """A local DFlash2 draft directory: config.json with dflash_config, and safetensors weights."""
    root = Path(path).expanduser().absolute()
    remediation = "Pass a local DFlash2 draft directory, such as a download of incoai/Qwen3.8-Flash-Next-DFlash2."
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        weights = sorted(root.glob("*.safetensors"))
        if not isinstance(config, dict) or not isinstance(config.get("dflash_config"), dict) or not weights:
            raise ValueError("The directory is not a DFlash2 draft (config.json dflash_config and safetensors weights).")
        identities = _identity([root / "config.json", *weights])
    except (OSError, ValueError, TypeError) as error:
        raise SplashError("splash_draft_invalid", str(error), remediation) from error
    return {"path": str(root), "fingerprint": _fingerprint(identities)}


def serve_arguments(draft=None, max_memory_gb=None, max_context_tokens=None):
    """Bounded Splash options appended after the recipe; sizes go as whole bytes, never binary-unit text."""
    args = []
    if draft is not None:
        args += ["--draft-model", draft["path"]]
    if max_memory_gb is not None:
        if type(max_memory_gb) not in (int, float) or not math.isfinite(max_memory_gb) or not 1 <= max_memory_gb <= 1024:
            raise SplashError("invalid_arguments", "Splash memory must be a finite value from 1 to 1024 GB.",
                              "Pass --splash-max-memory-gb in decimal GB.")
        args += ["--max-memory", str(math.floor(max_memory_gb * 1e9))]
    if max_context_tokens is not None:
        if type(max_context_tokens) is not int or not 512 <= max_context_tokens <= 1_010_000:
            raise SplashError("invalid_arguments", "Splash context must be an integer from 512 to 1010000 tokens.",
                              "Pass --splash-max-context in tokens.")
        args += ["--max-context", str(max_context_tokens)]
    return args
