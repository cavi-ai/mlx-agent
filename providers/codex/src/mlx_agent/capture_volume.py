"""Private, capacity-limited capture volumes with checked mount ownership."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import plistlib
import shutil
import stat
import subprocess
import sys
import time
from uuid import UUID

from .task_contracts import TaskError, encoded

HDIUTIL = "/usr/bin/hdiutil"
DISKUTIL = "/usr/sbin/diskutil"
LISTING_DELAYS = (0, 0.05, 0.1, 0.2, 0.4, 0.8)


def _run(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, timeout=30, umask=0o077)
        if result.returncode or len(result.stdout) > 65536 or len(result.stderr) > 65536:
            raise OSError("Volume tool failed")
        return plistlib.loads(result.stdout) if result.stdout else {}
    except (OSError, ValueError, subprocess.SubprocessError, plistlib.InvalidFileException) as error:
        raise TaskError("capture_volume_unavailable", "The private capture volume operation failed.",
                        "Preserve owned recovery data; capture requires an unprivileged private volume.") from error


def _identity(path):
    info = path.lstat()
    return [info.st_dev, info.st_ino]


def _private(path, directory=True):
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077
            or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))):
        raise TaskError("unsafe_capture_volume", "Capture storage must remain private and owned.")


def _root(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise TaskError("unsafe_capture_volume", "Capture storage cannot traverse links.")
    _private(path)
    return path


def _safe_image(image):
    _private(image)
    for item in image.rglob("*"):
        _private(item, directory=item.is_dir())


def _write_marker(root, record):
    payload = encoded(record)
    if len(payload) > 4096:
        raise TaskError("unsafe_capture_volume", "Capture ownership metadata exceeds its limit.")
    temporary = root / "volume.new"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, root / "volume.json")
        directory = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _load(root):
    root = _root(root)
    marker = root / "volume.json"
    _private(marker, directory=False)
    if marker.stat().st_size > 4096:
        raise TaskError("unsafe_capture_volume", "Capture ownership metadata is invalid.")
    try:
        record = json.loads(marker.read_bytes())
        if (set(record) != {"schema", "task_id", "capacity_bytes", "root_identity", "image_identity", "volume_uuid"}
                or record["schema"] != "capture-volume/1" or str(UUID(record["task_id"])) != root.name
                or record["root_identity"] != _identity(root)
                or record["image_identity"] != _identity(root / "capture.sparsebundle")
                or type(record["capacity_bytes"]) is not int or record["capacity_bytes"] <= 0
                or (record["volume_uuid"] is not None and str(UUID(record["volume_uuid"])).upper() != record["volume_uuid"].upper())):
            raise ValueError()
        _safe_image(root / "capture.sparsebundle")
        return record
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise TaskError("unsafe_capture_volume", "The owned capture volume identity changed.") from error


@contextmanager
def _locked(root):
    root = _root(root)
    descriptor = os.open(root / "volume.lock", os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise TaskError("unsafe_capture_volume", "Capture lock ownership changed.")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise TaskError("capture_volume_busy", "The owned capture volume is in use.") from error
        yield root
    finally:
        os.close(descriptor)


def _listing(root):
    """This root's image in one hdiutil listing: None, its one mounted entity, or False when inconsistent."""
    image, mounted = root / "capture.sparsebundle", root / "mounted"
    matches = [entry for entry in _run([HDIUTIL, "info", "-plist"]).get("images", [])
               if entry.get("image-path") == str(image)]
    if not matches:
        return None
    entities = [entry for image in matches for entry in image.get("system-entities", []) if entry.get("mount-point")]
    if len(matches) != 1 or len(entities) != 1 or entities[0]["mount-point"] != str(mounted):
        return False
    return entities[0]


def _attached(root, expect_mount=False):
    # The hdiutil listing can lag an attach or half-list an image while other images attach or detach,
    # so an inconsistent listing, or an empty one when a mount is expected, is read again before it is trusted.
    for delay in LISTING_DELAYS:
        time.sleep(delay)
        entity = _listing(root)
        if entity or (entity is None and not expect_mount):
            break
    if entity is None:
        return None
    if entity is False:
        raise TaskError("unsafe_capture_volume", "The capture image has an unexpected mount.")
    mounted = root / "mounted"
    device = entity.get("dev-entry", "")
    if not device.startswith("/dev/disk") or "/" in device[len("/dev/"):]:
        raise TaskError("unsafe_capture_volume", "The capture device identity is invalid.")
    info = _run([DISKUTIL, "info", "-plist", device])
    if info.get("MountPoint") != str(mounted) or not info.get("VolumeUUID"):
        raise TaskError("unsafe_capture_volume", "The mounted capture identity is unavailable.")
    return {"device": device, "volume_uuid": info["VolumeUUID"], "readonly": info.get("WritableVolume") is False}


def _detach(root, expected):
    observed = _attached(root, expect_mount=True)
    if observed != expected:
        raise TaskError("unsafe_capture_volume", "The capture mount changed before detach.")
    try:
        _run([HDIUTIL, "detach", observed["device"], "-quiet"])
    except TaskError as error:
        raise TaskError("capture_volume_unresolved", "The owned capture mount could not be detached.",
                        "Keep its ownership metadata and retry recovery; do not force-detach another volume.") from error


@contextmanager
def _mounted(root, record, readonly):
    existing = _attached(root)
    if existing:
        if record["volume_uuid"] and existing["volume_uuid"].upper() != record["volume_uuid"].upper():
            raise TaskError("unsafe_capture_volume", "The retained capture mount UUID changed.")
        _detach(root, existing)
    mount = root / "mounted"
    mount.mkdir(mode=0o700, exist_ok=True)
    _private(mount)
    command = [HDIUTIL, "attach", str(root / "capture.sparsebundle"), "-mountpoint", str(mount),
               "-nobrowse", "-noautoopen", "-owners", "on", "-plist"]
    if readonly:
        command.append("-readonly")
    try:
        _run(command)
    except TaskError:
        # A tool may fail after committing the mount. Only detach the image
        # whose backing path and mountpoint this operation already owns.
        attached = _attached(root)
        if attached is not None:
            _detach(root, attached)
        raise
    observed = _attached(root, expect_mount=True)
    if observed is None:
        raise TaskError("capture_volume_unavailable", "The capture image did not mount at its owned path.")
    try:
        if record["volume_uuid"] and observed["volume_uuid"].upper() != record["volume_uuid"].upper():
            raise TaskError("unsafe_capture_volume", "The capture filesystem UUID changed.")
        if readonly and observed["readonly"] is not True:
            raise TaskError("unsafe_capture_volume", "A read-only capture mount is required.")
        info = mount.stat()
        if info.st_uid != os.getuid():
            raise TaskError("capture_volume_unavailable", "The capture filesystem is not owned by this user.")
        if not readonly:
            mount.chmod(0o700)
        _private(mount)
        capacity = os.statvfs(mount)
        if readonly and not capacity.f_flag & os.ST_RDONLY:
            raise TaskError("unsafe_capture_volume", "The kernel did not mount capture data read-only.")
        if capacity.f_blocks * capacity.f_frsize > record["capacity_bytes"]:
            raise TaskError("unsafe_capture_volume", "The filesystem exceeds the accepted capture capacity.")
        yield mount, observed
    finally:
        _detach(root, observed)


class CaptureVolume:
    def __init__(self, root):
        self.root = root

    @classmethod
    def create(cls, root: Path, task_id: str, capacity_bytes: int):
        if sys.platform != "darwin":
            raise TaskError("capture_volume_unavailable", "Owned capture volumes require macOS.")
        root = Path(root)
        try:
            valid_id = str(UUID(task_id)) == task_id == root.name
        except (ValueError, TypeError, AttributeError):
            valid_id = False
        if (not root.is_absolute() or ".." in root.parts or not valid_id or type(capacity_bytes) is not int
                or not 0 < capacity_bytes <= 2 ** 63 - 1 or capacity_bytes % 512):
            raise TaskError("unsafe_capture_volume", "Capture identity or capacity is invalid.")
        if any(item.is_symlink() for item in (root, *root.parents)):
            raise TaskError("unsafe_capture_volume", "Capture storage cannot traverse links.")
        root.mkdir(mode=0o700)
        _root(root)
        created_identity = _identity(root)
        image = root / "capture.sparsebundle"
        try:
            # hdiutil's create -size suffix 'b' means sectors, not bytes.
            _run([HDIUTIL, "create", "-sectors", str(capacity_bytes // 512), "-fs", "HFS+", "-layout", "NONE",
                  "-type", "SPARSEBUNDLE", "-volname", "MLX Capture", "-nospotlight", "-quiet", str(image)])
            _safe_image(image)
            record = {"schema": "capture-volume/1", "task_id": task_id, "capacity_bytes": capacity_bytes,
                      "root_identity": created_identity, "image_identity": _identity(image), "volume_uuid": None}
            _write_marker(root, record)
        except BaseException:
            if not (root / "volume.json").exists():
                try:
                    _root(root)
                    if _identity(root) != created_identity:
                        raise OSError("Created storage was replaced")
                    for item in root.rglob("*"):
                        _private(item, directory=item.is_dir())
                        if item.lstat().st_dev != created_identity[0]:
                            raise OSError("Partial storage crosses a filesystem boundary")
                    if _identity(root) != created_identity:
                        raise OSError("Created storage was replaced")
                    shutil.rmtree(root)
                except (OSError, TaskError) as error:
                    raise TaskError("capture_volume_unresolved", "Partial capture storage could not be removed.",
                                    "Inspect {0} for recovery; do not delete a replaced path.".format(root)) from error
            raise
        with _locked(root):
            with _mounted(root, record, readonly=False) as (_, observed):
                record["volume_uuid"] = observed["volume_uuid"]
                _write_marker(root, record)
        return cls(root)

    @staticmethod
    @contextmanager
    def open_owned(root: Path, readonly: bool):
        with _locked(root) as root:
            record = _load(root)
            if not record["volume_uuid"]:
                raise TaskError("capture_volume_unavailable", "Capture creation needs recovery.")
            with _mounted(root, record, readonly) as (mounted, _):
                yield mounted

    @staticmethod
    def remove(root: Path):
        with _locked(root) as root:
            record = _load(root)
            with _mounted(root, record, readonly=bool(record["volume_uuid"])):
                pass  # Verify the UUID before deleting any retained file.
            _load(root)
            for item in root.rglob("*"):
                if item.is_symlink():
                    raise TaskError("unsafe_capture_volume", "Capture removal cannot traverse a link.")
            shutil.rmtree(root)
