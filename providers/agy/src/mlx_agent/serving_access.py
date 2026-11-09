"""Private file credentials for explicitly protected serving endpoints."""
import hashlib
import hmac
import os
import re
import stat


def read_access_key(path, expected_digest=None):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as source:
        metadata = os.fstat(source.fileno())
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077 or not 32 <= metadata.st_size <= 512):
            raise ValueError('The serving key requires an owned private regular file.')
        raw = source.read(513)
    try:
        key = raw.decode('ascii').strip()
    except UnicodeError:
        raise ValueError('Invalid serving key file.') from None
    if len(raw) > 512 or re.fullmatch(r'[A-Za-z0-9._~-]{32,256}', key) is None:
        raise ValueError('The serving key must contain 32–256 URL-safe ASCII characters.')
    digest = hashlib.sha256(key.encode()).hexdigest()
    if expected_digest is not None and not hmac.compare_digest(digest, expected_digest):
        raise ValueError('The serving key changed after review.')
    return key, digest
