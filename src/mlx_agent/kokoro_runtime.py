"""Keep packaged Kokoro phonemizer data below eSpeak's path-length limit."""

from contextlib import contextmanager
import os
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def kokoro_data_directory():
    import espeakng_loader

    data = Path(espeakng_loader.get_data_path())
    previous = os.environ.get("ESPEAK_DATA_PATH")
    # eSpeak's native path buffer is 160 bytes. A private directory under the
    # system's short temporary root avoids resolving a long venv path in C.
    with TemporaryDirectory(prefix="mlx-espeak-", dir="/tmp") as directory:
        (Path(directory) / "espeak-ng-data").symlink_to(data, target_is_directory=True)
        os.environ["ESPEAK_DATA_PATH"] = directory
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("ESPEAK_DATA_PATH", None)
            else:
                os.environ["ESPEAK_DATA_PATH"] = previous


def check_english():
    with kokoro_data_directory():
        import en_core_web_sm  # noqa: F401 -- require the installed pipeline before Misaki can download it
        from misaki import en, espeak

        g2p = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False))
        phonemes, _ = g2p("A local speech test.")
        if not phonemes:
            raise RuntimeError("The English phonemizer produced no phonemes")


if __name__ == "__main__":
    check_english()
