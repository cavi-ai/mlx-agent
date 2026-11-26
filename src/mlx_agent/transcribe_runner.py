"""Run inside a speech backend's environment: load one converted model, transcribe one clip, print JSON."""

import argparse
import inspect
import json
import sys
import time

if __package__:
    from .runner_model_cache import load_model as retained_load
else:
    from runner_model_cache import load_model as retained_load

# Audio left after the last heard segment that is decoded again.
RESUME_SECONDS = 1.0


def _restore_whisper_end_token(model):
    """Give a Whisper tokenizer the no-speech token its vocabulary has.

    mlx-audio looks up `<|nospeech|>`, which only large-v3 vocabularies carry; earlier ones call it
    `<|nocaptions|>`. The failed lookup returns the unknown token, `<|endoftext|>`, which decoding then
    suppresses as the no-speech token, so every transcript runs to the token limit.
    """
    get_tokenizer = getattr(model, "get_tokenizer", None)
    if not callable(get_tokenizer):
        return
    try:
        probe = get_tokenizer()
        vocabulary = probe.hf_tokenizer.get_vocab()
        if probe.no_speech != probe.eot:
            return
    except Exception:
        return
    no_speech = vocabulary.get("<|nospeech|>", vocabulary.get("<|nocaptions|>"))
    tokenizer_type = type(probe)
    restored = type(tokenizer_type.__name__, (tokenizer_type,), {"no_speech": property(lambda self: no_speech)})

    def restored_tokenizer(*args, **kwargs):
        tokenizer = get_tokenizer(*args, **kwargs)
        tokenizer.__class__ = restored
        return tokenizer

    model.get_tokenizer = restored_tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--language", default=None)
    parser.add_argument("--words", action="store_true")
    arguments = parser.parse_args(argv)

    from mlx_audio.stt.utils import load_audio, load_model

    started = time.time()
    model = retained_load(load_model, arguments.model)
    _restore_whisper_end_token(model)
    samples = load_audio(arguments.audio)
    kwargs = {}
    parameters = inspect.signature(model.generate).parameters
    if arguments.language and "language" in parameters:
        kwargs["language"] = arguments.language
    if arguments.words:
        if "word_timestamps" not in parameters:
            print(json.dumps({"error": "words_unsupported", "detail": type(model).__name__}), flush=True)
            return 2
        kwargs["word_timestamps"] = True
    output = model.generate(arguments.audio, **kwargs)
    text = str(getattr(output, "text", output if isinstance(output, str) else ""))
    segments = list(getattr(output, "segments", None) or [])
    audio_seconds = int(samples.shape[0]) / 16000
    # Whisper can end a clip's text early on sung audio and skip the rest; decode again from the last segment.
    while "clip_timestamps" in parameters and segments and audio_seconds - float(segments[-1]["end"]) >= RESUME_SECONDS:
        heard_until = float(segments[-1]["end"])
        later = list(getattr(model.generate(arguments.audio, **kwargs, clip_timestamps=[heard_until]), "segments", None) or [])
        if not later or float(later[-1]["end"]) <= heard_until:
            break
        segments += later
        # Whisper's decoded segments carry their own spacing.
        text += "".join(str(segment["text"]) for segment in later)
    result = {
        "text": text.strip(),
        "seconds": round(time.time() - started, 3),
        "audio_seconds": round(audio_seconds, 3),
    }
    if arguments.words:
        result["words"] = [
            {"word": str(word["word"]).strip(), "start": round(float(word["start"]), 3), "end": round(float(word["end"]), 3)}
            for segment in segments for word in segment.get("words") or []
        ]
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
