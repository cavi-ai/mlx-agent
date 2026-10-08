"""Run inside a speech backend's environment: load one converted model, transcribe one clip, print JSON."""

import argparse
import inspect
import json
import sys
import time


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
    arguments = parser.parse_args(argv)

    from mlx_audio.stt.utils import load_audio, load_model

    started = time.time()
    model = load_model(arguments.model)
    _restore_whisper_end_token(model)
    samples = load_audio(arguments.audio)
    kwargs = {}
    if arguments.language and "language" in inspect.signature(model.generate).parameters:
        kwargs["language"] = arguments.language
    output = model.generate(arguments.audio, **kwargs)
    text = getattr(output, "text", output if isinstance(output, str) else "")
    print(json.dumps({
        "text": str(text).strip(),
        "seconds": round(time.time() - started, 3),
        "audio_seconds": round(int(samples.shape[0]) / 16000, 3),
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
