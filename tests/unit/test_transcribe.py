import io
import json
import subprocess
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from mlx_agent import transcribe_runner
from mlx_agent.backends import INSTALL_MARKER
from mlx_agent.transcribe import TRANSCRIBE_RUNNER, TranscribeError, plan_transcribe, run_transcribe

from .backend_fixtures import synthetic_manifests, synthetic_registries


class Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


class TranscribeTests(unittest.TestCase):
    def setUp(self):
        from tests.unit.ownership_fixtures import isolate
        isolate(self)
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifests = synthetic_manifests()
        self.registries = synthetic_registries(self.manifests, installed=("mlx-audio",))
        venv = self.root / "backends" / "mlx-audio"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("", encoding="utf-8")
        (venv / INSTALL_MARKER).write_text(json.dumps({"id": "mlx-audio", "version": "0.0-test"}), encoding="utf-8")
        self.audio = self.root / "clip.wav"
        self.audio.write_bytes(b"RIFF")

    def model(self, model_type):
        directory = self.root / model_type
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
        return directory

    def plan(self, path, **kwargs):
        return plan_transcribe(path, kwargs.pop("audio", self.audio), manifests=self.manifests,
                               registries=self.registries, root=self.root / "backends", **kwargs)

    def test_plan_runs_the_runner_with_the_speech_backend_python(self):
        plan = self.plan(self.model("whisper"), language="en")
        self.assertEqual(plan["backend"], "mlx-audio")
        self.assertEqual(plan["argv"][:2], [str(self.root / "backends" / "mlx-audio" / "bin" / "python"), str(TRANSCRIBE_RUNNER)])
        self.assertEqual(plan["argv"][-2:], ["--language", "en"])
        self.assertNotIn("--trust-remote-code", plan["argv"])

    def test_plan_refusals(self):
        cases = [
            (lambda: self.plan(self.model("qwen2")), "not_speech_model"),
            (lambda: self.plan(self.root / "missing"), "model_unreadable"),
            (lambda: self.plan(self.model("voxtral"), audio=self.root / "none.wav"), "audio_not_found"),
            (lambda: self.plan(self.model("glmasr"), language="EN; rm"), "invalid_arguments"),
        ]
        for call, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(TranscribeError) as caught:
                    call()
                self.assertEqual(caught.exception.code, code)
        registries = synthetic_registries(self.manifests)
        with self.assertRaises(TranscribeError) as caught:
            plan_transcribe(self.model("voxtral_realtime"), self.audio, manifests=self.manifests,
                            registries=registries, root=self.root / "elsewhere")
        self.assertEqual(caught.exception.code, "backend_not_installed")

    def test_run_reads_the_last_json_line(self):
        plan = self.plan(self.model("whisper"))
        seen = []

        def runner(argv, **kwargs):
            seen.append((argv, kwargs))
            return Completed(stdout='loading...\n{"text": "hello there", "seconds": 1.5, "audio_seconds": 2.0}\n')

        result = run_transcribe(plan, manifests=self.manifests, root=self.root / "backends", runner=runner)
        self.assertEqual((result["schema"], result["text"], result["seconds"]), ("transcribe/1", "hello there", 1.5))
        self.assertEqual(seen[0][0], plan["argv"])
        self.assertEqual(seen[0][1]["stdin"], subprocess.DEVNULL)
        self.assertNotIn("HF_TOKEN", seen[0][1]["env"])

    def test_run_failures_are_classified(self):
        plan = self.plan(self.model("whisper"))
        failing = lambda argv, **kwargs: Completed(returncode=1, stderr="Traceback\nValueError: bad weights\n")
        with self.assertRaises(TranscribeError) as caught:
            run_transcribe(plan, manifests=self.manifests, root=self.root / "backends", runner=failing)
        self.assertEqual(caught.exception.code, "transcribe_failed")
        self.assertIn("bad weights", str(caught.exception))

        def slow(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

        with self.assertRaises(TranscribeError) as caught:
            run_transcribe(plan, manifests=self.manifests, root=self.root / "backends", runner=slow, timeout=5)
        self.assertEqual(caught.exception.code, "transcribe_timeout")

    def test_words_are_requested_and_returned_with_their_times(self):
        plan = self.plan(self.model("whisper"), words=True)
        self.assertEqual(plan["argv"][-1], "--words")
        self.assertNotIn("--words", self.plan(self.root / "whisper")["argv"])
        words = [{"word": "hello", "start": 0.12, "end": 0.48}, {"word": "there", "start": 0.48, "end": 0.9}]
        timed = lambda argv, **kwargs: Completed(stdout=json.dumps({"text": "hello there", "words": words}) + "\n")
        result = run_transcribe(plan, manifests=self.manifests, root=self.root / "backends", runner=timed)
        self.assertEqual(result["words"], words)
        untimed = lambda argv, **kwargs: Completed(stdout='{"text": "hello there"}\n')
        with self.assertRaises(TranscribeError) as caught:
            run_transcribe(plan, manifests=self.manifests, root=self.root / "backends", runner=untimed)
        self.assertEqual(caught.exception.code, "transcribe_failed")
        refusing = lambda argv, **kwargs: Completed(returncode=2, stdout='{"error": "words_unsupported", "detail": "Parakeet"}\n')
        with self.assertRaises(TranscribeError) as caught:
            run_transcribe(plan, manifests=self.manifests, root=self.root / "backends", runner=refusing)
        self.assertEqual(caught.exception.code, "words_unsupported")


PRE_V3_VOCABULARY = {"<|endoftext|>": 50257, "<|nocaptions|>": 50362}
V3_VOCABULARY = {"<|endoftext|>": 50257, "<|nospeech|>": 50363}


class FakeHFTokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def get_vocab(self):
        return dict(self.vocabulary)

    def convert_tokens_to_ids(self, token):
        return self.vocabulary.get(token, self.vocabulary["<|endoftext|>"])


class FakeWhisperTokenizer:
    """mlx-audio's wrapper: `no_speech` looks up `<|nospeech|>`, falling back to the unknown token, `<|endoftext|>`."""

    def __init__(self, hf_tokenizer):
        self.hf_tokenizer = hf_tokenizer

    @property
    def eot(self):
        return self.hf_tokenizer.convert_tokens_to_ids("<|endoftext|>")

    @property
    def no_speech(self):
        return self.hf_tokenizer.convert_tokens_to_ids("<|nospeech|>")


class FakeWhisper:
    """Decodes like mlx-audio: the no-speech token is suppressed, so a suppressed end token never stops the text."""

    def __init__(self, vocabulary):
        self.hf_tokenizer = FakeHFTokenizer(vocabulary)
        self.no_speech_seen = None

    def get_tokenizer(self, language=None, task="transcribe"):
        return FakeWhisperTokenizer(self.hf_tokenizer)

    def generate(self, audio, language=None):
        tokenizer = self.get_tokenizer(language=language, task="transcribe")
        self.no_speech_seen = tokenizer.no_speech
        tail = " . . . . . ." if tokenizer.no_speech == tokenizer.eot else ""
        return types.SimpleNamespace(text=" The quick brown fox jumps over the lazy dog." + tail)


class FakeParakeet:
    def generate(self, audio):
        return types.SimpleNamespace(text="The quick brown fox jumps over the lazy dog.")


class FakeTimedWhisper:
    """Returns mlx-audio Whisper segments, whose words carry padded text, times and probabilities."""

    def __init__(self):
        self.word_timestamps = None

    def generate(self, audio, language=None, word_timestamps=False):
        self.word_timestamps = word_timestamps
        words = [{"word": " Hello", "start": 0.1234, "end": 0.5, "probability": 0.9},
                 {"word": " there", "start": 0.5, "end": 0.8126, "probability": 0.8}]
        return types.SimpleNamespace(text=" Hello there", segments=[{"text": " Hello there", "words": words}])


def sung(text, start, end):
    return {"text": text, "start": start, "end": end, "words": [{"word": text, "start": start, "end": end}]}


class FakeSungWhisper:
    """Ends its text after a sung ad-lib, as Whisper does on music, unless decoding starts past it."""

    def __init__(self, texts=(" Yeah.", " Wake the laptop,", " lights down low")):
        self.texts = texts
        self.clips = []

    def generate(self, audio, language=None, word_timestamps=False, clip_timestamps="0"):
        self.clips.append(clip_timestamps)
        start = float(clip_timestamps) if isinstance(clip_timestamps, str) else clip_timestamps[0]
        first, second, third = self.texts
        if start < 2.36:
            segments = [sung(first, 0.0, 2.36)]
        elif start < 9.8:
            segments = [sung(second, 5.0, 7.3), sung(third, 7.6, 9.8)]
        else:
            segments = []
        return types.SimpleNamespace(text="".join(segment["text"] for segment in segments), segments=segments)


class TranscribeRunnerTests(unittest.TestCase):
    def transcribe(self, model, *extra, expected=0, samples=45000, language="en"):
        utils = types.ModuleType("mlx_audio.stt.utils")
        utils.load_model = lambda path: model
        utils.load_audio = lambda path: types.SimpleNamespace(shape=(samples,))
        modules = {"mlx_audio": types.ModuleType("mlx_audio"), "mlx_audio.stt": types.ModuleType("mlx_audio.stt"),
                   "mlx_audio.stt.utils": utils}
        buffer = io.StringIO()
        with mock.patch.dict(sys.modules, modules), redirect_stdout(buffer):
            code = transcribe_runner.main(["--model", "/m", "--audio", "/clip.wav", "--language", language, *extra])
        self.assertEqual(code, expected)
        return json.loads(buffer.getvalue())

    def test_words_carry_their_start_and_end_seconds(self):
        model = FakeTimedWhisper()
        result = self.transcribe(model, "--words")
        self.assertIs(model.word_timestamps, True)
        self.assertEqual(result["words"], [{"word": "Hello", "start": 0.123, "end": 0.5},
                                           {"word": "there", "start": 0.5, "end": 0.813}])
        self.assertNotIn("words", self.transcribe(FakeTimedWhisper()))

    def test_a_text_that_ends_early_resumes_from_its_last_segment_to_the_end_of_the_clip(self):
        model = FakeSungWhisper()
        result = self.transcribe(model, "--words", samples=20 * 16000)
        self.assertEqual(model.clips, ["0", [2.36], [9.8]])
        self.assertEqual(result["text"], "Yeah. Wake the laptop, lights down low")
        self.assertEqual([(word["word"], word["end"]) for word in result["words"]],
                         [("Yeah.", 2.36), ("Wake the laptop,", 7.3), ("lights down low", 9.8)])

    def test_resumed_text_preserves_japanese_segment_spacing(self):
        model = FakeSungWhisper(("こんにちは", "、世界", "！"))
        result = self.transcribe(model, samples=20 * 16000, language="ja")
        self.assertEqual(result["text"], "こんにちは、世界！")

    def test_a_model_without_word_times_refuses_words(self):
        self.assertEqual(self.transcribe(FakeParakeet(), "--words", expected=2),
                         {"error": "words_unsupported", "detail": "FakeParakeet"})

    def test_a_pre_v3_whisper_vocabulary_suppresses_no_captions_instead_of_the_end_token(self):
        model = FakeWhisper(PRE_V3_VOCABULARY)
        result = self.transcribe(model)
        self.assertEqual(result["text"], "The quick brown fox jumps over the lazy dog.")
        self.assertEqual(model.no_speech_seen, 50362)
        self.assertEqual(result["audio_seconds"], 2.812)

    def test_a_vocabulary_with_no_speech_and_a_model_without_a_tokenizer_are_left_alone(self):
        model = FakeWhisper(V3_VOCABULARY)
        self.assertEqual(self.transcribe(model)["text"], "The quick brown fox jumps over the lazy dog.")
        self.assertEqual(model.no_speech_seen, 50363)
        self.assertNotIn("get_tokenizer", vars(model))
        self.assertEqual(self.transcribe(FakeParakeet())["text"], "The quick brown fox jumps over the lazy dog.")


if __name__ == "__main__":
    unittest.main()
