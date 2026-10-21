import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.task_contracts import MEDIA_REQUEST_SCHEMA, MEDIA_SUFFIXES, REQUEST_SCHEMA, TaskError, capabilities
from mlx_agent.task_kinds import KINDS, find, get


class TaskKindTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_each_kind_declares_its_media_input_and_output(self):
        self.assertEqual(list(KINDS), ["classification", "music_generation", "transcribe", "speak", "image_generation",
                                       "image_understanding", "video_generation", "video_understanding"])
        inputs = {name: (kind.input_field, kind.input_media) for name, kind in KINDS.items() if kind.input_media}
        self.assertEqual(inputs, {"transcribe": ("audio_path", "audio"), "image_understanding": ("image_path", "image"),
                                  "video_understanding": ("video_path", "video")})
        outputs = {name: kind.output for name, kind in KINDS.items() if kind.output}
        self.assertEqual(outputs, {"music_generation": "output.wav", "speak": "output.wav",
                                   "image_generation": "output.png", "video_generation": "output.mp4"})
        for kind in KINDS.values():
            with self.subTest(kind=kind.name):
                self.assertIn(kind.input_media, (None, *MEDIA_SUFFIXES))
                self.assertEqual(kind.output is None, kind.output_check is None)
                self.assertEqual(kind.request_schema, REQUEST_SCHEMA if kind.legacy else MEDIA_REQUEST_SCHEMA)

    def test_capabilities_list_every_kind_in_table_order(self):
        tasks = capabilities()["tasks"]
        self.assertEqual([task["kind"] for task in tasks], list(KINDS))
        self.assertEqual({task["kind"]: task["result_schema"] for task in tasks},
                         {name: kind.result_schema for name, kind in KINDS.items()})
        self.assertEqual([task["kind"] for task in tasks if "request_schema" not in task], ["classification", "music_generation"])

    def test_unknown_or_unhashable_kind_names_are_unsupported(self):
        for name in ("missing", ["speak"], None):
            with self.subTest(name=name):
                self.assertIsNone(find(name))
                with self.assertRaises(TaskError) as caught:
                    get(name)
                self.assertEqual(caught.exception.code, "unsupported_task")

    def test_music_and_speech_reject_truncated_wav_output(self):
        path = self.root / "output.wav"
        with wave.open(str(path), "wb") as output:
            output.setparams((1, 2, 8000, 2000, "NONE", "not compressed"))
            output.writeframes(b"\x00\x01" * 2000)
        for name in ("music_generation", "speak"):
            KINDS[name].check_output(path, {})
        path.write_bytes(path.read_bytes()[:46])
        for name, activity in (("music_generation", "Music generation"), ("speak", "Speech synthesis")):
            with self.subTest(kind=name), self.assertRaises(TaskError) as caught:
                KINDS[name].check_output(path, {})
            self.assertEqual(caught.exception.code, "invalid_audio_output")
            self.assertEqual(str(caught.exception), activity + " produced a truncated WAV payload.")


if __name__ == "__main__":
    unittest.main()
