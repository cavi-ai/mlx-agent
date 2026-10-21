import sqlite3
import unittest
from contextlib import contextmanager

from mlx_agent import training_evaluation as evaluation


class _Database:
    def __init__(self):
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row

    @contextmanager
    def connect(self):
        yield self.connection


def _suite(**changes):
    value = {"id": "suite-1", "name": "Smoke", "prompts": [{"id": "p1", "text": "Say hi.", "max_tokens": 16}]}
    value.update(changes)
    return value


class EvaluationInputTests(unittest.TestCase):
    def test_suites_are_validated_and_copied(self):
        suite = _suite()
        validated = evaluation.validate_suite(suite)
        validated["prompts"][0]["text"] = "changed"
        self.assertEqual("Say hi.", suite["prompts"][0]["text"])
        prompt = {"id": "p1", "text": "Say hi.", "max_tokens": 16}
        cases = (
            ([], "An evaluation suite needs id, name and prompts only."),
            (_suite(extra=1), "An evaluation suite needs id, name and prompts only."),
            (_suite(id="has space"), "Invalid evaluation suite id."),
            (_suite(name=" "), "Evaluation suite name must contain 1..200 characters without controls."),
            (_suite(name="a" * 201), "Evaluation suite name must contain 1..200 characters without controls."),
            (_suite(name="tab\tname"), "Evaluation suite name must contain 1..200 characters without controls."),
            (_suite(prompts=[]), "An evaluation suite must contain 1..16 prompts."),
            (_suite(prompts=[dict(prompt, id="p{0}".format(n)) for n in range(17)]), "An evaluation suite must contain 1..16 prompts."),
            (_suite(prompts=[dict(prompt, seed=1)]), "Evaluation entries support id, text and max_tokens only."),
            (_suite(prompts=[prompt, prompt]), "Evaluation prompt ids must be valid and unique."),
            (_suite(prompts=[dict(prompt, text="")]), "Evaluation prompt must contain 1..8000 characters."),
            (_suite(prompts=[dict(prompt, text="x" * 8001)]), "Evaluation prompt must contain 1..8000 characters."),
            (_suite(prompts=[dict(prompt, max_tokens=0)]), "Evaluation token limit must be 1..512."),
            (_suite(prompts=[dict(prompt, max_tokens=True)]), "Evaluation token limit must be 1..512."),
        )
        for value, message in cases:
            with self.subTest(message=message, value=str(value)[:60]):
                with self.assertRaisesRegex(ValueError, "^" + message.replace(".", r"\.") + "$"):
                    evaluation.validate_suite(value)

    def test_preview_takes_a_suite_or_a_single_prompt(self):
        for arguments in ({"prompt": "x", "prompt_suite": _suite()}, {"max_tokens": 64, "prompt_suite": _suite()}):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "Choose either a prompt suite"):
                evaluation.preview("/missing/receipt.json", **arguments)
        with self.assertRaisesRegex(ValueError, "Evaluation prompt must contain"):
            evaluation.preview("/missing/receipt.json", prompt=None)

    def test_report_samples_read_both_schemas(self):
        samples = [{"id": "p1", "prompt": "Say hi.", "max_tokens": 16, "outputs": {}}]
        self.assertEqual(samples, evaluation.report_samples({"schema_version": "text-training-evaluation/2", "samples": samples}))
        self.assertEqual([{"id": "prompt", "prompt": "Say hi.", "max_tokens": 16, "outputs": {"base": "hi"}}],
                         evaluation.report_samples({"schema_version": "text-training-evaluation/1", "prompt": "Say hi.",
                                                    "max_tokens": 16, "outputs": {"base": "hi"}}))
        with self.assertRaisesRegex(ValueError, "Unsupported retained evaluation schema."):
            evaluation.report_samples({"schema_version": "text-training-evaluation/9"})

    def test_history_and_reviews_are_empty_before_any_evaluation(self):
        database = _Database()
        self.assertEqual([], evaluation.history(database))
        self.assertEqual([], evaluation.latest_reviews(database, "e1", "a" * 64))
        with self.assertRaisesRegex(ValueError, "Evaluation was not found in this workspace."):
            evaluation.inspect(database, None, "e1")

    def test_history_is_newest_first_and_reviews_keep_the_latest_per_prompt(self):
        database = _Database()
        connection = database.connection
        connection.execute("CREATE TABLE text_training_evaluations (id TEXT, created_at TEXT)")
        connection.executemany("INSERT INTO text_training_evaluations VALUES (?, ?)",
                               [("old", "2026-01-01"), ("new", "2026-02-01"), ("same-time", "2026-02-01")])
        connection.execute("CREATE TABLE text_training_evaluation_reviews "
                           "(evaluation_id TEXT, prompt_id TEXT, artifact_sha256 TEXT, verdict TEXT)")
        digest = "a" * 64
        connection.executemany("INSERT INTO text_training_evaluation_reviews VALUES (?, ?, ?, ?)", [
            ("e1", "p1", digest, "worse"), ("e1", "p1", digest, "better"), ("e1", "p2", digest, "same"),
            ("e1", "p1", "b" * 64, "other-artifact"), ("e2", "p1", digest, "other-evaluation"),
        ])
        self.assertEqual(["same-time", "new", "old"], [row["id"] for row in evaluation.history(database)])
        self.assertEqual([("p1", "better"), ("p2", "same")],
                         sorted((row["prompt_id"], row["verdict"]) for row in evaluation.latest_reviews(database, "e1", digest)))


if __name__ == "__main__":
    unittest.main()
