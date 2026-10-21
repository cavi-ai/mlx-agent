import io
import json
import re
import unittest
from contextlib import redirect_stderr, redirect_stdout

from mlx_agent.command_args import CommandArgumentError, main, parse_command_arguments


DIGEST = "a" * 64
TASK_ID = "0123abcd-0000-4000-8000-0123456789ab"


class CommandArgumentGrammarTests(unittest.TestCase):
    def assertAccepted(self, capability, raw, argv):
        with self.subTest(capability=capability, raw=raw):
            self.assertEqual(argv, parse_command_arguments(capability, raw))

    def assertRefused(self, capability, raw, message):
        with self.subTest(capability=capability, raw=raw):
            with self.assertRaisesRegex(CommandArgumentError, "^" + re.escape(message)):
                parse_command_arguments(capability, raw)

    def test_tokens_refuse_shell_syntax_control_characters_and_oversized_input(self):
        self.assertRefused("scout", None, "arguments must be text")
        self.assertRefused("scout", "--json\t", "arguments must not contain control characters")
        self.assertRefused("scout", "x" * 4097, "arguments must not contain control characters")
        for character in ";|&`$<>":
            self.assertRefused("scout", "--json {0} id".format(character), "shell metacharacters are not allowed")
        self.assertRefused("scout", "--license 'open", "arguments must use balanced shell-style quotes")
        self.assertRefused("scout", " ".join(["--json"] * 33), "arguments exceed the supported command grammar")
        self.assertRefused("scout", "x" * 513, "arguments exceed the supported command grammar")
        self.assertRefused("scout", '--license ""', "arguments exceed the supported command grammar")
        self.assertRefused("serve", "--json", "unknown command capability")

    def test_scout_accepts_every_documented_flag(self):
        self.assertAccepted("scout", " ".join((
            "--role coding --limit 3 --memory-gb 24 --quantization 4bit --runtime mlx_lm --context 8192",
            "--state-dir state --wire mlx-community/Qwen3-8B-4bit --target lmstudio --port 1234",
            "--license mit --license apache-2.0 --publisher mlx-community --exclude-gated --new --fast --json --offline",
        )), [
            "discover", "--role", "coding", "--limit", "3", "--memory-gb", "24", "--quantization", "4bit",
            "--runtime", "mlx_lm", "--context", "8192", "--state-dir", "state", "--wire", "mlx-community/Qwen3-8B-4bit",
            "--target", "lmstudio", "--port", "1234", "--license", "mit", "--license", "apache-2.0",
            "--publisher", "mlx-community", "--exclude-gated", "--new", "--fast", "--json", "--offline",
        ])
        self.assertAccepted("scout", "--memory-gb .5 --refresh", ["discover", "--memory-gb", ".5", "--refresh"])

    def test_scout_refusals_name_the_problem(self):
        cases = (
            ("--include-gated --exclude-gated", "gated-repository flags are mutually exclusive"),
            ("--offline --refresh", "--refresh and --offline are mutually exclusive"),
            ("--json --json", "duplicate flag: --json"),
            ("--limit 3 --limit 4", "duplicate flag: --limit"),
            ("--limit 0", "limit must be between 1 and 50"),
            ("--limit three", "limit must be an integer"),
            ("--limit", "--limit requires a value"),
            ("--limit --json", "--limit requires a value"),
            ("--memory-gb 0.1", "memory-gb must be between 0.25 and 1024"),
            ("--memory-gb 1e3", "memory-gb must be a decimal number"),
            ("--role wizard", "invalid role value"),
            ("--runtime vllm", "invalid runtime value"),
            ("--quantization -q", "invalid quantization value"),
            ("--license 'two words'", "--license must be a safe identifier"),
            ("--wire qwen", "model must be a publisher/model identifier"),
            ("--state-dir ../outside", "state-dir must not traverse parent directories"),
            ("--state-dir ~/state", "state-dir must be a portable non-empty path"),
            ("--bogus", "unsupported Scout argument: --bogus"),
        )
        for raw, message in cases:
            self.assertRefused("scout", raw, message)

    def test_adopt_requires_an_action_and_a_state_path(self):
        self.assertAccepted(
            "adopt", "start --state s.json --role coding --shortlist-limit 5 --offline --fast --no-network --measure --json",
            ["adopt", "start", "--state", "s.json", "--role", "coding", "--shortlist-limit", "5",
             "--offline", "--fast", "--no-network", "--measure", "--json"],
        )
        self.assertAccepted("adopt", "status --state s.json --json", ["adopt", "status", "--state", "s.json", "--json"])
        cases = (
            ("", "Adopt requires start, resume, or status"),
            ("start --json", "Adopt requires an explicit --state path"),
            ("start --state a --state b", "duplicate flag: --state"),
            ("start --state s --role wizard", "unknown adoption role"),
            ("start --state s --shortlist-limit 21", "shortlist-limit must be between 1 and 20"),
            ("start --state s --json --json", "unsupported Adopt argument: --json"),
            ("status --state s --fast", "unsupported Adopt argument: --fast"),
            ("resume --state s --role coding", "unsupported Adopt argument: --role"),
        )
        for raw, message in cases:
            self.assertRefused("adopt", raw, message)

    def test_wire_binds_models_paths_hashes_and_loopback_endpoints(self):
        self.assertAccepted(
            "wire", "render mlx-community/Qwen3-8B-4bit --target mlx_lm --path providers.json --json",
            ["wire", "render", "mlx-community/Qwen3-8B-4bit", "--target", "mlx_lm", "--path", "providers.json", "--json"],
        )
        self.assertAccepted(
            "wire", "apply org/model --path p.json --receipts-dir r --endpoint http://127.0.0.1:8080/v1 --preview-hash {0} --confirm".format(DIGEST),
            ["wire", "apply", "org/model", "--path", "p.json", "--receipts-dir", "r", "--endpoint",
             "http://127.0.0.1:8080/v1", "--preview-hash", DIGEST, "--confirm"],
        )
        self.assertAccepted("wire", "apply org/model --path p --endpoint http://[::1]:1234", [
            "wire", "apply", "org/model", "--path", "p", "--endpoint", "http://[::1]:1234"])
        self.assertAccepted("wire", "status receipts/r.json --json", ["wire", "status", "receipts/r.json", "--json"])
        self.assertAccepted("wire", "rollback r.json --preview-hash {0} --confirm".format(DIGEST),
                            ["wire", "rollback", "r.json", "--preview-hash", DIGEST, "--confirm"])
        cases = (
            ("", "Wire requires render, apply, status, or rollback"),
            ("status", "Wire status requires a receipt path"),
            ("render", "Wire render requires a model"),
            ("render org/model", "Wire render requires --path"),
            ("render org/model --path p --confirm", "unsupported Wire argument: --confirm"),
            ("status r --confirm", "unsupported Wire argument: --confirm"),
            ("render org/model --target vllm --path p", "unsupported Wire target"),
            ("render org/model --path p --path q", "duplicate flag: --path"),
            ("rollback r --preview-hash abc", "preview-hash must be a SHA-256 digest"),
            ("apply org/model --path p --endpoint http://example.com", "endpoint must be a credential-free local HTTP URL"),
            ("apply org/model --path p --endpoint http://user:pw@127.0.0.1", "endpoint must be a credential-free local HTTP URL"),
            ("apply org/model --path p --endpoint http://127.0.0.1/?key=1", "endpoint must be a credential-free local HTTP URL"),
            ("apply org/model --path p --endpoint ftp://127.0.0.1", "endpoint must be a credential-free local HTTP URL"),
            ("apply org/model --path p --endpoint http://127.0.0.1:0", "endpoint port must be between 1 and 65535"),
            ("apply org/model --path p --endpoint http://127.0.0.1:70000", "endpoint is not a valid local HTTP URL"),
        )
        for raw, message in cases:
            self.assertRefused("wire", raw, message)

    def test_bench_run_and_aggregate(self):
        self.assertAccepted(
            "bench", "run --repo org/model --runtime mlx_lm --role coding --runs 3 --gen-tokens 128 --timeout 30.5 --export out.json --json",
            ["bench", "run", "--repo", "org/model", "--runtime", "mlx_lm", "--role", "coding", "--runs", "3",
             "--gen-tokens", "128", "--timeout", "30.5", "--export", "out.json", "--json"],
        )
        self.assertAccepted("bench", "aggregate --exports e --out o --json",
                            ["bench", "aggregate", "--exports", "e", "--out", "o", "--json"])
        cases = (
            ("", "Bench requires the run or aggregate action"),
            ("run --repo org/model", "Bench run requires --runtime"),
            ("run --runtime mlx_lm", "Bench run requires --repo"),
            ("run --repo org/model --runtime vllm", "invalid runtime value"),
            ("run --repo org/model --runtime mlx_lm --runs 11", "runs must be between 1 and 10"),
            ("run --repo org/model --repo org/other --runtime mlx_lm", "duplicate flag: --repo"),
            ("run --json --json", "duplicate flag: --json"),
            ("run --bogus", "unsupported Bench argument: --bogus"),
            ("aggregate", "Bench aggregate requires --exports"),
            ("aggregate --exports e --exports f", "duplicate flag: --exports"),
            ("aggregate --json --json", "duplicate flag: --json"),
            ("aggregate --bogus", "unsupported Bench argument: --bogus"),
        )
        for raw, message in cases:
            self.assertRefused("bench", raw, message)

    def test_doctor_and_watch(self):
        self.assertAccepted("doctor", "models --wired-root a --wired-root b --hf-cache c --json",
                            ["doctor", "models", "--wired-root", "a", "--wired-root", "b", "--hf-cache", "c", "--json"])
        self.assertAccepted("watch", "diff --state-dir s --limit 5 --json",
                            ["watch", "diff", "--state-dir", "s", "--limit", "5", "--json"])
        cases = (
            ("doctor", "", "Doctor requires the models action"),
            ("doctor", "models --hf-cache a --hf-cache b", "duplicate flag: --hf-cache"),
            ("doctor", "models --json --json", "duplicate flag: --json"),
            ("doctor", "models --bogus", "unsupported Doctor argument: --bogus"),
            ("watch", "", "Watch requires the snapshot or diff action"),
            ("watch", "snapshot --limit 2 --limit 3", "duplicate flag: --limit"),
            ("watch", "snapshot --state-dir a --state-dir b", "duplicate flag: --state-dir"),
            ("watch", "snapshot --json --json", "duplicate flag: --json"),
            ("watch", "snapshot --limit 51", "limit must be between 1 and 50"),
            ("watch", "snapshot --bogus", "unsupported Watch argument: --bogus"),
        )
        for capability, raw, message in cases:
            self.assertRefused(capability, raw, message)

    def test_fleet_assignments_runtime_maps_and_required_path(self):
        self.assertAccepted(
            "fleet", " ".join((
                "apply --assign coding=org/model --runtime-map coding=mlx_lm --path f.json --from-adoption a.json",
                "--preview-hash {0} --receipts-dir r --endpoint http://localhost:8080 --confirm --allow-missing --json".format(DIGEST),
            )),
            ["fleet", "apply", "--assign", "coding=org/model", "--runtime-map", "coding=mlx_lm", "--path", "f.json",
             "--from-adoption", "a.json", "--preview-hash", DIGEST, "--receipts-dir", "r",
             "--endpoint", "http://localhost:8080", "--confirm", "--allow-missing", "--json"],
        )
        cases = (
            ("", "Fleet requires the render or apply action"),
            ("render --json", "Fleet render requires --path"),
            ("render --path f --assign coding", "--assign must use role=value form"),
            ("render --path f --assign wizard=org/model", "--assign must name a canonical role"),
            ("render --path f --assign coding=model", "model must be a publisher/model identifier"),
            ("render --path f --runtime-map coding=ollama", "--runtime-map must target mlx_lm or mlx-vlm"),
            ("render --path f --preview-hash zz", "invalid preview-hash value"),
            ("render --path f --path g", "duplicate flag: --path"),
            ("render --path f --json --json", "duplicate flag: --json"),
            ("render --path f --receipts-dir a --receipts-dir b", "duplicate flag: --receipts-dir"),
            ("render --path f --bogus", "unsupported Fleet argument: --bogus"),
        )
        for raw, message in cases:
            self.assertRefused("fleet", raw, message)

    def test_task_actions_require_ids_absolute_paths_and_always_emit_json(self):
        self.assertAccepted("task", "capabilities", ["task", "capabilities", "--json"])
        self.assertAccepted("task", "start --request /tmp/r.json --state-dir /tmp/s",
                            ["task", "start", "--request", "/tmp/r.json", "--state-dir", "/tmp/s", "--json"])
        self.assertAccepted("task", "status --id {0} --json".format(TASK_ID), ["task", "status", "--id", TASK_ID, "--json"])
        self.assertAccepted("task", "list --limit 5", ["task", "list", "--limit", "5", "--json"])
        cases = (
            ("", "Task requires capabilities, start, status, list, result, cancel or verify-model"),
            ("status", "Task status requires --id"),
            ("start --json", "Task start requires --request"),
            ("status --id nope", "Task ID must be a canonical UUID"),
            ("start --request rel.json", "Task paths must be absolute"),
            ("list --limit 101", "limit must be between 1 and 100"),
            ("status --id {0} --id {0}".format(TASK_ID), "Unsupported or duplicate Task argument: --id"),
            ("capabilities --limit 3", "Unsupported or duplicate Task argument: --limit"),
        )
        for raw, message in cases:
            self.assertRefused("task", raw, message)

    def test_lab_passes_tokens_through_after_shell_checks(self):
        self.assertAccepted("lab", "experiment list --json", ["lab", "experiment", "list", "--json"])
        self.assertRefused("lab", "experiment list ; id", "shell metacharacters are not allowed")

    def test_command_line_prints_argv_or_exits_with_the_refusal(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(0, main(["scout", "--limit 2 --json"]))
        self.assertEqual({"argv": ["discover", "--limit", "2", "--json"]}, json.loads(output.getvalue()))
        errors = io.StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit) as caught:
            main(["scout", "--bogus 1"])
        self.assertEqual(2, caught.exception.code)
        self.assertIn("unsupported Scout argument: --bogus", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
