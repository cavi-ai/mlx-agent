import unittest
from types import SimpleNamespace

from mlx_agent._prompt_lookup import PromptLookupIndex, request_eligible


class PromptLookupIndexTests(unittest.TestCase):
    def test_proposals_use_the_longest_matching_suffix_and_respect_limits(self):
        index = PromptLookupIndex([1, 2, 3, 4, 5, 6, 1, 2, 9, 8])
        self.assertEqual(index.propose([0, 1, 2, 3], 2), [4, 5])
        self.assertEqual(index.propose([1, 2, 7], 8), [])
        self.assertEqual(index.propose([1, 2], 8), [])
        self.assertEqual(index.propose([1, 2, 3], 0), [])

    def test_index_retains_only_the_bounded_prompt_window(self):
        index = PromptLookupIndex(list(range(10000)))
        self.assertEqual(index.propose([0, 1, 2], 8), [])
        self.assertEqual(index.propose([9990, 9991, 9992], 8), list(range(9993, 10000)))
        self.assertLessEqual(index.entry_count, 6 * 8192)


class PromptLookupEligibilityTests(unittest.TestCase):
    def arguments(self):
        return SimpleNamespace(
            model=SimpleNamespace(draft="default_model", adapter="default_model"),
            sampling=SimpleNamespace(temperature=0),
            logits=SimpleNamespace(logit_bias=None, repetition_penalty=0),
            seed=None, logprobs=-1,
        )

    def test_only_unmodified_greedy_requests_enter_single_sequence_mode(self):
        args = self.arguments()
        cli = SimpleNamespace(draft_model=None, adapter_path=None)
        self.assertTrue(request_eligible(args, cli))
        for parent, name, value in (
            (args.sampling, "temperature", 0.7),
            (args.logits, "logit_bias", {4: 2}),
            (args.logits, "repetition_penalty", 1.1),
            (args.model, "draft", "/models/draft"),
            (args.model, "adapter", "/models/adapter"),
            (args, "seed", 3), (args, "logprobs", 2),
            (cli, "draft_model", "/models/draft"),
            (cli, "adapter_path", "/models/adapter"),
        ):
            with self.subTest(name=name, value=value):
                previous = getattr(parent, name)
                setattr(parent, name, value)
                self.assertFalse(request_eligible(args, cli))
                setattr(parent, name, previous)
