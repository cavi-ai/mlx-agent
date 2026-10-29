import unittest
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mlx_agent import _prompt_lookup as lookup
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


class PromptLookupHookTests(unittest.TestCase):
    @contextmanager
    def runtime(self):
        class KVCache:
            offset = 0

        state = [KVCache()]
        prompt = SimpleNamespace(size=3)
        native_step = Mock(return_value=iter([(7, None)]))
        generation = SimpleNamespace(generate_step=native_step)
        cache = SimpleNamespace(fetch_nearest_cache=Mock(return_value=(state, prompt)),
                                insert_cache=Mock())

        class NativeGenerator:
            def _is_batchable(self, args):
                return True

            def _serve_single(self, item):
                if item[1] == "fail":
                    raise RuntimeError("request failed")
                state, prompt = self.prompt_cache.fetch_nearest_cache("model", [1, 2, 3])
                generator = generation.generate_step(prompt, "model", max_tokens=16, prompt_cache=state)
                token, _ = next(generator)
                # The native owner publishes before explicitly closing its generator.
                self.prompt_cache.insert_cache("model", [1, 2, 3, token], state)

        server = SimpleNamespace(ResponseGenerator=NativeGenerator)
        with patch("mlx_agent._prompt_lookup.importlib.import_module", return_value=generation), \
                patch.dict(sys.modules, {"mlx_lm.models.cache": SimpleNamespace(KVCache=KVCache)}):
            restore = lookup.install(server)
            owner = server.ResponseGenerator()
            owner.prompt_cache = cache
            owner.model_provider = SimpleNamespace(cli_args=SimpleNamespace(draft_model=None, adapter_path=None))
            try:
                yield owner, generation, state, cache, native_step
            finally:
                restore()
                self.assertIs(server.ResponseGenerator, NativeGenerator)
                self.assertIs(generation.generate_step, native_step)

    def test_request_closes_before_cache_publication_and_restores_native_owner(self):
        args = PromptLookupEligibilityTests().arguments()
        with self.runtime() as (owner, generation, state, cache, native):
            def verified(*_, **kwargs):
                self.assertEqual(kwargs["full_prompt"], [1, 2, 3])
                try:
                    yield 7, None
                finally:
                    state[0].offset = 4

            cache.insert_cache.side_effect = lambda _, tokens, state: self.assertEqual(state[0].offset, len(tokens))
            self.assertFalse(owner._is_batchable(args))
            with patch.object(lookup, "lookup_step", side_effect=verified):
                owner._serve_single((None, "request", args))
            cache.insert_cache.assert_called_once()
            native.assert_not_called()
            self.assertIs(owner.prompt_cache, cache)
            self.assertIsNone(lookup._request.get())
            args.sampling.temperature = .7
            self.assertTrue(owner._is_batchable(args))
            owner._serve_single((None, "request", args))
            native.assert_called_once()

    def test_request_error_restores_cache_and_context(self):
        with self.runtime() as (owner, _, _, cache, _):
            with self.assertRaisesRegex(RuntimeError, "request failed"):
                owner._serve_single((None, "fail", PromptLookupEligibilityTests().arguments()))
            self.assertIs(owner.prompt_cache, cache)
            self.assertIsNone(lookup._request.get())

    def test_unsupported_cache_and_generation_options_delegate_unchanged(self):
        with self.runtime() as (_, generation, state, _, native):
            context = lookup._Request(prompt=[1, 2, 3])
            token = lookup._request.set(context)
            try:
                for options in ({"prompt_cache": [SimpleNamespace(offset=0)]},
                                {"kv_bits": 4}, {"max_kv_size": 1024},
                                {"input_embeddings": object()}, {"logits_processors": [object()]},
                                {"max_tokens": -1}):
                    kwargs = dict(prompt_cache=state, max_tokens=16)
                    kwargs.update(options)
                    prompt = SimpleNamespace(size=3)
                    generation.generate_step(prompt, "model", **kwargs)
                    native.assert_called_with(prompt, "model", **kwargs)
                state[0].offset = 1
                generation.generate_step(SimpleNamespace(size=3), "model", prompt_cache=state)
                self.assertFalse(context.used)
            finally:
                lookup._request.reset(token)

    def test_failed_or_misaligned_cache_is_not_reused(self):
        original = SimpleNamespace(insert_cache=Mock(), other="delegated")
        context = lookup._Request(used=True)
        cache = lookup._PromptCache(original, context)
        self.assertEqual(cache.other, "delegated")
        with self.assertLogs(level="WARNING"):
            cache.insert_cache("model", [1, 2], [SimpleNamespace(offset=1)])
        context.failed = True
        with self.assertLogs(level="WARNING"):
            cache.insert_cache("model", [1, 2], [SimpleNamespace(offset=2)])
        original.insert_cache.assert_not_called()
