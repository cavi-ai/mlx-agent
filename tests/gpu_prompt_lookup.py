"""Run explicitly in an MLX child, never during portable test discovery."""
import unittest
import threading
import time
from types import SimpleNamespace


class FixedCostClock:
    ticks = 0
    def __call__(self):
        self.ticks += 1
        return self.ticks * .001


class PromptLookupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global mx, KVCache, generate_step, lookup_step, Request
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import KVCache
        from mlx_agent._prompt_lookup import lookup_step
        from mlx_agent._prompt_lookup import _Request as Request

    def test_idle_native_loop_releases_the_cpu_and_stops(self):
        from mlx_lm import server
        from mlx_agent._prompt_lookup import install
        restore = install(server)
        owner, observations = None, []
        original = server.ResponseGenerator._generate
        def measured(worker):
            began, cpu = time.monotonic(), time.thread_time()
            try:
                original(worker)
            finally:
                observations.append((time.monotonic() - began, time.thread_time() - cpu))
        server.ResponseGenerator._generate = measured
        try:
            owner = server.ResponseGenerator(SimpleNamespace(), SimpleNamespace())
            threading.Event().wait(.2)
        finally:
            if owner is not None:
                owner.stop_and_join()
            restore()
        self.assertEqual(len(observations), 1)
        wall, cpu = observations[0]
        self.assertLess(cpu / wall, .25, observations)
        self.assertFalse(owner._generation_thread.is_alive())

    def run_steps(self, prompt, maximum, stop=None, prefix=0, constant=False, clock=None):
        # Deterministic next-token model uses real MLX tensors and KVCache writes.
        def model(tokens, cache):
            values = tokens.astype(mx.float32)[..., None, None]
            cache[0].update_and_fetch(values.transpose(0, 2, 1, 3), values.transpose(0, 2, 1, 3))
            predicted = mx.zeros_like(tokens[..., None]) if constant else (tokens[..., None] + 1) % 32
            return mx.where(mx.arange(32)[None, None, :] == predicted, 10., -10.)

        caches = [[KVCache()], [KVCache()]]
        if prefix:
            for state in caches:
                model(mx.array([prompt[:prefix]]), state)
        request = Request()
        native = generate_step(mx.array(prompt[prefix:]), model, max_tokens=maximum, prompt_cache=caches[0])
        lookup = lookup_step(mx.array(prompt[prefix:]), model, full_prompt=prompt,
                             max_tokens=maximum, prompt_cache=caches[1], request=request,
                             **({"clock": clock} if clock is not None else {}))
        expected, actual, probabilities = [], [], [[], []]
        for index, (generator, results) in enumerate(((native, expected), (lookup, actual))):
            for token, logprobs in generator:
                # The server consumes the selected probability at every yield.
                self.assertTrue(mx.isfinite(logprobs[token]).item())
                probabilities[index].append(logprobs.tolist())
                results.append(token)
                if stop is not None and len(results) == stop:
                    break
            generator.close()
        self.assertEqual(actual, expected)
        self.assertEqual(probabilities[0], probabilities[1])
        for state in caches:
            mx.eval(state[0].state)
            self.assertEqual(state[0].offset, len(prompt) + len(actual))
        for expected_state, actual_state in zip(caches[0][0].state, caches[1][0].state):
            self.assertTrue(mx.array_equal(expected_state, actual_state).item())
        return request

    def test_lengths_and_cancellation_inside_verified_blocks(self):
        prompt = list(range(1, 25)) + [1, 2, 3]
        for maximum in (1, 2, 8, 19, 32):
            for stop in (None, 1, 3, 7):
                with self.subTest(maximum=maximum, stop=stop):
                    self.run_steps(prompt, maximum, stop)

    def test_cached_prompt_suffix_and_mismatching_proposals(self):
        for prompt in ([1, 2, 3, 4, 5, 6, 1, 2, 3], [1, 2, 3, 10, 11, 12, 1, 2, 3], [20, 4, 7]):
            with self.subTest(prompt=prompt):
                self.run_steps(prompt, 16, prefix=len(prompt) - 1)

    def test_first_reply_waits_for_a_complete_generated_lookup_prefix(self):
        request = self.run_steps(list(range(24)) + [30, 31], 16, clock=FixedCostClock())
        self.assertGreater(request.accepted_tokens, 0)

    def test_m5_copy_uses_wide_blocks_with_exact_tokens_probabilities_and_cache(self):
        if mx.device_info().get("architecture") != "applegpu_g17s":
            self.skipTest("Requires the M5 proposal profile")
        prompt = list(range(32)) * 3 + [0, 1, 2]
        request = self.run_steps(prompt, 48, clock=FixedCostClock())
        self.assertLessEqual(request.verification_calls, 5)
        for maximum in (33, 48, 64, 96):
            for stop in (3, 9, 31, 33):
                with self.subTest(maximum=maximum, stop=stop):
                    self.run_steps(prompt, maximum, stop=stop, clock=FixedCostClock())

    def test_m5_lookup_finds_copied_content_after_a_short_unmatched_prefix(self):
        if mx.device_info().get("architecture") != "applegpu_g17s":
            self.skipTest("Requires the M5 proposal profile")
        request = self.run_steps(list(range(24)) + [27, 28], 48, clock=FixedCostClock())
        self.assertGreater(request.accepted_tokens, 0)

    def test_native_handoff_after_misses_keeps_cache_exact_on_completion_and_cancel(self):
        prompt = [0, 0, 0, 9, 0, 0, 0]
        for stop in (None, 5, 6, 9):
            with self.subTest(stop=stop):
                request = self.run_steps(prompt, 16, stop=stop, prefix=len(prompt) - 1, constant=True, clock=FixedCostClock())
                self.assertEqual(request.fallback_reason, "low_acceptance")
                self.assertGreater(request.native_tokens, 0)

    def test_expensive_verification_hands_off_despite_accepted_tokens(self):
        class Clock:
            calls, value = 0, 0
            def __call__(self):
                self.calls += 1
                if self.calls % 2 == 0:
                    self.value += .001 if self.calls <= 4 else .1
                return self.value
        request = self.run_steps(list(range(32)) + [0, 1, 2], 48, clock=Clock())
        self.assertEqual(request.fallback_reason, "verification_cost")
        self.assertGreater(request.accepted_tokens, 0)
        self.assertGreater(request.native_tokens, 0)

    def test_failure_during_native_handoff_marks_cache_unusable(self):
        calls = 0
        def model(tokens, cache):
            nonlocal calls
            calls += 1
            if calls == 6:
                raise RuntimeError("native target failed")
            values = tokens.astype(mx.float32)[..., None, None].transpose(0, 2, 1, 3)
            cache[0].update_and_fetch(values, values)
            return mx.where(mx.arange(32)[None, None, :] == 0, 10., -10.) + mx.zeros((*tokens.shape, 1))
        request = Request()
        generator = lookup_step(mx.array([0, 0, 0, 9, 0, 0, 0]), model,
            full_prompt=[0, 0, 0, 9, 0, 0, 0], max_tokens=16, prompt_cache=[KVCache()], request=request,
            clock=FixedCostClock())
        with self.assertRaisesRegex(RuntimeError, "native target failed"):
            list(generator)
        self.assertTrue(request.failed)
        self.assertEqual(request.fallback_reason, "low_acceptance")


if __name__ == "__main__":
    unittest.main()
