"""Run explicitly in an MLX child, never during portable test discovery."""
import unittest
from types import SimpleNamespace


class PromptLookupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global mx, KVCache, generate_step, lookup_step
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import KVCache
        from mlx_agent._prompt_lookup import lookup_step

    def run_steps(self, prompt, maximum, stop=None, prefix=0):
        # Deterministic next-token model uses real MLX tensors and KVCache writes.
        def model(tokens, cache):
            values = tokens.astype(mx.float32)[..., None, None]
            cache[0].update_and_fetch(values.transpose(0, 2, 1, 3), values.transpose(0, 2, 1, 3))
            return mx.where(mx.arange(32)[None, None, :] == (tokens[..., None] + 1) % 32, 10., -10.)

        caches = [[KVCache()], [KVCache()]]
        if prefix:
            for state in caches:
                model(mx.array([prompt[:prefix]]), state)
        request = SimpleNamespace(failed=False, verification_calls=0, accepted_tokens=0)
        native = generate_step(mx.array(prompt[prefix:]), model, max_tokens=maximum, prompt_cache=caches[0])
        lookup = lookup_step(mx.array(prompt[prefix:]), model, full_prompt=prompt,
                             max_tokens=maximum, prompt_cache=caches[1], request=request)
        expected, actual = [], []
        for generator, results in ((native, expected), (lookup, actual)):
            for token, _ in generator:
                results.append(token)
                if stop is not None and len(results) == stop:
                    break
            generator.close()
        self.assertEqual(actual, expected)
        for state in caches:
            mx.eval(state[0].state)
            self.assertEqual(state[0].offset, len(prompt) + len(actual))
        self.assertTrue(mx.array_equal(caches[0][0].state[0], caches[1][0].state[0]).item())

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


if __name__ == "__main__":
    unittest.main()
