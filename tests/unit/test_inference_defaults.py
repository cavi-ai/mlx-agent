import math
import unittest

from mlx_agent import inference_defaults as defaults


class InferenceDefaultsTests(unittest.TestCase):
    def test_defaults_fill_missing_fields_without_mutating_input(self):
        body = {'messages': [{'role': 'user', 'content': 'Hello'}], 'temperature': 0}
        configured = defaults.validate('Be concise.', 0.7, 0.9)
        result = defaults.apply(body, configured)
        self.assertEqual(result['messages'][0], {'role': 'system', 'content': 'Be concise.'})
        self.assertEqual(result['temperature'], 0)
        self.assertEqual(result['top_p'], 0.9)
        self.assertEqual(body, {'messages': [{'role': 'user', 'content': 'Hello'}], 'temperature': 0})

    def test_explicit_system_or_developer_prompt_has_precedence(self):
        for role in ('system', 'developer'):
            body = {'messages': [{'role': role, 'content': 'Client policy'}]}
            self.assertEqual(defaults.apply(body, defaults.validate('Server policy')), body)
        self.assertEqual(defaults.apply({'prompt': 'Legacy completion'}, defaults.validate('Chat only')), {'prompt': 'Legacy completion'})

    def test_invalid_or_unbounded_settings_are_refused(self):
        for value in ('x' * 8193, 'x\x00', '\ud800', 3):
            with self.assertRaises(ValueError): defaults.validate(system_prompt=value)
        for value in (True, -1, 2.1, math.inf, math.nan, '0.5'):
            with self.assertRaises(ValueError): defaults.validate(temperature=value)
        with self.assertRaises(ValueError): defaults.validate(top_p=1.1)
        self.assertEqual(defaults.validate('', 0, 0), {'temperature': 0.0, 'top_p': 0.0})

    def test_status_summary_contains_no_prompt_content(self):
        result = defaults.summary(defaults.validate('Private instructions', 0.4))
        self.assertNotIn('system_prompt', result)
        self.assertEqual(len(result['system_prompt_sha256']), 64)
        self.assertEqual(result['temperature'], 0.4)
