import concurrent.futures
import json
import unittest
from unittest import mock

from mlx_agent.inference_telemetry import InferenceTelemetry


class InferenceTelemetryTests(unittest.TestCase):
    def test_totals_distinguish_missing_usage_from_reported_zero(self):
        telemetry = InferenceTelemetry(clock=mock.Mock(side_effect=[10, 12, 20, 24]))
        first = telemetry.begin()
        first.finish(success=True)
        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot['completed_requests'], 1)
        self.assertEqual(snapshot['measured_requests'], 0)
        self.assertIsNone(snapshot['input_tokens'])
        second = telemetry.begin()
        second.observe({'usage': {'prompt_tokens': 8, 'completion_tokens': 0}})
        second.finish(success=True)
        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot['requests'], 2)
        self.assertEqual(snapshot['measured_requests'], 1)
        self.assertEqual((snapshot['input_tokens'], snapshot['output_tokens']), (8, 0))
        self.assertEqual(snapshot['last_request']['response_seconds'], 4)
        self.assertEqual(snapshot['last_request']['output_tokens_per_second'], 0)

    def test_repeated_usage_and_finish_do_not_double_count(self):
        telemetry = InferenceTelemetry(clock=mock.Mock(side_effect=[1, 3]))
        request = telemetry.begin()
        for count in (1, 4, 4):
            request.observe({'usage': {'prompt_tokens': 6, 'completion_tokens': count}})
        request.finish(success=True)
        request.finish(success=True)
        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot['completed_requests'], 1)
        self.assertEqual(snapshot['output_tokens'], 4)
        self.assertEqual(snapshot['last_request']['output_tokens_per_second'], 2)

    def test_invalid_usage_and_failed_requests_do_not_enter_totals(self):
        telemetry = InferenceTelemetry()
        for value in (-1, True, '5', 1.5, 2**53):
            request = telemetry.begin()
            request.observe({'usage': {'prompt_tokens': 3, 'completion_tokens': value}})
            request.finish(success=True)
        request = telemetry.begin()
        request.observe({'usage': {'prompt_tokens': 3, 'completion_tokens': 5}})
        request.finish(success=False)
        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot['requests'], 6)
        self.assertEqual(snapshot['failed_requests'], 1)
        self.assertEqual(snapshot['measured_requests'], 0)
        self.assertIsNone(snapshot['output_tokens'])

    def test_stream_parser_handles_chunk_boundaries_and_bounded_lines(self):
        telemetry = InferenceTelemetry()
        request = telemetry.begin()
        line = b'data: ' + json.dumps({'usage': {'prompt_tokens': 7, 'completion_tokens': 2}}).encode() + b'\n\n'
        request.feed(b'data: ' + b'x' * 100_000 + b'\n')
        for index in range(0, len(line), 3):
            request.feed(line[index:index + 3])
        request.finish(success=True)
        self.assertEqual(telemetry.snapshot()['input_tokens'], 7)
        self.assertEqual(telemetry.snapshot()['output_tokens'], 2)
        self.assertNotIn('usage', json.dumps(telemetry.snapshot()))

    def test_concurrent_requests_and_snapshot_copies_are_independent(self):
        telemetry = InferenceTelemetry()
        def complete(_):
            request = telemetry.begin()
            request.observe({'usage': {'prompt_tokens': 2, 'completion_tokens': 3}})
            request.finish(success=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(complete, range(40)))
        snapshot = telemetry.snapshot()
        self.assertEqual((snapshot['measured_requests'], snapshot['input_tokens'], snapshot['output_tokens']), (40, 80, 120))
        snapshot['last_request']['input_tokens'] = 999
        self.assertEqual(telemetry.snapshot()['last_request']['input_tokens'], 2)
        self.assertIsNone(InferenceTelemetry().snapshot()['input_tokens'])
