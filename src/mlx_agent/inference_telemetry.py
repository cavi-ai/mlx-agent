"""Bounded numeric observations for one inference gateway process."""
import datetime
import json
import math
import threading
import time

MAX_LINE = 65536
MAX_TOKENS = 2**53 - 1


def _timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def _span(start, end):
    if start is None or end is None:
        return None
    seconds = end - start
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _rate(count, seconds):
    return count / seconds if count is not None and seconds else None


class InferenceTelemetry:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started_at = _timestamp()
        self.lock = threading.Lock()
        self.requests = self.completed = self.failed = self.measured = 0
        self.input_tokens = self.output_tokens = 0
        self.last = None

    def begin(self):
        request = RequestMeasurement(self, self.clock())
        with self.lock:
            self.requests += 1
        return request

    def snapshot(self):
        with self.lock:
            return {'schema': 1, 'scope': 'gateway-process', 'started_at': self.started_at,
                    'captured_at': _timestamp(), 'requests': self.requests,
                    'completed_requests': self.completed, 'failed_requests': self.failed,
                    'measured_requests': self.measured,
                    'input_tokens': self.input_tokens if self.measured else None,
                    'output_tokens': self.output_tokens if self.measured else None,
                    'last_request': dict(self.last) if self.last else None}

    def finish(self, request, success):
        with self.lock:
            if request.finished:
                return
            request.finished = True
            ended = self.clock()
            elapsed = _span(request.started, ended)
            measured = success and request.tokens is not None
            prompt_seconds = _span(request.ready_at, request.first_token_at)
            decode_seconds = _span(request.first_token_at, ended)
            decoded = request.tokens[1] - 1 if measured and request.tokens[1] > 1 else None
            if success:
                self.completed += 1
            else:
                self.failed += 1
            if measured:
                self.measured += 1
                self.input_tokens += request.tokens[0]
                self.output_tokens += request.tokens[1]
            self.last = {'completed_at': _timestamp(), 'outcome': 'completed' if success else 'failed',
                         'input_tokens': request.tokens[0] if measured else None,
                         'output_tokens': request.tokens[1] if measured else None,
                         'response_seconds': elapsed,
                         'output_tokens_per_second': request.tokens[1] / elapsed if measured and elapsed else None,
                         'wait_seconds': _span(request.started, request.ready_at),
                         'first_token_seconds': _span(request.started, request.first_token_at),
                         'prompt_tokens_per_second': _rate(request.tokens[0] if measured else None, prompt_seconds),
                         'decode_tokens_per_second': _rate(decoded, decode_seconds)}
            request.buffer.clear()


class RequestMeasurement:
    def __init__(self, owner, started):
        self.owner, self.started = owner, started
        self.tokens = None
        self.ready_at = self.first_token_at = None
        self.finished = False
        self.buffer = bytearray()
        self.discarding_line = False
        self.stream_done = False

    def mark_ready(self):
        """Record when a model worker accepts the request, after any load or queue wait."""
        if self.ready_at is None:
            self.ready_at = self.owner.clock()

    def _first_token(self, value):
        choices = value.get('choices') if isinstance(value, dict) else None
        if self.first_token_at is not None or not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return
        choice = choices[0]
        delta = choice.get('delta') if isinstance(choice.get('delta'), dict) else {}
        if choice.get('text') or delta.get('tool_calls') or any(delta.get(key) for key in ('content', 'reasoning', 'reasoning_content')):
            self.first_token_at = self.owner.clock()

    def observe(self, value):
        usage = value.get('usage') if isinstance(value, dict) else None
        if not isinstance(usage, dict):
            return
        pair = usage.get('prompt_tokens'), usage.get('completion_tokens')
        if all(type(value) is int and 0 <= value <= MAX_TOKENS for value in pair):
            self.tokens = pair

    def observe_json(self, payload):
        """Read a streamed chunk: its first generated text marks the end of prompt processing."""
        if self.first_token_at is not None and b'"usage"' not in payload:
            return
        try:
            value = json.loads(payload)
        except (ValueError, TypeError, UnicodeError):
            return
        self._first_token(value)
        self.observe(value)

    def feed(self, chunk):
        """Inspect SSE usage without retaining an unbounded line or changing the relay."""
        offset = 0
        while offset < len(chunk):
            end = chunk.find(b'\n', offset)
            boundary = len(chunk) if end < 0 else end
            length = boundary - offset
            if len(self.buffer) + length > MAX_LINE:
                self.buffer.clear()
                self.discarding_line = True
            elif not self.discarding_line:
                self.buffer.extend(chunk[offset:boundary])
            if end < 0:
                return
            if not self.discarding_line and self.buffer.startswith(b'data: '):
                payload = bytes(self.buffer[6:]).strip()
                if payload == b'[DONE]':
                    self.stream_done = True
                else:
                    self.observe_json(payload)
            self.buffer.clear()
            self.discarding_line = False
            offset = end + 1

    def finish(self, success):
        self.owner.finish(self, success)
