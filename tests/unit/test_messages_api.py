import asyncio
import http.client
import importlib.util
import json
import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.jit_serve import make_server, model_fingerprint


# Answers like mlx_lm.server: tool calls arrive whole (id null before 0.31), and assistant tool-call arguments must be JSON text.
WORKER = '''
import argparse, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
p = argparse.ArgumentParser()
p.add_argument('--model'); p.add_argument('--port', type=int); p.add_argument('--trace')
a = p.parse_args()
CALL = {'function': {'name': 'get_weather', 'arguments': json.dumps({'city': 'Paris'})}, 'type': 'function', 'id': None}
def response(kind, choice, usage=None):
    value = {'id': 'chatcmpl-1', 'system_fingerprint': 'fp', 'object': kind, 'model': 'local', 'created': 1, 'choices': [choice]}
    if usage: value['usage'] = usage
    return value
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with open(a.trace, 'a') as f: f.write(json.dumps(body) + '\\n')
        try:
            [json.loads(call['function']['arguments']) for m in body['messages'] for call in m.get('tool_calls', [])]
        except (TypeError, ValueError) as error:
            self.send_response(404); self.end_headers(); self.wfile.write(json.dumps({'error': str(error)}).encode()); return
        answered = body['messages'][-1]['role'] == 'tool'
        calls = [CALL] if body.get('tools') and not answered else []
        text = 'It is sunny.' if answered else ('' if calls else 'Hello from MLX')
        self.send_response(200); self.end_headers()
        if body.get('stream'):
            for piece in ('Hel', 'lo'):
                chunk = response('chat.completion.chunk', {'index': 0, 'finish_reason': None, 'delta': {'role': 'assistant', 'content': piece, 'tool_calls': []}})
                self.wfile.write(('data: ' + json.dumps(chunk) + '\\n\\n').encode())
            last = response('chat.completion.chunk', {'index': 0, 'finish_reason': 'stop', 'delta': {'role': 'assistant', 'content': '', 'tool_calls': calls}})
            self.wfile.write(('data: ' + json.dumps(last) + '\\n\\n').encode())
            if (body.get('stream_options') or {}).get('include_usage'):
                usage = {'id': 'chatcmpl-1', 'object': 'chat.completion', 'model': 'local', 'created': 1, 'choices': [],
                         'usage': {'prompt_tokens': 15, 'completion_tokens': 3, 'total_tokens': 18}}
                self.wfile.write(('data: ' + json.dumps(usage) + '\\n\\n').encode())
            self.wfile.write(b'data: [DONE]\\n\\n')
        else:
            usage = {'prompt_tokens': 20 if calls else 12, 'completion_tokens': 9 if calls else 4, 'total_tokens': 29 if calls else 16}
            self.wfile.write(json.dumps(response('chat.completion', {'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': text, 'tool_calls': calls}}, usage)).encode())
class Server(ThreadingHTTPServer):
    def server_bind(self):
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]
Server(('127.0.0.1', a.port), Handler).serve_forever()
'''

WEATHER = {"name": "get_weather", "description": "Current weather for a city.",
           "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}


class MessagesGatewayTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.model = root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        (self.model / "model.safetensors").write_bytes(b"fixture")
        (root / "worker.py").write_text(WORKER)
        self.trace = root / "requests.jsonl"
        config = {
            "model": str(self.model), "local_path": str(self.model), "fingerprint": model_fingerprint(self.model),
            "control_token": "test-control-token", "max_tokens": 8192,
            "worker_argv": [sys.executable, str(root / "worker.py"), "--model", str(self.model),
                            "--port", "{worker_port}", "--trace", str(self.trace)],
        }
        self.server = make_server(config, port=0)
        self.port = self.server.server_port
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.controller.close)
        self.addCleanup(self.server.shutdown)

    def post(self, body):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            connection.request("POST", "/v1/messages", json.dumps(body),
                               {"Content-Type": "application/json", "x-api-key": "local", "anthropic-version": "2023-06-01"})
            response = connection.getresponse()
            return response.status, response.getheader("Content-Type"), response.read().decode()
        finally:
            connection.close()

    def sent(self):
        return json.loads(self.trace.read_text().splitlines()[-1])

    @unittest.skipUnless(importlib.util.find_spec("cavi_runtime"), "requires requirements-dev.txt")
    def test_claude_client_completes_a_run_through_the_gateway(self):
        from cavi_runtime import create_runtime_client

        async def run():
            async with create_runtime_client("claude-sdk", api_key="local", base_url="http://127.0.0.1:{}".format(self.port),
                                             default_model=str(self.model)) as client:
                return await client.start_run({"input": "Say hello", "instructions": "Answer briefly.",
                                               "metadata": {"max_tokens": 64}})

        run = asyncio.run(run())
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["output"], "Hello from MLX")
        self.assertEqual((run["tokens"]["inputTokens"], run["tokens"]["outputTokens"]), (12, 4))
        sent = self.sent()
        self.assertEqual(sent["messages"], [{"role": "system", "content": "Answer briefly."},
                                            {"role": "user", "content": "Say hello"}])
        self.assertEqual((sent["max_tokens"], sent["model"], sent.get("stream", False)), (64, str(self.model), False))

    def test_tool_use_round_trip_maps_calls_and_results(self):
        question = {"role": "user", "content": "Weather in Paris?"}
        status, _, text = self.post({"model": str(self.model), "max_tokens": 128, "tools": [WEATHER], "messages": [question]})
        self.assertEqual(status, 200, text)
        message = json.loads(text)
        call = message["content"][0]
        self.assertEqual((message["type"], message["role"], message["stop_reason"]), ("message", "assistant", "tool_use"))
        self.assertEqual((call["type"], call["name"], call["input"]), ("tool_use", "get_weather", {"city": "Paris"}))
        self.assertTrue(call["id"].startswith("toolu_"))
        self.assertEqual(message["usage"], {"input_tokens": 20, "output_tokens": 9})
        self.assertEqual(self.sent()["tools"], [{"type": "function", "function": {
            "name": "get_weather", "description": "Current weather for a city.", "parameters": WEATHER["input_schema"]}}])

        status, _, text = self.post({"model": str(self.model), "max_tokens": 128, "tools": [WEATHER], "messages": [
            question, {"role": "assistant", "content": [call]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"],
                                          "content": [{"type": "text", "text": "18C and sunny"}]}]}]})
        self.assertEqual(status, 200, text)
        answer = json.loads(text)
        self.assertEqual((answer["content"], answer["stop_reason"]), ([{"type": "text", "text": "It is sunny."}], "end_turn"))
        self.assertEqual(self.sent()["messages"][1:], [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": call["id"], "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
            {"role": "tool", "tool_call_id": call["id"], "content": "18C and sunny"}])

    def test_streaming_emits_anthropic_events_for_text_and_tool_use(self):
        status, content_type, text = self.post({"model": str(self.model), "max_tokens": 128, "stream": True,
                                                "tools": [WEATHER], "messages": [{"role": "user", "content": "Hi"}]})
        self.assertEqual((status, content_type), (200, "text/event-stream"), text)
        events = []
        for frame in text.strip().split("\n\n"):
            name, data = frame.split("\n")
            events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
        self.assertEqual([name for name, _ in events], [
            "message_start", "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
            "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"])
        self.assertEqual([data["delta"]["text"] for name, data in events[2:4]], ["Hel", "lo"])
        tool = events[5][1]["content_block"]
        self.assertEqual((events[5][1]["index"], tool["type"], tool["name"]), (1, "tool_use", "get_weather"))
        self.assertEqual(json.loads(events[6][1]["delta"]["partial_json"]), {"city": "Paris"})
        self.assertEqual(events[8][1]["delta"]["stop_reason"], "tool_use")
        self.assertEqual(events[8][1]["usage"], {"input_tokens": 15, "output_tokens": 3})
        sent = self.sent()
        self.assertEqual((sent["stream"], sent["stream_options"]), (True, {"include_usage": True}))

    def test_unsupported_requests_return_anthropic_errors_without_loading(self):
        image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}
        for body in ({"model": str(self.model), "max_tokens": 16, "messages": [{"role": "user", "content": [image]}]},
                     {"model": "claude-sonnet", "max_tokens": 16, "messages": [{"role": "user", "content": "Hi"}]}):
            status, _, text = self.post(body)
            error = json.loads(text)
            self.assertEqual((status, error["type"], error["error"]["type"]), (400, "error", "invalid_request_error"), text)
        self.assertFalse(self.trace.exists())


if __name__ == "__main__":
    unittest.main()
