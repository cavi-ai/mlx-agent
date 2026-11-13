import http.client
import json
import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_agent.jit_serve import make_server, model_fingerprint


# Answers as mlx_lm 0.29.1 does: tool calls without id or stream index, finish_reason "stop", and the
# tokenizer's end-of-sequence text left at the end of unstreamed content.
WORKER = '''
import argparse, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
p = argparse.ArgumentParser()
p.add_argument('--model'); p.add_argument('--port', type=int)
a = p.parse_args()
CALL = {'function': {'name': 'get_weather', 'arguments': json.dumps({'city': 'Lisbon'})}, 'type': 'function', 'id': None}
def response(kind, choice):
    return {'id': 'chatcmpl-1', 'object': kind, 'model': 'local', 'created': 1, 'choices': [choice]}
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        calls = [CALL] if body.get('tools') and body['messages'][-1]['role'] != 'tool' else []
        # The native chat template serializes arguments itself. A JSON string
        # here becomes a quoted string instead of the generated argument object.
        history = [c for m in body.get('messages', []) for c in m.get('tool_calls', [])]
        answer = 'Double-encoded tool history' if any(isinstance(c['function']['arguments'], str) for c in history) else 'It is 23 degrees.'
        self.send_response(200); self.end_headers()
        if body.get('stream'):
            delta = {'role': 'assistant', 'content': '' if calls else 'It is 23 degrees.', 'tool_calls': calls}
            for choice in ({'index': 0, 'finish_reason': None, 'delta': delta},
                           {'index': 0, 'finish_reason': 'stop', 'delta': {'role': 'assistant', 'content': '', 'tool_calls': []}}):
                self.wfile.write(('data: ' + json.dumps(response('chat.completion.chunk', choice)) + '\\n\\n').encode())
            self.wfile.write(b'data: [DONE]\\n\\n')
        else:
            text = '<|im_end|>' if calls else answer + '<|im_end|>'
            self.wfile.write(json.dumps(response('chat.completion', {'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': text, 'tool_calls': calls}})).encode())
class Server(ThreadingHTTPServer):
    def server_bind(self):
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]
Server(('127.0.0.1', a.port), Handler).serve_forever()
'''

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
QUESTION = [{"role": "user", "content": "Weather in Lisbon?"}]
ANSWERED = QUESTION + [
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Lisbon"}'}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "Lisbon: 23 degrees Celsius."}]


# Answers with the tool_parser field of the request it received.
PARSER_WORKER = '''
import argparse, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer
p = argparse.ArgumentParser()
p.add_argument('--model'); p.add_argument('--port', type=int)
a = p.parse_args()
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        message = {'role': 'assistant', 'content': json.dumps(body.get('tool_parser'))}
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({'id': 'chatcmpl-1', 'object': 'chat.completion', 'model': 'local', 'created': 1,
                                     'choices': [{'index': 0, 'finish_reason': 'stop', 'message': message}]}).encode())
class Server(ThreadingHTTPServer):
    def server_bind(self):
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]
Server(('127.0.0.1', a.port), Handler).serve_forever()
'''

# MiMo-V2.6 and Qwen3.5 render Qwen3-Coder XML calls with no newline after <tool_call>.
ADJACENT_TOOL_CALL_TEMPLATE = "{{- '<tool_call><function=' ~ tool_call.name ~ '>' -}}"


class GatewayTestCase(unittest.TestCase):
    def serve(self, worker, runtime, chat_template=None):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.model = root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        (self.model / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>"}))
        if chat_template is not None:
            (self.model / "chat_template.jinja").write_text(chat_template)
        (self.model / "model.safetensors").write_bytes(b"fixture")
        (root / "worker.py").write_text(worker)
        config = {
            "runtime": runtime,
            "model": str(self.model), "local_path": str(self.model), "fingerprint": model_fingerprint(self.model),
            "control_token": "test-control-token", "max_tokens": 8192,
            "worker_argv": [sys.executable, str(root / "worker.py"), "--model", str(self.model), "--port", "{worker_port}"],
        }
        self.server = make_server(config, port=0)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.controller.close)
        self.addCleanup(self.server.shutdown)

    def post(self, path, body):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=30)
        try:
            connection.request("POST", path, json.dumps(dict(body, model="default_model")), {"Content-Type": "application/json"})
            response = connection.getresponse()
            text = response.read().decode()
            self.assertEqual(response.status, 200, text)
            return text
        finally:
            connection.close()


class ChatCompletionsGatewayTests(GatewayTestCase):
    def setUp(self):
        self.serve(WORKER, "mlx_lm")

    def test_unstreamed_tool_calls_have_ids_and_the_tool_calls_finish_reason(self):
        [choice] = json.loads(self.post("/v1/chat/completions", {"messages": QUESTION, "tools": TOOLS}))["choices"]
        [call] = choice["message"]["tool_calls"]
        self.assertRegex(str(call["id"]), r"^call_[0-9a-f]{24}$")
        self.assertEqual((call["function"]["name"], json.loads(call["function"]["arguments"])), ("get_weather", {"city": "Lisbon"}))
        self.assertEqual((choice["finish_reason"], choice["message"]["content"]), ("tool_calls", ""))

    def test_streamed_tool_calls_have_an_id_and_index_and_the_tool_calls_finish_reason(self):
        raw = self.post("/v1/chat/completions", {"messages": QUESTION, "tools": TOOLS, "stream": True})
        lines = [line[len("data: "):] for line in raw.splitlines() if line.startswith("data: ")]
        self.assertEqual(lines[-1], "[DONE]")
        choices = [json.loads(line)["choices"][0] for line in lines[:-1]]
        [call] = [call for choice in choices for call in choice["delta"]["tool_calls"]]
        self.assertRegex(str(call["id"]), r"^call_[0-9a-f]{24}$")
        self.assertEqual((call["index"], call["function"]["name"]), (0, "get_weather"))
        self.assertEqual(choices[-1]["finish_reason"], "tool_calls")

    def test_unstreamed_answers_end_without_the_end_of_sequence_token_in_every_protocol(self):
        chat = json.loads(self.post("/v1/chat/completions", {"messages": ANSWERED, "tools": TOOLS}))
        self.assertEqual((chat["choices"][0]["message"]["content"], chat["choices"][0]["finish_reason"]), ("It is 23 degrees.", "stop"))
        messages = json.loads(self.post("/v1/messages", {"max_tokens": 64, "messages": [{"role": "user", "content": "Weather?"}]}))
        self.assertEqual(messages["content"], [{"type": "text", "text": "It is 23 degrees."}])
        responses = json.loads(self.post("/v1/responses", {"input": "Weather?"}))
        self.assertEqual(responses["output"][0]["content"][0]["text"], "It is 23 degrees.")
        tool_turn = json.loads(self.post("/v1/messages", {"max_tokens": 64, "messages": [{"role": "user", "content": "Weather?"}],
                                                          "tools": [{"name": "get_weather", "input_schema": TOOLS[0]["function"]["parameters"]}]}))
        self.assertEqual([block["type"] for block in tool_turn["content"]], ["tool_use"])

    def test_other_runtimes_retain_openai_tool_argument_strings(self):
        self.server.config["runtime"] = "lmstudio"
        chat = json.loads(self.post("/v1/chat/completions", {"messages": ANSWERED, "tools": TOOLS}))
        self.assertEqual(chat["choices"][0]["message"]["content"], "Double-encoded tool history")

    def test_existing_native_console_receipts_decode_tool_history(self):
        self.post("/v1/chat/completions", {"messages": QUESTION})
        self.server.config.pop("runtime")
        self.server.config["worker_argv"][0] = "mlx_lm.server"
        chat = json.loads(self.post("/v1/chat/completions", {"messages": ANSWERED, "tools": TOOLS}))
        self.assertEqual(chat["choices"][0]["message"]["content"], "It is 23 degrees.")


class VisionToolParserTests(GatewayTestCase):
    def setUp(self):
        self.serve(PARSER_WORKER, "mlx-vlm", ADJACENT_TOOL_CALL_TEMPLATE)

    def sent_parser(self, body):
        return json.loads(json.loads(self.post("/v1/chat/completions", body))["choices"][0]["message"]["content"])

    def test_tool_requests_name_the_xml_parser_for_adjacent_tool_call_templates(self):
        self.assertEqual(self.sent_parser({"messages": QUESTION, "tools": TOOLS}), "qwen3_coder")
        messages = json.loads(self.post("/v1/messages", {"max_tokens": 64, "messages": [{"role": "user", "content": "Weather?"}],
                                                         "tools": [{"name": "get_weather", "input_schema": TOOLS[0]["function"]["parameters"]}]}))
        self.assertEqual(messages["content"], [{"type": "text", "text": '"qwen3_coder"'}])
        self.assertIsNone(self.sent_parser({"messages": QUESTION}))
        self.assertEqual(self.sent_parser({"messages": QUESTION, "tools": TOOLS, "tool_parser": "json_tools"}), "json_tools")
        self.server.config["runtime"] = "mlx_lm"
        self.assertIsNone(self.sent_parser({"messages": QUESTION, "tools": TOOLS}))


if __name__ == "__main__":
    unittest.main()
