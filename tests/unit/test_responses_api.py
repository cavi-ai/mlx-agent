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
from .test_messages_api import WORKER

WEATHER = {"type": "function", "name": "get_weather", "description": "Current weather for a city.", "strict": False,
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
# Fields Codex sends with every request to a local model.
CODEX = {"stream": False, "tool_choice": "auto", "parallel_tool_calls": False, "reasoning": {}, "store": False,
         "include": ["reasoning.encrypted_content"], "prompt_cache_key": "conversation-1",
         "client_metadata": {"session_id": "conversation-1"}}


class ResponsesGatewayTests(unittest.TestCase):
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

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            connection.request(method, path, None if body is None else json.dumps(body),
                               {"Content-Type": "application/json", "Authorization": "Bearer local"})
            response = connection.getresponse()
            return response.status, response.getheader("Content-Type"), response.read().decode()
        finally:
            connection.close()

    def post(self, body):
        return self.request("POST", "/v1/responses", body)

    def sent(self):
        return json.loads(self.trace.read_text().splitlines()[-1])

    @unittest.skipUnless(importlib.util.find_spec("cavi_runtime"), "requires requirements-dev.txt")
    def test_codex_client_completes_a_background_run_and_reads_it_back(self):
        from cavi_runtime import create_runtime_client

        async def run():
            async with create_runtime_client("codex-responses", api_key="local", base_url="http://127.0.0.1:{}".format(self.port),
                                             default_model=str(self.model)) as client:
                started = await client.start_run({"input": "Say hello", "instructions": "Answer briefly."})
                return started, await client.get_run(started["run_id"])

        started, read = asyncio.run(run())
        self.assertEqual((started["status"], started["output"]), ("completed", "Hello from MLX"))
        self.assertEqual((started["tokens"]["inputTokens"], started["tokens"]["outputTokens"]), (12, 4))
        self.assertEqual((read["run_id"], read["status"], read["output"]), (started["run_id"], "completed", "Hello from MLX"))
        sent = self.sent()
        self.assertEqual(sent["messages"], [{"role": "system", "content": "Answer briefly."},
                                            {"role": "user", "content": "Say hello"}])
        self.assertEqual((sent["model"], sent["max_tokens"], sent.get("stream", False)), (str(self.model), 8192, False))

    def test_function_call_round_trip_maps_calls_and_outputs(self):
        question = [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Use tools."}]},
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Weather in Paris?"}]}]
        status, _, text = self.post(dict(CODEX, model=str(self.model), input=question, tools=[WEATHER]))
        self.assertEqual(status, 200, text)
        response = json.loads(text)
        [call] = response["output"]
        self.assertEqual((response["object"], response["status"], response["store"]), ("response", "completed", False))
        self.assertEqual((call["type"], call["name"], json.loads(call["arguments"]), call["status"]),
                         ("function_call", "get_weather", {"city": "Paris"}, "completed"))
        self.assertTrue(call["call_id"].startswith("call_"))
        self.assertEqual(response["usage"], {"input_tokens": 20, "output_tokens": 9, "total_tokens": 29})
        sent = self.sent()
        self.assertEqual(sent["messages"], [{"role": "system", "content": "Use tools."},
                                            {"role": "user", "content": "Weather in Paris?"}])
        self.assertEqual(sent["tools"], [{"type": "function", "function": {
            "name": "get_weather", "parameters": WEATHER["parameters"], "description": "Current weather for a city."}}])
        status, _, text = self.request("GET", "/v1/responses/" + response["id"])
        self.assertEqual((status, json.loads(text)["error"]["type"]), (404, "invalid_request_error"), text)

        result = {"type": "function_call_output", "call_id": call["call_id"], "output": "18C and sunny"}
        status, _, text = self.post(dict(CODEX, model=str(self.model), input=question + [call, result], tools=[WEATHER]))
        self.assertEqual(status, 200, text)
        [message] = json.loads(text)["output"]
        self.assertEqual((message["type"], message["role"], message["content"][0]["text"]), ("message", "assistant", "It is sunny."))
        self.assertEqual(self.sent()["messages"][2:], [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": call["call_id"], "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
            {"role": "tool", "tool_call_id": call["call_id"], "content": "18C and sunny"}])

    def test_streaming_emits_responses_events_and_stores_the_result(self):
        status, content_type, text = self.post({"model": str(self.model), "stream": True, "input": "Hi", "tools": [WEATHER]})
        self.assertEqual((status, content_type), (200, "text/event-stream"), text)
        events = []
        for frame in text.strip().split("\n\n"):
            name, data = frame.split("\n")
            events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
        self.assertEqual([name for name, _ in events], [
            "response.created", "response.in_progress", "response.output_item.added", "response.content_part.added",
            "response.output_text.delta", "response.output_text.delta", "response.output_text.done",
            "response.content_part.done", "response.output_item.done", "response.output_item.added",
            "response.function_call_arguments.delta", "response.function_call_arguments.done",
            "response.output_item.done", "response.completed"])
        self.assertEqual([data["type"] for name, data in events], [name for name, _ in events])
        self.assertEqual([data["sequence_number"] for _, data in events], list(range(len(events))))
        self.assertEqual([data["delta"] for _, data in events[4:6]], ["Hel", "lo"])
        self.assertEqual(events[8][1]["item"]["content"][0]["text"], "Hello")
        call = events[12][1]["item"]
        self.assertEqual((events[12][1]["output_index"], call["name"], json.loads(call["arguments"])), (1, "get_weather", {"city": "Paris"}))
        completed = events[-1][1]["response"]
        self.assertEqual((completed["status"], [item["type"] for item in completed["output"]]), ("completed", ["message", "function_call"]))
        self.assertEqual(completed["usage"], {"input_tokens": 15, "output_tokens": 3, "total_tokens": 18})
        sent = self.sent()
        self.assertEqual((sent["stream"], sent["stream_options"]), (True, {"include_usage": True}))
        status, _, text = self.request("GET", "/v1/responses/" + completed["id"])
        self.assertEqual((status, json.loads(text)), (200, completed))

    def test_unsupported_requests_return_openai_errors_without_loading(self):
        image = {"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}
        for body in ({"model": str(self.model), "input": [image]},
                     {"model": str(self.model), "input": "Hi", "tools": [{"type": "web_search"}]},
                     {"model": str(self.model), "input": "Hi", "reasoning": {"effort": "high"}},
                     {"model": str(self.model), "input": "Hi", "previous_response_id": "resp_1"},
                     {"model": "gpt-5-codex", "input": "Hi"}):
            status, _, text = self.post(body)
            error = json.loads(text)["error"]
            self.assertEqual((status, error["type"]), (400, "invalid_request_error"), text)
        self.assertFalse(self.trace.exists())


if __name__ == "__main__":
    unittest.main()
