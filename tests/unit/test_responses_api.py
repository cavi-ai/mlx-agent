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
from mlx_agent import responses_api
from mlx_agent.splash_runtime import supports_reasoning_effort
from mlx_agent.probe_assets import vision_probe_image_base64
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
        self.server.controller.memory_probe = lambda: 100_000_000_000
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

    def test_namespaced_calls_round_trip_and_stream(self):
        worker = self.model.parent / "worker.py"
        worker.write_text(WORKER.replace("'name': 'get_weather'", "'name': 'weather__get_weather'"))
        namespace = {"type": "namespace", "name": "weather", "description": "Weather tools", "tools": [WEATHER]}
        body = {"model": str(self.model), "input": "Weather?", "tools": [namespace]}
        status, _, text = self.post(body)
        self.assertEqual(status, 200, text)
        [call] = json.loads(text)["output"]
        self.assertEqual((call["namespace"], call["name"]), ("weather", "get_weather"))
        self.assertEqual(self.sent()["tools"][0]["function"]["name"], "weather__get_weather")
        result = {"type": "function_call_output", "call_id": call["call_id"], "output": "sunny"}
        status, _, text = self.post(dict(body, input=[call, result]))
        self.assertEqual(status, 200, text)
        history = self.sent()["messages"]
        self.assertEqual(history[0]["tool_calls"][0]["function"]["name"], "weather__get_weather")
        self.assertEqual(history[1]["tool_call_id"], call["call_id"])
        status, _, text = self.post(dict(body, stream=True))
        self.assertEqual(status, 200, text)
        frames = [json.loads(frame.split("\n")[1].removeprefix("data: ")) for frame in text.strip().split("\n\n")]
        items = [f["item"] for f in frames if f["type"] in ("response.output_item.added", "response.output_item.done")
                 and f["item"]["type"] == "function_call"]
        self.assertEqual(len(items), 2)
        self.assertTrue(all((i["namespace"], i["name"]) == ("weather", "get_weather") for i in items))
        self.assertEqual(items[0]["call_id"], items[1]["call_id"])
        self.assertEqual(frames[-1]["response"]["output"][-1]["namespace"], "weather")

    def test_indexed_stream_assembles_interleaved_namespaced_calls(self):
        stream = responses_api.ResponseStream("local", {}, {"weather__lookup": ("weather", "lookup")})
        def chunk(*calls, reason=None):
            return stream.chunk({"choices": [{"delta": {"tool_calls": list(calls)}, "finish_reason": reason}]})
        def decoded(events):
            return [json.loads(event.decode().split("\n")[1].removeprefix("data: ")) for event in events]
        first = decoded(chunk({"index": 0, "id": "call_a", "type": "function", "function": {"name": "weather__lookup"}},
                              {"index": 1, "id": "call_b", "type": "function", "function": {"name": "plain"}}))
        self.assertEqual([(e["item"]["call_id"], e["item"]["status"]) for e in first],
                         [("call_a", "in_progress"), ("call_b", "in_progress")])
        self.assertEqual((first[0]["item"]["namespace"], first[0]["item"]["name"]), ("weather", "lookup"))
        self.assertEqual(chunk({"index": 0, "function": {"arguments": '{"city":'}},
                               {"index": 1, "function": {"arguments": '{"n":1}'}},
                               {"index": 0, "function": {"arguments": '"Paris"}'}}), [])
        chunk(reason="tool_calls")
        events = decoded(stream.finish())
        done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
        self.assertEqual([(i["call_id"], json.loads(i["arguments"])) for i in done],
                         [("call_a", {"city": "Paris"}), ("call_b", {"n": 1})])
        self.assertEqual((done[0]["namespace"], done[0]["name"]), ("weather", "lookup"))
        self.assertEqual(events[-1]["response"]["output"], done)

    def test_indexed_stream_accepts_complete_object_and_bounds_shared_bytes(self):
        def chunk(stream, call, reason=None):
            return stream.chunk({"choices": [{"delta": {"tool_calls": [call] if call else []}, "finish_reason": reason}]})
        header = {"index": 1000, "id": "call_a", "type": "function",
                  "function": {"name": "lookup", "arguments": {"city": "Paris"}}}
        stream = responses_api.ResponseStream("local", {})
        chunk(stream, header)
        chunk(stream, None, "tool_calls")
        stream.finish()
        self.assertEqual(json.loads(stream.response["output"][0]["arguments"]), {"city": "Paris"})
        stream = responses_api.ResponseStream("local", {})
        chunk(stream, header)
        with self.assertRaises(responses_api.WorkerResponseError):
            chunk(stream, {"index": 1000, "function": {"arguments": " "}})
        stream = responses_api.ResponseStream("local", {})
        chunk(stream, {"index": 0, "function": {"name": "first", "arguments": "{"}})
        stream.call_argument_bytes = 16 * 1024 * 1024 - 1
        with self.assertRaises(responses_api.WorkerResponseError):
            chunk(stream, {"index": 1, "function": {"name": "second", "arguments": "{}"}})

    def test_indexed_stream_rejects_invalid_or_incomplete_calls(self):
        def chunk(stream, call, reason=None):
            return stream.chunk({"choices": [{"delta": {"tool_calls": [call] if call else []}, "finish_reason": reason}]})
        header = {"index": 0, "id": "call_a", "type": "function", "function": {"name": "lookup"}}
        with self.assertRaises(responses_api.WorkerResponseError):
            chunk(responses_api.ResponseStream("local", {}), {"index": 0, "function": {"arguments": "{}"}})
        for fragment in ({"index": 0, "id": "changed", "function": {"arguments": "{}"}},
                         {"index": 0, "function": {"name": "changed"}}):
            stream = responses_api.ResponseStream("local", {})
            chunk(stream, header)
            with self.assertRaises(responses_api.WorkerResponseError): chunk(stream, fragment)
        for arguments, reason in (("{", "tool_calls"), ("[]", "tool_calls"), ("{}", None), ("{}", "length")):
            stream = responses_api.ResponseStream("local", {})
            chunk(stream, header)
            chunk(stream, {"index": 0, "function": {"arguments": arguments}}, reason)
            with self.assertRaises(responses_api.WorkerResponseError): stream.finish()

    def test_none_choice_ignores_inactive_unsupported_tools(self):
        body = {"model": str(self.model), "input": "Hi", "tool_choice": "none",
                "tools": [{"type": "web_search"}, {"type": "namespace", "name": "invalid.name", "tools": []}]}
        status, _, text = self.post(body)
        self.assertEqual(status, 200, text)
        self.assertNotIn("tools", self.sent())

    def test_namespace_validation_precedes_worker_load(self):
        def ns(name, tools): return {"type": "namespace", "name": name, "tools": tools}
        for tools in ([ns("bad.name", [WEATHER])], [ns("ok", [])], [ns("ok", [{"type": "web_search"}])],
                      [ns("ok", [dict(WEATHER, strict=True)])],
                      [ns("a", [dict(WEATHER, name="b__c")]), ns("a__b", [dict(WEATHER, name="c")])],
                      [ns("a", [dict(WEATHER, name="b")]), dict(WEATHER, name="a__b")],
                      [ns("a", [dict(WEATHER, name="bad.name")])]):
            with self.subTest(tools=tools):
                status, _, text = self.post({"model": str(self.model), "input": "Hi", "tools": tools})
                self.assertEqual(status, 400, text)
        self.assertFalse(self.trace.exists())

    def test_splash_reasoning_effort_translation(self):
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            with self.subTest(effort=effort):
                request = responses_api.to_chat_request({"input": "Hi", "reasoning": {"effort": effort, "summary": None}},
                                                        allow_reasoning_effort=True)
                self.assertEqual(request["reasoning_effort"], effort)
        for reasoning in (None, {}, {"effort": None}, {"effort": None, "summary": None}):
            request = responses_api.to_chat_request({"input": "Hi", "reasoning": reasoning}, allow_reasoning_effort=True)
            self.assertNotIn("reasoning_effort", request)
        for reasoning in ({"effort": "unknown"}, {"effort": True}, {"summary": "auto"},
                          {"context": "other"}, {"effort": "high", "summary": "concise"}, []):
            with self.subTest(reasoning=reasoning), self.assertRaises(responses_api.ResponsesRequestError):
                responses_api.to_chat_request({"input": "Hi", "reasoning": reasoning}, allow_reasoning_effort=True)
        self.assertFalse(supports_reasoning_effort("1.6.0"))
        self.assertFalse(supports_reasoning_effort("1.6.1-rc1"))
        self.assertFalse(supports_reasoning_effort(None))
        self.assertTrue(supports_reasoning_effort("1.6.1"))

    def test_effort_route_is_version_gated_before_loading(self):
        self.addCleanup(self.server.config.update, runtime="mlx_lm")
        body = {"model": str(self.model), "input": "Hi", "reasoning": {"effort": "medium"}}
        for runtime, version in (("mlx_lm", "1.6.1"), ("splash", "1.6.0"), ("splash", None)):
            with self.subTest(runtime=runtime, version=version):
                self.server.config.update(runtime=runtime, splash_version=version)
                self.assertEqual(self.post(body)[0], 400)
                self.assertFalse(self.trace.exists())
        self.server.config.update(runtime="mlx_lm")
        self.assertEqual(self.post({"model": str(self.model), "input": "warm"})[0], 200)
        self.server.config.update(runtime="splash", splash_version="1.6.1", splash_served_model=str(self.model))
        self.assertEqual(self.post(body)[0], 200)
        self.assertEqual(self.sent()["reasoning_effort"], "medium")

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

    def test_vision_worker_receives_images_in_streamed_and_complete_requests(self):
        self.server.config["runtime"] = "mlx-vlm"
        image = "data:image/png;base64,"+vision_probe_image_base64()
        for stream in (False, True):
            with self.subTest(stream=stream):
                status, _, text = self.post({"model": str(self.model), "max_output_tokens": 64, "stream": stream,
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": "Read this."},
                        {"type": "input_image", "image_url": image, "detail": "auto"}]}]})
                self.assertEqual(status, 200, text)
                self.assertEqual(self.sent()["messages"], [{"role": "user", "content": [
                    {"type": "text", "text": "Read this."},
                    {"type": "image_url", "image_url": {"url": image, "detail": "auto"}}]}])

    def test_invalid_vision_source_is_rejected_before_loading(self):
        self.server.config["runtime"] = "mlx-vlm"
        for image in ({"type": "input_image", "file_id": "file_1"},
                      {"type": "input_image", "image_url": "data:image/png;base64,!"},
                      {"type": "input_image", "image_url": "\x00https://example.invalid/i.png"},
                      {"type": "input_image", "image_url": "https://example.invalid/i.png", "detail": []}):
            with self.subTest(image=image):
                status, _, text = self.post({"model": str(self.model),
                    "input": [{"role": "user", "content": [image]}]})
                self.assertEqual(status, 400, text)
                self.assertEqual(json.loads(text)["error"]["type"], "invalid_request_error")
        self.assertFalse(self.trace.exists())


if __name__ == "__main__":
    unittest.main()
