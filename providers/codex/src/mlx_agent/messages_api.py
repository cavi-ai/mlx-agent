"""Anthropic Messages requests and responses over a local chat-completions worker.

Requests the worker cannot represent are refused rather than reduced.
"""
from __future__ import annotations

import json
import secrets

PARAMETERS = {"model", "messages", "max_tokens", "system", "tools", "tool_choice", "stream",
              "temperature", "top_p", "top_k", "stop_sequences", "metadata"}
SAMPLING = {"temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "stop_sequences": "stop"}
STOP_REASONS = {"stop": "end_turn", "length": "max_tokens"}
ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
               404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error"}


class MessagesRequestError(ValueError):
    """A Messages request the local worker cannot represent."""


class WorkerResponseError(ValueError):
    """A worker response that has no faithful Messages form."""


def error_body(status, message):
    return {"type": "error", "error": {"type": ERROR_TYPES.get(status, "api_error"), "message": message}}


def _text(content, where):
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(block, dict) and block.get("type") == "text"
                                         and isinstance(block.get("text"), str) for block in content):
        return "".join(block["text"] for block in content)
    raise MessagesRequestError("{} supports only text content.".format(where))


def _string(block, key):
    value = block.get(key)
    if not isinstance(value, str) or not value:
        raise MessagesRequestError("{} blocks need a {}.".format(block.get("type"), key))
    return value


def _messages(message):
    if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
        raise MessagesRequestError("Each message needs the user or assistant role.")
    role, content = message["role"], message.get("content")
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        raise MessagesRequestError("Message content must be a string or a list of blocks.")
    results, text, calls = [], [], []
    for block in content:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text" and isinstance(block.get("text"), str):
            text.append(block["text"])
        elif kind == "tool_use" and role == "assistant" and isinstance(block.get("input", {}), dict):
            calls.append({"id": _string(block, "id"), "type": "function",
                          "function": {"name": _string(block, "name"), "arguments": json.dumps(block.get("input", {}))}})
        elif kind == "tool_result" and role == "user":
            results.append({"role": "tool", "tool_call_id": _string(block, "tool_use_id"),
                            "content": _text(block.get("content", ""), "tool_result")})
        else:
            raise MessagesRequestError("Unsupported {} content block: {}.".format(role, kind))
    if calls:
        return results + [{"role": role, "content": "".join(text), "tool_calls": calls}]
    return results + ([{"role": role, "content": "".join(text)}] if text or not results else [])


def _tools(tools):
    if not isinstance(tools, list):
        raise MessagesRequestError("tools must be a list.")
    functions = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type", "custom") != "custom" or not isinstance(tool.get("input_schema"), dict):
            raise MessagesRequestError("Only custom tools with an input_schema are supported.")
        function = {"name": _string(tool, "name"), "parameters": tool["input_schema"]}
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        functions.append({"type": "function", "function": function})
    return functions


def to_chat_request(body):
    """Translate a Messages request into the worker's chat-completions request."""
    unknown = sorted(set(body) - PARAMETERS)
    if unknown:
        raise MessagesRequestError("Unsupported parameters: {}.".format(", ".join(unknown)))
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise MessagesRequestError("messages must be a non-empty list.")
    if "max_tokens" not in body:
        raise MessagesRequestError("max_tokens is required.")
    chat = [{"role": "system", "content": _text(body["system"], "system")}] if body.get("system") else []
    for message in body["messages"]:
        chat.extend(_messages(message))
    request = {"model": body.get("model"), "messages": chat, "max_tokens": body["max_tokens"]}
    choice = (body.get("tool_choice") or {"type": "auto"}).get("type")
    if choice not in ("auto", "none"):
        raise MessagesRequestError("Forced tool use is not supported by the local runtime.")
    if body.get("tools") and choice == "auto":
        request["tools"] = _tools(body["tools"])
    for key, target in SAMPLING.items():
        if key in body:
            request[target] = body[key]
    if body.get("stream"):
        request.update(stream=True, stream_options={"include_usage": True})
    return request


def tool_call(call):
    """Return a worker tool call's id (None when absent), name and parsed arguments."""
    function = call.get("function") if isinstance(call, dict) else None
    arguments = function.get("arguments") if isinstance(function, dict) else None
    try:
        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
    except ValueError:
        arguments = None
    if not isinstance(arguments, dict) or not isinstance(function.get("name"), str):
        raise WorkerResponseError("The model produced an invalid tool call.")
    identifier = call.get("id") if isinstance(call.get("id"), str) and call.get("id") else None
    return identifier, function["name"], arguments


def _tool_use(call):
    identifier, name, arguments = tool_call(call)
    return {"type": "tool_use", "id": identifier or "toolu_" + secrets.token_hex(12), "name": name, "input": arguments}


def _usage(usage):
    usage = usage if isinstance(usage, dict) else {}
    value = {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if isinstance(cached, int):
        value["cache_read_input_tokens"] = cached
    return value


def _stop_reason(finish_reason, tool_used):
    return "tool_use" if tool_used else STOP_REASONS.get(finish_reason, "end_turn")


def from_chat_response(response, model):
    """Translate a complete chat-completions response into a Messages response."""
    choice = response["choices"][0]
    message = choice.get("message") or {}
    calls = [_tool_use(call) for call in message.get("tool_calls") or []]
    text = [{"type": "text", "text": message["content"]}] if message.get("content") else []
    return {"id": "msg_" + secrets.token_hex(12), "type": "message", "role": "assistant", "model": model,
            "content": text + calls, "stop_reason": _stop_reason(choice.get("finish_reason"), bool(calls)),
            "stop_sequence": None, "usage": _usage(response.get("usage"))}


def sse(name, data):
    return "event: {}\ndata: {}\n\n".format(name, json.dumps(data)).encode()


class StreamTranslator:
    """Turns chat-completion chunks into Messages stream events, one content block at a time."""

    def __init__(self, model):
        self.model, self.index, self.text_open, self.tool_used = model, -1, False, False
        self.finish_reason, self.usage = None, None

    def start(self):
        return [sse("message_start", {"type": "message_start", "message": {
            "id": "msg_" + secrets.token_hex(12), "type": "message", "role": "assistant", "model": self.model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}}})]

    def _close_text(self):
        if not self.text_open:
            return []
        self.text_open = False
        return [sse("content_block_stop", {"type": "content_block_stop", "index": self.index})]

    def chunk(self, value):
        events = []
        if isinstance(value.get("usage"), dict):
            self.usage = value["usage"]
        for choice in value.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                if not self.text_open:
                    self.index, self.text_open = self.index + 1, True
                    events.append(sse("content_block_start", {"type": "content_block_start", "index": self.index,
                                                              "content_block": {"type": "text", "text": ""}}))
                events.append(sse("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                                          "delta": {"type": "text_delta", "text": delta["content"]}}))
            for call in delta.get("tool_calls") or []:
                block = _tool_use(call)
                events += self._close_text()
                self.index, self.tool_used = self.index + 1, True
                events += [
                    sse("content_block_start", {"type": "content_block_start", "index": self.index,
                                                "content_block": dict(block, input={})}),
                    sse("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                                "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}}),
                    sse("content_block_stop", {"type": "content_block_stop", "index": self.index})]
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        return events

    def finish(self):
        return self._close_text() + [
            sse("message_delta", {"type": "message_delta", "usage": _usage(self.usage),
                                  "delta": {"stop_reason": _stop_reason(self.finish_reason, self.tool_used), "stop_sequence": None}}),
            sse("message_stop", {"type": "message_stop"})]

    def fail(self, message):
        return [sse("error", error_body(502, message))]
