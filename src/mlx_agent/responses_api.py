"""OpenAI Responses requests and responses over a local chat-completions worker.

Requests the worker cannot represent are refused rather than reduced.
"""
from __future__ import annotations

import json
import secrets
import time

from mlx_agent.messages_api import sse, tool_call

PARAMETERS = {"model", "input", "instructions", "max_output_tokens", "tools", "tool_choice", "parallel_tool_calls",
              "stream", "temperature", "top_p", "store", "background", "metadata", "include", "reasoning", "text",
              "prompt_cache_key", "client_metadata"}
# Encrypted reasoning is never produced locally, so asking for it is satisfied by its absence.
INCLUDES = {"reasoning.encrypted_content"}
ROLES = {"user": "user", "assistant": "assistant", "system": "system", "developer": "system"}
TEXT_PARTS = {"input_text", "output_text"}


class ResponsesRequestError(ValueError):
    """A Responses request the local worker cannot represent."""


def error_body(status, code, message):
    return {"error": {"message": message, "type": "invalid_request_error" if status < 500 else "server_error",
                      "param": None, "code": code}}


def _identifier(prefix):
    return prefix + secrets.token_hex(12)


def _text(content, where):
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(part, dict) and part.get("type") in TEXT_PARTS
                                         and isinstance(part.get("text"), str) for part in content):
        return "".join(part["text"] for part in content)
    raise ResponsesRequestError("{} supports only text content.".format(where))


def _string(item, key):
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ResponsesRequestError("{} items need a {}.".format(item.get("type", "message"), key))
    return value


def _call(item):
    arguments = _string(item, "arguments")
    try:
        parsed = json.loads(arguments)
    except ValueError:
        parsed = None
    if not isinstance(parsed, dict):
        raise ResponsesRequestError("function_call arguments must be a JSON object.")
    return {"id": _string(item, "call_id"), "type": "function",
            "function": {"name": _string(item, "name"), "arguments": arguments}}


def _input(items):
    if isinstance(items, str):
        return [{"role": "user", "content": items}]
    if not isinstance(items, list) or not items:
        raise ResponsesRequestError("input must be a string or a non-empty list of items.")
    chat = []
    for item in items:
        kind = item.get("type", "message") if isinstance(item, dict) else None
        if kind == "message":
            if item.get("role") not in ROLES:
                raise ResponsesRequestError("Each message needs the user, assistant, system or developer role.")
            chat.append({"role": ROLES[item["role"]], "content": _text(item.get("content"), "message")})
        elif kind == "function_call":
            # Chat completions carry an assistant turn's text and its calls in one message.
            if chat and chat[-1]["role"] == "assistant":
                chat[-1].setdefault("tool_calls", []).append(_call(item))
            else:
                chat.append({"role": "assistant", "content": "", "tool_calls": [_call(item)]})
        elif kind == "function_call_output":
            chat.append({"role": "tool", "tool_call_id": _string(item, "call_id"),
                         "content": _text(item.get("output"), "function_call_output")})
        else:
            raise ResponsesRequestError("Unsupported input item: {}.".format(kind))
    return chat


def _tools(tools):
    if not isinstance(tools, list):
        raise ResponsesRequestError("tools must be a list.")
    functions = []
    for tool in tools:
        kind = tool.get("type") if isinstance(tool, dict) else None
        if kind != "function":
            raise ResponsesRequestError("Only function tools are supported, not {}.".format(kind))
        if tool.get("strict"):
            raise ResponsesRequestError("Strict function schemas are not enforced by the local runtime.")
        parameters = tool.get("parameters") or {"type": "object", "properties": {}}
        if not isinstance(parameters, dict):
            raise ResponsesRequestError("Function parameters must be a JSON schema object.")
        function = {"name": _string(tool, "name"), "parameters": parameters}
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        functions.append({"type": "function", "function": function})
    return functions


def to_chat_request(body):
    """Translate a Responses request into the worker's chat-completions request."""
    unknown = sorted(set(body) - PARAMETERS)
    if unknown:
        raise ResponsesRequestError("Unsupported parameters: {}.".format(", ".join(unknown)))
    if body.get("reasoning") not in (None, {}):
        raise ResponsesRequestError("Reasoning controls are not supported by the local runtime.")
    include = body.get("include") or []
    if not isinstance(include, list) or not all(isinstance(value, str) and value in INCLUDES for value in include):
        raise ResponsesRequestError("Unsupported include values.")
    text = body.get("text") or {}
    output = text.get("format") or {"type": "text"} if isinstance(text, dict) else None
    if not isinstance(output, dict) or set(text) - {"format"} or output.get("type") != "text":
        raise ResponsesRequestError("Only plain text output is supported by the local runtime.")
    choice = body.get("tool_choice", "auto")
    if choice not in ("auto", "none"):
        raise ResponsesRequestError("Forced tool use is not supported by the local runtime.")
    chat = [{"role": "system", "content": _text(body["instructions"], "instructions")}] if body.get("instructions") else []
    request = {"messages": chat + _input(body.get("input"))}
    if "model" in body:
        request["model"] = body["model"]
    if body.get("max_output_tokens") is not None:
        request["max_tokens"] = body["max_output_tokens"]
    if body.get("tools") and choice == "auto":
        request["tools"] = _tools(body["tools"])
    for key in ("temperature", "top_p"):
        if key in body:
            request[key] = body[key]
    if body.get("stream"):
        request.update(stream=True, stream_options={"include_usage": True})
    return request


def envelope(body):
    """Request fields a Responses object reports back. Background requests complete before returning."""
    return {"background": bool(body.get("background")), "instructions": body.get("instructions"),
            "max_output_tokens": body.get("max_output_tokens"), "metadata": body.get("metadata") or {},
            "parallel_tool_calls": body.get("parallel_tool_calls", True), "previous_response_id": None,
            "store": body.get("store", True) is not False, "temperature": body.get("temperature"),
            "top_p": body.get("top_p"), "tool_choice": body.get("tool_choice", "auto"), "tools": body.get("tools") or []}


def _response(model, fields, status, output):
    return dict(fields, id=_identifier("resp_"), object="response", created_at=int(time.time()), model=model,
                status=status, output=output, usage=None, error=None, incomplete_details=None)


def _usage(usage):
    if not isinstance(usage, dict):
        return None
    value = {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}
    value["total_tokens"] = usage.get("total_tokens", value["input_tokens"] + value["output_tokens"])
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if isinstance(cached, int):
        value["input_tokens_details"] = {"cached_tokens": cached}
    return value


def _outcome(finish_reason):
    if finish_reason == "length":
        return {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
    return {"status": "completed"}


def _message(text, status="completed"):
    content = [] if text is None else [{"type": "output_text", "text": text, "annotations": []}]
    return {"type": "message", "id": _identifier("msg_"), "status": status, "role": "assistant", "content": content}


def _function_call(call):
    identifier, name, arguments = tool_call(call)
    return {"type": "function_call", "id": _identifier("fc_"), "call_id": identifier or _identifier("call_"),
            "name": name, "arguments": json.dumps(arguments), "status": "completed"}


def from_chat_response(response, model, fields):
    """Translate a complete chat-completions response into a Responses object."""
    choice = response["choices"][0]
    message = choice.get("message") or {}
    output = [_message(message["content"])] if message.get("content") else []
    output += [_function_call(call) for call in message.get("tool_calls") or []]
    result = _response(model, fields, "completed", output)
    result.update(_outcome(choice.get("finish_reason")), usage=_usage(response.get("usage")))
    return result


class ResponseStream:
    """Turns chat-completion chunks into Responses stream events, one output item at a time."""

    def __init__(self, model, fields):
        self.response = _response(model, fields, "in_progress", [])
        self.sequence, self.text, self.finish_reason, self.usage = 0, None, None, None

    def _event(self, kind, **data):
        self.sequence += 1
        return sse(kind, dict(type=kind, sequence_number=self.sequence - 1, **data))

    def _add(self, item):
        self.response["output"].append(item)
        return self._event("response.output_item.added", output_index=len(self.response["output"]) - 1, item=item)

    def start(self):
        return [self._event("response.created", response=self.response),
                self._event("response.in_progress", response=self.response)]

    def _close_text(self):
        if self.text is None:
            return []
        index, self.text = self.text, None
        item = self.response["output"][index]
        item["status"] = "completed"
        where = {"item_id": item["id"], "output_index": index, "content_index": 0}
        return [self._event("response.output_text.done", text=item["content"][0]["text"], **where),
                self._event("response.content_part.done", part=item["content"][0], **where),
                self._event("response.output_item.done", output_index=index, item=item)]

    def chunk(self, value):
        events = []
        if isinstance(value.get("usage"), dict):
            self.usage = value["usage"]
        for choice in value.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                if self.text is None:
                    item = _message(None, "in_progress")
                    events.append(self._add(item))
                    self.text = len(self.response["output"]) - 1
                    part = {"type": "output_text", "text": "", "annotations": []}
                    events.append(self._event("response.content_part.added", item_id=item["id"],
                                              output_index=self.text, content_index=0, part=part))
                    item["content"].append(part)
                item = self.response["output"][self.text]
                item["content"][0]["text"] += delta["content"]
                events.append(self._event("response.output_text.delta", item_id=item["id"], output_index=self.text,
                                          content_index=0, delta=delta["content"]))
            for call in delta.get("tool_calls") or []:
                item = _function_call(call)
                events += self._close_text()
                events.append(self._add(dict(item, arguments="", status="in_progress")))
                index = len(self.response["output"]) - 1
                self.response["output"][index] = item
                events += [
                    self._event("response.function_call_arguments.delta", item_id=item["id"], output_index=index,
                                delta=item["arguments"]),
                    self._event("response.function_call_arguments.done", item_id=item["id"], output_index=index,
                                arguments=item["arguments"]),
                    self._event("response.output_item.done", output_index=index, item=item)]
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        return events

    def finish(self):
        events = self._close_text()
        self.response.update(_outcome(self.finish_reason), usage=_usage(self.usage))
        return events + [self._event("response." + self.response["status"], response=self.response)]

    def fail(self, message):
        self.response.update(status="failed", error={"code": "server_error", "message": message})
        return [self._event("response.failed", response=self.response)]
