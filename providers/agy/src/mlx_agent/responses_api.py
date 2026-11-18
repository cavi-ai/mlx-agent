"""OpenAI Responses requests and responses over a local chat-completions worker.

Requests the worker cannot represent are refused rather than reduced.
"""
from __future__ import annotations

import json
import re
import secrets
import time

from mlx_agent.messages_api import WorkerResponseError, sse, tool_call
from mlx_agent.image_inputs import responses_image

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


def _call(item, aliases):
    arguments = _string(item, "arguments")
    try:
        parsed = json.loads(arguments)
    except ValueError:
        parsed = None
    if not isinstance(parsed, dict):
        raise ResponsesRequestError("function_call arguments must be a JSON object.")
    name = _string(item, "name")
    if item.get("namespace") is not None:
        namespace = _namespace_part(item["namespace"])
        name = _alias(namespace, _namespace_part(name))
        if name in aliases and aliases[name] != (namespace, item["name"]):
            raise ResponsesRequestError("Function call namespace conflicts with a tool.")
    return {"id": _string(item, "call_id"), "type": "function",
            "function": {"name": name, "arguments": arguments}}


def _content(content, allow_images):
    if not isinstance(content, list) or not any(isinstance(p, dict) and p.get("type") == "input_image" for p in content):
        return _text(content, "message")
    if not allow_images:
        raise ResponsesRequestError("Images require a user message and a vision runtime.")
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in TEXT_PARTS and isinstance(part.get("text"), str):
            parts.append({"type": "text", "text": part["text"]})
        elif isinstance(part, dict) and part.get("type") == "input_image":
            try:
                parts.append(responses_image(part))
            except ValueError as error:
                raise ResponsesRequestError(str(error)) from error
        else:
            raise ResponsesRequestError("Unsupported message content.")
    return parts


def _input(items, allow_images=False, aliases=None):
    aliases = aliases or {}
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
            chat.append({"role": ROLES[item["role"]],
                         "content": _content(item.get("content"), allow_images and item["role"] == "user")})
        elif kind == "function_call":
            # Chat completions carry an assistant turn's text and its calls in one message.
            if chat and chat[-1]["role"] == "assistant":
                chat[-1].setdefault("tool_calls", []).append(_call(item, aliases))
            else:
                chat.append({"role": "assistant", "content": "", "tool_calls": [_call(item, aliases)]})
        elif kind == "function_call_output":
            chat.append({"role": "tool", "tool_call_id": _string(item, "call_id"),
                         "content": _text(item.get("output"), "function_call_output")})
        else:
            raise ResponsesRequestError("Unsupported input item: {}.".format(kind))
    return chat


def _namespace_part(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise ResponsesRequestError("Invalid function namespace or name.")
    return value


def _alias(namespace, name):
    value = namespace + "__" + name
    if len(value) > 64:
        raise ResponsesRequestError("Namespaced function name is too long.")
    return value


def _tools(tools):
    if not isinstance(tools, list):
        raise ResponsesRequestError("tools must be a list.")
    functions, aliases = [], {}

    def add(tool, namespace=None):
        kind = tool.get("type") if isinstance(tool, dict) else None
        if kind != "function":
            raise ResponsesRequestError("Only function tools are supported, not {}.".format(kind))
        if tool.get("strict"):
            raise ResponsesRequestError("Strict function schemas are not enforced by the local runtime.")
        parameters = tool.get("parameters") or {"type": "object", "properties": {}}
        if not isinstance(parameters, dict):
            raise ResponsesRequestError("Function parameters must be a JSON schema object.")
        name = _string(tool, "name")
        alias = _alias(namespace, _namespace_part(name)) if namespace is not None else name
        if alias in aliases:
            raise ResponsesRequestError("Function tool names collide.")
        aliases[alias] = (namespace, name)
        function = {"name": alias, "parameters": parameters}
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        functions.append({"type": "function", "function": function})

    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "namespace":
            namespace = _namespace_part(tool.get("name"))
            nested = tool.get("tools")
            if not isinstance(nested, list) or not nested:
                raise ResponsesRequestError("Namespace tools must be a non-empty list.")
            for child in nested:
                add(child, namespace)
        else:
            add(tool)
    return functions, aliases


def to_chat_request(body, *, allow_images=False, allow_reasoning_effort=False, aliases=None):
    """Translate a Responses request into the worker's chat-completions request."""
    unknown = sorted(set(body) - PARAMETERS)
    if unknown:
        raise ResponsesRequestError("Unsupported parameters: {}.".format(", ".join(unknown)))
    reasoning = body.get("reasoning")
    effort = None
    if reasoning not in (None, {}):
        if (not allow_reasoning_effort or not isinstance(reasoning, dict)
                or set(reasoning) - {"effort", "summary"} or reasoning.get("summary") is not None):
            raise ResponsesRequestError("Reasoning controls are not supported by the local runtime.")
        effort = reasoning.get("effort")
        if effort is not None and (not isinstance(effort, str) or effort not in
                                   {"none", "minimal", "low", "medium", "high", "xhigh", "max"}):
            raise ResponsesRequestError("Unsupported reasoning effort.")
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
    functions, resolved_aliases = _tools(body.get("tools") or []) if choice == "auto" else ([], {})
    if aliases is not None:
        aliases.update(resolved_aliases)
    request = {"messages": chat + _input(body.get("input"), allow_images, resolved_aliases)}
    if effort is not None:
        request["reasoning_effort"] = effort
    if "model" in body:
        request["model"] = body["model"]
    if body.get("max_output_tokens") is not None:
        request["max_tokens"] = body["max_output_tokens"]
    if body.get("tools") and choice == "auto":
        request["tools"] = functions
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


def _function_call(call, aliases=None):
    identifier, name, arguments = tool_call(call)
    item = {"type": "function_call", "id": _identifier("fc_"), "call_id": identifier or _identifier("call_"),
            "name": name, "arguments": json.dumps(arguments), "status": "completed"}
    if name in (aliases or {}) and aliases[name][0] is not None:
        item["namespace"], item["name"] = aliases[name]
    return item


def from_chat_response(response, model, fields, aliases=None):
    """Translate a complete chat-completions response into a Responses object."""
    choice = response["choices"][0]
    message = choice.get("message") or {}
    output = [_message(message["content"])] if message.get("content") else []
    output += [_function_call(call, aliases) for call in message.get("tool_calls") or []]
    result = _response(model, fields, "completed", output)
    result.update(_outcome(choice.get("finish_reason")), usage=_usage(response.get("usage")))
    return result


class ResponseStream:
    """Turns chat-completion chunks into Responses stream events, one output item at a time."""

    def __init__(self, model, fields, aliases=None):
        self.aliases = aliases or {}
        self.response = _response(model, fields, "in_progress", [])
        self.sequence, self.text, self.finish_reason, self.usage = 0, None, None, None
        self.calls = {}
        self.call_argument_bytes = 0

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

    def _indexed_call(self, call):
        index = call.get("index")
        function = call.get("function")
        if (type(index) is not int or index < 0 or not isinstance(function, dict)
                or call.get("type") not in (None, "function")):
            raise WorkerResponseError("The model produced an invalid tool call.")
        name, identifier, arguments = function.get("name"), call.get("id"), function.get("arguments")
        if (name is not None and (not isinstance(name, str) or not name)
                or identifier is not None and (not isinstance(identifier, str) or not identifier)
                or arguments is not None and not isinstance(arguments, (str, dict))):
            raise WorkerResponseError("The model produced an invalid tool call.")
        events = []
        current = self.calls.get(index)
        if current is None:
            if not name or len(self.calls) >= 256:
                raise WorkerResponseError("The model produced an invalid tool call.")
            item = {"type": "function_call", "id": _identifier("fc_"),
                    "call_id": identifier or _identifier("call_"), "name": name,
                    "arguments": "", "status": "in_progress"}
            if name in self.aliases and self.aliases[name][0] is not None:
                item["namespace"], item["name"] = self.aliases[name]
            events += self._close_text()
            events.append(self._add(item))
            current = {"name": name, "worker_id": identifier, "output_index": len(self.response["output"]) - 1,
                       "fragments": [], "object_argument": False}
            self.calls[index] = current
        elif (name is not None and name != current["name"] or
              identifier is not None and identifier != current["worker_id"]):
            raise WorkerResponseError("The model produced an invalid tool call.")
        if arguments is not None:
            if isinstance(arguments, dict):
                if current["fragments"] or current["object_argument"]:
                    raise WorkerResponseError("The model produced an invalid tool call.")
                try:
                    fragment = json.dumps(arguments)
                except (TypeError, ValueError):
                    raise WorkerResponseError("The model produced an invalid tool call.") from None
                current["object_argument"] = True
            else:
                if current["object_argument"]:
                    raise WorkerResponseError("The model produced an invalid tool call.")
                fragment = arguments
            self.call_argument_bytes += len(fragment.encode("utf-8"))
            if self.call_argument_bytes > 16 * 1024 * 1024:
                raise WorkerResponseError("The model produced an invalid tool call.")
            current["fragments"].append(fragment)
        return events

    def _finish_calls(self):
        if self.calls and self.finish_reason in (None, "length"):
            raise WorkerResponseError("The model produced an invalid tool call.")
        # Validate every call before completing any of them.
        for current in self.calls.values():
            try:
                current["arguments"] = "".join(current.pop("fragments"))
                arguments = json.loads(current["arguments"])
            except (ValueError, TypeError):
                raise WorkerResponseError("The model produced an invalid tool call.") from None
            if not isinstance(arguments, dict):
                raise WorkerResponseError("The model produced an invalid tool call.")
        events = []
        for current in self.calls.values():
            index = current["output_index"]
            item = self.response["output"][index]
            item.update(arguments=current["arguments"], status="completed")
            events += [self._event("response.function_call_arguments.delta", item_id=item["id"],
                                   output_index=index, delta=item["arguments"]),
                       self._event("response.function_call_arguments.done", item_id=item["id"],
                                   output_index=index, arguments=item["arguments"]),
                       self._event("response.output_item.done", output_index=index, item=item)]
        return events

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
                if isinstance(call, dict) and "index" in call:
                    events += self._indexed_call(call)
                    continue
                item = _function_call(call, self.aliases)
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
        events = self._close_text() + self._finish_calls()
        self.response.update(_outcome(self.finish_reason), usage=_usage(self.usage))
        return events + [self._event("response." + self.response["status"], response=self.response)]

    def keepalive(self):
        # An SSE comment: Responses defines no ping event, and comments leave sequence numbers unchanged.
        return [b": keepalive\n\n"]

    def fail(self, message):
        self.response.update(status="failed", error={"code": "server_error", "message": message})
        return [self._event("response.failed", response=self.response)]
