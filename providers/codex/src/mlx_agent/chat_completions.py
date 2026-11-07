"""Native chat requests and worker responses in the shape OpenAI clients expect.

mlx_lm 0.29.1 sends tool calls without ids, stream indexes or the ``tool_calls``
finish reason, and leaves the tokenizer's end-of-sequence text on unstreamed
answers. Responses that already conform are relayed byte for byte.
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path


def decode_tool_arguments(body):
    """Decode OpenAI history arguments for MLX-LM's template serializer.

    The native server passes history straight to the tokenizer. Its templates
    serialize function arguments themselves, so retaining the API's JSON string
    double-encodes them. Already structured or malformed values stay unchanged.
    """
    messages = body.get("messages")
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        for call in calls if isinstance(calls, list) else []:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
                continue
            try:
                arguments = json.loads(function["arguments"])
            except (ValueError, RecursionError):
                continue
            if isinstance(arguments, dict):
                function["arguments"] = arguments


def end_of_sequence(model_path):
    """Return the model tokenizer's end-of-sequence token text, or None when it declares none."""
    try:
        config = json.loads((Path(model_path) / "tokenizer_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    token = config.get("eos_token") if isinstance(config, dict) else None
    token = token.get("content") if isinstance(token, dict) else token
    return token if isinstance(token, str) and token else None


def _identifier():
    return "call_" + secrets.token_hex(12)


def _choices(response):
    choices = response.get("choices") if isinstance(response, dict) else None
    return [choice for choice in choices if isinstance(choice, dict)] if isinstance(choices, list) else []


def trim_end(response, end_text):
    """Drop the end-of-sequence text from stopped unstreamed answers; return whether anything changed."""
    changed = False
    for choice in _choices(response):
        message = choice.get("message")
        holder, key = (message, "content") if isinstance(message, dict) else (choice, "text")
        text = holder.get(key)
        if end_text and choice.get("finish_reason") == "stop" and isinstance(text, str) and text.endswith(end_text):
            holder[key], changed = text[:-len(end_text)], True
    return changed


def complete(response, end_text):
    """Trim the answer, give every tool call an id and report tool turns as tool_calls; return whether anything changed."""
    changed = trim_end(response, end_text)
    for choice in _choices(response):
        message = choice.get("message")
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        calls = [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []
        for call in calls:
            if not call.get("id"):
                call["id"], changed = _identifier(), True
        if calls and choice.get("finish_reason") == "stop":
            choice["finish_reason"], changed = "tool_calls", True
    return changed


def complete_body(payload, end_text):
    """Return an unstreamed worker body with complete() applied, or the same bytes when nothing changes."""
    try:
        response = json.loads(payload)
    except (ValueError, UnicodeError):
        return payload
    return json.dumps(response).encode() if complete(response, end_text) else payload


class ChunkStream:
    """Relays streamed chat-completion lines, giving each tool call one id and index and tool turns the tool_calls finish reason."""

    def __init__(self):
        self.calls = {}

    def _chunk(self, value):
        changed = False
        for choice in _choices(value):
            ids = self.calls.setdefault(choice.get("index", 0), {})
            delta = choice.get("delta")
            calls = delta.get("tool_calls") if isinstance(delta, dict) else None
            for call in calls if isinstance(calls, list) else []:
                if not isinstance(call, dict):
                    continue
                index = call.get("index")
                if not isinstance(index, int) or isinstance(index, bool):
                    index, call["index"], changed = len(ids), len(ids), True
                if call.get("id"):
                    ids.setdefault(index, call["id"])
                elif index not in ids or "id" in call:
                    call["id"], changed = ids.setdefault(index, _identifier()), True
            if ids and choice.get("finish_reason") == "stop":
                choice["finish_reason"], changed = "tool_calls", True
        return changed

    def line(self, raw):
        """Return the line to relay for one worker stream line."""
        if not raw.startswith(b"data: "):
            return raw
        body = raw[len(b"data: "):]
        try:
            value = json.loads(body)
        except (ValueError, UnicodeError):
            return raw
        if not self._chunk(value):
            return raw
        return b"data: " + json.dumps(value).encode() + body[len(body.rstrip(b"\r\n")):]
