"""A bounded local tool-and-edit workload with fixed correctness checks."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import secrets
import subprocess
import sys
import tempfile
import time
from contextlib import closing

from .bench import BenchError, _measurement_timeout, _safe_error, _stream_event_usage
from .verification import _chat_request_payload, _is_installed, installed_model_ids


BASE_SOURCE = "def clamp(value, lower, upper):\n    return min(value, upper)\n"
READ_FILE = {"type": "function", "function": {
    "name": "read_file", "description": "Read the fixed clamp.py benchmark fixture.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string", "enum": ["clamp.py"]}},
                   "required": ["path"], "additionalProperties": False}}}
_ALLOWED = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.If, ast.IfExp,
            ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Call,
            ast.Name, ast.Load, ast.Constant, ast.Expr, ast.Raise,
            ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
            )
_CHECK = '''import json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
# Darwin does not support lowering RLIMIT_AS for this interpreter. The grammar
# prohibits allocation-producing operations; Linux additionally bounds address space.
if sys.platform == 'linux':
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
payload = json.load(sys.stdin)
namespace = {'__builtins__': {'min': min, 'max': max, 'ValueError': ValueError}}
cases = [((-2, 0, 5), 0), ((10, 0, 5), 5), ((2, 0, 5), 2),
         ((-10, -5, -2), -5), ((20, 3, 3), 3), ((0, -2, 2), 0)]
results = []
try:
    exec(compile(payload['source'], '<coding-workload>', 'exec'), namespace)
    if payload['reversed_bounds']:
        cases += [((1, 5, 0), 'ValueError'), ((0, 2, -2), 'ValueError')]
    for args, expected in cases:
        actual, error = None, None
        try:
            value = namespace['clamp'](*args)
            actual = value if type(value) in (int, float, str, bool, type(None)) else {'type': type(value).__name__}
        except Exception as exception:
            error = type(exception).__name__
        success = error == 'ValueError' if expected == 'ValueError' else error is None and actual == expected
        results.append({'arguments': args, 'expected': expected, 'actual': actual, 'error': error, 'passed': success})
except Exception:
    pass
total = 8 if payload['reversed_bounds'] else 6
passed = sum(case['passed'] for case in results)
print(json.dumps({'passed': passed == total, 'cases_passed': passed, 'cases_total': total,
                  'cases': results, 'reason': 'valid' if passed == total else 'wrong_answer'}))
'''


def check_edit(content, reversed_bounds=False, timeout=3):
    """Run only the fixed, restricted single-function micro-workload.

    This grammar and child process are not a general Python sandbox.
    """
    rejected = {"passed": False, "reason": "unsupported_code", "cases_passed": 0,
                "cases_total": 8 if reversed_bounds else 6, "cases": []}
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        return dict(rejected, reason="check_timeout")
    if not isinstance(content, str) or len(content) > 4000:
        return rejected
    source = content.strip()
    if source.startswith("```python\n") and source.endswith("```") and source.count("```") == 2:
        source = source[len("```python\n"):-3].strip()
    try:
        tree = ast.parse(source)
        nodes = list(ast.walk(tree))
        if (len(nodes) > 256 or len(tree.body) != 1 or type(tree.body[0]) is not ast.FunctionDef
                or any(type(node) not in _ALLOWED for node in nodes)):
            return rejected
        function = tree.body[0]
        args = function.args
        if (function.name != "clamp" or function.decorator_list or function.returns
                or args.posonlyargs or args.kwonlyargs or args.defaults or args.vararg or args.kwarg
                or [arg.arg for arg in args.args] != ["value", "lower", "upper"]
                or any(arg.annotation for arg in args.args)
                or sum(type(node) is ast.FunctionDef for node in nodes) != 1):
            return rejected
        for node in nodes:
            if isinstance(node, ast.Name) and "__" in node.id:
                return rejected
            if isinstance(node, ast.Call) and (type(node.func) is not ast.Name
                    or node.func.id not in ("min", "max", "ValueError") or node.keywords):
                return rejected
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if not -1e6 <= node.value <= 1e6:
                    return rejected
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) > 128:
                return rejected
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return rejected
    try:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-I", "-S", "-c", _CHECK],
                input=json.dumps({"source": source, "reversed_bounds": bool(reversed_bounds)}),
                capture_output=True, text=True, cwd=directory, env={}, timeout=min(3, timeout), start_new_session=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return dict(rejected, reason="execution_failed")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _chat(repo, runtime, messages, tools, gen_tokens, deadline, clock):
    started = clock()
    if started >= deadline:
        raise _measurement_timeout()
    chunks, calls, usage, progress = [], {}, {}, []
    first = None
    native_progress = runtime.name == "mlx_lm"
    with closing(runtime.stream_chat(repo, messages, gen_tokens, timeout=deadline - started,
            tools=tools, temperature=0, seed=0, include_progress=native_progress)) as stream:
        for event in stream:
            observed = clock()
            if observed > deadline:
                raise _measurement_timeout()
            if not isinstance(event, dict):
                continue
            if native_progress and "prompt_progress" in event:
                progress.append(event["prompt_progress"])
            usage.update(_stream_event_usage(event))
            choices = event.get("choices")
            delta = choices[0].get("delta", {}) if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
            if not isinstance(delta, dict):
                continue
            text = delta.get("content")
            tools_delta = delta.get("tool_calls") or []
            if not isinstance(tools_delta, list) or any(not isinstance(item, dict) for item in tools_delta):
                raise BenchError("invalid_response", "The runtime returned malformed tool calls.", "Inspect the runtime response.")
            if text or tools_delta:
                first = observed if first is None else first
            if isinstance(text, str):
                chunks.append(text)
            for item in tools_delta:
                if "type" in item and item["type"] != "function":
                    raise BenchError("invalid_response", "The runtime returned an invalid tool-call type.", "Use a function-calling runtime.")
                index = item.get("index", 0)
                if type(index) is not int or not 0 <= index < 8:
                    raise BenchError("invalid_response", "The runtime returned an invalid tool-call index.", "Inspect the runtime response.")
                call = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                function = item.get("function") or {}
                if not isinstance(function, dict):
                    raise BenchError("invalid_response", "The runtime returned a malformed function call.", "Inspect the runtime response.")
                for destination, key, value in ((call, "id", item.get("id")),
                        (call["function"], "name", function.get("name")),
                        (call["function"], "arguments", function.get("arguments"))):
                    if isinstance(value, str):
                        destination[key] += value
                        if len(destination[key]) > 4000:
                            raise BenchError("invalid_response", "The tool response exceeds the workload bound.", "Use the fixed workload output format.")
            if sum(map(len, chunks)) > 8000:
                raise BenchError("invalid_response", "The coding response exceeds its size bound.", "Use a smaller output-token limit.")
    finished = clock()
    if finished > deadline:
        raise _measurement_timeout()
    prompt = usage.get("prompt_tokens")
    cached = usage.get("cached_tokens")
    cache_source = "server_usage" if cached is not None else None
    if cached is None and prompt is not None and progress:
        valid = all(isinstance(p, dict) and not p.get("invalid") and p.get("source") == "mlx_lm_keepalive"
                    and type(p.get("processed")) is int and type(p.get("total")) is int
                    and 0 <= p["processed"] <= p["total"] <= prompt for p in progress)
        if (valid and progress[0]["processed"] == 0 and progress[-1]["processed"] == progress[-1]["total"]
                and len({p["total"] for p in progress}) == 1
                and all(left["processed"] <= right["processed"] for left, right in zip(progress, progress[1:]))):
            cached, cache_source = prompt - progress[0]["total"], "mlx_lm_keepalive"
    if cached is not None and (prompt is None or not 0 <= cached <= prompt):
        cached, cache_source = None, None
    response = {"role": "assistant", "content": "".join(chunks)}
    if calls:
        response["tool_calls"] = [calls[index] for index in sorted(calls)]
    summary = {"elapsed_ms": round((finished - started) * 1000, 1),
               "first_response_ms": round((first - started) * 1000, 1) if first is not None else None,
               "prompt_tokens": prompt, "completion_tokens": usage.get("completion_tokens"),
               "cached_tokens": cached, "cache_source": cache_source,
               "request_sha256": _digest(_chat_request_payload(repo, messages, gen_tokens, tools)),
               "response_sha256": _digest(response)}
    return response, summary


def validate_coding_bounds(gen_tokens, timeout):
    if type(gen_tokens) is not int or not 256 <= gen_tokens <= 1024:
        raise ValueError("coding gen_tokens must be between 256 and 1024")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise ValueError("coding timeout must be positive and at most 600 seconds")


def run_coding_workload(repo, runtime, gen_tokens=512, timeout=180, clock=time.monotonic):
    """Run at most three requests against an already-served local chat model."""
    validate_coding_bounds(gen_tokens, timeout)
    try:
        installed = installed_model_ids(runtime.list_models())
    except Exception as error:
        raise BenchError("runtime_unreachable", "The runtime inventory could not be read: " + _safe_error(error),
                         "Confirm the local runtime is running.") from error
    if not _is_installed(repo, installed):
        raise BenchError("model_not_serving", "The coding model is not served by this runtime.", "Serve the cached model first.")
    started = clock()
    deadline = started + timeout
    messages = [{"role": "system", "content": "Session " + secrets.token_hex(8) +
        ". You edit the fixed clamp.py fixture. First call read_file for clamp.py. "
        "Then return only its complete clamp(value, lower, upper) function, without imports, "
        "annotations, decorators, loops, or helper functions. Use min/max and ValueError as needed."},
        {"role": "user", "content": "Read clamp.py, then fix it to enforce both lower and upper bounds for integer inputs."}]
    tools, turns, transcript = [READ_FILE], [], []

    def request(name):
        try:
            response, summary = _chat(repo, runtime, messages, tools, gen_tokens, deadline, clock)
        except BenchError:
            raise
        except TimeoutError as error:
            raise _measurement_timeout() from error
        except Exception as error:
            raise BenchError("measurement_failed", "The coding request failed: " + _safe_error(error),
                             "Confirm the model supports streaming chat and tools.") from error
        turns.append(dict(summary, name=name))
        transcript.append({"request": json.loads(json.dumps(_chat_request_payload(repo, messages, gen_tokens, tools))),
                           "response": response})
        return response

    def result(reason):
        return {"workload": "coding-edit-v1", "repo": repo, "runtime": runtime.name,
                "quality_passed": reason == "valid", "reason": reason, "turns": turns, "gen_tokens": gen_tokens,
                "elapsed_ms": round((clock() - started) * 1000, 1),
                "sampling": {"temperature": 0, "seed": 0}, "transcript": transcript}

    def evaluate(content, reversed_bounds=False):
        remaining = deadline - clock()
        if remaining <= 0:
            raise _measurement_timeout()
        quality = check_edit(content, reversed_bounds=reversed_bounds, timeout=remaining)
        if clock() > deadline:
            raise _measurement_timeout()
        return quality

    response = request("read_file")
    calls = response.get("tool_calls", [])
    try:
        valid = (len(calls) == 1 and bool(calls[0]["id"])
                 and calls[0]["function"]["name"] == "read_file"
                 and json.loads(calls[0]["function"]["arguments"]) == {"path": "clamp.py"})
    except (KeyError, ValueError, TypeError):
        valid = False
    if not valid:
        return result("invalid_tool_call")
    messages.extend([response, {"role": "tool", "tool_call_id": calls[0]["id"],
                               "content": json.dumps({"path": "clamp.py", "source": BASE_SOURCE})}])
    tools = None
    response = request("first_edit")
    if response.get("tool_calls"):
        return result("unexpected_tool_call")
    quality = evaluate(response["content"])
    turns[-1]["quality"] = quality
    messages.extend([response, {"role": "user", "content": json.dumps({"test_results": quality,
        "request": "Now also raise ValueError when lower > upper, preserving all passing boundary cases. Return only the complete function."})}])
    response = request("followup_edit")
    if response.get("tool_calls"):
        return result("unexpected_tool_call")
    turns[-1]["quality"] = evaluate(response["content"], reversed_bounds=True)
    if clock() > deadline:
        raise _measurement_timeout()
    return result("valid" if quality["passed"] and turns[-1]["quality"]["passed"] else "wrong_answer")
