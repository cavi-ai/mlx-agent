"""Bounded prompt lookup for single-sequence greedy decoding."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import importlib
import logging

_request = ContextVar("prompt_lookup_request", default=None)


class PromptLookupIndex:
    def __init__(self, tokens):
        self.tokens = list(tokens[-8192:])
        self.index = {}
        for size in range(3, 9):
            for position in range(len(self.tokens) - size):
                self.index[tuple(self.tokens[position:position + size])] = position + size

    @property
    def entry_count(self):
        return len(self.index)

    def propose(self, history, limit):
        for size in range(min(8, len(history)), 2, -1):
            position = self.index.get(tuple(history[-size:]))
            if position is not None:
                return self.tokens[position:position + max(0, min(8, limit))]
        return []


def request_eligible(arguments, cli_arguments):
    return (arguments.sampling.temperature == 0
            and not arguments.logits.logit_bias
            and not arguments.logits.repetition_penalty
            and arguments.seed is None and arguments.logprobs <= 0
            and arguments.model.draft in (None, "default_model")
            and arguments.model.adapter in (None, "default_model")
            and cli_arguments.draft_model is None
            and cli_arguments.adapter_path is None)


@dataclass
class _Request:
    prompt: list | None = None
    generator: object = None
    used: bool = False
    failed: bool = False
    verification_calls: int = 0
    accepted_tokens: int = 0

    def close(self):
        if self.generator is not None:
            self.generator.close()


class _PromptCache:
    def __init__(self, original, request):
        self.original = original
        self.request = request

    def __getattr__(self, name):
        return getattr(self.original, name)

    def fetch_nearest_cache(self, model, tokens):
        self.request.prompt = list(tokens)
        return self.original.fetch_nearest_cache(model, tokens)

    def insert_cache(self, model, tokens, state):
        # Native stream_generate may finish while its token generator is suspended.
        # Close it before publishing the cache, including EOS and cancelled replies.
        self.request.close()
        if self.request.used:
            if self.request.failed or any(c.offset != len(tokens) for c in state):
                logging.warning("Code-edit acceleration discarded an unaligned prompt cache.")
                return
            logging.info("Code-edit decoding: %d verification calls, %d accepted prompt tokens.",
                         self.request.verification_calls, self.request.accepted_tokens)
        return self.original.insert_cache(model, tokens, state)


def lookup_step(prompt, model, *, full_prompt, max_tokens, prompt_cache,
                prefill_step_size=2048, prompt_progress_callback=None, request):
    """Verify prompt continuations with the target model before yielding tokens.

    Only ordinary KVCache states enter here. The pending output token is committed
    on close so MLX-LM can retain its existing full-token cache key convention.
    """
    import mlx.core as mx
    from mlx_lm.generate import generation_stream
    from mlx_lm.models.cache import trim_prompt_cache

    index = PromptLookupIndex(full_prompt)
    history = list(full_prompt[-8:])
    progress = prompt_progress_callback or (lambda *_: None)
    y = prompt
    generated = 0
    unyielded = 0
    last_token = None
    try:
        with mx.stream(generation_stream):
            total = y.size
            progress(0, total)
            while y.size > 1:
                count = min(prefill_step_size, y.size - 1)
                model(y[:count][None], cache=prompt_cache)
                mx.eval([c.state for c in prompt_cache])
                y = y[count:]
                progress(total - y.size, total)
                mx.clear_cache()
        while generated < max_tokens:
            proposal = index.propose(history, max_tokens - generated - 1)
            with mx.stream(generation_stream):
                inputs = mx.concatenate([y, mx.array(proposal, mx.uint32)]) if proposal else y
                logits = model(inputs[None], cache=prompt_cache).squeeze(0)
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                predictions = mx.argmax(logprobs, axis=-1)
                mx.eval(predictions)
            predicted = predictions.tolist()
            accepted = 0
            while accepted < len(proposal) and predicted[accepted] == proposal[accepted]:
                accepted += 1
            count = accepted + 1
            request.verification_calls += 1
            trim_prompt_cache(prompt_cache, len(proposal) + 1 - count)
            if generated == 0:
                progress(total, total)
            unyielded = count
            for position in range(count):
                last_token = predicted[position]
                history = (history + [last_token])[-8:]
                generated += 1
                unyielded -= 1
                if position < accepted:
                    request.accepted_tokens += 1
                yield last_token, logprobs[position]
            y = mx.array([last_token], mx.uint32)
    except GeneratorExit:
        raise
    except BaseException:
        request.failed = True
        raise
    finally:
        if not request.failed and last_token is not None:
            trim_prompt_cache(prompt_cache, unyielded)
            with mx.stream(generation_stream):
                model(mx.array([[last_token]], mx.uint32), cache=prompt_cache)
                mx.eval([c.state for c in prompt_cache])


def install(server):
    """Install request-scoped dispatch, preserving the upstream server lifecycle."""
    generate = importlib.import_module("mlx_lm.generate")
    from mlx_lm.models.cache import KVCache

    original_step = generate.generate_step
    original_generator = server.ResponseGenerator

    def dispatch(prompt, model, **kwargs):
        context = _request.get()
        state = kwargs.get("prompt_cache")
        supported = (context is not None and context.prompt is not None and state
                     and prompt.size > 0 and kwargs.get("max_tokens", 256) > 0
                     and all(type(c) is KVCache for c in state)
                     and all(c.offset + prompt.size == len(context.prompt) for c in state)
                     and not kwargs.get("logits_processors")
                     and kwargs.get("max_kv_size") is None
                     and kwargs.get("kv_bits") is None
                     and kwargs.get("input_embeddings") is None)
        if not supported:
            return original_step(prompt, model, **kwargs)
        context.used = True
        context.generator = lookup_step(
            prompt, model, full_prompt=context.prompt, prompt_cache=state,
            max_tokens=kwargs.get("max_tokens", 256), request=context,
            prefill_step_size=kwargs.get("prefill_step_size", 2048),
            prompt_progress_callback=kwargs.get("prompt_progress_callback"))
        return context.generator

    class ResponseGenerator(original_generator):
        def _is_batchable(self, args):
            if request_eligible(args, self.model_provider.cli_args):
                return False
            return super()._is_batchable(args)

        def _serve_single(self, item):
            if not request_eligible(item[2], self.model_provider.cli_args):
                return super()._serve_single(item)
            context = _Request()
            token = _request.set(context)
            original_cache = self.prompt_cache
            self.prompt_cache = _PromptCache(original_cache, context)
            try:
                return super()._serve_single(item)
            finally:
                try:
                    context.close()
                finally:
                    self.prompt_cache = original_cache
                    _request.reset(token)

    generate.generate_step = dispatch
    server.ResponseGenerator = ResponseGenerator

    def restore():
        generate.generate_step = original_step
        server.ResponseGenerator = original_generator

    return restore
