"""Bounded prompt lookup for single-sequence greedy decoding."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import importlib
import logging
import time

if __package__:
    from ._decode_policy import LookupPolicy
else:
    from _decode_policy import LookupPolicy

_request = ContextVar("prompt_lookup_request", default=None)
MIN_MATCH_TOKENS = 3


class PromptLookupIndex:
    def __init__(self, tokens):
        self.tokens = list(tokens[-8192:])
        self.index = {}
        for size in range(MIN_MATCH_TOKENS, 9):
            for position in range(len(self.tokens) - size):
                self.index[tuple(self.tokens[position:position + size])] = position + size

    @property
    def entry_count(self):
        return len(self.index)

    def propose(self, history, limit):
        for size in range(min(8, len(history)), MIN_MATCH_TOKENS - 1, -1):
            position = self.index.get(tuple(history[-size:]))
            if position is not None:
                return self.tokens[position:position + max(0, min(8, limit))]
        return []


def request_eligible(arguments, cli_arguments):
    return (arguments.sampling.temperature == 0
            and not arguments.logits.logit_bias
            and not arguments.logits.repetition_penalty
            and arguments.logprobs <= 0
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
    proposed_tokens: int = 0
    native_tokens: int = 0
    fallback_reason: str | None = None
    verification_work_seconds: float = 0

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
            logging.info("Code-edit decoding: calls=%d accepted=%d proposed=%d native=%d fallback=%s verification_work_ms=%.3f",
                         self.request.verification_calls, self.request.accepted_tokens,
                         self.request.proposed_tokens, self.request.native_tokens,
                         self.request.fallback_reason or "none", self.request.verification_work_seconds * 1000)
        return self.original.insert_cache(model, tokens, state)


def lookup_step(prompt, model, *, full_prompt, max_tokens, prompt_cache,
                prefill_step_size=2048, prompt_progress_callback=None, request,
                native_step=None, clock=time.perf_counter):
    """Verify prompt continuations with the target model before yielding tokens.

    Only ordinary KVCache states enter here. The pending output token is committed
    on close so MLX-LM can retain its existing full-token cache key convention.
    """
    import mlx.core as mx
    from mlx_lm.generate import generation_stream
    from mlx_lm.models.cache import trim_prompt_cache
    if native_step is None:
        from mlx_lm.generate import generate_step as native_step

    index = PromptLookupIndex(full_prompt)
    policy = LookupPolicy()
    history = list(full_prompt[-8:])
    progress = prompt_progress_callback or (lambda *_: None)
    y = prompt
    generated = 0
    last_token = None
    native_owns_cache = False

    def schedule(pending, history, generated):
        limit = min(policy.proposal_limit, max_tokens - generated - 1)
        proposal = index.propose(history, limit) if limit > 0 else []
        if policy.native_reason or (limit > 0 and not proposal and generated >= MIN_MATCH_TOKENS):
            return None, policy.native_reason or "no_prompt_match"
        started = clock()
        with mx.stream(generation_stream):
            inputs = mx.concatenate([pending, mx.array(proposal, mx.uint32)]) if proposal else pending
            logits = model(inputs[None], cache=prompt_cache).squeeze(0)
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            predictions = mx.argmax(logprobs, axis=-1)
            mx.async_eval(predictions, logprobs)
        request.verification_calls += 1
        request.proposed_tokens += len(proposal)
        scheduling_seconds = clock() - started
        request.verification_work_seconds += scheduling_seconds
        return (scheduling_seconds, proposal, predictions, logprobs), None

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
        block, reason = schedule(y, history, generated)
        while generated < max_tokens:
            if block is None:
                request.fallback_reason = reason
                # Native generation pipelines one token ahead and owns its final
                # cache commit. Hand over the pending token exactly once.
                native_owns_cache = True
                suffix = native_step(y, model, max_tokens=max_tokens - generated,
                    prompt_cache=prompt_cache, prefill_step_size=prefill_step_size,
                    prompt_progress_callback=lambda *_: None)
                try:
                    for token, logprobs in suffix:
                        request.native_tokens += 1
                        yield token, logprobs
                finally:
                    suffix.close()
                return
            scheduling_seconds, proposal, predictions, logprobs = block
            started = clock()
            mx.eval(predictions, logprobs)
            predicted = predictions.tolist()
            accepted = 0
            while accepted < len(proposal) and predicted[accepted] == proposal[accepted]:
                accepted += 1
            count = accepted + 1
            trim_prompt_cache(prompt_cache, len(proposal) + 1 - count)
            # Include host scheduling and exposed target waits; exclude the
            # consumer interval while a scheduled block runs asynchronously.
            collection_seconds = clock() - started
            elapsed = scheduling_seconds + collection_seconds
            request.verification_work_seconds += collection_seconds
            policy.observe(len(proposal), accepted, elapsed)
            if generated == 0:
                progress(total, total)
            history = (history + predicted[:count])[-8:]
            y = mx.array([predicted[count - 1]], mx.uint32)
            # The whole block is verified. Overlap the next GPU evaluation with
            # probability reads and streaming of the current block, as native
            # generation does for individual tokens.
            if generated + count < max_tokens:
                block, reason = schedule(y, history, generated + count)
            for position in range(count):
                last_token = predicted[position]
                generated += 1
                if position < accepted:
                    request.accepted_tokens += 1
                yield last_token, logprobs[position]
    except GeneratorExit:
        raise
    except BaseException:
        request.failed = True
        raise
    finally:
        if not request.failed and last_token is not None and not native_owns_cache:
            # A scheduled block can be ahead of the last delivered token. Drop
            # all unconsumed work before committing that token for cache reuse.
            trim_prompt_cache(prompt_cache, prompt_cache[0].offset - (len(full_prompt) + generated - 1))
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
            native_step=original_step,
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
