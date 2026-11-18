"""Source-bound instruction-prefix reuse for the private MLX-LM worker."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

VERSIONS = {'mlx': '0.32.0', 'mlx-metal': '0.32.0', 'mlx-lm': '0.31.3'}
UPSTREAM = {
    'mlx_lm/server.py': 'cdfcb4ac848636f9927851a0ec7a951584526530cb7832ba58049e4a9144db8b',
    'mlx_lm/generate.py': '270778ad53eaca55a8533d82e6752660fe5d2605c4aa0879b48a50a91f69345f',
    'mlx_lm/models/cache.py': '819ed95dcbf755652363cfdb15a639890447abb534a06dcefd52c7fff5055750',
}


def supported(facts):
    return facts['versions'] == VERSIONS and all(facts['sources'].get(k) == v for k, v in UPSTREAM.items())


def identity_here():
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    dist = importlib.metadata.distribution('mlx-lm')
    sources = {}
    for name in ('mlx_lm/server.py', 'mlx_lm/generate.py', 'mlx_lm/models/cache.py'):
        sources[name] = hashlib.sha256(Path(dist.locate_file(name)).read_bytes()).hexdigest()
    sources['adapter'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    facts = {'versions': versions, 'sources': sources}
    facts['fingerprint'] = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
    return facts


def identity(python):
    code = ('import sys,json;sys.path.insert(0,sys.argv[1]);'
            'from mlx_agent.shared_prefix_cache import identity_here;print(json.dumps(identity_here()))')
    result = subprocess.run([python, '-I', '-B', '-c', code, str(Path(__file__).resolve().parents[1])],
                            check=True, capture_output=True, text=True, timeout=10)
    facts = json.loads(result.stdout)
    if not supported(facts):
        raise ValueError('Unsupported MLX-LM runtime versions.')
    return facts


class ScopedPrompt(list):
    def __init__(self, tokens, prefix):
        super().__init__(tokens)
        self.prefix = prefix


class SystemSegment(str):
    def __new__(cls, prefix):
        obj = super().__new__(cls, 'system')
        obj.prefix = prefix
        return obj


def eligible_prefix(prompt, segments, types, rendered, messages):
    if not messages or messages[0].get('role') != 'system':
        return None
    count = 0
    for message in messages:
        if message.get('role') != 'system':
            break
        count += 1
    if count >= len(messages) or messages[count].get('role') != 'user':
        return None
    if len(segments) < 2 or not types or types[0] != 'system':
        return None
    prefix = segments[0]
    if not prefix or len(prefix) >= len(prompt) or prompt[:len(prefix)] != prefix:
        return None
    common = 0
    for a, b in zip(prompt, rendered):
        if a != b:
            break
        common += 1
    return tuple(prefix) if common == len(prefix) else None


class PrefixCache:
    def __init__(self, native, cache_type, max_bytes):
        self.native, self.cache_type, self.max_bytes = native, cache_type, max_bytes

    def __len__(self):
        return len(self.native)

    def fetch_nearest_cache(self, model, tokens):
        if not isinstance(tokens, ScopedPrompt) or tokens[:len(tokens.prefix)] != list(tokens.prefix):
            return None, tokens
        cache, rest = self.native.fetch_nearest_cache(model, list(tokens.prefix))
        if cache is None or rest:
            return None, tokens
        return cache, list(tokens[len(tokens.prefix):])

    def insert_cache(self, model, tokens, state, *, cache_type='assistant'):
        if (not isinstance(cache_type, SystemSegment) or tuple(tokens) != cache_type.prefix or not state
                or any(type(c) is not self.cache_type or c.offset != len(tokens) for c in state)
                or sum(c.nbytes for c in state) > self.max_bytes):
            return
        self.native.insert_cache(model, tokens, state, cache_type='system')

    def __getattr__(self, name):
        return getattr(self.native, name)


def install(server, budget):
    from mlx_lm.models.cache import KVCache, LRUPromptCache
    original = server.ResponseGenerator

    class ResponseGenerator(original):
        def __init__(self, provider, unused):
            super().__init__(provider, PrefixCache(LRUPromptCache(max_size=128, max_bytes=budget), KVCache, budget))

        def _tokenize(self, tokenizer, request, args):
            prompt, segments, types, initial = super()._tokenize(tokenizer, request, args)
            if request.request_type != 'chat' or not tokenizer.has_chat_template:
                return prompt, segments, types, initial
            messages = request.messages
            if not messages or messages[0].get('role') != 'system':
                return prompt, segments, types, initial
            count = 0
            while count < len(messages) and messages[count].get('role') == 'system':
                count += 1
            kwargs = self.model_provider.cli_args.chat_template_args.copy()
            if args.chat_template_kwargs:
                kwargs.update(args.chat_template_kwargs)
            try:
                rendered = tokenizer.apply_chat_template(messages[:count] + [{'role': 'user', 'content': ''}],
                    add_generation_prompt=False, tools=request.tools, tokenize=True, **kwargs)
                prefix = eligible_prefix(prompt, segments, types, rendered, messages)
            except Exception:
                prefix = None
            if prefix:
                prompt = ScopedPrompt(prompt, prefix)
                types[0] = SystemSegment(prefix)
            return prompt, segments, types, initial

        def _serve_single(self, request):
            # Single requests have no native segment boundary callback.
            original_cache = self.prompt_cache
            self.prompt_cache = PrefixCache(LRUPromptCache(max_size=0, max_bytes=0), KVCache, 0)
            try:
                return super()._serve_single(request)
            finally:
                self.prompt_cache = original_cache

    server.ResponseGenerator = ResponseGenerator
    return lambda: setattr(server, 'ResponseGenerator', original)
