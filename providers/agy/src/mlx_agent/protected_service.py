"""Foreground authenticated MLX service for an existing service manager."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import secrets
import signal
import tempfile
import threading

PROTOCOL = "protected-service-v1"


def identity():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("cli.py", "identity_pass.py", "protected_service.py", "private_serve.py", "serving_access.py",
                 "jit_serve.py", "serve.py", "inference_auth.py", "protected_worker.py", "worker_transport.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return {"protocol": PROTOCOL, "source_sha256": digest.hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--api-key-file")
    parser.add_argument("--port", type=int)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--runtime-executable", default="mlx_lm.server")
    args = parser.parse_args(argv)
    if args.describe:
        if args.model or args.api_key_file or args.port is not None:
            parser.error("--describe accepts no service inputs")
        print(json.dumps(identity()))
        return 0
    if not args.model or not args.api_key_file or args.port is None:
        parser.error("--model, --api-key-file and --port are required")
    if not Path(args.model).is_dir():
        parser.error("--model must be an existing local model directory")
    from .identity_pass import recheck_bound_identity
    from .serve import load_recipes, plan_start
    from .jit_serve import make_server
    with tempfile.TemporaryDirectory(prefix="mlx-service-", dir="/tmp") as directory:
        # The secure state writer refuses symlink components, including macOS /tmp.
        directory = str(Path(directory).resolve())
        plan = plan_start(None, "mlx_lm", load_recipes(), path=args.model, jit=True,
                          receipts_dir=directory, port=args.port, max_tokens=args.max_tokens,
                          api_key_file=args.api_key_file, runtime_executable=args.runtime_executable)
        from .worker_transport import gateway_config
        config = gateway_config(plan, control_token=secrets.token_hex(32),
                                state_path=str(Path(directory) / "state.json"))
        recheck_bound_identity(plan)
        server = make_server(config, args.port)
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for sig in previous:
                signal.signal(sig, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
            server.serve_forever()
        finally:
            server.controller.close()
            server.server_close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return 0
