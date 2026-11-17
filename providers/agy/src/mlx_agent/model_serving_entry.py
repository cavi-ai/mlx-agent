"""Claim a local model before replacing this process with its serving runtime."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_agent.compute import process_birth_identity
from mlx_agent.model_serving import ModelServingError, acquire, resolve_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--runtime', required=True, choices=('mlx_lm', 'mlx-vlm', 'strata', 'splash', 'tasks'))
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--auth-required', required=True, choices=('true', 'false'))
    parser.add_argument('--hf-cache')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not 1 <= args.port <= 65535 or not command or len(command) > 32 or any(not item or '\x00' in item for item in command):
        raise ValueError('Invalid serving runtime command.')
    flag = '--model-dir' if args.runtime == 'splash' else '--model'
    if command.count(flag) != 1 or command.count('--port') != 1:
        raise ValueError('Serving runtime identity is unavailable.')
    try:
        selector = command[command.index(flag) + 1]
        port = int(command[command.index('--port') + 1])
    except (IndexError, ValueError):
        raise ValueError('Serving runtime identity is unavailable.') from None
    if port != args.port or resolve_model(selector, args.hf_cache) != resolve_model(args.artifact, args.hf_cache):
        raise ValueError('Serving runtime identity changed.')
    if os.getpgrp() != os.getpid():
        os.setpgid(0, 0)
    if os.getpgrp() != os.getpid():
        raise ValueError('Serving process group identity is unavailable.')
    service = {'port': args.port, 'runtime': args.runtime, 'auth_required': args.auth_required == 'true'}
    claim = acquire(args.artifact, kind='service', service=service)
    birth = process_birth_identity(os.getpid())
    if not birth:
        claim.cancel_reserved()
        raise ValueError('Serving process identity is unavailable.')
    claim.update(state='active', pid=os.getpid(), birth=birth, pgid=os.getpgrp(), jit=False, ready=False)
    try:
        os.execvpe(command[0], command, os.environ)
    except BaseException:
        claim.cancel_failed_exec()
        raise


if __name__ == '__main__':
    try:
        main()
    except ModelServingError as error:
        print(json.dumps({'code': error.code, 'message': str(error), 'remediation': error.remediation,
                          'data': error.data}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)
