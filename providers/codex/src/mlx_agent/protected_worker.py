"""Run an owned HTTP model worker with mandatory private upstream authentication."""
import os
from pathlib import Path
import runpy
import shutil
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_agent.inference_auth import protect_worker

WORKER_DIGEST = 'MLX_AGENT_WORKER_KEY_SHA256'


def command(argv, runtime):
    executable = shutil.which(argv[0])
    if executable is None:
        raise ValueError('The authenticated model worker executable is unavailable.')
    if Path(executable).name.startswith('python'):
        arguments = list(argv[1:])
        while arguments and arguments[0] in ('-I', '-B', '-u'):
            arguments.pop(0)
        if not arguments or arguments[0].startswith('-'):
            raise ValueError('Authenticated workers require a Python server entry point.')
        entry, arguments = arguments[0], arguments[1:]
        python = executable
    else:
        entry, arguments = executable, list(argv[1:])
        with open(entry, 'rb') as stream:
            shebang = stream.readline(4096).decode('utf-8').strip()
        python = shebang[2:] if shebang.startswith('#!/') else ''
        if not Path(python).is_file() or not Path(python).name.startswith('python'):
            raise ValueError('Authenticated workers require a Python runtime with an absolute interpreter.')
    if runtime not in ('mlx_lm', 'mlx-vlm', 'strata'):
        raise ValueError('This HTTP runtime does not support protected serving.')
    return [python, '-I', '-B', str(Path(__file__).resolve()), runtime, entry, *arguments]


def main():
    runtime, entry, *arguments = sys.argv[1:]
    digest = os.environ.pop(WORKER_DIGEST)  # missing configuration must never start an open server
    if runtime == 'strata':
        os.environ.pop('STRATA_API_KEY', None)  # this process uses the mandatory upstream verifier instead
    if runtime == 'mlx-vlm':
        os.environ.pop('MLX_VLM_SERVER_API_KEY', None)
    protect_worker(digest, asgi=runtime == 'mlx-vlm')
    sys.argv = [entry, *arguments]
    runpy.run_path(entry, run_name='__main__')


if __name__ == '__main__':
    main()
