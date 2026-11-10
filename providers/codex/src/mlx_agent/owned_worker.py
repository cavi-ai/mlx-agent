"""Gate native startup behind host ownership and contain the backend process group."""
import os
from pathlib import Path
import runpy
import shutil
import signal
import sys
import threading
import time
import traceback


def command(argv, ready, gate):
    executable = shutil.which(argv[0])
    if executable is None:
        raise ValueError('The model worker executable is unavailable.')
    if Path(executable).name.startswith('python'):
        python, arguments = executable, list(argv[1:])
        while arguments and arguments[0] in ('-I', '-B', '-u'):
            arguments.pop(0)
    else:
        with open(executable, 'rb') as stream:
            shebang = stream.readline(4096).decode().strip()
        python = shebang[2:] if shebang.startswith('#!/') else ''
        if not Path(python).is_file() or not Path(python).name.startswith('python'):
            raise ValueError('Model workers require an absolute Python interpreter.')
        arguments = [executable, *argv[1:]]
    if not arguments or (arguments[0].startswith('-') and arguments[0] not in ('-c', '-m')):
        raise ValueError('Model workers require a Python entry point.')
    return [python, '-I', '-B', str(Path(__file__).resolve()), str(ready), str(gate), str(os.getpid()), *arguments]


def main():
    ready, gate, parent = map(int, sys.argv[1:4])
    arguments = sys.argv[4:]
    if os.getpid() != os.getpgrp() or os.getsid(0) != os.getpid():
        raise ValueError('Model workers require a private process group.')
    if os.getppid() != parent:
        return

    def watch_parent():
        while os.getppid() == parent:
            time.sleep(.25)
        os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=watch_parent, daemon=True).start()
    os.write(ready, b'R')
    os.close(ready)
    try:
        if os.read(gate, 1) != b'G':
            return
    finally:
        os.close(gate)
    try:
        if arguments[0] == '-c':
            sys.argv = ['-c', *arguments[2:]]
            exec(compile(arguments[1], '<string>', 'exec'), {'__name__': '__main__'})
        elif arguments[0] == '-m':
            sys.argv = arguments[1:]
            runpy.run_module(arguments[1], run_name='__main__', alter_sys=True)
        else:
            sys.argv = arguments
            sys.path.insert(0, str(Path(arguments[0]).resolve().parent))
            runpy.run_path(arguments[0], run_name='__main__')
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        # A frontend that exits must not leave an engine consuming compute.
        os.killpg(os.getpgrp(), signal.SIGKILL)


if __name__ == '__main__':
    main()
