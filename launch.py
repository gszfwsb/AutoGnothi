import pathlib
import sys
import unittest
from typing import List, Optional, Type

import rich
import typer


def launch(script: str) -> Optional[int]:
    repo_root = pathlib.Path(__file__).resolve().parent
    sys.path.append(repo_root.parent.as_posix())
    script_path = pathlib.Path(script).resolve()
    script_rel = script_path.relative_to(repo_root)
    script_module = [i.strip().split(".")[0] for i in script_rel.as_posix().split("/")]
    sys.stdout.write("loading libraries...")
    sys.stdout.flush()
    module = __import__(".".join([repo_root.name] + script_module), fromlist=[""])
    sys.stdout.write("\r                    \r")
    sys.stdout.flush()

    m_app = getattr(module, "app", None)
    m_main = getattr(module, "main", None)
    m_tests: List[Type[unittest.TestCase]] = []
    for k in dir(module):
        try:
            if issubclass(v := getattr(module, k), unittest.TestCase):
                m_tests.append(v)
        except TypeError:
            pass

    if isinstance(m_app, typer.Typer):
        ret = m_app()
    elif callable(m_main):
        ret = m_main()
    elif m_tests:
        runner = unittest.TextTestRunner()
        suite = unittest.TestSuite()
        for t in m_tests:
            suite.addTest(unittest.TestLoader().loadTestsFromTestCase(t))
        ret = runner.run(suite)
    else:
        raise RuntimeError(f"script `{script}` has no entrypoint")

    if isinstance(ret, int):
        return ret
    elif isinstance(ret, bool):
        return 0 if ret else 1
    elif ret is None:
        return 0
    return 0x3F3F3F3F


if __name__ == "__main__":
    # python launch.py /path/to/script *args -> /path/to/script *args
    script = sys.argv[1]
    sys.argv = sys.argv[1:]

    console = rich.console.Console()
    console.print(f"< autognothi module launcher: {script} >", style="italic grey35")

    ret = launch(script)
    sys.exit(ret)
