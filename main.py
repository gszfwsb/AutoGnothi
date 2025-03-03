import pathlib
import sys
from typing import Callable

if __name__ == "__main__":
    print("loading libraries... ", end="\r")
    repo_root = pathlib.Path(__file__).resolve().parent
    sys.path.append(repo_root.parent.absolute().as_posix())
    module = __import__(f"{repo_root.name}.scripts.shell", fromlist=[""])
    main: Callable[[], None] = module.main
    print("                     ", end="\r")
    main()
