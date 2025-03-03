import datetime
import json
import pathlib
from io import TextIOWrapper
from typing import Any, Callable, Dict, Optional, Tuple

import rich
import wandb

from .types import Config_Logger, ExpConfig


class ExpEnv:
    def __init__(
        self,
        model_path: pathlib.Path,
        get_logger_opts: Callable[[ExpConfig], Optional[Config_Logger]],
        _forked: Optional[Tuple[ExpConfig, TextIOWrapper]] = None,
    ) -> None:
        self.model_path = model_path
        self._get_logger_opts = get_logger_opts
        self._console = rich.get_console()
        # config
        if not _forked:
            with open(self.model_path / ".hparams.json", "r", encoding="utf-8") as f:
                r = f.read()
                j = json.loads(r)
                config = ExpConfig.model_validate(j)
            self.config = config
            self._log_fd = open(self.model_path / ".log.txt", "a", encoding="utf-8")
            self.log(
                f"[[[ NEW RUN: load config from {self.model_path.absolute().as_posix()} ]]]"
            )
        else:
            self.config, self._log_fd = _forked

    def fork(
        self, get_logger_opts: Callable[[ExpConfig], Optional[Config_Logger]]
    ) -> "ExpEnv":
        """Create a new environment that is exactly the same model except that
        the options (e.g. wandb) are different."""

        return ExpEnv(
            self.model_path,
            get_logger_opts,
            _forked=(self.config, self._log_fd),
        )

    def log(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        msg = f"[{ts}] {msg}"

        if "[[[" in msg and "]]]" in msg:
            if "!!!" in msg or "error" in msg or "failed" in msg:
                self._console.print(msg, style="bold red1")
            elif "..." in msg or "ing " in msg:
                self._console.print(msg, style="bold sky_blue2")
            elif "ok" in msg or "done" in msg or "ed " in msg:
                self._console.print(msg, style="bold green1")
            else:
                self._console.print(msg, style="pale_violet_red1")
        else:
            if "!!! " in msg:
                self._console.print(msg, style="indian_red1")
            else:
                self._console.print(msg)

        if not self._log_fd.closed:
            self._log_fd.write(msg + "\n")
            self._log_fd.flush()
        return

    def metrics(self, data: Dict[str, Any]) -> None:
        opts = self._get_logger_opts(self.config)
        if opts is not None and opts.wandb_enabled:
            step = opts.wandb_global_step or 0
            step += 1
            wandb.log(data, step=step)
            opts.wandb_global_step = step
        else:
            printable_data: Dict[str, Any] = {}
            for k, v in data.items():
                if isinstance(v, (float, int, str)):
                    printable_data[k] = v
                else:
                    printable_data[k] = f"<{type(v).__name__}>"
            self.log(f"METRICS: {printable_data}")
        return

    def __enter__(self) -> "ExpEnv":
        opts = self._get_logger_opts(self.config)
        flattened = self.config.flatten_dump()
        self.log("CONFIG: " + json.dumps(flattened, indent=2))

        if opts is not None and opts.wandb_enabled:
            wandb.init(
                id=opts.wandb_run_id,
                project=opts.wandb_project,
                name=opts.wandb_name,
                config=flattened,
                resume="allow",
            )
            if wandb.run is not None:
                opts.wandb_run_id = wandb.run.id
                self.flush_cfg()
            self.log(
                f"[[[ wandb enabled: {opts.wandb_project} / {opts.wandb_name} / {opts.wandb_run_id} ]]]"
            )
        return self

    def __exit__(self, *args) -> None:
        opts = self._get_logger_opts(self.config)
        if opts is not None and opts.wandb_enabled:
            if wandb.run is not None:
                wandb.run.finish()
                self.log("[[[ wandb finished ]]]")
        return

    def flush_cfg(self) -> None:
        with open(self.model_path / ".hparams.json", "w", encoding="utf-8") as f:
            r = self.config.model_dump_json(by_alias=True, exclude_unset=True)
            j = json.loads(r)
            r = json.dumps(j, indent=2)
            f.write(r + "\n")
        self.log("[i] updated config file")

    pass
