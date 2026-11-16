"""
日志管理模块
提供训练过程的文本日志写入与读取能力
"""

from pathlib import Path
from typing import Union, List, Any
from datetime import datetime
from threading import Lock
import sys

from agent.utils.path_tool import resolve_project_path


def progressbar_enabled(run_opts=None) -> bool:
    """Keep interactive progress while redirected logs contain only events."""
    return sys.stderr.isatty() and not bool((run_opts or {}).get("noprogressbar"))


class Logger:
    """文本日志记录器类"""

    def __init__(self, log_path: Union[str, Path]):
        self.log_path = resolve_project_path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def write(self, message: str) -> None:
        """追加一行日志文本"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} {message}\n"
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as fout:
                fout.write(line)

    def info(self, message: str) -> None:
        self.write(f"INFO {message}")

    def warning(self, message: str) -> None:
        self.write(f"WARN {message}")

    def error(self, message: str) -> None:
        self.write(f"ERROR {message}")

    def tail(self, n: int = 20) -> str:
        """读取最后 n 行日志"""
        if not self.log_path.exists():
            return ""
        with self.log_path.open("r", encoding="utf-8", errors="ignore") as fin:
            lines = fin.readlines()
        return "".join(lines[-n:])


def get_logger(log_path: Union[str, Path]) -> Logger:
    """快速获取文本日志记录器"""
    return Logger(log_path)


class AgentLogger(Logger):
    """智能体日志记录器，提供中间件集成。"""

    def __init__(self, log_path: Union[str, Path], truncate_limit: int = 800):
        super().__init__(log_path)
        self.truncate_limit = truncate_limit

    append = Logger.write

    def build_middleware(self) -> List[Any]:
        from agent.utils.agent_middleware import build_agent_logging_middleware

        return build_agent_logging_middleware(self, truncate_limit=self.truncate_limit)
