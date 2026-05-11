"""
日志管理模块
提供训练过程的文本日志写入与读取能力
"""

from pathlib import Path
from typing import Union, Optional, List, Any
from datetime import datetime

from agent.utils.agent_middleware import build_agent_logging_middleware


class Logger:
    """文本日志记录器类"""

    def __init__(self, log_path: Union[str, Path]):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        """追加一行日志文本"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} {message}\n"
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


class AgentLogger:
    """智能体日志记录器，提供中间件集成。"""

    def __init__(self, log_path: Union[str, Path], truncate_limit: int = 800):
        self.log_path = Path(log_path)
        self.truncate_limit = truncate_limit
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.log_path.open("a", encoding="utf-8") as fout:
            fout.write(f"{timestamp} {message}\n")

    def _truncate(self, text: str) -> str:
        text = text.strip()
        if len(text) <= self.truncate_limit:
            return text
        return f"{text[:self.truncate_limit]}..."

    def build_middleware(self) -> List[Any]:
        return build_agent_logging_middleware(self, truncate_limit=self.truncate_limit)