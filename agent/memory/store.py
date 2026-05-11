"""
Lightweight memory storage for agent optimization runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional
import json


@dataclass
class MemoryUpdate:
    """Payload for memory updates."""
    model_key: str
    metadata: Dict[str, Any]
    history_entry: Optional[Dict[str, Any]] = None


class MemoryStore:
    """Persisted memory for model-specific optimization notes."""

    def __init__(self, file_path: Optional[Path] = None, max_history: int = 50) -> None:
        base_dir = Path(__file__).parent
        self.file_path = file_path or (base_dir / "agent_memory.json")
        self.max_history = max_history
        self._lock = Lock()

    def load(self) -> Dict[str, Any]:
        """Load the memory file content."""
        with self._lock:
            return self._read_file()

    def get_model(self, model_key: str) -> Dict[str, Any]:
        """Get memory for a specific model key."""
        data = self.load()
        return data.get("models", {}).get(model_key, {})

    def update_model(self, update: MemoryUpdate) -> None:
        """Update model memory with metadata and optional history entry."""
        with self._lock:
            data = self._read_file()
            models = data.setdefault("models", {})
            record = models.setdefault(update.model_key, {"history": []})

            record.update(update.metadata)

            if update.history_entry:
                history = record.setdefault("history", [])
                history.append(update.history_entry)
                if len(history) > self.max_history:
                    record["history"] = history[-self.max_history :]

            self._write_file(data)

    def _read_file(self) -> Dict[str, Any]:
        if not self.file_path.exists():
            return {"models": {}}

        try:
            content = self.file_path.read_text(encoding="utf-8").strip()
            if not content:
                return {"models": {}}
            return json.loads(content)
        except (json.JSONDecodeError, OSError):
            return {"models": {}}

    def _write_file(self, data: Dict[str, Any]) -> None:
        temp_path = self.file_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(self.file_path)


def build_history_entry(
    objective: str,
    best_config: Dict[str, Any],
    best_metrics: Optional[Dict[str, Any]],
    summary: str,
    total_steps: int,
) -> Dict[str, Any]:
    """Create a history entry payload."""
    return {
        "timestamp": datetime.now().isoformat(),
        "objective": objective,
        "best_config": best_config,
        "best_metrics": best_metrics or {},
        "summary": summary,
        "total_steps": total_steps,
    }
