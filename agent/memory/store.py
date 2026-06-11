"""
Lightweight memory storage for agent optimization runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
import json
import uuid

from agent.utils.path_tool import ensure_dir, get_memory_dir, resolve_project_path


@dataclass
class MemoryScope:
    """Structured scope used to isolate and share agent memories."""

    agent_type: str
    visibility: str = "shared"
    task_type: Optional[str] = None
    model_family: Optional[str] = None
    dataset_key: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class EpisodeMemory:
    """Reusable summary of one meaningful agent event."""

    agent_type: str
    objective: str
    outcome: Dict[str, Any]
    summary: str = ""
    action: Dict[str, Any] = field(default_factory=dict)
    experiment_ids: List[str] = field(default_factory=list)
    scope: Optional[MemoryScope] = None
    status: str = "success"
    importance: float = 0.5
    memory_id: str = ""
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["scope"] = asdict(self.scope) if self.scope else {}
        return data


@dataclass
class MemoryQuery:
    """Structured filters for episodic memory retrieval."""

    agent_type: Optional[str] = None
    task_type: Optional[str] = None
    model_family: Optional[str] = None
    dataset_key: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    visibility: Optional[str] = None
    limit: int = 5


class MemoryService:
    """Unified first-stage memory service for working and episodic memory."""

    _shared_lock = Lock()

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = resolve_project_path(root_dir) if root_dir else get_memory_dir()
        self.working_dir = self.root_dir / "working"
        self.episodes_file = self.root_dir / "episodes.jsonl"
        self._lock = self._shared_lock

    def get_working_state(self, orchestration_id: str) -> Dict[str, Any]:
        """Load isolated working state for one orchestration run."""
        path = self.working_dir / f"{orchestration_id}.json"
        with self._lock:
            return self._read_json(path)

    def update_working_state(self, orchestration_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Merge and atomically persist working state."""
        path = self.working_dir / f"{orchestration_id}.json"
        with self._lock:
            state = self._read_json(path)
            state.update(updates)
            state["orchestration_id"] = orchestration_id
            state["updated_at"] = datetime.now().isoformat()
            self._write_json(path, state)
            return state

    def clear_working_state(self, orchestration_id: str) -> None:
        """Remove completed working state when explicitly requested."""
        path = self.working_dir / f"{orchestration_id}.json"
        with self._lock:
            if path.exists():
                path.unlink()

    def remember_episode(self, memory: EpisodeMemory, force: bool = False) -> Optional[str]:
        """Append a meaningful episode and return its memory ID."""
        if not force and not self.should_remember(memory):
            return None

        memory.memory_id = memory.memory_id or f"episode_{uuid.uuid4().hex}"
        memory.created_at = memory.created_at or datetime.now().isoformat()
        ensure_dir(self.episodes_file.parent)
        line = json.dumps(memory.to_dict(), ensure_ascii=False, default=str)
        with self._lock:
            with self.episodes_file.open("a", encoding="utf-8") as fout:
                fout.write(line + "\n")
        return memory.memory_id

    def search(self, query: MemoryQuery) -> List[Dict[str, Any]]:
        """Return recent episodes matching structured scope filters."""
        episodes = self._read_episodes()
        matches = [item for item in episodes if self._matches(item, query)]
        matches.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return matches[: max(query.limit, 0)]

    def get_model(
        self,
        model_key: str,
        dataset_key: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a compatibility summary derived from the latest HPO episode."""
        episodes = self.search(MemoryQuery(
            agent_type="hpo_agent",
            task_type=task_type,
            model_family=model_key,
            dataset_key=dataset_key,
            limit=1,
        ))
        if not episodes:
            return {}
        episode = episodes[0]
        action = episode.get("action") or {}
        outcome = episode.get("outcome") or {}
        return {
            "last_objective": episode.get("objective"),
            "last_best_config": action.get("best_config") or {},
            "last_best_metrics": outcome.get("best_metrics") or {},
            "last_summary": episode.get("summary") or "",
            "last_changes": action.get("changes") or [],
            "last_outcomes": outcome,
        }

    def format_context(self, query: MemoryQuery, max_chars: int = 1200) -> str:
        """Format matching episodes as compact prompt context."""
        memories = self.search(query)
        if not memories:
            return "Memory context: no relevant episodes."

        lines = ["Relevant memory episodes:"]
        for item in memories:
            experiment_ids = item.get("experiment_ids") or []
            lines.append(
                f"- [{item.get('agent_type')}/{item.get('status')}] "
                f"objective={item.get('objective', '')}; "
                f"outcome={item.get('outcome', {})}; "
                f"experiments={experiment_ids}; "
                f"summary={item.get('summary', '')}"
            )
        text = "\n".join(lines)
        return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def should_remember(memory: EpisodeMemory) -> bool:
        """Keep failures and reusable outcomes; ignore empty conversational events."""
        return (
            memory.status == "failed"
            or memory.importance >= 0.7
            or bool(memory.experiment_ids)
            or bool(memory.action)
            or bool(memory.outcome)
        )

    def _read_episodes(self) -> List[Dict[str, Any]]:
        with self._lock:
            if not self.episodes_file.exists():
                return []
            episodes: List[Dict[str, Any]] = []
            with self.episodes_file.open("r", encoding="utf-8") as fin:
                for line in fin:
                    try:
                        episodes.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return episodes

    @staticmethod
    def _matches(item: Dict[str, Any], query: MemoryQuery) -> bool:
        scope = item.get("scope") or {}
        if query.agent_type and item.get("agent_type") != query.agent_type:
            return False
        for key in ("task_type", "model_family", "dataset_key", "visibility"):
            expected = getattr(query, key)
            if expected and scope.get(key) != expected:
                return False
        if query.tags and not set(query.tags).intersection(scope.get("tags") or []):
            return False
        return True

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def _write_json(path: Path, data: Dict[str, Any]) -> None:
        ensure_dir(path.parent)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        temp_path.replace(path)


@dataclass
class MemoryUpdate:
    """Payload for memory updates."""
    model_key: str
    metadata: Dict[str, Any]
    history_entry: Optional[Dict[str, Any]] = None


class MemoryStore:
    """Persisted memory for model-specific optimization notes."""

    def __init__(self, file_path: Optional[Path] = None, max_history: int = 50) -> None:
        self.file_path = (
            resolve_project_path(file_path)
            if file_path
            else get_memory_dir() / "agent_memory.json"
        )
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
        ensure_dir(self.file_path.parent)
        temp_path = self.file_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(self.file_path)


def build_history_entry(
    objective: str,
    best_config: Dict[str, Any],
    best_metrics: Optional[Dict[str, Any]],
    summary: str,
    total_steps: int,
    changes: Optional[list[Dict[str, Any]]] = None,
    outcomes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a history entry payload."""
    return {
        "timestamp": datetime.now().isoformat(),
        "objective": objective,
        "best_config": best_config,
        "best_metrics": best_metrics or {},
        "summary": summary,
        "total_steps": total_steps,
        "changes": changes or [],
        "outcomes": outcomes or {},
    }
