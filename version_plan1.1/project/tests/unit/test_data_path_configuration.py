from pathlib import Path

from agent.utils.path_tool import get_datasets_dir, resolve_data_path


def test_data_root_env_controls_default_dataset_root(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "server-data"
    monkeypatch.setenv("SR_AGENT_DATA_ROOT", str(data_root))
    monkeypatch.delenv("SR_AGENT_DATASET_PATH", raising=False)

    assert get_datasets_dir() == data_root.resolve()
    assert resolve_data_path("voxceleb1") == (data_root / "voxceleb1").resolve()
    assert resolve_data_path() == (data_root / "voxceleb1").resolve()


def test_dataset_path_env_overrides_placeholder(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "tmp" / "voxceleb1"
    monkeypatch.setenv("SR_AGENT_DATA_ROOT", str(tmp_path / "data-root"))
    monkeypatch.setenv("SR_AGENT_DATASET_PATH", str(dataset))

    assert resolve_data_path("!PLACEHOLDER") == dataset.resolve()
    assert resolve_data_path(None) == dataset.resolve()


def test_absolute_dataset_path_is_preserved(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "external" / "voxceleb1"
    monkeypatch.setenv("SR_AGENT_DATA_ROOT", str(tmp_path / "data-root"))

    assert resolve_data_path(dataset) == dataset.resolve()