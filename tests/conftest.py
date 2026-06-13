from pathlib import Path

import pytest


@pytest.fixture
def minimal_config(tmp_path: Path) -> Path:
    config = tmp_path / "minimal.yaml"
    config.write_text(
        "\n".join(
            [
                "embedding_model: fake_embedding",
                "classifier: fake_classifier",
                f"data_folder: {tmp_path / 'dataset'}",
                f"output_folder: {tmp_path / 'output'}",
            ]
        ),
        encoding="utf-8",
    )
    return config


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "sample.txt").write_text("sample", encoding="utf-8")
    return dataset
