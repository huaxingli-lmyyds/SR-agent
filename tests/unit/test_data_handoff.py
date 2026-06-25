from pathlib import Path

import pytest

from agent.data_processing.handoff import build_data_handoff, resolve_data_handoff


def _context(handoff, status="success"):
    return {
        "previous_results": {
            "data_processing_agent": {
                "status": status,
                "summary": {"data_handoff": handoff},
                "experiment_ids": {"data_processing": "dp_1"},
            }
        }
    }


def test_ready_data_handoff_is_selected_for_hpo(tmp_path: Path) -> None:
    derived = tmp_path / "derived"
    derived.mkdir()

    resolved = resolve_data_handoff(_context({
        "consumer_uri": str(derived),
        "consumption_status": "ready",
        "dataset_version": "v1",
        "data_processing_experiment_id": "dp_1",
    }), tmp_path / "source")

    assert resolved["consumer_uri"] == str(derived)
    assert resolved["source"] == "data_processing_agent"
    assert resolved["dataset_version"] == "v1"


def test_handoff_builder_keeps_stable_public_fields() -> None:
    handoff = build_data_handoff({
        "dataset_id": "demo",
        "version": "v1",
        "source_uri": "source",
        "output_uri": "output",
        "consumer_uri": "consumer",
        "consumption_status": "ready",
        "consumption_reason": "complete",
    }, "dp_1")

    assert set(handoff) == {
        "dataset_id",
        "dataset_version",
        "source_uri",
        "output_uri",
        "consumer_uri",
        "consumption_status",
        "consumption_reason",
        "data_processing_experiment_id",
    }


def test_non_consumable_processing_output_cannot_silently_fall_back(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not ready"):
        resolve_data_handoff(_context({
            "consumer_uri": None,
            "consumption_status": "not_ready",
            "consumption_reason": "manifests only",
        }), tmp_path / "source")


def test_successful_data_processing_must_publish_handoff(tmp_path: Path) -> None:
    context = {
        "previous_results": {
            "data_processing_agent": {
                "status": "success",
                "summary": {},
            }
        }
    }

    with pytest.raises(ValueError, match="does not provide"):
        resolve_data_handoff(context, tmp_path / "source")


def test_failed_data_processing_blocks_downstream_training(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="did not complete"):
        resolve_data_handoff(_context({}, status="failed"), tmp_path / "source")
