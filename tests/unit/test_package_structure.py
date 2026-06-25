from pathlib import Path

import agent.models
import agent.runners
import agent.tasks
import agent.tools
import agent.utils
from agent.utils import runner as legacy_runner


def test_public_packages_import_without_loading_optional_tool_modules() -> None:
    assert "TrainModel" not in agent.tools.__dict__
    assert "MetricsVisualizer" not in agent.utils.__dict__
    assert agent.runners.RUNNER_REGISTRY.get("speechbrain").runner == "speechbrain"
    assert agent.models.get_model_adapter("ecapa_tdnn").implementation == "speechbrain"
    assert agent.tasks.get_task_adapter("speaker_verification").primary_metric == "eer"


def test_legacy_runner_path_is_a_thin_compatibility_layer() -> None:
    source = Path(legacy_runner.__file__).read_text(encoding="utf-8")

    assert "speechbrain_backend" in source
    assert "recipes.voxceleb" not in source
    assert callable(legacy_runner.run_training)


def test_domain_and_infrastructure_modules_do_not_depend_on_agent_or_tool_layers() -> None:
    package_root = Path(agent.runners.__file__).parents[1]
    for module_name in ("core", "data_processing", "hpo", "models", "runners", "tasks", "utils"):
        for path in (package_root / module_name).glob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "from agent.agents" not in source, path
            assert "from agent.tools" not in source, path


def test_speechbrain_is_an_external_dependency() -> None:
    project_root = Path(__file__).parents[2]
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")

    assert not (project_root / "speechbrain").exists()
    assert 'name = "sr-agent"' in pyproject
    assert '"speechbrain==1.0.3"' in pyproject
    assert 'speechbrain/version.txt' not in pyproject
