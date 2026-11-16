from __future__ import annotations

from types import SimpleNamespace

from scripts.tools import check_remote_environment as check


def test_major_minor_ignores_cuda_build_suffix() -> None:
    assert check._major_minor("2.5.1+cu121") == "2.5"
    assert check._major_minor("2.9") == "2.9"
    assert check._major_minor(None) is None


def test_collect_environment_reports_mismatch_and_missing_cuda(monkeypatch) -> None:
    versions = {
        "langgraph": "1.2.7",
        "langchain-core": "1.4.8",
        "langchain-openai": "1.3.3",
        "optuna": "4.9.0",
        "speechbrain": "1.0.3",
        "torch": "2.5.1",
        "torchaudio": "2.4.1",
    }

    def fake_package_status(name: str) -> check.PackageStatus:
        return check.PackageStatus(name=name, installed=True, version=versions[name])

    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.1"),
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            get_device_name=lambda index: None,
        ),
    )

    monkeypatch.setattr(check, "package_status", fake_package_status)
    monkeypatch.setattr(check.importlib, "import_module", lambda name: fake_torch)

    report, issues = check.collect_environment(require_cuda=True)

    assert report["cuda"]["available"] is False
    assert "CUDA is required" in "\n".join(issues)
    assert "torch and torchaudio major.minor versions differ" in "\n".join(issues)