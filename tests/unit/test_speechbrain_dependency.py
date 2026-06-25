from importlib.metadata import PackageNotFoundError

import pytest

from agent.runners import speechbrain_dependency


def test_speechbrain_dependency_accepts_pinned_version(monkeypatch) -> None:
    monkeypatch.setattr(
        speechbrain_dependency,
        "version",
        lambda package: speechbrain_dependency.SUPPORTED_SPEECHBRAIN_VERSION,
    )

    assert (
        speechbrain_dependency.require_speechbrain()
        == speechbrain_dependency.SUPPORTED_SPEECHBRAIN_VERSION
    )


def test_speechbrain_dependency_reports_missing_installation(monkeypatch) -> None:
    def missing(_package):
        raise PackageNotFoundError()

    monkeypatch.setattr(speechbrain_dependency, "version", missing)

    with pytest.raises(RuntimeError, match="pip install -e"):
        speechbrain_dependency.require_speechbrain()


def test_speechbrain_dependency_rejects_unverified_version(monkeypatch) -> None:
    monkeypatch.setattr(speechbrain_dependency, "version", lambda package: "9.9.9")

    with pytest.raises(RuntimeError, match="expected 1.0.3"):
        speechbrain_dependency.require_speechbrain()
