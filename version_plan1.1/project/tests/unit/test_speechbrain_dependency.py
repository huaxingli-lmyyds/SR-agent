from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

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

    with pytest.raises(RuntimeError, match=r"pip install -e .\[speech\]"):
        speechbrain_dependency.require_speechbrain()


def test_speechbrain_dependency_rejects_unverified_version(monkeypatch) -> None:
    monkeypatch.setattr(speechbrain_dependency, "version", lambda package: "9.9.9")

    with pytest.raises(RuntimeError, match="expected 1.0.3"):
        speechbrain_dependency.require_speechbrain()


def test_torchaudio_compatibility_patch_restores_removed_backend_api(monkeypatch) -> None:
    fake_torchaudio = SimpleNamespace()
    monkeypatch.setitem(__import__("sys").modules, "torchaudio", fake_torchaudio)

    patched = speechbrain_dependency.patch_torchaudio_compatibility()

    assert patched == ["list_audio_backends", "get_audio_backend", "set_audio_backend", "load"]
    assert fake_torchaudio.list_audio_backends() == ["ffmpeg", "soundfile"]
    assert fake_torchaudio.get_audio_backend() == "soundfile"
    assert fake_torchaudio.set_audio_backend("soundfile") is None
    assert fake_torchaudio.load.__module__ == "agent.runners.soundfile_audio"


def test_torchaudio_compatibility_patch_keeps_existing_backend_api(monkeypatch) -> None:
    def legacy_load(*args, **kwargs):
        return args, kwargs

    fake_torchaudio = SimpleNamespace(
        list_audio_backends=lambda: ["existing"],
        get_audio_backend=lambda: "existing",
        set_audio_backend=lambda backend: backend,
        load=legacy_load,
    )
    monkeypatch.setitem(__import__("sys").modules, "torchaudio", fake_torchaudio)

    patched = speechbrain_dependency.patch_torchaudio_compatibility()

    assert patched == ["load"]
    assert fake_torchaudio.list_audio_backends() == ["existing"]
    assert fake_torchaudio.get_audio_backend() == "existing"
    assert fake_torchaudio.set_audio_backend("x") == "x"
    assert fake_torchaudio.load.__module__ == "agent.runners.soundfile_audio"
