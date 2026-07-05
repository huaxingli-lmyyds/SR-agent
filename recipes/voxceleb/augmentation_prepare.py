"""SpeechBrain-compatible augmentation data preparation without torchaudio.load."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from agent.runners.speechbrain_dependency import patch_torchaudio_compatibility

patch_torchaudio_compatibility()


def prepare_dataset_from_URL(
    URL: str,
    dest_folder: str,
    ext: str = "wav",
    csv_file: str | None = None,
    max_length: float | None = None,
    **_kwargs: Any,
) -> None:
    """Download/extract augmentation audio and write a SpeechBrain CSV.

    SpeechBrain 1.0.3 prepares these CSV files with ``torchaudio.load``. Newer
    torchaudio releases may require TorchCodec for that call, so SR-agent uses
    soundfile here while preserving the same configuration surface.
    """
    destination = Path(dest_folder)
    destination.mkdir(parents=True, exist_ok=True)
    zip_path = destination / "data.zip"

    if not _has_audio_files(destination, ext) and zip_path.exists():
        _extract_archive(zip_path, destination)

    if not zip_path.exists() and not _has_audio_files(destination, ext):
        from speechbrain.utils.data_utils import download_file

        download_file(URL, str(zip_path))
        _extract_archive(zip_path, destination)

    if not _has_audio_files(destination, ext) and zip_path.exists():
        _extract_archive(zip_path, destination)

    if csv_file is None:
        csv_file = str(destination / f"{ext}.csv")
    _write_augmentation_csv(destination, Path(csv_file), ext, max_length)


def _has_audio_files(folder: Path, ext: str) -> bool:
    return any(folder.rglob(f"*.{ext.lstrip('.')}"))


def _extract_archive(archive: Path, destination: Path) -> None:
    try:
        shutil.unpack_archive(str(archive), str(destination))
    except (shutil.ReadError, ValueError):
        return


def _soundfile_module():
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "soundfile is required to prepare SpeechBrain augmentation CSV files. "
            "Install project dependencies with: pip install -e .[speech]"
        ) from exc
    return sf


def _write_augmentation_csv(
    audio_root: Path,
    csv_path: Path,
    ext: str,
    max_length: float | None,
) -> None:
    sf = _soundfile_module()
    files = sorted(audio_root.rglob(f"*.{ext.lstrip('.')}"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["ID", "duration", "wav", "wav_format", "wav_opts"])
        for index, filename in enumerate(files):
            try:
                info = sf.info(str(filename))
            except RuntimeError:
                continue
            duration = float(info.frames) / float(info.samplerate)
            if max_length is not None and duration > float(max_length):
                duration = float(max_length)
            writer.writerow([
                f"{filename.stem}-{index}",
                duration,
                str(filename),
                ext.lstrip("."),
                "",
            ])

    if len(files) == 0:
        raise RuntimeError(
            f"No '*.{ext}' files found under augmentation folder: {audio_root}"
        )


__all__ = ["prepare_dataset_from_URL"]