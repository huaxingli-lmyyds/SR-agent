"""Deterministic speaker inventory helpers for VoxCeleb training."""

from __future__ import annotations

from pathlib import Path

from recipes.voxceleb.verification_compat import parse_verification_pair


def dataset_speaker_ids(data_folder: str | Path) -> set[str]:
    """Return speaker directory names available below one or more wav roots."""
    roots = [
        Path(item.strip()).expanduser()
        for item in str(data_folder).split(",")
        if item.strip()
    ]
    if not roots:
        raise ValueError("data_folder does not contain a VoxCeleb root")
    speakers: set[str] = set()
    for root in roots:
        wav_root = root.resolve() / "wav"
        if not wav_root.is_dir():
            raise ValueError(f"missing VoxCeleb wav directory: {wav_root}")
        speakers.update(path.name for path in wav_root.iterdir() if path.is_dir())
    if not speakers:
        raise ValueError(f"no speaker directories found below: {roots}")
    return speakers


def excluded_speaker_ids(pairs_file: str | Path) -> set[str]:
    """Return all speakers referenced by a verification/exclusion protocol."""
    path = Path(pairs_file).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"training exclusion pairs file not found: {path}")
    speakers: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        pair = parse_verification_pair(
            line,
            source=str(path),
            line_number=line_number,
        )
        if pair is None:
            continue
        _, left, right = pair
        speakers.update((left.split("/", 1)[0], right.split("/", 1)[0]))
    if not speakers:
        raise ValueError(f"training exclusion pairs contain no speakers: {path}")
    return speakers


def eligible_training_speaker_ids(
    data_folder: str | Path,
    exclusion_pairs: str | Path,
) -> list[str]:
    """Return sorted dataset speakers after applying the training protocol."""
    dataset = dataset_speaker_ids(data_folder)
    excluded = excluded_speaker_ids(exclusion_pairs)
    unknown = excluded - dataset
    if unknown:
        preview = ", ".join(sorted(unknown)[:5])
        raise ValueError(
            f"training exclusion protocol references {len(unknown)} unknown "
            f"speakers; examples: {preview}"
        )
    eligible = sorted(dataset - excluded)
    if not eligible:
        raise ValueError("training exclusion protocol removes every speaker")
    return eligible


__all__ = [
    "dataset_speaker_ids",
    "eligible_training_speaker_ids",
    "excluded_speaker_ids",
]
