"""Protect held-out identities when a source contains the complete corpus."""

from hashlib import sha256
from pathlib import Path

from recipes.voxceleb.verification_compat import parse_verification_pair

from .contracts import DatasetSpec


def bind_evaluation_scope(
    dataset: DatasetSpec, exclusion_pairs: str | None
) -> None:
    if dataset.task_type != "speaker_verification" or not exclusion_pairs:
        return
    protocol = Path(exclusion_pairs).expanduser().resolve()
    speakers = set()
    for index, line in enumerate(
        protocol.read_text(encoding="utf-8").splitlines(), 1
    ):
        pair = parse_verification_pair(
            line, source=str(protocol), line_number=index
        )
        if pair is not None:
            speakers.update(path.split("/")[0] for path in pair[1:])
    source = Path(dataset.source_uri)
    speech = source / "wav" if (source / "wav").is_dir() else source
    dataset.metadata.update(
        {
            "protected_speakers": sorted(speakers),
            "exclusion_protocol_sha256": sha256(
                protocol.read_bytes()
            ).hexdigest(),
            "contains_protected_speakers": any(
                (speech / speaker).exists() for speaker in speakers
            ),
        }
    )


def validate_mutation_scope(dataset: DatasetSpec) -> None:
    if dataset.metadata.get("contains_protected_speakers"):
        raise ValueError(
            "dataset contains protected evaluation speakers; filtering or rewriting "
            "requires a training-only view. Keep validation/test audio immutable."
        )
