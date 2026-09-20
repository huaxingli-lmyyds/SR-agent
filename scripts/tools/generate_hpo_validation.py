#!/usr/bin/env python3
"""Generate a deterministic, speaker-disjoint VoxCeleb HPO protocol.

The script only inventories audio paths and writes protocol metadata. It does
not read audio, modify the dataset, prepare CSV files, or start training.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
import random
import sys
from typing import Iterable


SCHEMA_VERSION = 2
GENERATOR_VERSION = "speaker_disjoint_v2"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--data-folder",
        required=True,
        help="Extracted VoxCeleb root containing wav/<speaker>/<session>/*.wav",
    )
    result.add_argument(
        "--test-pairs",
        required=True,
        help="Final held-out test protocol (normally veri_test2.txt)",
    )
    result.add_argument(
        "--output-dir",
        required=True,
        help="New/empty directory for the generated immutable protocol",
    )
    result.add_argument("--validation-speakers", type=int, default=100)
    result.add_argument("--positive-pairs", type=int, default=10_000)
    result.add_argument("--negative-pairs", type=int, default=10_000)
    result.add_argument(
        "--max-utterances-per-speaker",
        type=int,
        default=20,
        help="Caps embedding extraction cost while retaining session diversity",
    )
    result.add_argument("--min-sessions", type=int, default=2)
    result.add_argument(
        "--speaker-prefix",
        default="id1",
        help="VoxCeleb1 speaker prefix; prevents accidental VoxCeleb2 mixing",
    )
    result.add_argument("--seed", type=int, default=2025)
    return result


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(values: Iterable[str]) -> str:
    digest = sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _speaker(path: str) -> str:
    return path.split("/", 1)[0]


def _session(path: str) -> str:
    parts = path.split("/")
    if len(parts) != 3:
        raise ValueError(f"expected speaker/session/file VoxCeleb path: {path}")
    return parts[1]


def read_test_pairs(path: Path) -> tuple[set[str], set[str], list[str]]:
    """Return test speakers, utterance paths, and normalized pair rows."""
    speakers: set[str] = set()
    utterances: set[str] = set()
    labels: set[str] = set()
    rows: list[str] = []
    for number, raw in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 3 or parts[0] not in {"0", "1"}:
            raise ValueError(f"invalid verification pair at {path}:{number}")
        for index in (1, 2):
            value = parts[index]
            value = value.replace("\\", "/")
            components = value.split("/")
            if (
                len(components) != 3
                or value.startswith("/")
                or ":" in value
                or any(item in {"", ".", ".."} for item in components)
                or not components[-1].lower().endswith(".wav")
            ):
                raise ValueError(
                    f"unsafe/non-VoxCeleb path at {path}:{number}: {value}"
                )
            parts[index] = value
            speakers.add(components[0])
            utterances.add(value)
        labels.add(parts[0])
        rows.append(" ".join(parts))
    if labels != {"0", "1"}:
        raise ValueError(f"test pairs must contain labels 0 and 1: {path}")
    return speakers, utterances, rows


def inventory_audio(wav_root: Path) -> dict[str, list[str]]:
    """Inventory relative paths without opening audio files."""
    by_speaker: dict[str, list[str]] = defaultdict(list)
    for path in sorted(wav_root.rglob("*.wav")):
        relative = path.relative_to(wav_root).as_posix()
        parts = relative.split("/")
        if len(parts) != 3:
            raise ValueError(
                "audio must use wav/<speaker>/<session>/<utterance>.wav; "
                f"found {relative}"
            )
        by_speaker[parts[0]].append(relative)
    if not by_speaker:
        raise ValueError(f"no .wav audio found below {wav_root}")
    return dict(by_speaker)


def _speaker_seed(seed: int, speaker: str) -> int:
    value = sha256(f"{seed}:{speaker}".encode()).digest()[:8]
    return int.from_bytes(value, "big")


def choose_utterances(
    utterances: list[str], maximum: int, seed: int, speaker: str
) -> list[str]:
    """Choose a deterministic session-balanced utterance subset."""
    sessions: dict[str, list[str]] = defaultdict(list)
    for utterance in utterances:
        sessions[_session(utterance)].append(utterance)
    rng = random.Random(_speaker_seed(seed, speaker))
    queues = []
    for session in sorted(sessions):
        queue = sorted(sessions[session])
        rng.shuffle(queue)
        queues.append(queue)
    selected: list[str] = []
    while queues and len(selected) < maximum:
        next_queues = []
        for queue in queues:
            selected.append(queue.pop())
            if queue:
                next_queues.append(queue)
            if len(selected) == maximum:
                break
        queues = next_queues
    return sorted(selected)


def positive_candidates(utterances: list[str]) -> list[tuple[str, str]]:
    sessions: dict[str, list[str]] = defaultdict(list)
    for utterance in utterances:
        sessions[_session(utterance)].append(utterance)
    candidates = []
    for left, right in combinations(sorted(sessions), 2):
        candidates.extend(product(sessions[left], sessions[right]))
    return sorted(tuple(sorted(pair)) for pair in candidates)


def allocate_positive_quotas(
    capacities: dict[str, int], total: int, rng: random.Random
) -> dict[str, int]:
    """Allocate pairs fairly while ensuring every held-out speaker appears."""
    if total < len(capacities):
        raise ValueError(
            "positive-pairs must be at least validation-speakers so every "
            "held-out speaker is represented"
        )
    if sum(capacities.values()) < total:
        raise ValueError(
            f"requested {total} positive pairs but the selected capped audio "
            f"provides only {sum(capacities.values())}; increase "
            "--max-utterances-per-speaker or reduce --positive-pairs"
        )
    quotas = {speaker: 1 for speaker in capacities}
    remaining = total - len(quotas)
    order = sorted(capacities)
    while remaining:
        rng.shuffle(order)
        progressed = False
        for speaker in order:
            if quotas[speaker] < capacities[speaker]:
                quotas[speaker] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("positive-pair quota allocation stalled")
    return quotas


def sample_positive_pairs(
    pools: dict[str, list[str]], total: int, rng: random.Random
) -> list[tuple[int, str, str]]:
    candidates = {
        speaker: positive_candidates(utterances)
        for speaker, utterances in pools.items()
    }
    quotas = allocate_positive_quotas(
        {speaker: len(pairs) for speaker, pairs in candidates.items()},
        total,
        rng,
    )
    rows = []
    for speaker in sorted(candidates):
        for left, right in rng.sample(candidates[speaker], quotas[speaker]):
            rows.append((1, left, right))
    return rows


def sample_negative_pairs(
    pools: dict[str, list[str]], total: int, rng: random.Random
) -> list[tuple[int, str, str]]:
    speakers = sorted(pools)
    speaker_pairs = list(combinations(speakers, 2))
    capacity = sum(
        len(pools[left]) * len(pools[right])
        for left, right in speaker_pairs
    )
    if capacity < total:
        raise ValueError(
            f"requested {total} negative pairs but only {capacity} are possible"
        )
    chosen: set[tuple[str, str]] = set()
    attempts = 0
    maximum_attempts = max(10_000, total * 100)
    while len(chosen) < total and attempts < maximum_attempts:
        rng.shuffle(speaker_pairs)
        for left_speaker, right_speaker in speaker_pairs:
            left = rng.choice(pools[left_speaker])
            right = rng.choice(pools[right_speaker])
            chosen.add(tuple(sorted((left, right))))
            attempts += 1
            if len(chosen) == total or attempts == maximum_attempts:
                break
    if len(chosen) != total:
        raise RuntimeError(
            "negative-pair sampling did not converge; reduce --negative-pairs"
        )
    return [(0, left, right) for left, right in sorted(chosen)]


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def generate_protocol(
    *,
    data_folder: Path,
    test_pairs: Path,
    output_dir: Path,
    validation_speakers: int = 100,
    positive_pairs: int = 10_000,
    negative_pairs: int = 10_000,
    max_utterances_per_speaker: int = 20,
    min_sessions: int = 2,
    speaker_prefix: str = "id1",
    seed: int = 2025,
) -> dict:
    for value, name in (
        (validation_speakers, "validation-speakers"),
        (positive_pairs, "positive-pairs"),
        (negative_pairs, "negative-pairs"),
        (max_utterances_per_speaker, "max-utterances-per-speaker"),
        (min_sessions, "min-sessions"),
    ):
        _positive_int(value, name)
    if validation_speakers < 2:
        raise ValueError("validation-speakers must be at least 2")
    if max_utterances_per_speaker < min_sessions:
        raise ValueError(
            "max-utterances-per-speaker must be at least min-sessions"
        )
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not speaker_prefix or "/" in speaker_prefix or "\\" in speaker_prefix:
        raise ValueError("speaker-prefix must be a nonempty speaker-id prefix")

    data_folder = data_folder.resolve()
    test_pairs = test_pairs.resolve()
    output_dir = output_dir.resolve()
    wav_root = data_folder / "wav"
    if not wav_root.is_dir():
        raise ValueError(f"missing VoxCeleb wav directory: {wav_root}")
    if not test_pairs.is_file():
        raise ValueError(f"test pair file not found: {test_pairs}")
    targets = {
        "pairs": output_dir / "hpo_validation.txt",
        "training_exclusions": output_dir / "training_exclusions.txt",
        "speakers": output_dir / "validation_speakers.txt",
        "utterances": output_dir / "validation_utterances.txt",
        "manifest": output_dir / "hpo_validation_manifest.json",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise ValueError(
            "refusing to overwrite an existing validation protocol: "
            + ", ".join(existing)
        )

    test_speakers, test_utterances, test_pair_lines = read_test_pairs(test_pairs)
    inventory = inventory_audio(wav_root)
    missing_test_audio = sorted(
        utterance
        for utterance in test_utterances
        if not (wav_root / utterance).is_file()
    )
    if missing_test_audio:
        preview = ", ".join(missing_test_audio[:3])
        raise ValueError(
            f"{len(missing_test_audio)} test-pair audio files are absent below "
            f"{wav_root}; examples: {preview}"
        )

    candidate_pools = {}
    for speaker, utterances in sorted(inventory.items()):
        if speaker in test_speakers or not speaker.startswith(speaker_prefix):
            continue
        if len({_session(item) for item in utterances}) < min_sessions:
            continue
        pool = choose_utterances(
            utterances, max_utterances_per_speaker, seed, speaker
        )
        if len({_session(item) for item in pool}) >= min_sessions:
            candidate_pools[speaker] = pool
    if len(candidate_pools) < validation_speakers:
        raise ValueError(
            f"only {len(candidate_pools)} eligible non-test speakers found; "
            f"cannot select {validation_speakers} validation speakers"
        )

    selection_rng = random.Random(seed)
    selected_speakers = sorted(
        selection_rng.sample(sorted(candidate_pools), validation_speakers)
    )
    pools = {
        speaker: candidate_pools[speaker] for speaker in selected_speakers
    }
    pair_rng = random.Random(seed + 1)
    rows = sample_positive_pairs(pools, positive_pairs, pair_rng)
    rows.extend(sample_negative_pairs(pools, negative_pairs, pair_rng))
    pair_rng.shuffle(rows)

    used_speakers = {
        _speaker(path) for _, left, right in rows for path in (left, right)
    }
    overlap = used_speakers & test_speakers
    if overlap:
        raise RuntimeError(
            f"internal error: validation/test speakers overlap: {sorted(overlap)}"
        )
    if used_speakers != set(selected_speakers):
        raise RuntimeError("internal error: not every selected speaker has a pair")
    pair_lines = [f"{label} {left} {right}" for label, left, right in rows]
    selected_utterances = sorted(
        {utterance for pool in pools.values() for utterance in pool}
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text(targets["pairs"], "\n".join(pair_lines) + "\n")
    # Training must exclude both HPO-validation and final-test speakers.  The
    # final-test rows are copied only into this exclusion protocol; HPO never
    # evaluates or selects candidates with them.
    training_exclusion_lines = pair_lines + test_pair_lines
    _write_text(
        targets["training_exclusions"],
        "\n".join(training_exclusion_lines) + "\n",
    )
    _write_text(
        targets["speakers"], "\n".join(selected_speakers) + "\n"
    )
    _write_text(
        targets["utterances"], "\n".join(selected_utterances) + "\n"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "data_folder": str(data_folder),
        "wav_root": str(wav_root),
        "test_pairs": str(test_pairs),
        "seed": seed,
        "speaker_prefix": speaker_prefix,
        "settings": {
            "validation_speakers": validation_speakers,
            "positive_pairs": positive_pairs,
            "negative_pairs": negative_pairs,
            "max_utterances_per_speaker": max_utterances_per_speaker,
            "min_sessions": min_sessions,
            "positive_pairs_cross_session_only": True,
        },
        "counts": {
            "dataset_audio_files": sum(map(len, inventory.values())),
            "dataset_speakers": len(inventory),
            "eligible_speakers": len(candidate_pools),
            "validation_speakers": len(selected_speakers),
            "validation_utterances": len(selected_utterances),
            "validation_pairs": len(rows),
            "positive_pairs": positive_pairs,
            "negative_pairs": negative_pairs,
            "test_speakers": len(test_speakers),
            "test_pairs": len(test_pair_lines),
            "training_exclusion_pairs": len(training_exclusion_lines),
        },
        "integrity": {
            "validation_test_speaker_overlap": 0,
            "test_pairs_sha256": _file_sha256(test_pairs),
            "dataset_inventory_sha256": _text_sha256(
                utterance
                for speaker in sorted(inventory)
                for utterance in inventory[speaker]
            ),
            "hpo_validation_sha256": _file_sha256(targets["pairs"]),
            "training_exclusions_sha256": _file_sha256(
                targets["training_exclusions"]
            ),
            "validation_speakers_sha256": _file_sha256(targets["speakers"]),
            "validation_utterances_sha256": _file_sha256(
                targets["utterances"]
            ),
        },
        "usage": {
            "validation_pairs": str(targets["pairs"]),
            "training_exclusion_pairs": str(targets["training_exclusions"]),
            "final_test_pairs": str(test_pairs),
        },
    }
    _write_text(
        targets["manifest"],
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        manifest = generate_protocol(
            data_folder=Path(args.data_folder),
            test_pairs=Path(args.test_pairs),
            output_dir=Path(args.output_dir),
            validation_speakers=args.validation_speakers,
            positive_pairs=args.positive_pairs,
            negative_pairs=args.negative_pairs,
            max_utterances_per_speaker=args.max_utterances_per_speaker,
            min_sessions=args.min_sessions,
            speaker_prefix=args.speaker_prefix,
            seed=args.seed,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"HPO validation generation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
