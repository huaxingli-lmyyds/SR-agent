"""Prepare labelled raw speech as a portable, frozen verification protocol."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import gcd, isfinite
from pathlib import Path
import random
import tempfile

from .contracts import DataOperationResult, OperationImpact


@dataclass(frozen=True)
class PreparationOptions:
    seed: int = 0
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    dev_fraction: float = 0.2
    sample_rate: int = 16000
    min_duration_seconds: float = 0.1
    labels_csv: str | None = None

    def __post_init__(self):
        for name in ("seed", "sample_rate"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (1 if name == "sample_rate" else 0)
            ):
                raise ValueError(f"{name} must be a valid integer")
        for name in ("validation_fraction", "test_fraction", "dev_fraction"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0 < value < 1
            ):
                raise ValueError(f"{name} must be between 0 and 1")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError(
                "validation and test fractions must leave training speakers"
            )
        if (
            isinstance(self.min_duration_seconds, bool)
            or not isinstance(self.min_duration_seconds, (int, float))
            or not isfinite(self.min_duration_seconds)
            or self.min_duration_seconds <= 0
        ):
            raise ValueError("min_duration_seconds must be finite and positive")

    @classmethod
    def from_dict(cls, values):
        if values is not None and not isinstance(values, dict):
            raise ValueError("preparation options must be an object")
        values = dict(values or {})
        unknown = values.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"unknown preparation options: {sorted(unknown)}")
        return cls(**values)


def _inventory(source: Path, options: PreparationOptions):
    from .audio import AUDIO_EXTENSIONS

    root = source / "wav" if (source / "wav").is_dir() else source
    rows = []
    if options.labels_csv:
        with (
            Path(options.labels_csv)
            .expanduser()
            .open(encoding="utf-8-sig", newline="") as stream
        ):
            reader = csv.DictReader(stream)
            if not {"path", "speaker_id"} <= set(reader.fieldnames or []):
                raise ValueError(
                    "labels_csv requires path,speaker_id and optional session_id columns"
                )
            for item in reader:
                relative = Path(item["path"])
                if relative.is_absolute():
                    raise ValueError(
                        "label paths must be relative to the raw audio root"
                    )
                rows.append(
                    (
                        root / relative,
                        item["speaker_id"].strip(),
                        item.get("session_id", "").strip() or relative.stem,
                    )
                )
    else:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                parts = path.relative_to(root).parts
                if len(parts) < 2:
                    raise ValueError(
                        "raw audio needs speaker directories or a labels_csv; identities cannot be inferred from unlabelled audio"
                    )
                rows.append(
                    (path, parts[0], "/".join(parts[1:-1]) or path.stem)
                )
    if not rows:
        raise ValueError("no labelled audio files found")
    seen = set()
    result = []
    for path, speaker, session in rows:
        resolved = path.resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError("audio path escapes raw audio root")
        if not speaker or not session or resolved in seen:
            raise ValueError(
                "speaker/session labels must be nonempty and audio paths unique"
            )
        if (
            not resolved.is_file()
            or resolved.suffix.lower() not in AUDIO_EXTENSIONS
        ):
            raise ValueError(
                f"labelled audio file missing or unsupported: {path}"
            )
        seen.add(resolved)
        result.append(
            {
                "source": resolved,
                "relative": path.relative_to(root).as_posix(),
                "speaker": speaker,
                "session": session,
            }
        )
    return sorted(result, key=lambda row: row["relative"])


def _group(rows, key):
    groups = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    return groups


def _split(rows, options):
    rng = random.Random(options.seed)
    speakers = _group(rows, "speaker")
    if any(len(items) < 2 for items in speakers.values()):
        raise ValueError(
            "every retained speaker needs at least two distinct recordings"
        )
    identities = sorted(speakers)
    rng.shuffle(identities)
    n_valid = max(2, round(len(identities) * options.validation_fraction))
    n_test = max(2, round(len(identities) * options.test_fraction))
    if len(identities) - n_valid - n_test < 2:
        raise ValueError(
            "speaker-disjoint train/validation/test requires at least six eligible speakers and two training speakers"
        )
    partitions = {
        "hpo_validation": identities[:n_valid],
        "final_test": identities[n_valid : n_valid + n_test],
        "train": identities[n_valid + n_test :],
    }
    splits = {"train": [], "dev": [], "hpo_validation": [], "final_test": []}
    for role, names in partitions.items():
        for name in names:
            if role != "train":
                splits[role].extend(speakers[name])
                continue
            sessions = _group(speakers[name], "session")
            groups = sorted(sessions)
            rng.shuffle(groups)
            count = min(
                len(groups) - 1,
                max(1, round(len(groups) * options.dev_fraction)),
            )
            for index, session in enumerate(groups):
                splits["dev" if index < count else "train"].extend(
                    sessions[session]
                )
    if not splits["dev"]:
        raise ValueError(
            "internal dev needs separate recording sessions; supply session_id or separate recording files"
        )
    return splits


def _pairs(rows):
    """Linear-size balanced protocol; every held-out speaker is represented."""
    speakers = _group(rows, "speaker")
    names = sorted(speakers)
    lines = []
    for index, name in enumerate(names):
        own = sorted(speakers[name], key=lambda row: row["utterance"])
        other = sorted(
            speakers[names[(index + 1) % len(names)]],
            key=lambda row: row["utterance"],
        )
        for offset, row in enumerate(own):
            lines.append(
                f"1 {row['utterance']} {own[(offset + 1) % len(own)]['utterance']}"
            )
            lines.append(
                f"0 {row['utterance']} {other[offset % len(other)]['utterance']}"
            )
    # Two-utterance positive pairs otherwise occur in both directions.
    unique = {}
    for line in lines:
        label, left, right = line.split()
        key = (label, *sorted((left, right)))
        unique.setdefault(key, line)
    return list(unique.values())


def prepare_audio(source, output, options=None):
    """Never modify source audio; publish only a complete derived directory."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    source, output = Path(source).resolve(), Path(output).resolve()
    options = (
        options
        if isinstance(options, PreparationOptions)
        else PreparationOptions.from_dict(options)
    )
    if (
        output == source
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError("prepared output must be separate from raw audio")
    if output.exists():
        raise ValueError(
            "prepared output already exists; use a new dataset version directory"
        )
    inventory = _inventory(source, options)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".prepare-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        retained, rejected, digests = [], [], {}
        for item in inventory:
            try:
                audio, rate = sf.read(
                    item["source"], dtype="float32", always_2d=True
                )
                if (
                    not np.isfinite(audio).all()
                    or len(audio) / rate < options.min_duration_seconds
                ):
                    raise ValueError("nonfinite or too short audio")
                audio = audio.mean(axis=1)
                if rate != options.sample_rate:
                    divisor = gcd(rate, options.sample_rate)
                    audio = resample_poly(
                        audio, options.sample_rate // divisor, rate // divisor
                    )
                # Hash the actual published representation, also finding duplicate encodings.
                pcm = (np.clip(audio, -1, 1) * 32767).round().astype("<i2")
                digest = sha256(pcm.tobytes()).hexdigest()
            except (OSError, RuntimeError, ValueError) as exc:
                rejected.append({"path": item["relative"], "reason": str(exc)})
                continue
            if digest in digests:
                if digests[digest] != item["speaker"]:
                    raise ValueError(
                        "identical audio has conflicting speaker labels"
                    )
                rejected.append(
                    {"path": item["relative"], "reason": "duplicate_audio"}
                )
                continue
            digests[digest] = item["speaker"]
            speaker = "id" + sha256(item["speaker"].encode()).hexdigest()[:16]
            session = sha256(item["session"].encode()).hexdigest()[:16]
            utterance = f"{speaker}/{session}/{digest}.wav"
            target = staging / "wav" / utterance
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target, pcm, options.sample_rate, subtype="PCM_16")
            retained.append(
                {
                    "speaker": speaker,
                    "session": session,
                    "utterance": utterance,
                    "duration": len(pcm) / options.sample_rate,
                    "frames": len(pcm),
                    "source": item["relative"],
                    "source_label": item["speaker"],
                    "sha256": sha256(target.read_bytes()).hexdigest(),
                }
            )
        if not retained:
            raise ValueError("no usable audio remains after validation")
        splits = _split(retained, options)
        for role, rows in splits.items():
            with (staging / f"{role}.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["ID", "duration", "wav", "start", "stop", "spk_id"]
                )
                for row in sorted(rows, key=lambda row: row["utterance"]):
                    writer.writerow(
                        [
                            row["utterance"][:-4],
                            row["duration"],
                            "$data_root/wav/" + row["utterance"],
                            0,
                            row["frames"],
                            row["speaker"],
                        ]
                    )
        validation, final_test = (
            _pairs(splits["hpo_validation"]),
            _pairs(splits["final_test"]),
        )
        for name, lines in (
            ("validation_pairs.txt", validation),
            ("final_test_pairs.txt", final_test),
            ("training_exclusion_pairs.txt", validation + final_test),
        ):
            (staging / name).write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
        files = {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in staging.iterdir()
            if p.is_file()
        }
        protocol = {
            "schema_version": 1,
            "options": {
                k: v for k, v in asdict(options).items() if k != "labels_csv"
            },
            "files": files,
            "splits": {
                role: {
                    "samples": len(rows),
                    "speakers": len(_group(rows, "speaker")),
                }
                for role, rows in splits.items()
            },
            "audio": sorted(retained, key=lambda row: row["utterance"]),
            "rejected": rejected,
        }
        (staging / "prepared_protocol.json").write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        staging.rename(output)
    return {
        "root": str(output),
        "protocol_sha256": sha256(
            (output / "prepared_protocol.json").read_bytes()
        ).hexdigest(),
        "sample_rate": options.sample_rate,
        "training_speaker_count": protocol["splits"]["train"]["speakers"],
        "rejected_count": len(rejected),
    }


class PrepareAudioProcessor:
    operation_name = "prepare_speaker_dataset"
    advisor_allowed = False
    supported_data_types = {"audio"}
    parameter_schema = {
        name: {"advisor_allowed": False, "default": field.default}
        for name, field in PreparationOptions.__dataclass_fields__.items()
    }

    def validate(self, dataset, parameters):
        PreparationOptions.from_dict(
            {k: v for k, v in parameters.items() if not k.startswith("_")}
        )
        if dataset.task_type != "speaker_verification":
            raise ValueError(
                "speaker dataset preparation requires speaker_verification"
            )

    def preview(self, dataset, parameters):
        # Normalization may rewrite all samples. A bounded scan cannot prove size.
        return OperationImpact(
            self.operation_name,
            creates_new_version=True,
            details={"scan_complete": False},
        )

    def execute(self, dataset, parameters):
        prepared = prepare_audio(
            dataset.source_uri,
            parameters["_output_uri"],
            {k: v for k, v in parameters.items() if not k.startswith("_")},
        )
        return DataOperationResult(
            status="success",
            operation=self.operation_name,
            output_dataset_uri=prepared["root"],
            consumer_ready=True,
            artifacts=[
                {
                    "type": "data_protocol",
                    "path": str(
                        Path(prepared["root"]) / "prepared_protocol.json"
                    ),
                }
            ],
            details={"prepared_data": prepared},
        )


def validate_prepared_data(prepared, *, verify_audio=False):
    """Verify the published protocol before consuming its split assignments."""
    root = Path(prepared["root"]).resolve()
    path = root / "prepared_protocol.json"
    if sha256(path.read_bytes()).hexdigest() != prepared["protocol_sha256"]:
        raise ValueError("prepared data protocol changed")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported prepared data protocol")
    for name, expected in protocol["files"].items():
        target = (root / name).resolve()
        if (
            not target.is_relative_to(root)
            or sha256(target.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"prepared data manifest changed: {name}")
    if verify_audio:
        for row in protocol["audio"]:
            target = (root / "wav" / row["utterance"]).resolve()
            if (
                not target.is_relative_to(root)
                or sha256(target.read_bytes()).hexdigest() != row["sha256"]
            ):
                raise ValueError("prepared audio changed")
    return protocol


def prepared_data_reference(root):
    """Adopt an existing prepared directory while retaining its frozen splits."""
    root = Path(root).resolve()
    path = root / "prepared_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    reference = {
        "root": str(root),
        "protocol_sha256": sha256(path.read_bytes()).hexdigest(),
        "sample_rate": protocol["options"]["sample_rate"],
        "training_speaker_count": protocol["splits"]["train"]["speakers"],
        "rejected_count": len(protocol.get("rejected", [])),
    }
    validate_prepared_data(reference)
    return reference


def prepared_training_overrides(prepared):
    protocol = validate_prepared_data(prepared)
    root = Path(prepared["root"]).resolve()
    return {
        "data_folder": str(root),
        "train_annotation": str(root / "train.csv"),
        "valid_annotation": str(root / "dev.csv"),
        "sample_rate": protocol["options"]["sample_rate"],
        "out_n_neurons": protocol["splits"]["train"]["speakers"],
        "skip_prep": True,
    }
