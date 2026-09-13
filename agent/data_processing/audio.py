"""Audio and speaker-verification quality controls."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .contracts import (
    DataIssue,
    DataOperationResult,
    DataProfile,
    DatasetSpec,
    OperationImpact,
    QualityPolicy,
)
from .materialization import (
    materialize_file,
    materialize_tree,
    validate_materialization_mode,
)
from .registry import register_processor, register_profiler

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
SPEAKER_FIELDS = ("spk_id", "speaker_id", "speaker", "speaker-id")
PATH_FIELDS = ("wav", "audio", "audio_path", "path", "file")
SPLITS = {"train", "valid", "validation", "dev", "test", "enrol"}
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class AudioDataProfiler:
    data_type = "audio"

    def profile(
        self, dataset: DatasetSpec, policy: QualityPolicy
    ) -> DataProfile:
        source = Path(dataset.source_uri).resolve()
        files = list(_iter_audio_files(source, policy.max_audio_files))
        manifest = _profile_manifests(source, policy)
        referenced = [Path(value) for value in manifest["existing_audio_paths"]]
        probes = list(dict.fromkeys([*files, *referenced]))[
            : policy.max_audio_files
        ]
        infos, unreadable = [], []
        for path in probes:
            try:
                import soundfile as sf

                info = sf.info(str(path))
                infos.append(
                    {
                        "path": str(path),
                        "duration": float(info.duration),
                        "sample_rate": int(info.samplerate),
                        "channels": int(info.channels),
                        "format": str(info.format),
                        "bytes": path.stat().st_size,
                        "speaker": _speaker(source, path),
                    }
                )
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                unreadable.append(f"{path}: {exc}")

        signal = _profile_signals(infos[: policy.max_signal_files], policy)
        issues = list(manifest["issues"])
        if not source.exists():
            issues.append(
                _issue(
                    "source_missing", "error", "Dataset source does not exist."
                )
            )
        if not probes:
            issues.append(
                _issue("audio_empty", "error", "No audio files were found.")
            )
        if unreadable:
            issues.append(
                _issue(
                    "audio_unreadable",
                    "error",
                    "Audio files cannot be decoded.",
                    "filter_unreadable_audio",
                    {"count": len(unreadable), "examples": unreadable[:20]},
                )
            )

        durations = [item["duration"] for item in infos]
        short = sum(value < policy.min_duration_seconds for value in durations)
        long = sum(value > policy.max_duration_seconds for value in durations)
        if short or long:
            issues.append(
                _issue(
                    "duration_out_of_range",
                    "warning",
                    "Audio duration is outside policy bounds.",
                    "filter_by_duration",
                    {"short": short, "long": long},
                )
            )
        rates = Counter(item["sample_rate"] for item in infos)
        channels = Counter(item["channels"] for item in infos)
        unexpected_rates = sum(
            count
            for value, count in rates.items()
            if value not in policy.expected_sample_rates
        )
        unexpected_channels = sum(
            count
            for value, count in channels.items()
            if value not in policy.expected_channels
        )
        if unexpected_rates:
            issues.append(
                _issue(
                    "sample_rate_mismatch",
                    "warning",
                    "Unexpected sample rates were found.",
                    evidence={"count": unexpected_rates},
                )
            )
        if unexpected_channels:
            issues.append(
                _issue(
                    "channel_mismatch",
                    "warning",
                    "Unexpected channel counts were found.",
                    evidence={"count": unexpected_channels},
                )
            )
        for code, key, message in (
            (
                "excessive_silence",
                "excessive_silence_count",
                "Excessively silent audio was found.",
            ),
            (
                "audio_clipping",
                "clipped_file_count",
                "Clipped audio was found.",
            ),
            (
                "near_zero_energy",
                "near_zero_file_count",
                "Near-zero-energy audio was found.",
            ),
        ):
            if signal[key]:
                issues.append(
                    _issue(
                        code,
                        "warning",
                        message,
                        evidence={"count": signal[key]},
                    )
                )

        speaker_counts = Counter(manifest["speaker_counts"])
        if not speaker_counts:
            speaker_counts.update(
                item["speaker"] for item in infos if item["speaker"]
            )
        low_resource = sum(
            count < policy.min_samples_per_speaker
            for count in speaker_counts.values()
        )
        if low_resource:
            issues.append(
                _issue(
                    "speaker_long_tail",
                    "warning",
                    "Some speakers have too few samples.",
                    evidence={"speaker_count": low_resource},
                )
            )
        issues = _deduplicate_issues(issues)
        metrics = {
            "missing_source": not source.exists(),
            "audio_file_count": len(probes),
            "decoded_audio_count": len(infos),
            "unreadable_audio_count": len(unreadable),
            "missing_audio_count": manifest["missing_audio_count"],
            "path_violation_count": manifest["path_violation_count"],
            "missing_speaker_count": manifest["missing_speaker_count"],
            "invalid_speaker_count": manifest["invalid_speaker_count"],
            "duplicate_sample_count": manifest["duplicate_sample_count"],
            "speaker_overlap_count": manifest["speaker_overlap_count"],
            "file_overlap_count": manifest["file_overlap_count"],
            "invalid_trial_count": manifest["invalid_trial_count"],
            "duplicate_trial_count": manifest["duplicate_trial_count"],
            "duration_outlier_count": short + long,
            "unexpected_sample_rate_count": unexpected_rates,
            "unexpected_channel_count": unexpected_channels,
            **signal,
            "issue_count": len(issues),
            "error_count": sum(
                item.severity in {"error", "critical"} for item in issues
            ),
            "warning_count": sum(item.severity == "warning" for item in issues),
        }
        return DataProfile(
            dataset=dataset,
            sample_count=len(probes),
            schema=manifest["schema"],
            distributions={
                "duration_seconds": _summary(durations),
                "sample_rates": dict(rates),
                "channels": dict(channels),
                "formats": dict(Counter(item["format"] for item in infos)),
                "samples_per_speaker": _summary(list(speaker_counts.values())),
                "speakers_per_split": manifest["speakers_per_split"],
                "trial_labels": manifest["trial_labels"],
            },
            quality_metrics=metrics,
            issues=issues,
            extensions={
                "audio": {
                    "probe_limit": policy.max_audio_files,
                    "probe_limited": len(files) >= policy.max_audio_files,
                    "signal_probe_count": signal["signal_probe_count"],
                    "decoded_examples": infos[:20],
                    "unreadable_examples": unreadable[:20],
                },
                "manifest": manifest["details"],
                "quality_policy": policy.to_dict(),
            },
        )


class _AuditProcessor:
    supported_data_types = {"audio"}
    parameter_schema: Dict[str, Any] = {}
    metric_keys: tuple[str, ...] = ()
    error_message = "data quality validation failed"

    def validate(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> None:
        if not dataset.source_uri:
            raise ValueError("dataset source_uri is required")

    def preview(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> OperationImpact:
        profile = self._profile(dataset, parameters)
        affected = sum(profile.quality_metrics[key] for key in self.metric_keys)
        return OperationImpact(
            self.operation_name, profile.sample_count, affected
        )

    def _blocking_count(
        self,
        profile: DataProfile,
        policy: QualityPolicy,
    ) -> int:
        return sum(profile.quality_metrics[key] for key in self.metric_keys)

    @staticmethod
    def _profile(
        dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> DataProfile:
        cached = parameters.get("_preview_profile")
        if isinstance(cached, DataProfile):
            return cached
        profile = AudioDataProfiler().profile(dataset, _policy(parameters))
        parameters["_preview_profile"] = profile
        return profile

    def execute(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> DataOperationResult:
        self.validate(dataset, parameters)
        policy = _policy(parameters)
        profile = self._profile(dataset, parameters)
        blocking = self._blocking_count(profile, policy)
        return DataOperationResult(
            status="failed" if blocking else "success",
            operation=self.operation_name,
            after_metrics=profile.quality_metrics,
            error=self.error_message if blocking else None,
        )


class ValidateAudioFilesProcessor(_AuditProcessor):
    operation_name = "validate_audio_files"
    metric_keys = (
        "unreadable_audio_count",
        "missing_audio_count",
        "path_violation_count",
    )
    error_message = "audio file validation failed"


class CheckSpeakerSplitProcessor(_AuditProcessor):
    operation_name = "check_speaker_split"
    metric_keys = ("speaker_overlap_count", "file_overlap_count")
    error_message = "speaker or file leakage detected across splits"

    def _blocking_count(
        self,
        profile: DataProfile,
        policy: QualityPolicy,
    ) -> int:
        metrics = profile.quality_metrics
        speaker = (
            metrics["speaker_overlap_count"]
            if policy.require_disjoint_speakers
            else 0
        )
        files = (
            metrics["file_overlap_count"]
            if policy.require_disjoint_files
            else 0
        )
        return speaker + files


class ValidateVerificationTrialsProcessor(_AuditProcessor):
    operation_name = "validate_verification_trials"
    metric_keys = ("invalid_trial_count", "duplicate_trial_count")
    error_message = "verification trial validation failed"

    def _blocking_count(
        self,
        profile: DataProfile,
        policy: QualityPolicy,
    ) -> int:
        affected = sum(profile.quality_metrics[key] for key in self.metric_keys)
        return affected if policy.require_valid_trials else 0


class NormalizeManifestProcessor:
    operation_name = "normalize_manifest"
    supported_data_types = {"audio"}
    parameter_schema = {
        "csv_glob": {"type": "string", "default": "**/*.csv"},
        "drop_invalid": {"type": "boolean", "default": True},
        "materialization_mode": {
            "type": "string",
            "default": "hardlink",
            "description": (
                "Reuse immutable audio bytes; fall back to copy when linking "
                "is unavailable."
            ),
        },
        "materialize_complete_dataset": {
            "type": "boolean",
            "default": False,
            "advisor_allowed": False,
        },
    }

    def validate(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> None:
        _validate_output(parameters)
        if "drop_invalid" in parameters and not isinstance(
            parameters["drop_invalid"], bool
        ):
            raise ValueError("drop_invalid must be a boolean")
        validate_materialization_mode(
            str(parameters.get("materialization_mode", "hardlink"))
        )

    def preview(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> OperationImpact:
        source = Path(dataset.source_uri).resolve()
        audit = _profile_manifests(source)
        manifest_size = sum(
            path.stat().st_size
            for path in source.glob(
                str(parameters.get("csv_glob") or "**/*.csv")
            )
        )
        mode = validate_materialization_mode(
            str(parameters.get("materialization_mode", "hardlink"))
        )
        size = manifest_size
        if parameters.get("materialize_complete_dataset") and mode == "copy":
            size = _tree_size(source)
        affected = sum(
            audit[key]
            for key in (
                "missing_audio_count",
                "path_violation_count",
                "duplicate_sample_count",
            )
        )
        return OperationImpact(
            self.operation_name,
            audit["row_count"],
            affected,
            size,
            True,
            {"materialization_mode": mode},
        )

    def execute(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> DataOperationResult:
        self.validate(dataset, parameters)
        source, output = _safe_output(dataset, parameters)
        materialize = bool(
            parameters.get("materialize_complete_dataset", False)
        )
        mode = validate_materialization_mode(
            str(parameters.get("materialization_mode", "hardlink"))
        )
        if materialize:
            materialization = materialize_tree(source, output, mode)
        else:
            output.mkdir(parents=True, exist_ok=True)
            materialization = {"hardlink": 0, "copy": 0}
        materialization["generated"] = 0
        manifests = list(
            source.glob(str(parameters.get("csv_glob") or "**/*.csv"))
        )
        before = after = dropped = 0
        artifacts = []
        for path in manifests:
            relative = path.relative_to(source)
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if materialize and destination.exists():
                method = "copy"
                try:
                    if path.samefile(destination):
                        method = "hardlink"
                except OSError:
                    pass
                destination.unlink()
                materialization[method] = max(
                    0, materialization[method] - 1
                )
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames or [], list(reader)
            before += len(rows)
            cleaned, seen = [], set()
            for row in rows:
                normalized = {
                    key: str(value or "").strip() for key, value in row.items()
                }
                path_key = _first_field(normalized, PATH_FIELDS)
                audio_path = (
                    _resolve_audio_path(
                        source, path, normalized.get(path_key, "")
                    )
                    if path_key
                    else None
                )
                invalid = bool(
                    path_key and (audio_path is None or not audio_path.exists())
                )
                fingerprint = tuple(sorted(normalized.items()))
                if fingerprint in seen or (
                    invalid and parameters.get("drop_invalid", True)
                ):
                    dropped += 1
                    continue
                seen.add(fingerprint)
                cleaned.append(normalized)
            with destination.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(cleaned)
            materialization["generated"] += 1
            after += len(cleaned)
            artifacts.append(
                {
                    "type": "manifest",
                    "name": relative.as_posix(),
                    "path": str(destination),
                }
            )
        if not manifests:
            return DataOperationResult(
                status="failed",
                operation=self.operation_name,
                error="no CSV manifests found",
            )
        return DataOperationResult(
            status="success",
            operation=self.operation_name,
            output_dataset_uri=str(output),
            consumer_ready=materialize,
            before_metrics={"manifest_row_count": before},
            after_metrics={
                "manifest_row_count": after,
                "dropped_row_count": dropped,
            },
            artifacts=artifacts,
            details={
                "materialization_mode": mode,
                "materialization": materialization,
            },
        )


class _AudioSubsetProcessor:
    supported_data_types = {"audio"}
    parameter_schema = {
        "materialization_mode": {
            "type": "string",
            "default": "hardlink",
            "description": (
                "Reuse immutable audio bytes; fall back to copy when linking "
                "is unavailable."
            ),
        }
    }

    def validate(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> None:
        _validate_output(parameters)
        validate_materialization_mode(
            str(parameters.get("materialization_mode", "hardlink"))
        )

    @staticmethod
    def _relative_path(source: Path, path: Path) -> Path:
        return Path(path.name) if source.is_file() else path.relative_to(source)

    def _selection_path(self, parameters: Dict[str, Any]) -> Path:
        output = Path(str(parameters["_output_uri"])).resolve()
        return output.parent / ".preview-cache" / (
            f"{output.name}.selection.jsonl"
        )

    def _accept(self, path: Path, parameters: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def _scan_selection(
        self,
        dataset: DatasetSpec,
        parameters: Dict[str, Any],
        cache_path: Path,
    ) -> Dict[str, Any]:
        source = Path(dataset.source_uri).resolve()
        scanned = selected = estimated = 0
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as stream:
            for path in _iter_audio_files(source, None):
                scanned += 1
                if not self._accept(path, parameters):
                    continue
                relative = self._relative_path(source, path)
                stream.write(json.dumps(relative.as_posix()) + "\n")
                selected += 1
                estimated += path.stat().st_size
        return {
            "scanned_samples": scanned,
            "selected_samples": selected,
            "selected_audio_bytes": estimated,
            "scan_complete": True,
        }

    def _selection(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> tuple[Path, Dict[str, Any], bool]:
        source = str(Path(dataset.source_uri).resolve())
        cache_path = self._selection_path(parameters)
        cached_source = parameters.get("_selection_source_uri")
        stats = parameters.get("_selection_stats")
        if cache_path.exists() and cached_source == source and isinstance(
            stats, dict
        ):
            return cache_path, stats, True
        stats = self._scan_selection(dataset, parameters, cache_path)
        parameters["_selection_source_uri"] = source
        parameters["_selection_stats"] = stats
        return cache_path, stats, False

    def _selected_paths(
        self, source: Path, cache_path: Path
    ) -> Iterable[Path]:
        root = source.parent if source.is_file() else source
        with cache_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                relative = Path(json.loads(line))
                candidate = (root / relative).resolve()
                candidate.relative_to(root)
                yield candidate

    def preview(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> OperationImpact:
        self.validate(dataset, parameters)
        cache_path, stats, _ = self._selection(dataset, parameters)
        mode = validate_materialization_mode(
            str(parameters.get("materialization_mode", "hardlink"))
        )
        selected_bytes = int(stats["selected_audio_bytes"])
        estimated_bytes = (
            selected_bytes if mode == "copy" else cache_path.stat().st_size
        )
        return OperationImpact(
            self.operation_name,
            int(stats["scanned_samples"]),
            int(stats["scanned_samples"] - stats["selected_samples"]),
            estimated_bytes,
            True,
            {
                "scan_complete": bool(stats["scan_complete"]),
                "materialization_mode": mode,
                "selected_audio_bytes": selected_bytes,
            },
        )

    def execute(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> DataOperationResult:
        self.validate(dataset, parameters)
        source, output = _safe_output(dataset, parameters)
        cache_path, stats, reused = self._selection(dataset, parameters)
        output.mkdir(parents=True, exist_ok=True)
        mode = validate_materialization_mode(
            str(parameters.get("materialization_mode", "hardlink"))
        )
        materialization = {"hardlink": 0, "copy": 0, "generated": 1}
        manifest = output / "manifest.csv"
        try:
            with manifest.open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["speaker_id", "wav", "split"]
                )
                writer.writeheader()
                for path in self._selected_paths(source, cache_path):
                    relative = self._relative_path(source, path)
                    destination = output / relative
                    method = materialize_file(path, destination, mode)
                    materialization[method] += 1
                    writer.writerow(
                        {
                            "speaker_id": _speaker(source, path) or "unknown",
                            "wav": relative.as_posix(),
                            "split": _split(source, path) or "unspecified",
                        }
                    )
        finally:
            cache_path.unlink(missing_ok=True)
        scanned = int(stats["scanned_samples"])
        selected = int(stats["selected_samples"])
        return DataOperationResult(
            status="success" if selected else "failed",
            operation=self.operation_name,
            output_dataset_uri=str(output),
            consumer_ready=bool(selected),
            before_metrics={"audio_file_count": scanned},
            after_metrics={
                "audio_file_count": selected,
                "filtered_audio_count": scanned - selected,
            },
            artifacts=[
                {
                    "type": "manifest",
                    "name": "manifest.csv",
                    "path": str(manifest),
                }
            ],
            details={
                "materialization": materialization,
                "materialization_mode": mode,
                "selection_reused": reused,
                "scan_complete": bool(stats["scan_complete"]),
            },
            error=None if selected else "operation removed every audio file",
        )


class FilterUnreadableAudioProcessor(_AudioSubsetProcessor):
    operation_name = "filter_unreadable_audio"

    def _accept(self, path: Path, parameters: Dict[str, Any]) -> bool:
        import soundfile as sf

        try:
            sf.info(str(path))
            return True
        except (OSError, RuntimeError, ValueError):
            return False


class FilterByDurationProcessor(_AudioSubsetProcessor):
    operation_name = "filter_by_duration"
    parameter_schema = {
        **_AudioSubsetProcessor.parameter_schema,
        "min_seconds": {"type": "number", "default": 0.1},
        "max_seconds": {"type": "number", "default": 30.0},
    }

    def validate(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> None:
        super().validate(dataset, parameters)
        minimum, maximum = (
            float(parameters.get("min_seconds", 0.1)),
            float(parameters.get("max_seconds", 30.0)),
        )
        if minimum < 0 or maximum <= minimum:
            raise ValueError("duration bounds are invalid")

    def _accept(self, path: Path, parameters: Dict[str, Any]) -> bool:
        import soundfile as sf

        minimum, maximum = (
            float(parameters.get("min_seconds", 0.1)),
            float(parameters.get("max_seconds", 30.0)),
        )
        try:
            return minimum <= float(sf.info(str(path)).duration) <= maximum
        except (OSError, RuntimeError, ValueError):
            return False


class BuildDebugSubsetProcessor(_AudioSubsetProcessor):
    operation_name = "build_debug_subset"
    parameter_schema = {
        **_AudioSubsetProcessor.parameter_schema,
        "max_samples": {"type": "integer", "default": 100},
        "max_per_speaker": {"type": "integer", "default": 2},
    }

    def validate(
        self, dataset: DatasetSpec, parameters: Dict[str, Any]
    ) -> None:
        super().validate(dataset, parameters)
        for name, default in (("max_samples", 100), ("max_per_speaker", 2)):
            if int(parameters.get(name, default)) <= 0:
                raise ValueError(f"{name} must be positive")

    def _scan_selection(
        self,
        dataset: DatasetSpec,
        parameters: Dict[str, Any],
        cache_path: Path,
    ) -> Dict[str, Any]:
        source = Path(dataset.source_uri).resolve()
        limit = int(parameters.get("max_samples", 100))
        per_speaker = int(parameters.get("max_per_speaker", 2))
        counts: Counter[str] = Counter()
        scanned = selected = estimated = 0
        scan_complete = True
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as stream:
            for path in _iter_audio_files(source, None):
                scanned += 1
                speaker = _speaker(source, path) or "unknown"
                if counts[speaker] >= per_speaker:
                    continue
                counts[speaker] += 1
                relative = self._relative_path(source, path)
                stream.write(json.dumps(relative.as_posix()) + "\n")
                selected += 1
                estimated += path.stat().st_size
                if selected >= limit:
                    scan_complete = False
                    break
        return {
            "scanned_samples": scanned,
            "selected_samples": selected,
            "selected_audio_bytes": estimated,
            "scan_complete": scan_complete,
        }


def _profile_signals(
    infos: List[Dict[str, Any]], policy: QualityPolicy
) -> Dict[str, Any]:
    silence = clipping = near_zero = 0
    silence_values, clipping_values = [], []
    for item in infos:
        try:
            import numpy as np
            import soundfile as sf

            frames = int(item["sample_rate"] * policy.max_signal_seconds)
            values, _ = sf.read(
                item["path"], frames=frames, dtype="float32", always_2d=False
            )
            array = np.asarray(values, dtype="float32")
            if not array.size:
                near_zero += 1
                continue
            absolute = np.abs(array)
            silence_ratio = float(np.mean(absolute <= 1e-4))
            clipping_ratio = float(np.mean(absolute >= 0.999))
            rms = float(np.sqrt(np.mean(np.square(array))))
            silence_values.append(silence_ratio)
            clipping_values.append(clipping_ratio)
            silence += silence_ratio > policy.max_silence_ratio
            clipping += clipping_ratio > policy.max_clipping_ratio
            near_zero += rms <= policy.near_zero_rms
        except (ImportError, OSError, RuntimeError, ValueError):
            continue
    return {
        "signal_probe_count": len(silence_values),
        "mean_silence_ratio": sum(silence_values) / len(silence_values)
        if silence_values
        else 0.0,
        "mean_clipping_ratio": sum(clipping_values) / len(clipping_values)
        if clipping_values
        else 0.0,
        "excessive_silence_count": silence,
        "clipped_file_count": clipping,
        "near_zero_file_count": near_zero,
    }


def _profile_manifests(
    source: Path,
    policy: QualityPolicy | None = None,
) -> Dict[str, Any]:
    active_policy = policy or QualityPolicy()
    manifests = list(source.rglob("*.csv"))[:50] if source.is_dir() else []
    schema, details, issues = {}, {"files": []}, []
    speakers_by_split: Dict[str, set[str]] = defaultdict(set)
    files_by_split: Dict[str, set[str]] = defaultdict(set)
    speaker_counts: Counter[str] = Counter()
    existing_audio_paths: List[str] = []
    missing = path_violations = missing_speaker = invalid_speaker = (
        duplicates
    ) = rows_total = 0
    seen_paths: set[str] = set()
    for manifest in manifests:
        try:
            with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames or [], list(reader)
                schema[manifest.relative_to(source).as_posix()] = fields
        except (OSError, UnicodeError, csv.Error) as exc:
            issues.append(
                _issue(
                    "manifest_unreadable",
                    "error",
                    f"Manifest cannot be read: {manifest.name}",
                    evidence={"error": str(exc)},
                )
            )
            continue
        split = (
            manifest.stem.lower()
            if manifest.stem.lower() in SPLITS
            else "unspecified"
        )
        local_invalid = 0
        for row in rows:
            rows_total += 1
            speaker_key = _first_field(row, SPEAKER_FIELDS)
            speaker = (
                str(row.get(speaker_key) or "").strip() if speaker_key else ""
            )
            if not speaker:
                missing_speaker += 1
                local_invalid += 1
            elif not LABEL_RE.fullmatch(speaker):
                invalid_speaker += 1
                local_invalid += 1
            else:
                speakers_by_split[split].add(speaker)
                speaker_counts[speaker] += 1
            path_key = _first_field(row, PATH_FIELDS)
            if not path_key:
                continue
            resolved = _resolve_audio_path(
                source, manifest, str(row.get(path_key) or "").strip()
            )
            if resolved is None:
                path_violations += 1
                local_invalid += 1
                continue
            normalized = str(resolved).lower()
            if normalized in seen_paths:
                duplicates += 1
            seen_paths.add(normalized)
            files_by_split[split].add(normalized)
            if not resolved.exists():
                missing += 1
                local_invalid += 1
            elif resolved.suffix.lower() in AUDIO_EXTENSIONS:
                existing_audio_paths.append(str(resolved))
        details["files"].append(
            {
                "path": str(manifest),
                "rows": len(rows),
                "invalid_rows": local_invalid,
            }
        )

    speaker_overlap = sum(
        len(speakers_by_split[a] & speakers_by_split[b])
        for a, b in combinations(sorted(speakers_by_split), 2)
    )
    file_overlap = sum(
        len(files_by_split[a] & files_by_split[b])
        for a, b in combinations(sorted(files_by_split), 2)
    )
    checks = (
        (
            "manifest_audio_missing",
            missing,
            "error",
            "Manifest references missing audio files.",
            "normalize_manifest",
        ),
        (
            "manifest_path_violation",
            path_violations,
            "error",
            "Manifest paths escape the dataset root.",
            "normalize_manifest",
        ),
        (
            "speaker_id_missing",
            missing_speaker,
            "error",
            "Manifest rows are missing speaker IDs.",
            "normalize_manifest",
        ),
        (
            "speaker_id_invalid",
            invalid_speaker,
            "error",
            "Manifest rows contain invalid speaker IDs.",
            "normalize_manifest",
        ),
        (
            "duplicate_samples",
            duplicates,
            "warning",
            "Duplicate audio samples were found.",
            "normalize_manifest",
        ),
        (
            "speaker_split_overlap",
            speaker_overlap,
            "error" if active_policy.require_disjoint_speakers else "warning",
            "Speakers overlap across splits.",
            "check_speaker_split",
        ),
        (
            "file_split_overlap",
            file_overlap,
            "error" if active_policy.require_disjoint_files else "warning",
            "Audio files overlap across splits.",
            "check_speaker_split",
        ),
    )
    for code, count, severity, message, operation in checks:
        if count:
            issues.append(
                _issue(code, severity, message, operation, {"count": count})
            )
    trials = _profile_trials(source)
    if not active_policy.require_valid_trials:
        for issue in trials["issues"]:
            if issue.severity == "error":
                issue.severity = "warning"
    issues.extend(trials["issues"])
    return {
        "schema": schema,
        "details": details,
        "issues": issues,
        "row_count": rows_total,
        "speaker_counts": speaker_counts,
        "speakers_per_split": {
            key: len(value) for key, value in speakers_by_split.items()
        },
        "existing_audio_paths": existing_audio_paths,
        "missing_audio_count": missing,
        "path_violation_count": path_violations,
        "missing_speaker_count": missing_speaker,
        "invalid_speaker_count": invalid_speaker,
        "duplicate_sample_count": duplicates,
        "speaker_overlap_count": speaker_overlap,
        "file_overlap_count": file_overlap,
        **trials,
    }


def _profile_trials(source: Path) -> Dict[str, Any]:
    candidates = (
        [
            path
            for path in source.rglob("*.txt")
            if any(
                token in path.name.lower()
                for token in ("trial", "veri", "test_list")
            )
        ][:20]
        if source.is_dir()
        else []
    )
    invalid = duplicates = 0
    labels: Counter[str] = Counter()
    seen: Dict[tuple[str, str], str] = {}
    issues = []
    for path in candidates:
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError):
            invalid += 1
            continue
        for line in lines:
            parts = line.split()
            if len(parts) != 3 or parts[0] not in {"0", "1"}:
                invalid += 1
                continue
            label, left, right = parts
            pair = tuple(sorted((left, right)))
            previous_label = seen.get(pair)
            if previous_label == label:
                duplicates += 1
            elif previous_label is not None:
                invalid += 1
            seen[pair] = label
            labels[label] += 1
            for raw_path in (left, right):
                resolved = _resolve_audio_path(source, path, raw_path)
                if resolved is None or not resolved.exists():
                    invalid += 1
    if invalid:
        issues.append(
            _issue(
                "verification_trials_invalid",
                "error",
                "Verification trials contain invalid rows or paths.",
                "validate_verification_trials",
                {"count": invalid},
            )
        )
    if duplicates:
        issues.append(
            _issue(
                "verification_trials_duplicate",
                "error",
                "Verification trials contain duplicate pairs.",
                "validate_verification_trials",
                {"count": duplicates},
            )
        )
    if labels and (not labels.get("0") or not labels.get("1")):
        issues.append(
            _issue(
                "verification_trials_unbalanced",
                "warning",
                "Verification trials contain only one class.",
                evidence={"labels": dict(labels)},
            )
        )
    return {
        "invalid_trial_count": invalid,
        "duplicate_trial_count": duplicates,
        "trial_labels": dict(labels),
        "issues": issues,
    }


def _resolve_audio_path(
    source: Path, manifest: Path, raw_path: str
) -> Path | None:
    if not raw_path:
        return None
    value = raw_path.replace("{data_root}", str(source)).replace(
        "{dataset_root}", str(source)
    )
    candidate = Path(value)
    if not candidate.is_absolute():
        root_candidate = (source / candidate).resolve()
        local_candidate = (manifest.parent / candidate).resolve()
        candidate = (
            root_candidate
            if root_candidate.exists() or not local_candidate.exists()
            else local_candidate
        )
    else:
        candidate = candidate.resolve()
    try:
        candidate.relative_to(source)
    except ValueError:
        return None
    return candidate


def _iter_audio_files(
    source: Path, limit: int | None
) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() in AUDIO_EXTENSIONS:
            yield source.resolve()
        return
    if not source.is_dir():
        return
    yielded = 0
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        yield path.resolve()
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def _first_field(row: Dict[str, Any], candidates: Iterable[str]) -> str | None:
    lookup = {str(key).lower(): key for key in row}
    return next((lookup[name] for name in candidates if name in lookup), None)


def _speaker(source: Path, path: Path) -> str | None:
    try:
        parts = path.resolve().relative_to(source).parts
    except ValueError:
        return path.parent.name or None
    if len(parts) >= 3 and parts[0].lower() in SPLITS:
        return parts[1]
    return path.parent.name or None


def _split(source: Path, path: Path) -> str | None:
    try:
        first = path.resolve().relative_to(source).parts[0].lower()
    except (ValueError, IndexError):
        return None
    return first if first in SPLITS else None


def _summary(values: List[float] | List[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(values) / len(values),
        "median": ordered[len(ordered) // 2],
    }


def _issue(
    code: str,
    severity: str,
    message: str,
    operation: str | None = None,
    evidence: Dict[str, Any] | None = None,
) -> DataIssue:
    return DataIssue(code, severity, message, evidence or {}, operation)


def _deduplicate_issues(issues: List[DataIssue]) -> List[DataIssue]:
    result, seen = [], set()
    for issue in issues:
        if issue.code not in seen:
            seen.add(issue.code)
            result.append(issue)
    return result


def _policy(parameters: Dict[str, Any]) -> QualityPolicy:
    return QualityPolicy.from_dict(parameters.get("_quality_policy"))


def _validate_output(parameters: Dict[str, Any]) -> None:
    if not parameters.get("_output_uri"):
        raise ValueError("operation requires an execution output directory")
    if "csv_glob" in parameters and ".." in str(parameters["csv_glob"]):
        raise ValueError("csv_glob cannot traverse outside the dataset")


def _safe_output(
    dataset: DatasetSpec, parameters: Dict[str, Any]
) -> tuple[Path, Path]:
    source = Path(dataset.source_uri).resolve()
    output = Path(str(parameters["_output_uri"])).resolve()
    if output == source or source in output.parents:
        raise ValueError("output directory must be outside the source dataset")
    return source, output


def _tree_size(source: Path) -> int:
    return (
        source.stat().st_size
        if source.is_file()
        else sum(
            path.stat().st_size for path in source.rglob("*") if path.is_file()
        )
    )


def register_audio_components() -> None:
    register_profiler(AudioDataProfiler())
    for processor in (
        ValidateAudioFilesProcessor(),
        NormalizeManifestProcessor(),
        FilterUnreadableAudioProcessor(),
        FilterByDurationProcessor(),
        CheckSpeakerSplitProcessor(),
        BuildDebugSubsetProcessor(),
        ValidateVerificationTrialsProcessor(),
    ):
        register_processor(processor)


__all__ = ["AudioDataProfiler", "register_audio_components"]
