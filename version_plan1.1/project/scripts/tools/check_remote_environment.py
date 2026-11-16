"""Check that a remote GPU image can run SR-agent experiments."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


REQUIRED_PACKAGES = (
    "langgraph",
    "langchain-core",
    "langchain-openai",
    "optuna",
    "speechbrain",
    "torch",
    "torchaudio",
)


@dataclass
class PackageStatus:
    name: str
    installed: bool
    version: str | None = None
    error: str | None = None


def package_status(name: str) -> PackageStatus:
    try:
        return PackageStatus(name=name, installed=True, version=version(name))
    except PackageNotFoundError:
        return PackageStatus(name=name, installed=False, error="not installed")


def _major_minor(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.split("+", 1)[0]
    parts = normalized.split(".")
    if len(parts) < 2:
        return normalized
    return ".".join(parts[:2])


def collect_environment(require_cuda: bool = False) -> tuple[dict[str, Any], list[str]]:
    packages = [package_status(name) for name in REQUIRED_PACKAGES]
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "dataset_env": {
            "SR_AGENT_DATASET_PATH": os.getenv("SR_AGENT_DATASET_PATH"),
            "SR_AGENT_DATA_ROOT": os.getenv("SR_AGENT_DATA_ROOT"),
        },
        "packages": [asdict(item) for item in packages],
        "cuda": {},
    }

    issues = [
        f"{item.name} is not installed"
        for item in packages
        if not item.installed
    ]

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - depends on remote image
        issues.append(f"torch import failed: {exc}")
    else:
        cuda_available = bool(torch.cuda.is_available())
        report["cuda"] = {
            "available": cuda_available,
            "torch_cuda": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "device_name": (
                torch.cuda.get_device_name(0) if cuda_available else None
            ),
        }
        if require_cuda and not cuda_available:
            issues.append("CUDA is required but torch.cuda.is_available() is false")

    torch_version = next((p.version for p in packages if p.name == "torch"), None)
    torchaudio_version = next(
        (p.version for p in packages if p.name == "torchaudio"), None
    )
    if _major_minor(torch_version) != _major_minor(torchaudio_version):
        issues.append(
            "torch and torchaudio major.minor versions differ: "
            f"torch={torch_version}, torchaudio={torchaudio_version}"
        )

    return report, issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    report, issues = collect_environment(require_cuda=args.require_cuda)
    report["ok"] = not issues
    report["issues"] = issues

    if args.json_output:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("SR-agent remote environment check")
        print(f"Python: {report['python']} ({report['executable']})")
        for item in report["packages"]:
            status = item["version"] if item["installed"] else item["error"]
            print(f"- {item['name']}: {status}")
        cuda = report["cuda"]
        if cuda:
            print(
                "CUDA: "
                f"available={cuda['available']}, "
                f"torch_cuda={cuda['torch_cuda']}, "
                f"device={cuda['device_name']}"
            )
        if issues:
            print("\nIssues:")
            for issue in issues:
                print(f"- {issue}")
        else:
            print("\nOK: environment is ready for SR-agent experiments.")

    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())