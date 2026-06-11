"""
数据处理工具集合
提供数据准备与检查能力
使用 LangChain 工具接口
"""

from langchain_core.tools import tool
from typing import Optional, List, Any
from pathlib import Path
import json

from agent.utils import ConfigParser, ExperimentTracker
from agent.core.contracts import Artifact, OperationResult
from agent.utils.path_tool import (
    get_experiment_artifact_dir,
    is_remote_path,
    resolve_config_path,
    resolve_config_value_path,
    resolve_data_path,
)
from agent.utils import runner

def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_list(value: Optional[str]) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in text.split(",") if item.strip()]


def _count_csv_rows(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fin:
            first = fin.readline()
            if not first:
                return 0
            count = sum(1 for _ in fin)
        if not first:
            return 0
        header = first.lower()
        if "id" in header and "wav" in header:
            return count
        return count + 1
    except Exception:
        return None


@tool
def PrepareVoxCelebData(
    config_path: Optional[str] = None,
    experiment_id: Optional[str] = None,
    split_ratio: Optional[str] = None,
    sentence_len: Optional[str] = None,
    skip_prep: Optional[str] = None,
    random_segment: Optional[str] = None,
    split_speaker: Optional[str] = None,
) -> str:
    """
    准备 VoxCeleb 数据，生成 train/dev/test/enrol CSV。

    参数:
        config_path: 配置文件路径
        experiment_id: 数据处理实验 ID，用于保存本次数据准备结果
        split_ratio: 训练/验证比例（JSON 列表或 "90,10"）
        sentence_len: 片段长度（秒）
        skip_prep: 是否跳过准备
        random_segment: 是否随机切片
        split_speaker: 是否按说话人划分

    Returns:
        str: 数据准备结果与统计摘要
    """
    try:
        tracker = ExperimentTracker()
        cfg_path = str(resolve_config_path(config_path))
        parser = ConfigParser(cfg_path)
        config_data = parser.load_config(resolve_references=True)

        split_ratio_val = _parse_list(split_ratio) or config_data.get("split_ratio") or [90, 10]
        sentence_len_val = _parse_float(sentence_len)
        if sentence_len_val is None:
            sentence_len_val = config_data.get("sentence_len") or 3.0

        skip_prep_val = _parse_bool(skip_prep)
        if skip_prep_val is None:
            skip_prep_val = bool(config_data.get("skip_prep", False))

        random_segment_val = _parse_bool(random_segment)
        if random_segment_val is None:
            random_segment_val = False
        split_speaker_val = _parse_bool(split_speaker)
        if split_speaker_val is None:
            split_speaker_val = False

        df = str(resolve_data_path(config_data.get("data_folder")))

        sf = config_data.get("save_folder")
        if not sf:
            return OperationResult(
                status="failed",
                stage="data_preparation",
                error="save_folder is not configured",
                experiment_id=experiment_id,
            ).to_json()
        if experiment_id:
            sf = str(get_experiment_artifact_dir(
                experiment_id,
                "data",
                "data_processing",
                create=True,
            ))
        else:
            resolved_save_folder = resolve_config_value_path(sf)
            sf = str(resolved_save_folder) if resolved_save_folder is not None else None

        vf = config_data.get("verification_file")
        if not vf:
            return OperationResult(
                status="failed",
                stage="data_preparation",
                error="verification_file is not configured",
                experiment_id=experiment_id,
            ).to_json()

        splits_val = config_data.get("splits") or ["train", "dev"]
        source_val = config_data.get("voxceleb_source")
        if source_val and not is_remote_path(source_val):
            resolved_source = resolve_config_value_path(source_val)
            source_val = str(resolved_source) if resolved_source is not None else None
        amp_th_val = config_data.get("amp_th") or 5e-04

        prep_result = runner.run_data_prep(
            data_folder=df,
            save_folder=sf,
            verification_file=str(vf),
            split_ratio=split_ratio_val,
            sentence_len=sentence_len_val,
            splits=splits_val,
            skip_prep=skip_prep_val,
            source=source_val,
            random_segment=random_segment_val,
            amp_th=amp_th_val,
            split_speaker=split_speaker_val,
            signal = "prep_only"
        )

        if prep_result.get("status") != "success":
            if experiment_id:
                tracker.update_experiment(
                    experiment_id=experiment_id,
                    experiment_type="data_processing",
                    stage="data_preparation",
                    status="failed",
                    error=prep_result.get("error"),
                    parameters={
                        "data_folder": df,
                        "save_folder": sf,
                        "verification_file": str(vf),
                        "split_ratio": split_ratio_val,
                        "sentence_len": sentence_len_val,
                        "skip_prep": skip_prep_val,
                        "random_segment": random_segment_val,
                        "split_speaker": split_speaker_val,
                    },
                )
            return OperationResult(
                status="failed",
                stage="data_preparation",
                error=prep_result.get("error"),
                experiment_id=experiment_id,
            ).to_json()

        sf_path = Path(prep_result.get("save_folder") or sf)
        verification_local = prep_result.get("verification_local")

        manifest_names = list(dict.fromkeys([*splits_val, "enrol"] if "test" in splits_val else splits_val))
        csv_files = {
            name: str(sf_path / f"{name}.csv")
            for name in manifest_names
            if (sf_path / f"{name}.csv").exists()
        }
        stats = {
            name: _count_csv_rows(Path(path))
            for name, path in csv_files.items()
        }

        if experiment_id:
            tracker.update_experiment(
                experiment_id=experiment_id,
                experiment_type="data_processing",
                stage="data_preparation",
                status="success",
                parameters={
                    "data_folder": df,
                    "save_folder": str(sf_path),
                    "output_folder": str(sf_path),
                    "verification_file": str(verification_local) if verification_local else None,
                    "split_ratio": split_ratio_val,
                    "sentence_len": sentence_len_val,
                    "splits": splits_val,
                    "skip_prep": skip_prep_val,
                    "random_segment": random_segment_val,
                    "split_speaker": split_speaker_val,
                    "amp_th": amp_th_val,
                },
                metrics={"summary": stats},
                artifacts=[
                    {"type": "manifest", "name": name, "path": path}
                    for name, path in csv_files.items()
                ],
                extensions={
                    "data_preparation": {
                        "verification_file": str(verification_local) if verification_local else None,
                        "prepare_status": "success",
                    }
                },
            )

        return OperationResult(
            status="success",
            stage="data_preparation",
            task={"type": "speaker_verification", "dataset": df},
            execution={"runner": "speechbrain", "output_folder": str(sf_path)},
            metrics={"summary": stats},
            artifacts=[
                Artifact("manifest", name, path)
                for name, path in csv_files.items()
            ],
            parameters={
                "data_folder": df,
                "save_folder": str(sf_path),
                "verification_file": str(verification_local) if verification_local else None,
                "split_ratio": split_ratio_val,
                "sentence_len": sentence_len_val,
                "skip_prep": skip_prep_val,
                "random_segment": random_segment_val,
                "split_speaker": split_speaker_val,
            },
            extensions={"data_preparation": {"prepare_status": "success"}},
            experiment_id=experiment_id,
        ).to_json()
    except Exception as e:
        if experiment_id:
            tracker = ExperimentTracker()
            tracker.update_experiment(
                experiment_id=experiment_id,
                experiment_type="data_processing",
                stage="data_preparation",
                status="failed",
                error=str(e),
                extensions={"data_preparation": {"prepare_status": "failed"}},
            )
        return OperationResult(
            status="failed",
            stage="data_preparation",
            error=str(e),
            experiment_id=experiment_id,
        ).to_json()


__all__ = [
    "PrepareVoxCelebData",
]
