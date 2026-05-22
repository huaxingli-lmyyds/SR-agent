"""
数据处理工具集合
提供数据准备与检查能力
使用 LangChain 工具接口
"""

from langchain_core.tools import tool
from typing import Optional, List, Any
from pathlib import Path
import os
import json

from agent.utils import ConfigParser, get_config_file
from agent.utils.path_tool import get_datasets_dir, get_project_root
from agent.utils import runner

CONFIG_PATH = str(get_config_file("train_ecapa_tdnn.yaml"))


def _resolve_data_folder(path_value: Optional[str]) -> str:
    if not path_value or path_value == "!PLACEHOLDER":
        return str(get_datasets_dir() / "voxceleb1")

    path = Path(path_value)
    if path.is_absolute():
        return str(path)

    project_root = get_project_root()
    return str((project_root / path).resolve())


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
            rows = fin.readlines()
        if not rows:
            return 0
        header = rows[0].lower()
        if "id" in header and "wav" in header:
            return max(len(rows) - 1, 0)
        return len(rows)
    except Exception:
        return None


@tool
def PrepareVoxCelebData(
    config_path: Optional[str] = None,
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
        split_ratio: 训练/验证比例（JSON 列表或 "90,10"）
        sentence_len: 片段长度（秒）
        skip_prep: 是否跳过准备
        random_segment: 是否随机切片
        split_speaker: 是否按说话人划分

    Returns:
        str: 数据准备结果与统计摘要
    """
    try:
        cfg_path = config_path if config_path else CONFIG_PATH
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

        df = _resolve_data_folder(config_data.get("data_folder"))

        sf = config_data.get("save_folder")
        if not sf:
            return "❌ 无法确定 save_folder，请在配置中设置"

        vf = config_data.get("verification_file")
        if not vf:
            return "❌ 无法确定 verification_file，请在配置中设置"

        splits_val = config_data.get("splits") or ["train", "dev"]
        source_val = config_data.get("voxceleb_source")
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
            return f"❌ 数据准备失败: {prep_result.get('error')}"

        sf_path = Path(prep_result.get("save_folder") or sf)
        verification_local = prep_result.get("verification_local")

        train_csv = sf_path / "train.csv"
        dev_csv = sf_path / "dev.csv"


        stats = {
            "train": _count_csv_rows(train_csv),
            "dev": _count_csv_rows(dev_csv),
        }

        summary = (
            "✅ 数据准备完成！\n"
            f"数据目录: {df}\n"
            f"保存目录: {sf_path}\n"
            f"验证列表: {verification_local}\n"
            f"拆分比例: {split_ratio_val}\n"
            f"片段长度: {sentence_len_val}s\n"
            f"随机切片: {random_segment_val}\n"
            f"按说话人划分: {split_speaker_val}\n"
            f"跳过准备: {skip_prep_val}\n\n"
            "📊 CSV 统计:\n"
            f"  - train: {stats['train']}\n"
            f"  - dev: {stats['dev']}\n"
            "📁 CSV 路径:\n"
            f"  - train: {train_csv}\n"
            f"  - dev: {dev_csv}\n"
        )

        return summary
    except Exception as e:
        import traceback
        return f"❌ 数据准备失败: {str(e)}\n{traceback.format_exc()}"


__all__ = [
    "PrepareVoxCelebData",
]
