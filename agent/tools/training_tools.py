"""
训练管理工具集合
提供模型训练、训练监控、训练控制等功能
使用 LangChain 工具接口，基于 utils 模块实现
"""

from langchain_core.tools import tool
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import json

# 导入 utils 模块
from agent.utils import (
    get_experiment_dir,
    resolve_config_path,
    resolve_data_path,
    resolve_optional_project_path,
    ExperimentTracker,
)
from agent.core.adapters import resolve_adapter_bundle
from agent.core.contracts import OperationResult
from agent.core.experiment_service import ExperimentService
from agent.hpo import FailurePolicy, HPOService
from agent.runners import collect_training_result

@tool
def TrainModel(config_path: Optional[str] = None,
               data_folder: Optional[str] = None,
               description: Optional[str] = None,
               experiment_id: Optional[str] = None,
               trial_id: Optional[str] = None,
               parameters_json: Optional[str] = None,
               budget_json: Optional[str] = None,
               task_type: Optional[str] = None,
               model_family: Optional[str] = None,
               implementation: Optional[str] = None,
               runner: Optional[str] = None,
               experiments_dir: Optional[str] = None,
               device: Optional[str] = None,
               precision: Optional[str] = None,
               eval_precision: Optional[str] = None) -> str:
    """
    通过注册的模型、任务和 Runner 适配器执行训练。
    训练完成后会自动保存实验记录，包括配置、指标和产物。
    
    参数:
        config_path: 配置文件路径，如果为 None 则使用默认配置
        data_folder: 数据文件夹路径，如果为 None 则使用配置文件中的设置
        description: 实验描述（可选）
        experiment_id: 实验 ID（可选，存在时继续训练，不存在时报错）
    
    Returns:
        str: 训练输出或错误信息
    """
    try:
        # 参数校验
        config_path_str = str(resolve_config_path(config_path))

        # experiment_tracker 创建记录或复用已有记录
        start_time = datetime.now()
        tracker = ExperimentTracker(experiments_dir)
        record = None
        if experiment_id:
            record = tracker.get_experiment(experiment_id)
            if record is None:
                return OperationResult(
                    status="failed",
                    stage="training",
                    error=f"experiment_id not found: {experiment_id}",
                    experiment_id=experiment_id,
                ).to_json()

            execution_info = record.get("execution", {})
            backup_path = execution_info.get("config_backup_path")
            record_config_path = record.get("config_path")
            if not config_path:
                if backup_path and Path(backup_path).exists():
                    config_path_str = backup_path
                elif record_config_path and Path(record_config_path).exists():
                    config_path_str = record_config_path

        if not Path(config_path_str).exists():
            return OperationResult(
                status="failed",
                stage="training",
                error=f"config_path not found: {config_path_str}",
                experiment_id=experiment_id,
            ).to_json()

        # 使用 ConfigParser 读取配置（支持 YAML 引用解析）
        from agent.utils import ConfigParser
        parser = ConfigParser(config_path_str)
        config_data = parser.load_config(resolve_references=True)
        task_type = task_type or ((record or {}).get("task") or {}).get("type") or "speaker_verification"
        model_family = model_family or ((record or {}).get("model") or {}).get("family") or "ecapa_tdnn"
        implementation = implementation or ((record or {}).get("model") or {}).get("implementation") or "speechbrain"
        runner_name = runner or ((record or {}).get("execution") or {}).get("runner") or implementation
        adapters = resolve_adapter_bundle(task_type, model_family, implementation, runner_name)
        task_adapter, model_adapter, runner_adapter = adapters.task, adapters.model, adapters.runner
        model_adapter.validate_config(config_data)

        # 确定数据文件夹
        if data_folder:
            df = str(resolve_data_path(data_folder))
        else:
            df = str(resolve_data_path(config_data.get("data_folder")))

        output_folder = config_data.get("output_folder")
        if record:
            execution_info = record.get("execution", {})
            if execution_info.get("output_folder"):
                output_folder = execution_info.get("output_folder")
            if not data_folder and (record.get("task") or {}).get("dataset"):
                df = str(resolve_data_path(record["task"]["dataset"]))
        else:
            experiment_id = tracker.create_hpo_experiment(
                config_path=config_path_str,
                data_folder=df,
                output_folder=str(output_folder) if output_folder else None,
                description=description or "",
                task={
                    "type": task_type,
                    "dataset": df,
                    "primary_metric": task_adapter.primary_metric,
                    "metric_mode": task_adapter.metric_mode,
                },
                model={
                    "family": model_family,
                    "implementation": implementation,
                    "config_path": config_path_str,
                },
                execution={"runner": runner_name, "output_folder": str(output_folder) if output_folder else None},
            )

        if experiments_dir:
            exp_dir = Path(experiments_dir).resolve() / experiment_id
            exp_dir.mkdir(parents=True, exist_ok=True)
        else:
            exp_dir = get_experiment_dir(experiment_id, "hpo", create=True)
        if trial_id:
            hpo_service = HPOService(tracker)
            study = hpo_service.load_study(experiment_id)
            hpo_service.record_trial(study, trial_id, status="running")

        output_folder_path = resolve_optional_project_path(output_folder)
        if trial_id:
            output_folder_path = exp_dir / "trials" / trial_id / "output"
            output_folder = str(output_folder_path)
        elif output_folder_path is None or exp_dir not in output_folder_path.parents:
            output_folder_path = exp_dir / "output"
            output_folder = str(output_folder_path)

        # runner.run_training()
        overrides = {
            "data_folder": df,
            "output_folder": str(output_folder_path),
        }
        trial_parameters = json.loads(parameters_json) if parameters_json else {}
        trial_budget = json.loads(budget_json) if budget_json else {}
        if not isinstance(trial_parameters, dict) or not isinstance(trial_budget, dict):
            raise ValueError("parameters_json and budget_json must be JSON objects")
        parameter_validator = getattr(model_adapter, "validate_parameters", None)
        if callable(parameter_validator):
            parameter_validator(trial_parameters)
        data_fraction = trial_budget.get("data_fraction")
        if data_fraction is not None and not 0 < float(data_fraction) <= 1:
            raise ValueError("budget data_fraction must be in (0, 1]")
        max_duration = trial_budget.get("max_duration_seconds")
        if max_duration is not None and float(max_duration) <= 0:
            raise ValueError("budget max_duration_seconds must be positive")
        overrides.update(trial_parameters)
        if trial_budget.get("epochs") is not None:
            overrides["number_of_epochs"] = trial_budget["epochs"]
        if data_fraction is not None:
            overrides["_hpo_data_fraction"] = float(data_fraction)
        if max_duration is not None:
            overrides["_hpo_max_duration_seconds"] = float(max_duration)
        run_opts: Dict[str, Any] = {}
        if device:
            run_opts["device"] = device
        if precision:
            run_opts["precision"] = precision
        if eval_precision:
            run_opts["eval_precision"] = eval_precision
        if run_opts:
            overrides["_run_opts"] = run_opts
        train_result = runner_adapter.run_training(config_path_str, overrides)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        status = "success"
        error_msg = None
        if isinstance(train_result, dict) and train_result.get("status") == "failed":
            status = "failed"
            error_msg = train_result.get("error")

        collected = collect_training_result(runner_adapter, train_result, output_folder_path, exp_dir)
        collected["status"] = status
        collected["error"] = error_msg
        output_folder = collected.get("output_folder") or (
            str(output_folder_path) if output_folder_path else None
        )
        training_metrics = dict(collected.get("metrics") or {})
        epoch_data = list(collected.get("epoch_data") or [])

        # experiment_tracker 更新记录
        operation_result = runner_adapter.normalize_training_result(collected)
        operation_result.task = {
            **((record or {}).get("task") or {}),
            "type": task_type,
            "dataset": df,
            "primary_metric": task_adapter.primary_metric,
            "metric_mode": task_adapter.metric_mode,
        }
        operation_result.model = {
            **((record or {}).get("model") or {}),
            "family": model_family,
            "implementation": implementation,
            "config_path": config_path_str,
        }
        operation_result.execution.update({
            "runner": runner_name,
            "output_folder": output_folder,
            "trial_id": trial_id,
            "budget": trial_budget,
            "runtime_options": dict(overrides.get("_run_opts") or {}),
        })
        for artifact in operation_result.artifacts:
            if trial_id:
                artifact.metadata["trial_id"] = trial_id
        operation_result.parameters = dict(overrides)
        ExperimentService(tracker).record_result(
            experiment_id,
            operation_result,
            duration_seconds=duration,
            actor={"type": "hpo_agent", "name": "model_optimizer"},
            update_status=trial_id is None,
        )
        if trial_id and status == "success":
            tracker.update_hpo_experiment(
                experiment_id,
                extensions={"optimization": {
                    "latest_trial": {
                        "trial_id": trial_id,
                        "phase": "training",
                        "status": "training_completed",
                        "updated_at": datetime.now().isoformat(),
                    }
                }},
            )
        if trial_id and study is not None:
            trial_metrics: Dict[str, Any] = {}
            for split_metrics in operation_result.metrics.values():
                trial_metrics.update(split_metrics or {})
            failure = FailurePolicy().classify(error_msg) if status == "failed" else None
            HPOService(tracker).record_trial(
                study,
                trial_id,
                status="running" if status == "success" else "failed",
                metrics=trial_metrics,
                intermediate_metrics=epoch_data,
                cost={
                    "training": {
                        "duration_seconds": duration,
                        "status": status,
                        "metrics": training_metrics,
                    },
                    "failure_category": failure.category if failure else None,
                    "recoverable": failure.recoverable if failure else None,
                },
                artifacts=[artifact.to_dict() for artifact in operation_result.artifacts],
                stop_reason=error_msg,
            )
        return operation_result.to_json()
    except Exception as e:
        if trial_id and locals().get("study") is not None and locals().get("tracker") is not None:
            try:
                failure = FailurePolicy().classify(str(e))
                HPOService(tracker).record_trial(
                    study,
                    trial_id,
                    status="failed",
                    cost={
                        "failure_category": failure.category,
                        "recoverable": failure.recoverable,
                    },
                    stop_reason=str(e),
                )
            except Exception:
                pass
        return OperationResult(
            status="failed",
            stage="training",
            error=str(e),
            experiment_id=experiment_id,
        ).to_json()


@tool
def EvaluateModel(experiment_id: Optional[str] = None) -> str:
    """Evaluate through the registered runner and return OperationResult JSON."""
    try:
        from agent.tools.evaluation_tools import RunEvaluation
        return RunEvaluation.invoke({"experiment_id": experiment_id})
    except Exception as exc:
        return OperationResult(
            status="failed",
            stage="evaluation",
            error=str(exc),
            experiment_id=experiment_id,
        ).to_json()


@tool
def AnalyzeResults(experiment_id: Optional[str] = None) -> str:
    """Return recorded metrics plus metrics parsed from a training log."""
    try:
        from agent.utils import extract_log_metrics
        
        tracker = ExperimentTracker()
        if experiment_id is None:
            recent = tracker.list_experiments(limit=1)
            experiment_id = recent[0]["experiment_id"] if recent else None
        record = tracker.get_experiment(experiment_id) if experiment_id else None
        if not record:
            return OperationResult(
                status="failed",
                stage="analysis",
                error="experiment not found",
                experiment_id=experiment_id,
            ).to_json()

        log_path = next((
            Path(item["path"])
            for item in record.get("artifacts") or []
            if item.get("type") == "log" and item.get("path")
        ), None)
        parsed_metrics = extract_log_metrics(log_path) if log_path and log_path.exists() else {}
        return json.dumps({
            "status": "success",
            "experiment_id": experiment_id,
            "recorded_metrics": record.get("metrics") or {},
            "parsed_log_metrics": parsed_metrics,
            "log_path": str(log_path) if log_path else None,
        }, ensure_ascii=False, default=str)
    
    except Exception as e:
        return OperationResult(
            status="failed",
            stage="analysis",
            error=str(e),
            experiment_id=experiment_id,
        ).to_json()


# 导出所有工具
__all__ = [
    'TrainModel',
    'EvaluateModel',
    'AnalyzeResults',
]
