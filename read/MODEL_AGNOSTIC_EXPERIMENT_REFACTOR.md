# 模型无关实验协议重构说明

## 目标

本次重构将实验记录、工具返回值和具体 ECAPA-TDNN/SpeechBrain 实现解耦。核心代码只描述一次操作发生了什么；任务、模型和运行框架的差异由适配器负责。

## 核心结构

### 实验记录

实验记录统一使用 `schema_version: "2.0"`，主要字段如下：

```json
{
  "stage": "training",
  "actor": {"type": "hpo_agent", "name": "model_optimizer"},
  "task": {
    "type": "speaker_verification",
    "dataset": "...",
    "primary_metric": "eer",
    "metric_mode": "min"
  },
  "model": {
    "family": "ecapa_tdnn",
    "implementation": "speechbrain",
    "config_path": "..."
  },
  "execution": {
    "runner": "speechbrain",
    "output_folder": "..."
  },
  "metrics": {
    "validation": {},
    "test": {},
    "summary": {}
  },
  "artifacts": [],
  "parameters": {},
  "extensions": {}
}
```

- `stage` 表示数据准备、训练、评估、优化或协调等实验阶段，与执行该阶段的智能体类型分离。
- `metrics` 按数据切分保存通用指标，不再把 `eer`、`min_dcf` 等写成固定顶层字段。
- `artifacts` 统一保存 checkpoint、日志、预测结果、manifest 和报告等产物。
- `extensions` 保存 SpeechBrain、声纹任务或智能体特有信息，不污染核心协议。

### 工具输入输出

`agent/core/contracts.py` 定义模型无关的 `OperationRequest`、`OperationResult` 和 `Artifact`。训练、评估、数据准备工具成功时均返回序列化后的 `OperationResult`，调用方可以稳定读取：

- `status`、`stage`、`error`
- `task`、`model`、`execution`
- `metrics`、`artifacts`、`parameters`、`extensions`
- `experiment_id`

### 适配器

`agent/core/adapters.py` 定义三类接口：

- `TaskAdapter`：定义主指标、优化方向和指标校验。
- `ModelAdapter`：定义模型配置校验和模型特有约束。
- `RunnerAdapter`：将训练框架原始结果转换为统一 `OperationResult`。

当前首个实现为：

- `SpeakerVerificationTaskAdapter`
- `SpeechBrainEcapaAdapter`
- `SpeechBrainRunnerAdapter`

新增模型或框架时，通过 `register_task_adapter`、`register_model_adapter`、`register_runner_adapter` 注册，不需要修改实验记录器。

## 主要代码调整

- 新增 `agent/core/`，集中保存通用协议、适配器和 `ExperimentService`。
- `ExperimentTracker` 改为保存通用字段，删除旧的固定 `training`、`evaluation`、`results`、`data_processing`、`optimization` 记录结构。
- `ExperimentService` 成为统一写入口，并可按实际实验类型更新记录。
- 训练和评估工具先由 `SpeechBrainRunnerAdapter` 归一化，再写实验记录。
- 数据处理工具使用同一 `OperationResult` 成功返回结构，并将 CSV 保存为 `manifest` 产物。
- 历史查询、奖励计算、训练诊断和智能体改为读取 `metrics`、`artifacts`、`extensions`。

## 扩展方式

接入新模型时，建议只增加对应模型适配器；接入新任务时增加任务适配器；接入新训练框架时增加运行器适配器。模型或框架专有字段必须进入 `extensions`，通用工具不得直接依赖这些字段。

例如，分类任务可以将 `task.primary_metric` 设置为 `accuracy`，目标检测任务可以设置为 `map`；实验记录器和历史比较逻辑无需增加新字段。

## 兼容性说明

新产生的记录以 2.0 结构为准。旧实验 JSON 不会自动迁移；需要继续使用旧记录时，应单独编写一次性迁移脚本，将旧指标和路径转换到 `metrics`、`artifacts` 与 `extensions`。
