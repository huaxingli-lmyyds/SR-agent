# 数据处理与超参数智能体优化路线

本文档用于约束数据处理智能体与超参数智能体的后续优化方向，避免重新退化为绑定单一模型、任务或训练框架的工具调用脚本。

## 总体原则

- 智能体负责决策、计划、编排和结果解释。
- 普通服务负责确定性算法、状态管理和持久化。
- 适配器负责隔离数据类型、任务、模型与训练框架差异。
- 实验记录保存发生了什么，专用信息进入 `extensions`。
- 原始数据和主配置默认不可被单次候选实验覆盖。

## 数据处理智能体

### 当前阶段：通用数据生命周期

已实现：

```text
InspectDataset
  -> BuildDataProcessingPlan
  -> ExecuteDataProcessingPlan
  -> PublishDatasetVersion
```

关键接口：

- `DatasetSpec`：数据类型无关的数据集描述。
- `DataProfile`：质量指标、分布和问题证据。
- `DataProcessingPlan`：操作、原因、预期效果和验证规则。
- `DataProcessor`：数据处理器协议。
- `DatasetVersion`：数据版本和血缘记录。

### 后续阶段

1. 增加音频、图像、文本和表格专用 profiler。
2. 增加只读数据搜索与许可证检查。
3. 增加实际修复处理器，并保证输出新数据版本。
4. 增加数据泄漏、标签一致性和任务特定质量检查。
5. 使用轻量模型验证数据处理是否真正改善下游效果。

数据处理智能体不负责模型结构和训练超参数优化。

## 超参数智能体

### 当前阶段：低成本结构化搜索

已实现：

- `SearchSpace`：结构化参数范围和约束。
- `Trial`：独立候选记录。
- `TrialBudget`：screening、promotion、confirmation 阶段预算。
- `RandomSearchStrategy`：生成初始候选。
- `SuccessiveHalvingStrategy`：晋级高潜力候选。
- `EarlyStoppingPolicy`：根据中间指标提前终止。
- `HPOService`：保存 Study、Trial 和最佳候选。

Study 与 Trial 状态由服务层校验；Study 结束时必须写入明确终态和停止原因。

推荐流程：

```text
Create Study
  -> Suggest Trials
  -> Screening Training
  -> Record Intermediate Metrics
  -> Early Stop / Complete
  -> Promote Trials
  -> Confirmation Training and Evaluation
  -> Recommend Best Recorded Trial
```

关键接口：

- `SearchParameter` / `SearchSpace`
- `Objective`
- `TrialBudget`
- `Trial`
- `HPOStudy`
- `HPOService`
- `OptimizationStrategy`
- `EarlyStoppingPolicy`

每个 Trial 保存为独立 JSON；父 HPO 实验只保存 Study 和 Trial 摘要。
不同 Trial 的训练输出保存在独立目录，Study 在服务层强制限制最大 Trial 数。

当前阶段限制：

- `epochs` 预算已经由 `TrainModel` 转换为训练 override。
- `data_fraction` 和 `max_duration_seconds` 当前只记录，不会自动控制训练。
- 提前终止当前是决策接口，真正运行中的即时中止需要 Runner 后续提供指标回调。

### 后续阶段

1. 让 Runner 支持真正的运行中指标回调和即时中止。
2. 支持数据比例预算和时间预算的框架适配。
3. 增加并发 Trial 调度与资源限制。
4. 增加多目标评分、约束优化和 Pareto 前沿。
5. 接入贝叶斯优化或 Optuna 适配器。
6. 根据模型适配器自动生成推荐搜索空间。

超参数智能体不负责解析框架专用日志、实现训练代码或处理数据质量问题。

## 记录结构

数据处理生命周期写入：

```text
metrics.quality_before
metrics.quality_after
extensions.data_lifecycle
artifacts.dataset_version
```

HPO 生命周期写入：

```text
extensions.optimization.study
extensions.optimization.trial_summary
artifacts.hpo_study
hpo_study/trials/trial_*.json
```

后续功能必须优先扩展这些稳定协议，避免重新增加 ECAPA、SpeechBrain、EER 或音频专用顶层字段。
