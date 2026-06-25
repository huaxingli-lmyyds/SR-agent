# 系统接口审查、AAAI 创新路线与多模型接入指南

> 审查日期：2026-06-22  
> 审查范围：`agent/`、`configs/`、`recipes/`、`scripts/`、`tests/`、项目依赖与实验记录接口；不审查外部 SpeechBrain 源码。  
> 目标会议：AAAI-27。官方页面给出的摘要截止日期为 2026-07-21、全文截止日期为 2026-07-28、补充材料和代码截止日期为 2026-07-31，均为 UTC-12。提交前仍须以 [AAAI-27 官方页面](https://aaai.org/conference/aaai/aaai-27/) 和当届 Author Kit 为准。

## 1. 总体结论

当前项目已经形成四层边界：LangGraph 描述工作流，HPOService/DataProcessingService 执行业务约束，DecisionPolicy 执行确定性分支，LLM 只产生结构化策略提案和诊断建议。该设计方向合理，也比“让 LLM 自由调用工具并维护状态”更适合作为可审计实验系统。

系统目前可以支持 ECAPA-TDNN + SpeechBrain 的单机 HPO 闭环，也具备注册其他 Task、Model、Runner 和 DataProcessor 的接口。此次审查修复了三个直接影响结果可信度的问题：

1. 单个 Trial 的训练或评估结果不再提前把父实验写成 `success/failed`；Study/Campaign 才拥有父实验终态的决定权。
2. ECAPA 的默认搜索空间从 HPOAgent 硬编码迁移到 ModelAdapter；其他模型可以声明自己的搜索空间和参数校验。
3. Task、Model、Runner 通过 `resolve_adapter_bundle()` 统一解析；HPO 实验记录新增数据集 ID、数据版本和上游数据处理实验 ID。

当前代码仍不能支撑“方法优于现有 HPO”或“跨模型通用”的论文结论，因为仓库中没有足够的真实重复实验、显著性检验和第二种模型端到端证据。论文应把系统能力写成实现，把经过对照实验验证的决策机制写成贡献，不能反过来。

## 2. 接口审查结果

| 边界 | 当前接口 | 审查结果 | 状态 |
|---|---|---|---|
| 协调器 -> 智能体 | `AgentTaskRequest/AgentTaskResult` | 请求 ID、预算、上下文、结果可 JSON 化；状态值尚未统一为枚举 | 可用，建议增加 schema version |
| 数据处理 -> HPO | `data_handoff` | 校验 `consumer_uri`、消费状态和本地存在性；已继续传递 dataset ID/version/upstream experiment | 已对齐 |
| HPO -> 调度器 | `HPOStudy/Trial/TrialBudget` | Trial 状态机、rung、成本、产物、失败原因均结构化 | 已对齐 |
| LLM -> 决策层 | `StrategyProposal` | LLM 不能直接更改 Trial；提案由 StrategyDecisionPolicy 审批并记录拒绝原因 | 已对齐 |
| Task/Model/Runner | `resolve_adapter_bundle()` | 单入口解析并验证 Runner 对 implementation/model family 的兼容性 | 已修复 |
| Model -> HPO | `default_search_space()`、`validate_parameters()` | 模型可以定义搜索参数，训练前再次校验候选 | 已修复 |
| Runner -> 服务层 | `OperationResult` | 训练/评估结果统一为 metrics/artifacts/execution/error | 可用 |
| Trial -> Experiment | `ExperimentService.record_result(update_status=False)` | Trial 更新明细但不能结束父实验 | 已修复 |
| Coordinator -> 运行记录 | `OrchestrationResult` | 返回状态与落盘状态统一为 `success/failed`；详细完成原因保留在 completion | 已修复 |

### 2.1 仍需继续收紧的格式

P0（正式大规模实验前）：

- 为 `AgentTaskRequest`、`AgentTaskResult`、`OperationResult`、Study 文件增加统一 `schema_version`。
- 定义状态枚举和合法迁移表，拒绝 `complete/completed/success` 混用。
- 给 Runner 增加显式 capability 描述，例如 timeout、resume、intermediate metrics、distributed、supported budget fields；启动 Study 前完成能力协商。
- 保存代码版本、依赖锁文件哈希、配置内容哈希、数据清单哈希、随机种子和硬件信息。
- 增加原子写入和文件锁，避免中断或并发 Trial 破坏 JSON。

P1（跨模型实验前）：

- 将模型注册键从单一 `model_family` 扩展为 `(model_family, implementation, version)`，支持同一模型的 PyTorch/SpeechBrain 等多实现并存。
- 为模型参数增加 JSON Schema/Pydantic 级别的类型、范围、条件依赖校验。
- 将 checkpoint selector 变成 Runner/Model 可覆盖策略，不假设所有框架都使用 `.ckpt` 或 SpeechBrain 目录。
- 把 `OperationRequest` 真正作为 Train/Evaluate 的公共内部输入，减少工具函数参数漂移。

## 3. 当前完整执行路径

```text
CoordinatorAgent
  -> OrchestrationWorkflow (LangGraph)
  -> DataProcessingAgent
     -> profile -> deterministic plan validation -> execute operations
     -> DatasetVersion -> data_handoff
  -> HPOAgent
     -> resolve_data_handoff
     -> ModelAdapter.default_search_space
     -> optional LLM StrategyProposal
     -> StrategyDecisionPolicy.review
     -> HPOService.create_study
     -> HPOScheduler (LangGraph)
        -> strategy.suggest
        -> TrainModel
           -> resolve_adapter_bundle
           -> ModelAdapter.validate_config/validate_parameters
           -> RunnerAdapter.run_training
           -> OperationResult
           -> HPOService.record_trial
        -> RunEvaluation(trial_id)
           -> exact trial checkpoint
           -> RunnerAdapter.run_evaluation
           -> TaskAdapter.validate_metrics
           -> HPOService.record_trial
        -> rung review / promote / retry / terminate
     -> CampaignPolicy
        -> target/patience/cost/min-improvement decision
        -> optional next Study
  -> CompletionPolicy
  -> orchestration experiment record
```

关键记录应至少包含：`experiment_id`、`campaign_id`、`study_id`、`trial_id`、`task`、`model.family`、`model.implementation`、`execution.runner`、`dataset_id`、`dataset_version`、`parameters`、`budget`、`metrics`、`cost`、`artifacts`、`strategy_decision`、`failure_category`、`status` 和时间戳。

## 4. AAAI 创新点选择

### 4.1 主创新：可验证的反馈驱动策略元控制器

不要把“LangGraph + LLM + Optuna”本身作为创新。建议研究问题定义为：

> 在昂贵、反馈稀疏且可能失败的模型训练中，如何让 LLM 利用语义经验提出策略调整，同时通过确定性约束投影保证搜索过程有效、可复现且不超预算？

#### 实现方法

1. 构造观测状态 `s_t`：当前 rung 完成率、主指标分布、最优值改进、边界命中率、失败聚类、剩余 GPU 时间、历史提案批准率和相似 Campaign 经验。
2. LLM 仅生成 `StrategyProposal`：建议搜索策略、边界调整、预算分配和理由，不携带工具调用或状态变更。
3. 确定性投影器 `P_C` 根据约束集合 `C` 审核提案：参数类型/范围、预算上限、Runner capability、最小 rung 样本数、合法状态迁移、数据版本一致性。
4. 生成 `StrategyDecisionRecord`：逐字段记录 approved/rejected、原因码、原提案和最终采用方案。
5. 元控制动作只影响尚未生成的候选；已启动 Trial 和历史记录不可回写。
6. 使用成本感知效用评价动作：

```text
utility = normalized_improvement
          - lambda_cost * normalized_gpu_hours
          - lambda_fail * failure_rate
          - lambda_risk * invalid_proposal_rate
```

7. 每个 rung 或固定 Trial 窗口触发一次评审；使用 patience 和最小改进抑制频繁震荡。

论文真正需要证明的是：与 TPE/Random/ASHA 以及无投影的 LLM-HPO 相比，该机制在相同预算下获得更低 regret、更少无效 Trial 或更高成功率。

### 4.2 第二创新：失败与边界感知的层次化反馈

当前系统已有失败分类、边界检测和 rung 评审基础，但需要把它们形成可复现实验算法。

#### 实现方法

- 将失败划分为 OOM、timeout、data、configuration、numerical、runtime、unknown，并区分 recoverable。
- 对参数空间计算边界命中率、局部改进方向和有效样本密度。
- OOM 触发 batch size 上界收缩或梯度累积建议；timeout 触发低保真预算；数值失败收缩学习率/损失缩放范围；数据错误不得通过改超参数重试。
- 只有同一 rung 的完成 Trial 可以参与晋升，且满足最小完成数。
- 将失败样本纳入策略模型，而不是从 TPE/经验池中静默删除。

可报告的研究指标包括 failure recovery precision、恢复后的有效 Trial 比例、浪费 GPU 小时和边界收缩后的最优值变化。

### 4.3 支撑创新：带负迁移保护的跨 Study 经验复用

当前 Memory 已形成反馈回路，但主要是按 scope 检索历史记录，尚不能声称具有可靠迁移能力。

#### 实现方法

- 经验单元保存任务、模型、数据统计、搜索空间、预算、决策、结果和置信度。
- 根据 task/model/dataset meta-feature 计算经验相似度。
- 仅当相似度和历史收益超过阈值时允许 warm start；否则退化为无先验策略。
- 用在线收益更新经验权重，对负收益经验降权并记录 negative-transfer event。
- 对比 no-memory、recent-memory、similarity-memory、oracle-memory。

如果跨模型/跨数据集没有稳定正迁移，这一项应降为系统功能，不列为论文主贡献。

### 4.4 高风险进阶创新：数据处理与 HPO 联合优化

当前数据处理智能体负责生成可消费的数据版本，HPO 只消费固定版本。若要成为创新，需要把数据决策纳入联合动作空间，例如采样比例、增强强度、分段长度和类别平衡策略，并使用双层或多目标优化：外层选择数据策略，内层优化模型参数，同时约束数据处理成本和公平性。该方向潜力大，但变量和实验成本都会明显增加，不建议在 AAAI-27 截止前替代主创新。

## 5. 必做实验

### 5.1 基线

- Random Search、Grid Search、Optuna TPE。
- Successive Halving/ASHA（相同 fidelity 和总训练预算）。
- 固定 TPE + Campaign，不使用 LLM。
- LLM 直接提案但无确定性投影。
- 完整系统。

### 5.2 消融

- 去掉 rung feedback。
- 去掉 failure/boundary features。
- 去掉策略投影，只保留格式校验。
- 去掉跨 Study memory。
- 去掉 Campaign，仅运行一个 Study。
- LLM 替换为固定规则提案，隔离语言模型本身的贡献。

### 5.3 数据与模型覆盖

最低可信配置是两个模型族、两个数据条件：

- 已有：ECAPA-TDNN + SpeechBrain。
- 新增：x-vector、ResNet speaker encoder 或其他训练成本可控且具有公开基线的模型，至少选一个完成全流程。
- 数据：VoxCeleb 标准协议；再增加一个域移位条件，例如跨数据集测试、噪声/短语音子集或另一个合法可用的说话人数据集。

每种关键比较至少使用 3 个独立随机种子；若预算允许使用 5 个。报告 mean/std、bootstrap 置信区间或配对显著性检验。除 EER/minDCF 外，必须报告 GPU-hours、wall-clock、Trial 数、失败率、达到目标所需成本和 anytime best-so-far 曲线。

### 5.4 论文证据包

- 冻结环境锁文件、容器或可复现安装说明。
- 每张表可追溯到 Campaign/Study/Trial ID。
- 导出原始 Trial 表、聚合脚本和绘图脚本。
- 保存 LLM 模型标识、temperature、prompt version、原始提案和决策记录。
- 记录数据许可、隐私风险、算力和碳成本边界。
- 根据 AAAI reproducibility checklist 准备 Code and Data Appendix。
- AAAI 要求作者对全文负责，AI 系统不能作为作者或可引用来源，且在出版物开发中的角色需要按官方政策妥善记录，见 [AAAI Publication Policies & Guidelines](https://aaai.org/aaai-publications/aaai-publication-policies-guidelines/)。

## 6. 接入其他模型的实现步骤

### 6.1 同一任务、同一 Runner，仅新增模型

1. 在 `agent/models/` 新建 ModelAdapter。
2. 实现配置校验、默认搜索空间和候选参数校验。
3. 在 `agent/models/__init__.py` 注册。
4. 确认 Runner 的 `supported_model_families` 包含新模型。
5. 提供训练配置和评估配置。
6. 添加契约测试和一个最小训练/评估集成测试。

```python
class ResNetSpeakerAdapter:
    model_family = "resnet_speaker"
    implementation = "speechbrain"
    default_evaluation_config = "verification_resnet.yaml"

    def validate_config(self, config):
        required = {"embedding_model", "classifier", "output_folder"}
        missing = sorted(required - config.keys())
        if missing:
            raise ValueError(f"missing ResNet config fields: {missing}")

    def default_search_space(self):
        return {
            "parameters": [
                {"name": "lr", "parameter_type": "float",
                 "low": 1e-5, "high": 3e-3, "scale": "log"},
                {"name": "batch_size", "parameter_type": "categorical",
                 "choices": [16, 32]},
            ],
            "constraints": [],
        }

    def validate_parameters(self, parameters):
        if "lr" in parameters and float(parameters["lr"]) <= 0:
            raise ValueError("lr must be positive")
```

### 6.2 新增训练框架或外部执行器

实现 RunnerAdapter 的五个边界：

- `run_training(config_path, overrides)`
- `collect_training_result(raw, output_folder, experiment_dir)`
- `normalize_training_result(raw) -> OperationResult`
- `run_evaluation(config_path, model_path, data_path, overrides)`
- `normalize_evaluation_result(raw) -> OperationResult`

Runner 必须把框架特有输出归一化，业务层不应解析框架日志。至少声明 `runner`、`supported_implementations`、`supported_model_families` 和默认评估配置。checkpoint 必须以 `Artifact(type="checkpoint", ...)` 返回，主指标必须进入 `metrics[split][metric]`。

### 6.3 新增任务

若主指标或指标校验方式变化，再实现 TaskAdapter：声明 `task_type`、`primary_metric`、`metric_mode`，并在 `validate_metrics()` 中拒绝缺失、NaN 或方向不明确的指标。模型结构和任务指标不要混入 Runner。

### 6.4 接入验收门槛

一个新模型只有同时满足以下条件才算“系统已支持”，不能以成功注册类作为完成标准：

- 适配器组合可由 `resolve_adapter_bundle()` 解析。
- 默认搜索空间能生成合法候选，非法参数在训练前被拒绝。
- 低预算训练产生 checkpoint、指标、成本和终态 Trial。
- `RunEvaluation(trial_id=...)` 只选择该 Trial 的 checkpoint。
- 数据版本字段出现在 HPO 实验记录中。
- Campaign 能结束并给出 best Trial 或明确失败原因。
- 至少一条 CPU/mock 契约测试和一条真实 GPU smoke test 通过。

## 7. 优先级与投稿判断

AAAI-27 时间很紧。建议优先顺序：

1. 两周内完成 schema version、状态枚举、Runner capabilities、原子记录与实验导出。
2. 同时接入第二模型并完成低保真端到端 smoke test。
3. 固化元控制算法和对照预算，禁止实验中继续改规则。
4. 先跑 Random/TPE/ASHA 和无 LLM 消融，再决定完整 LLM 实验规模。
5. 只有在至少两个场景中观察到稳定收益后，才把跨 Study 迁移写为贡献。

当前准备度判断：代码架构已经接近可开展系统实验，但论文证据仍处于“未建立”状态。最稳妥的 AAAI 主线不是宣传多智能体数量，而是证明“受约束的 LLM 策略元控制”在昂贵 HPO 中同时改善效果、成本和可靠性。
## 8. 本次验证记录

- `python -m compileall -q agent tests`：通过。
- `python -m pytest -q`：75 passed，LangGraph、Optuna TPE、工具边界和新增模型契约测试均实际执行，无跳过。
- `python -m ruff check agent tests --select F,E9`：通过；未定义名称、无效导入、语法类错误已清零。
- `git diff --check`：本次相关文件无新增补丁格式错误；仓库中一个既有测试文件仍有 EOF 空行提示，不影响运行。
- 全量 Ruff 规则首次扫描得到 1755 项，主要是存量 Python 3.10 注解现代化、导入排序和空白样式。该债务应单独建立格式化提交，避免与实验逻辑修改混合。
- 为执行完整测试，当前全局 Anaconda 环境安装了 LangGraph、LangChain Core、Optuna 和 Ruff。环境原有 `googletrans 4.0.0rc1` 要求旧版 `httpx==0.13.3`，与当前 LangChain 的 `httpx` 依赖冲突；本项目测试不受影响，但正式实验必须使用独立虚拟环境或容器。