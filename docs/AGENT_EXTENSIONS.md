# 数据准备、编排扩展与训练中决策

本次扩展直接作用于主项目。保留 `version_plan1.1/` 的历史交付副本，主项目复用了其中已验证的依赖编排、数据影响约束、咨询机会和反馈校验实现。

## 原始音频入口

```bash
python main.py --data-folder /data/raw --prepare-audio \
  --data-preparation-json '{"seed":42,"validation_fraction":0.2,"test_fraction":0.2,"dev_fraction":0.2}' \
  --output-dir agent/experiments/raw_run \
  --training-control-json '{"min_epochs":3,"patience":3}'
```

使用项目训练配置和正常安装的训练环境；模型、增强资源、硬件和预算仍由训练配置控制。`--dry-run` 只预览 HPO 计划，不读取或处理整个音频库，也不证明数据已准备成功。

音频布局支持 `说话人/录音会话/音频文件`、`说话人/音频文件`，以及带 `wav/` 包装的目录。同一会话下的片段保持在同一训练或内部验证分区。没有会话目录时，每个文件视为独立录音；同源分段应提供会话标签以防泄漏。

平铺音频通过 `labels_csv` 提供标签：

```csv
path,speaker_id,session_id
recording1.wav,speaker_a,session_1
recording2.wav,speaker_a,session_2
```

`path` 相对于原始音频根目录；不接受越界路径或重复文件条目。系统不从无标签声音推断说话人身份。准备完整的说话人验证协议至少需要六个说话人：训练、HPO 验证和最终测试各至少两个；每人至少两份清理后不同录音，训练部分还需有可留出的录音会话。比例按说话人数取整，并受这些最小数量约束，不保证严格等于音频条数比例。

处理流程使用独立输出目录，保留原音频：解码检查、最短时长过滤、统一单声道/采样率、精确重复音频去重、说话人隔离、训练内部会话划分、协议生成、质量门和原子版本发布。统一为 PCM16 WAV，默认 16 kHz；解码失败、过短或同标签重复音频记录在协议的 `rejected` 中。相同音频对应不同说话人标签会拒绝发布。每个文件独立解码，超长录音建议先按有标签的录音片段组织。无完整扫描证据时，可选的操作影响/输出大小限制会阻止准备操作，不能把估算当成磁盘配额。

输出包含：

| 文件 | 用途 |
| --- | --- |
| `train.csv`、`dev.csv` | 分类训练和内部验证，使用可迁移的 `$data_root` 路径 |
| `hpo_validation.csv`、`validation_pairs.txt` | HPO 的独立说话人验证集 |
| `final_test.csv`、`final_test_pairs.txt` | 模型选择结束后的最终评估集 |
| `training_exclusion_pairs.txt` | 从训练排除两种外部评估人群 |
| `prepared_protocol.json` | 划分参数、清单和音频哈希、来源标签、过滤记录 |

生成的正负试验对数量随数据量线性增长，是本地实验协议，不声称等同于 VoxCeleb 官方基准协议。最终测试结果不得用于 HPO 选参。

## 数据与 HPO 的契约

配置文件中的 `data_processing.prepare` 与 `context.data_processing.prepare` 支持同一组选项，后者优先。已有操作注册接口保持不变；底层操作名为 `prepare_speaker_dataset`。该操作必须是最终变更操作，LLM 不得创建或改写划分。数据智能体仍可通过既有操作和咨询接口执行训练前优化。

数据智能体在 `summary.data_handoff.prepared_data` 返回协议根目录、协议哈希、采样率和训练说话人数。HPO 核验协议、清单和音频，绑定验证/排除文件；显式提供冲突协议会报错。内置 ECAPA、ResNet、x-vector 配有默认验证配置，外部模型可显式传入 `verification_config`。训练工具将准备好的训练清单、内部验证清单、采样率和分类器人数交给 Runner；SpeechBrain 跳过再次划分与数据下载准备步骤。模型增强资源的准备仍服从模型配置。

传入已有准备目录时，保留原划分和协议；已有外部评估说话人受到保护，过滤或重新划分该目录会被阻止。若确实要建立新实验协议，应从原始目录创建新版本。HPO 恢复使用持久化的协议和训练规则，不重新运行数据智能体，并重新检查准备数据。迁移新实验可整体复制准备目录；正在恢复的实验还受既有绝对路径和配置快照约束。

## 编排与后续智能体

```python
coordinator.register_agent(AgentRegistration(
    agent_type="evaluation_agent",
    actions=("evaluate_system",),
    factory=EvaluationAgent,
    requires=("hpo_agent",),
))
```

扩展只需实现 `execute_task(AgentTaskRequest) -> AgentTaskResult`。编排器在执行前检查缺失依赖和循环依赖，按稳定拓扑顺序调度；失败节点阻止其依赖节点，独立节点仍可执行。上下文的 `previous_results` 提供完整结果，`data_handoff` 和 `hpo_feedback` 提供 HPO 输出的共享契约。注册时 `required=False` 可避免将缺少该智能体结果列为未完成；实际任务失败仍遵循既有 `CompletionPolicy`。

恢复 HPO 时也运行其已注册的下游智能体。扩展评估智能体应从 `context.data_handoff` 读取数据，而不依赖数据智能体在本次恢复中重新执行。当前执行模型是有向无环依赖图，一次调度每个角色；没有增加任意循环、自动重新划分或自动启动最终测试的隐式行为。

## 训练中 HPO 决策

`runtime_options.training_control`（CLI：`--training-control-json`）启用规则，省略时维持旧行为：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `metric` | `valid_loss` | 仅允许内部验证 `valid_loss` / `valid_error_rate` |
| `min_epochs` | 3 | 平台/同预算比较的预热轮数 |
| `patience` | 3 | 预热结束后允许无足够改善的轮数 |
| `min_delta` | 0.001 | 视为改善的最小绝对变化 |
| `peer_min_trials` | 3 | 建立同预算逐轮阈值所需的已完成试验数 |
| `peer_patience` | 2 | 连续超过同预算阈值的轮数 |
| `peer_factor` | 2.0 | 同预算、同轮次中位指标的倍数门槛，加上 `min_delta` |

HPO 在派发 Trial 前结合主目标、训练预算、已完成同预算曲线生成可序列化规则，并写入 Trial 的 `cost.training_control_spec`。规则在训练验证边界检查平台、同预算落后和无效数值，不访问最终测试集，也不在训练进程中调用外部 LLM。内部分类验证指标是代理信号，不冒充 EER 或最终目标改善。缺少足够历史时不启用同预算比较。

训练进程写入 `training_control.json`，记录观察、原因与证据。分布式进程在同一验证边界聚合各 rank 的指标均值，rank 0 写入决策并广播；这是 rank 指标均值，不宣称是任意不等长数据分片的严格样本加权均值。

早停在 Trial 层为 `stopped`，保留检查点、曲线、成本与原因，不按运行失败重试，不进入完整预算候选晋级/主指标排序，也不继续花费该 Trial 的验证评估成本。达到预算终点的正常训练仍按完整 Trial 评估。多种子确认阶段禁用早停、使用完整固定预算。普通与管理式独立进程执行共享这些语义。

外部 Runner 可以声明 `supports_training_control=True`，消费训练覆盖参数 `_hpo_training_control`，使用 `EpochTrainingController`，并在归一化结果的 `extensions.training_control` 中返回 `runtime.training_control` 决策和 `epoch_data`。未声明支持的 Runner 会明确拒绝该选项，不会悄悄忽略。

## 反馈与验证

HPO 结果通过 `summary.hpo_feedback` 导出带 Study、数据版本/协议哈希和验证角色的证据，编排器传给下游智能体。调用方也可将它作为后续数据请求的 `context.hpo_feedback`。准备协议哈希不同或提供最终测试证据时拒绝采用。诊断是观察，不能据此自动认定数据导致了模型效果变化。不会在当前 Study 中修改冻结的数据或协议。

策略评审和 Campaign 汇总也保留早停次数、原因，以及最多五个带参数、分配预算、实际停止轮次和曲线摘要的实例。提前结束的内部验证曲线不会伪装成完整预算 EER，但可以作为后续策略建议的诊断证据。

主要验收入口：

```bash
python -m pytest tests/unit/test_audio_preparation.py tests/unit/test_training_control.py \
  tests/integration/test_prepared_data_system.py \
  tests/integration/test_epoch_training_backend.py \
  tests/integration/test_training_control_system.py
python -m pytest tests
```

覆盖真实小 WAV 数据准备、目录迁移、标签清单、泄漏/重复防护、真实 SpeechBrain CPU 训练与早停、普通/独立进程 HPO、双进程 Gloo 决策一致性、恢复和扩展评估智能体。没有将这些工程测试表述为真实数据集 EER 提升或 GPU 性能实验。

2026-11-17 在 Windows 本地 `.venv-hpo` 环境完成全量回归：**786 通过、2 跳过**，耗时 356.25 秒。跳过项为需显式启用的外部 LLM 冒烟测试、仅适用于 POSIX 的共享文件锁测试。69 条警告来自 Optuna 实验性参数、PyTorch 加载接口和 SpeechBrain 弃用提示。变更 Python 文件的 Ruff `F,E9` 检查及 `git diff --check` 均通过；测试日志、JUnit XML、临时音频和模型文件保留在被忽略的 `.tmp/` 中。
