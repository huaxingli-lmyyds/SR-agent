# 基础架构测试说明

本测试集用于验证系统接口、字段、工具边界、实验记录、记忆系统与智能体通信。
测试不会执行真实训练、评估、GPU 任务或真实 LLM 调用。

## 测试范围

- `tests/unit/test_contracts.py`：核心协议和领域对象的字段及 JSON 序列化。
- `tests/unit/test_communication.py`：注册表、调度器、消息关联和完成策略。
- `tests/unit/test_memory.py`：按任务、模型和数据集隔离记忆。
- `tests/unit/test_experiment_tracker.py`：实验生命周期和历史最佳范围过滤。
- `tests/unit/test_data_processing.py`：通用数据检查、计划执行和版本发布。
- `tests/unit/test_hpo_strategies.py`：验证随机、网格、自适应和自动策略选择。
- `tests/integration/test_fake_runner_tool_boundary.py`：使用 Fake Runner 检测评估工具与实验记录字段传递。
- `tests/integration/test_hpo_service.py`：不训练情况下检测 HPO Study 和 Trial 生命周期。
- `tests/unit/test_langgraph_workflows.py`：验证数据处理图和注册表驱动协调图。

## 隔离原则

1. 所有测试输出写入 pytest 提供的 `tmp_path`。
2. Fake Runner 仅返回固定结构化指标，不调用 SpeechBrain。
3. Fake Agent 仅验证调度和通信协议，不创建真实 LangChain 智能体。
4. 工具集成测试保留真实工具、适配器注册和实验记录链路，只替换资源执行边界。
5. LangChain 属于智能体和工具层的可选测试依赖；缺少时，对应测试会显示为 `skipped`，纯架构测试仍会执行。

## 运行方式

运行全部基础测试：

```powershell
python -m pytest tests -q
```

只运行单元测试：

```powershell
python -m pytest tests/unit -q
```

只运行工具和服务集成测试：

```powershell
python -m pytest tests/integration -q
```

安装 LangGraph 和 LangChain 相关依赖后，工作流与工具边界测试会自动从
`skipped` 变为正常执行；未安装时不会影响纯服务和策略测试。

同时执行静态语法检查：

```powershell
python -m compileall -q -x "speechbrain|recipes" agent main.py tests
```

## 通过标准

- 所有结构化结果均可转换为 JSON。
- 请求和结果通过 `request_id`、`correlation_id` 正确关联。
- 不同任务、模型和数据集的实验与记忆不会串用。
- Fake Runner 指标可以经工具写入实验记录。
- HPO Study、Trial、最佳 Trial 的状态与字段保持一致。
- 测试期间不创建真实训练进程，不访问 GPU。
