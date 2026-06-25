# 协调层第一阶段重构说明

## 本阶段目标

本次重构完成四项基础能力：

1. 智能体注册表：协调器通过注册信息发现智能体，不再为每个智能体编写专用调用工具。
2. 通用任务调度：所有智能体统一通过 `AgentTaskRequest -> execute_task -> AgentTaskResult` 调用。
3. 完成策略：协调器结束前必须检查必要智能体是否成功执行，不能仅凭大模型返回文本判定成功。
4. 可序列化结果：跨智能体传递和实验记录只使用 JSON 原生结构，不再依赖进程内 Python 对象。

## 核心接口

### 注册表

```python
registry.register(AgentRegistration(
    agent_type="example_agent",
    actions=("analyze", "optimize"),
    description="...",
    factory=create_example_agent,
))
```

新增智能体只需要实现：

```python
class ExampleAgent:
    def execute_task(self, request: AgentTaskRequest) -> AgentTaskResult:
        ...
```

注册表负责校验 `agent_type` 和 `action`，并在调度时创建对应智能体。协调器不需要新增 `LaunchExampleRound` 一类工具。

### 通用调度

协调器只向大模型暴露一个执行入口：

```text
DispatchAgentTask(agent_type, action, objective, context_json, budget_json)
```

调用链如下：

```text
CoordinatorAgent
  -> TaskDispatcher
  -> AgentRegistry.create(agent_type, action)
  -> SpecializedAgent.execute_task(request)
  -> AgentTaskResult
  -> TaskExecutionRecord
```

`TaskExecutionRecord` 保存请求、结果、智能体类型、动作和开始/结束时间，可直接写入实验记录。

### 可序列化结果协议

`AgentTaskResult` 的固定结构：

```json
{
  "status": "success",
  "summary": {},
  "metrics": {},
  "artifacts": [],
  "recommendations": [],
  "experiment_ids": {},
  "error": null,
  "request_id": "request_xxx"
}
```

- `summary`：模型无关摘要和后续智能体需要的上下文。
- `metrics`：结构化指标，允许不同任务使用不同指标名。
- `artifacts`：数据版本、检查点、报告等产物描述。
- `recommendations`：建议的后续动作，不直接等同于执行命令。
- `experiment_ids`：本次任务关联的实验记录。
- `error`：失败原因。

协议边界通过 `json_safe` 将数据类、路径、时间等转换为 JSON 原生值。已删除 `runtime_result`，因此后续可替换为远程调用或异步消息系统。

### 完成策略

默认 `CompletionPolicy` 要求：

- `data_processing_agent` 至少成功一次。
- `hpo_agent` 至少成功一次。
- 任一已调度任务失败时，整体不能标记为成功。

协调器结束时生成完成判定并写入协调实验记录：

```json
{
  "complete": false,
  "status": "incomplete",
  "reasons": ["required agents not completed: hpo_agent"]
}
```

## 当前代码职责

- `agent/agents/communication.py`：请求、结果、消息协议以及 JSON 边界转换。
- `agent/agents/coordination.py`：注册表、调度器、执行记录和完成策略。
- `agent/agents/orchestrator.py`：维护协调上下文、提供通用工具、保存协调实验。
- `data_processing_agent.py` 与 `hpo_agent.py`：执行领域任务并返回统一结果协议。

## 后续优化方向

### 第二阶段：目标与决策策略

- 增加模型无关的 `OrchestrationGoal`，统一描述任务、目标指标、约束和预算。
- 增加 `DecisionPolicy`，对智能体的 `recommendations` 做选择、拒绝和排序。
- 将完成策略扩展为可检查必需指标、必需产物和目标阈值。

### 第三阶段：预算、恢复与持久通信

- 增加 `BudgetPolicy`，强制限制总调用次数、单智能体调用次数、训练预算和耗时。
- 将消息历史由内存列表升级为 JSONL 或数据库存储。
- 保存工作流检查点，支持失败后从未完成任务继续执行。

### 第四阶段：工作流图与并发

- 引入任务依赖图，表达串行、并行和条件分支。
- 为可重试任务增加退避、幂等键和超时策略。
- 将同步 `TaskDispatcher` 扩展为异步或远程调度器，保持请求结果协议不变。

### 第五阶段：更多智能体

建议优先增加评估智能体和诊断智能体。新增智能体应遵守以下边界：

- 只通过注册表暴露能力。
- 只接收 `AgentTaskRequest`。
- 只返回 `AgentTaskResult`。
- 不直接修改协调器状态。
- 领域特有数据放入 `summary/metrics/artifacts`，不要增加协调器专用字段。
