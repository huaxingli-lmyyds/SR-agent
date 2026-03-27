# LangChain v1.0 智能体使用指南

本文档介绍如何使用基于 LangChain v1.0 最新架构的声纹识别超参数优化智能体。

## 概述

`LangChainHPOAgent` 是使用 LangChain v1.0 最新标准架构实现的智能体，具有以下特点：

- ✅ 使用 `create_agent` 统一 API（无需 AgentExecutor）
- ✅ 直接导入已用 `@tool` 修饰的工具
- ✅ 使用标准消息格式进行调用
- ✅ 支持自定义优化目标
- ✅ 提供详细的执行日志和结果分析

## 快速开始

### 基本使用

```python
from agent.agents import LangChainHPOAgent, create_react_agent

# 方式 1: 直接创建智能体
agent = LangChainHPOAgent(
    model_name="GLM-4.7",
    temperature=0.2,
    max_iterations=10,
    verbose=True
)

# 方式 2: 使用便捷函数
agent = create_react_agent(
    model_name="GLM-4.7",
    temperature=0.2,
    max_iterations=10
)
```

### 优化超参数

```python
# 自动优化超参数，目标 EER < 0.08
result = agent.optimize_hyperparameters(
    target_eer=0.08,
    max_experiments=5
)

# 查看结果
print(f"最佳配置: {result.best_config}")
print(f"执行总结: {result.execution_summary}")
print(f"最终答案: {result.final_answer}")
```

### 自定义任务

```python
# 运行自定义优化任务
objective = """
优化 ECAPA-TDNN 模型，重点关注批次大小对性能的影响。
请尝试不同的批次大小（32, 64, 128），并找出最佳配置。
"""

result = agent.run_custom_task(objective)
```

## 可用工具

智能体集成了以下工具（已在 tools 包中用 @tool 修饰）：

### 配置工具（config_tools）

- `ReadConfig` - 读取当前配置
- `UpdateConfig` - 更新配置参数（需要 JSON 字符串参数）
- `GetConfigStructure` - 获取配置结构
- `ListConfigParameters` - 列出所有参数
- `ResetConfig` - 重置配置到默认值

### 训练工具（training_tools）

- `TrainModel` - 训练模型
- `EvaluateModel` - 评估模型性能
- `AnalyzeResults` - 分析实验结果
- `CompareExperiments` - 比较多个实验

### 评估工具（evaluation_tools）

- `RunEvaluation` - 运行评估实验
- `GetEvaluationResults` - 获取评估结果
- `CompareEvaluations` - 比较多个评估
- `ListEvaluations` - 列出所有评估

## 架构对比

### 旧版架构（已弃用）

```python
from langchain.agents import create_react_agent, AgentExecutor

agent = create_react_agent(model, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "..."})
```

### 新版架构（LangChain v1.0）

```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=tools,
    prompt=system_prompt
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "..."}
    ]
})
```

## 关键差异

| 特性 | 旧版 (AgentExecutor) | 新版 (create_agent) |
|------|---------------------|-------------------|
| API | 两步创建 | 一步创建 |
| 输入格式 | `{"input": "..."}` | `{"messages": [{"role": "user", "content": "..."}]}` |
| 执行器 | 需要 AgentExecutor | 内置执行循环 |
| 工具装饰 | 需要手动包装 | 直接使用 @tool 修饰的工具 |
| 中间件 | 回调函数 | 中间件系统 |

## 参数说明

### LangChainHPOAgent 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | str | "GLM-4.7" | 使用的模型名称 |
| `temperature` | float | 0.2 | 温度参数，控制随机性 |
| `max_iterations` | int | 10 | 最大迭代次数 |
| `verbose` | bool | True | 是否显示详细输出 |
| `config_path` | str | "../configs/train_ecapa_tdnn.yaml" | 配置文件路径 |

### optimize_hyperparameters 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_eer` | float | 0.08 | 目标 EER 值 |
| `max_experiments` | int | None | 最大实验次数 |
| `custom_objective` | str | None | 自定义优化目标 |

## 返回结果

### OptimizationResult

```python
@dataclass
class OptimizationResult:
    objective: str              # 优化目标
    final_answer: str           # 最终答案
    total_steps: int            # 总步骤数
    intermediate_steps: List    # 中间步骤
    best_config: Dict           # 最佳配置
    execution_summary: str      # 执行总结
```

## 工作流程

智能体会按照以下流程自动执行：

1. **读取配置** - 使用 `ReadConfig` 了解当前配置
2. **分析参数** - 使用 `ListConfigParameters` 列出可调参数
3. **提出策略** - 基于知识提出优化建议
4. **更新配置** - 使用 `UpdateConfig` 调整参数（JSON 字符串格式）
5. **训练模型** - 使用 `TrainModel` 执行训练
6. **评估性能** - 使用 `EvaluateModel` 获取结果
7. **分析结果** - 使用 `AnalyzeResults` 分析性能
8. **迭代优化** - 重复步骤 4-7 直到达到目标

## 高级用法

### 自定义优化策略

```python
# 专注于特定参数
objective = """
优化 ECAPA-TDNN 模型的学习率。
请尝试以下学习率：[0.0001, 0.0003, 0.0005, 0.001]
记录每次实验的 EER，并找出最佳值。
"""

result = agent.run_custom_task(objective)
```

### 批量实验

```python
# 限制最大实验次数
result = agent.optimize_hyperparameters(
    target_eer=0.07,
    max_experiments=10
)
```

### 多轮对话

```python
# 支持多轮对话，智能体会记住上下文
history = [
    {"role": "user", "content": "请读取当前配置"},
    {"role": "assistant", "content": "已读取配置..."},
    {"role": "user", "content": "现在将学习率调整为 0.0005"}
]

result = agent.invoke({"messages": history})
```

## 工具使用示例

### 直接使用工具

```python
from agent.tools import ReadConfig, UpdateConfig, TrainModel

# 读取配置
config_tool = ReadConfig()
config = config_tool.invoke({})
print(config)

# 更新配置（注意参数必须是字符串）
update_tool = UpdateConfig()
result = update_tool.invoke('{"lr": 0.0005, "batch_size": 64}')
print(result)

# 训练模型
train_tool = TrainModel()
result = train_tool.invoke({})
print(result)
```

## 错误处理

智能体会自动处理工具执行错误：

```python
try:
    result = agent.optimize_hyperparameters(target_eer=0.05)
except Exception as e:
    print(f"优化失败: {e}")
    # 查看中间步骤了解问题
    for step in result.intermediate_steps:
        print(f"步骤: {step}")
```

## 性能优化建议

1. **合理的迭代次数** - 根据任务复杂度设置 `max_iterations`
2. **温度参数** - 较低的 `temperature`（0.1-0.3）产生更稳定的结果
3. **批量操作** - 在一次任务中尽量完成多个相关操作
4. **工具选择** - 让智能体自动选择最合适的工具

## 与旧代码的兼容性

为了保持向后兼容，我们提供了 `ReActHPOAgent` 别名：

```python
from agent.agents import ReActHPOAgent  # 等同于 LangChainHPOAgent

agent = ReActHPOAgent(
    model_name="GLM-4.7",
    temperature=0.2
)
```

## 常见问题

### Q: 为什么移除了 AgentExecutor？

A: LangChain v1.0 将执行循环内置到 `create_agent` 返回的对象中，不再需要单独的执行器。这使得 API 更简洁，使用更方便。

### Q: 如何使用中间件？

A: 中间件可以通过 `middleware` 参数添加：

```python
from langchain.agents.middleware import summarizationMiddleware

agent = create_agent(
    model=model,
    tools=tools,
    prompt=system_prompt,
    middleware=[
        summarizationMiddleware(trigger={"tokens": 500})
    ]
)
```

### Q: 工具参数为什么必须是字符串？

A: 由于工具使用 `@tool` 装饰器修饰，参数类型被定义为字符串。例如 `UpdateConfig` 需要 JSON 字符串格式的参数。

## 示例代码

完整示例请参考 `agent/tests/test_react_agent.py`。

## 更新日志

- **v1.0** - 采用 LangChain v1.0 架构，移除 AgentExecutor，直接使用 create_agent
- **v0.9** - 初版 ReAct 智能体实现