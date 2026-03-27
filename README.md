# ECAPA-TDNN 超参数优化智能体

基于 LangChain 构建的智能体系统，用于自动化调整 ECAPA-TDNN 声纹识别模型的超参数。

## 功能特性

- ✅ **智能配置解析**：支持 SpeechBrain 的 YAML 特殊语法（!ref, !new:, !apply: 等）
- ✅ **自动化超参数搜索**：通过对话式交互优化模型参数
- ✅ **实验管理**：自动记录、追踪和比较实验结果
- ✅ **训练控制**：启动、监控和分析训练任务
- ✅ **性能评估**：自动提取和比较模型性能指标

## 快速开始

### 1. 环境配置

```bash
pip install langchain langchain-openai
pip install ruamel.yaml
```

### 2. 基本使用

```python
from agent.hpo_agent import create_hpo_agent

# 创建智能体
agent = create_hpo_agent()

# 开始对话优化
response = agent.chat(
    "我想优化 ECAPA-TDNN 模型的学习率和批量大小，" +
    "请帮我设计实验方案"
)
```

### 3. 更新配置

```python
# 通过工具更新配置
agent.chat(
    "请将学习率调整为 0.0005，批量大小调整为 32"
)
```

### 4. 运行训练

```python
# 启动训练任务
agent.chat(
    "使用当前配置开始训练"
)
```

### 5. 分析结果

```python
# 比较实验结果
agent.chat(
    "比较最近 5 个实验的结果，告诉我哪个配置最好"
)
```

## 核心工具

### 配置管理工具

- `ReadConfig`: 读取当前配置
- `UpdateConfig`: 更新配置参数
- `ListConfigParameters`: 列出所有配置参数
- `GetConfigStructure`: 获取配置结构信息
- `ResetConfig`: 重置配置到默认值

### 训练管理工具

- `TrainModel`: 运行模型训练
- `EvaluateModel`: 评估模型性能
- `AnalyzeResults`: 分析实验结果
- `CompareExperiments`: 比较多个实验

### 评估工具

- `RunEvaluation`: 运行模型评估
- `GetEvaluationResults`: 获取评估结果
- `CompareEvaluations`: 比较评估结果
- `ListEvaluations`: 列出所有评估

## 配置文件示例

```yaml
# configs/train_ecapa_tdnn.yaml

# 基础参数
lr: 0.001
batch_size: 24
seed: 1986

# 变量引用
max_lr: !ref <lr>  # 解析为 0.001
output_folder: results/ecapa_augment/<seed>  # 解析为 results/ecapa_augment/1986

# 模型配置
embedding_model: !new:speechbrain.lobes.models.ECAPA_TDNN.ECAPA_TDNN
    input_size: !ref <n_mels>  # 解析为 80
    channels: [1024, 1024, 1024, 1024, 3072]
    kernel_sizes: [5, 3, 3, 3, 1]
    lin_neurons: 192

# 优化器配置
opt_class: !name:torch.optim.Adam
    lr: !ref <lr>
    weight_decay: 0.000002
```

## 项目结构

```
SR-agent/
├── agent/
│   ├── utils/              # 核心工具模块
│   │   ├── config_parser.py      # YAML 配置解析
│   │   ├── experiment_tracker.py # 实验追踪
│   │   ├── logger.py             # 日志记录
│   │   └── metrics.py            # 指标计算
│   ├── tools/              # LangChain 工具
│   │   ├── config_tools.py       # 配置管理工具
│   │   ├── training_tools.py     # 训练控制工具
│   │   └── evaluation_tools.py   # 评估工具
│   ├── experiments/        # 实验历史记录
│   ├── prompts/            # 智能体提示词
│   └── hpo_agent.py        # 主智能体
├── configs/
│   └── train_ecapa_tdnn.yaml   # ECAPA-TDNN 配置文件
└── recipes/                 # SpeechBrain 训练脚本
```

## 关键超参数

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `lr` | 学习率 | 0.001 | 0.0001-0.01 |
| `batch_size` | 批次大小 | 24 | 8-64 |
| `embedding_model.channels` | 通道数 | [1024, 1024, 1024, 1024, 3072] | [512-2048] |
| `embedding_model.lin_neurons` | 线性层神经元数 | 192 | 128-512 |

## 测试验证

运行测试脚本验证 YAML 解析功能：

```bash
python agent/test_yaml_parser.py
```

运行完整测试：

```bash
python agent/test_all_modules.py
```

## 最佳实践

1. **逐步调整**：一次只调整 1-2 个超参数
2. **合理范围**：基于经验设置合理的参数范围
3. **监控训练**：定期检查训练日志和性能指标
4. **系统搜索**：使用网格搜索或随机搜索方法

## 注意事项

- 确保配置文件使用正确的 YAML 语法
- 训练前检查数据路径是否正确
