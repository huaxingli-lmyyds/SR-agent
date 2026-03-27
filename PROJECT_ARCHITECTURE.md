# SR-agent 项目架构设计文档

## 1. 项目概述

SR-agent 是一个基于 LangChain 的智能体系统，用于自动优化 ECAPA-TDNN 声纹识别模型的超参数。

### 1.1 项目目标
- 提供自然语言交互的超参数优化界面
- 自动化模型训练和评估流程
- 记录和管理实验历史
- 智能推荐最优超参数配置

### 1.2 技术栈
- **语言**: Python 3.9+
- **深度学习框架**: PyTorch 2.1+
- **语音处理**: SpeechBrain
- **智能体框架**: LangChain
- **大语言模型**: ChatGLM (智谱 AI)
- **配置管理**: YAML, ruamel.yaml, hyperpyyaml

---

## 2. 目录结构设计

```
SR-agent/
│
├── agent/                          # 智能体系统核心目录
│   ├── __init__.py                 # 包初始化文件
│   ├── hpo_agent.py                # 超参数优化智能体主程序
│   ├── main.py                     # 程序入口
│   ├── config.py                   # 配置管理
│   │
│   ├── utils/                      # 工具模块
│   │   ├── __init__.py
│   │   ├── path_tool.py            # ✅ 路径管理工具（已实现）
│   │   ├── config_parser.py        # YAML 配置解析
│   │   ├── logger.py               # 日志管理
│   │   ├── experiment_tracker.py   # 实验跟踪
│   │   └── metrics.py              # 性能指标计算
│   │
│   ├── tools/                      # LangChain 工具集合
│   │   ├── __init__.py
│   │   ├── config_tools.py         # 配置相关工具
│   │   ├── training_tools.py       # 训练相关工具
│   │   ├── evaluation_tools.py     # 评估相关工具
│   │   └── analysis_tools.py       # 分析相关工具
│   │
│   ├── agents/                     # 智能体实现
│   │   ├── __init__.py
│   │   ├── base_agent.py           # 基础智能体类
│   │   ├── hpo_agent.py            # 超参数优化智能体
│   │   └── expert_agent.py         # 专家智能体
│   │
│   ├── prompts/                    # 提示词管理
│   │   ├── __init__.py
│   │   ├── hpo_prompt.txt          # ✅ 超参数优化提示词（已实现）
│   │   ├── system_prompt.txt       # 系统提示词
│   │   └── templates/              # 提示词模板
│   │       ├── analysis.txt
│   │       └── optimization.txt
│   │
│   ├── experiments/                # 实验管理
│   │   ├── experiments_history.json # ✅ 实验历史记录（已实现）
│   │   ├── configs/                # 实验配置备份
│   │   └── results/                # 实验结果存储
│   │
│   ├── data/                       # 数据管理
│   │   ├── __init__.py
│   │   ├── data_loader.py          # 数据加载器
│   │   └── preprocess.py           # 数据预处理
│   │
│   ├── tests/                      # 测试模块
│   │   ├── __init__.py
│   │   ├── test_path_tool.py       # ✅ 路径工具测试
│   │   ├── test_config_parser.py   # 配置解析测试
│   │   └── test_agent.py           # 智能体测试
│   │
│   ├── scripts/                    # 脚本工具
│   │   ├── run_experiment.py       # 运行实验
│   │   ├── analyze_results.py      # 分析结果
│   │   └── cleanup.py              # 清理工具
│   │
│   ├── environment.yml             # Conda 环境配置
│   ├── requirements.txt             # Python 依赖
│   └── README.md                   # Agent 模块文档
│
├── configs/                        # 模型配置文件
│   ├── train_ecapa_tdnn.yaml       # ✅ ECAPA-TDNN 训练配置
│   ├── train_xvector.yaml          # X-Vector 训练配置
│   └── verification_ecapa.yaml      # 验证配置
│
├── models/                         # 模型定义
│   ├── __init__.py
│   ├── ECAPA_TDNN.py               # ✅ ECAPA-TDNN 模型
│   ├── XVector.py                  # X-Vector 模型
│   └── ResNet.py                   # ResNet 模型
│
├── recipes/                        # 训练和评估脚本
│   ├── voxceleb/                   # VoxCeleb 数据集相关
│   │   ├── train_speaker_embeddings.py  # ✅ 训练脚本
│   │   ├── speaker_verification_cosine.py # ✅ 评估脚本
│   │   └── voxceleb_prepare.py     # ✅ 数据准备脚本
│   │
│   ├── librispeech/                # LibriSpeech 数据集相关
│   └── ...                         # 其他数据集
│
├── speechbrain/                    # SpeechBrain 框架（核心库）
│   ├── core.py                     # ✅ 核心
│   ├── lobes/                      # 模型组件
│   ├── processing/                 # 数据处理
│   ├── augment/                    # 数据增强
│   ├── utils/                      # 工具函数
│   └── ...
│
├── datasets/                       # 数据集存储
│   ├── voxceleb1/                  # VoxCeleb1 数据集
│   ├── voxceleb2/                  # VoxCeleb2 数据集
│   └── librispeech/                # LibriSpeech 数据集
│
├── docs/                           # 文档
│   ├── architecture.md             # 架构文档（本文件）
│   ├── api.md                      # API 文档
│   ├── user_guide.md               # 用户指南
│   └── development.md              # 开发指南
│
├── .env                            # 环境变量配置
├── .gitignore                      # Git 忽略文件
├── pyproject.toml                  # 项目配置
└── README.md                       # 项目主文档
```

---

## 3. 模块设计

### 3.1 核心模块 (agent/)

#### 3.1.1 路径管理 (utils/path_tool.py) ✅
**已实现功能**:
- 项目基础路径获取
- 路径验证和创建
- 文件和目录操作
- 路径信息获取
- 备份管理
- 实用工具函数

**使用示例**:
```python
from agent.utils.path_tool import (
    get_project_root,
    get_config_file,
    ensure_dir,
    backup_file,
    list_files
)

# 获取项目根目录
root = get_project_root()

# 获取配置文件
config_path = get_config_file("train_ecapa_tdnn.yaml")

# 确保目录存在
exp_dir = ensure_dir("experiments/20240325")

# 备份文件
backup_path = backup_file(config_path)

# 列出所有 YAML 文件
yaml_files = list_files(configs_dir, pattern="*.yaml")
```

#### 3.1.2 配置解析 (utils/config_parser.py)
**待实现功能**:
- YAML 配置文件加载和解析
- 支持超参数引用 (!ref)
- 配置验证
- 配置比较和差异分析
- 配置序列化和反序列化

```python
class ConfigParser:
    def load_config(self, config_path: Path) -> dict
    def validate_config(self, config: dict) -> bool
    def compare_configs(self, config1: dict, config2: dict) -> dict
    def update_config(self, config: dict, updates: dict) -> dict
```

#### 3.1.3 实验跟踪 (utils/experiment_tracker.py)
**待实现功能**:
- 实验记录管理
- 实验状态跟踪
- 结果比较
- 最佳实验查找
- 实验统计分析

```python
class ExperimentTracker:
    def create_experiment(self, config: dict) -> str
    def update_experiment(self, exp_id: str, results: dict)
    def get_experiment(self, exp_id: str) -> dict
    def list_experiments(self, filters: dict) -> List[dict]
    def find_best_experiment(self, metric: str) -> dict
```

#### 3.1.4 日志管理 (utils/logger.py)
**待实现功能**:
- 结构化日志记录
- 日志轮转
- 不同级别日志输出
- 日志查询和分析

```python
class Logger:
    def __init__(self, log_path: Path, level: str = "INFO")
    def info(self, message: str)
    def error(self, message: str)
    def warning(self, message: str)
    def get_logs(self, start_time: datetime, end_time: datetime)
```

### 3.2 工具模块 (agent/tools/)

#### 3.2.1 配置工具 (config_tools.py)
```python
@tool
def read_config() -> str:
    """读取当前配置"""

@tool
def modify_config(config_json: str) -> str:
    """修改配置"""

@tool
def compare_configs(config1: str, config2: str) -> str:
    """比较两个配置的差异"""

@tool
def validate_config(config_path: str) -> str:
    """验证配置文件"""
```

#### 3.2.2 训练工具 (training_tools.py)
```python
@tool
def run_training(config: str) -> str:
    """运行训练"""

@tool
def stop_training() -> str:
    """停止训练"""

@tool
def get_training_status() -> str:
    """获取训练状态"""
```

#### 3.2.3 评估工具 (evaluation_tools.py)
```python
@tool
def run_evaluation(model_path: str, test_set: str) -> str:
    """运行评估"""

@tool
def compute_metrics(predictions: List, labels: List) -> str:
    """计算性能指标"""

@tool
def compare_models(model_paths: List[str]) -> str:
    """比较多个模型"""
```

### 3.3 智能体模块 (agent/agents/)

#### 3.3.1 基础智能体 (base_agent.py)
```python
class BaseAgent:
    def __init__(self, llm, tools, system_prompt: str)
    def invoke(self, user_input: str) -> dict
    def add_tool(self, tool)
    def remove_tool(self, tool_name: str)
```

#### 3.3.2 超参数优化智能体 (hpo_agent.py)
```python
class HPOAgent(BaseAgent):
    def analyze_config(self, config: dict) -> dict
    def suggest_improvements(self, config: dict, history: List[dict]) -> List[dict]
    def optimize(self, objective: str, max_iterations: int) -> dict
```

---

## 4. 数据流设计

### 4.1 训练流程

```
用户输入
    ↓
智能体解析意图
    ↓
选择工具 (read_config / modify_config)
    ↓
修改配置文件
    ↓
备份配置
    ↓
运行训练脚本
    ↓
监控训练过程
    ↓
记录实验结果
    ↓
评估模型性能
    ↓
分析结果
    ↓
返回建议
```

### 4.2 配置管理流程

```
加载原始配置
    ↓
解析 YAML (ruamel.yaml)
    ↓
转换为标准字典
    ↓
用户修改
    ↓
验证配置
    ↓
保存到文件
    ↓
创建备份
```

---

## 5. API 设计

### 5.1 路径管理 API

```python
# 项目基础路径
get_project_root() -> Path
get_agent_dir() -> Path
get_configs_dir() -> Path
get_datasets_dir() -> Path
get_recipes_dir() -> Path

# 特定文件路径
get_config_file(config_name: str) -> Path
get_train_script(script_name: str) -> Path
get_eval_script(script_name: str) -> Path
get_system_prompt(prompt_name: str) -> Path

# 路径验证和创建
ensure_dir(directory: Union[str, Path]) -> Path
path_exists(path: Union[str, Path]) -> bool
file_exists(file_path: Union[str, Path]) -> bool

# 文件操作
list_files(directory, pattern=None, recursive=False) -> List[Path]
backup_file(file_path, backup_dir=None, suffix=None) -> Path

# 路径信息
get_path_info(path) -> dict
format_size(size_bytes) -> str
```

### 5.2 智能体 API

```python
# 初始化智能体
agent = HPOAgent(
    llm=ChatOpenAI(model="GLM-4.7"),
    tools=[read_config, modify_config, run_training],
    system_prompt="..."
)

# 执行任务
result = agent.invoke("将学习率调整为 0.001 并训练模型")

# 获取实验历史
history = agent.get_experiment_history()

# 查找最佳实验
best = agent.find_best_experiment(metric="eer")
```

---

## 6. 配置管理策略

### 6.1 配置文件层次

1. **基础配置**: `configs/train_ecapa_tdnn.yaml`
2. **实验配置**: `agent/experiments/configs/config_*.yaml`
3. **用户配置**: `agent/config/user_config.yaml`
4. **系统配置**: `.env`

### 6.2 配置优先级

用户配置 > 实验配置 > 基础配置 > 系统配置

### 6.3 配置版本控制

- 每次实验自动备份配置
- 保留最近 N 个备份
- 使用时间戳标识版本

---

## 7. 实验管理策略

### 7.1 实验生命周期

```
创建实验 → 配置修改 → 训练 → 评估 → 结果记录 → 分析
```

### 7.2 实验记录结构

```json
{
  "experiment_id": "20240325_143022",
  "timestamp": "2024-03-25T14:30:22",
  "config": {
    "lr": 0.001,
    "batch_size": 32,
    ...
  },
  "results": {
    "accuracy": 0.95,
    "eer": 0.05,
    ...
  },
  "status": "success",
  "duration": 3600,
  "config_backup": "experiments/configs/config_20240325_143022.yaml"
}
```

### 7.3 实验查询

- 按时间范围查询
- 按配置参数查询
- 按性能指标查询
- 按状态查询

---

## 8. 扩展性设计

### 8.1 添加新工具

1. 在 `agent/tools/` 下创建工具文件
2. 使用 `@tool` 装饰器定义工具
3. 在智能体初始化时注册工具

### 8.2 添加新智能体

1. 继承 `BaseAgent` 类
2. 实现特定逻辑
3. 注册自定义工具

### 8.3 添加新模型

1. 在 `models/` 下定义模型
2. 在 `configs/` 下创建配置
3. 在 `recipes/` 下创建训练脚本

---

## 9. 测试策略

### 9.1 单元测试

- 路径工具测试
- 配置解析测试
- 工具函数测试

### 9.2 集成测试

- 智能体流程测试
- 训练流程测试
- 实验管理测试

### 9.3 端到端测试

- 完整优化流程测试
- 多智能体协作测试

---

## 10. 部署建议

### 10.1 开发环境

```bash
# 创建虚拟环境
conda env create -f agent/environment.yml

# 激活环境
conda activate agent
```

### 10.2 生产环境

```bash
# 安装依赖
pip install -r agent/requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件

# 运行智能体
python agent/main.py
```

### 10.3 Docker 部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install -r agent/requirements.txt

CMD ["python", "agent/main.py"]
```

---

## 11. 最佳实践

### 11.1 路径管理

- 始终使用 `path_tool.py` 中的函数获取路径
- 避免硬编码路径
- 使用 `ensure_dir()` 确保目录存在
- 重要操作前使用 `backup_file()` 备份

### 11.2 配置管理

- 使用结构化配置
- 保持配置可读性
- 定期备份配置
- 记录配置变更

### 11.3 实验管理

- 每个实验有明确的 ID
- 记录完整的实验信息
- 保留实验配置和结果
- 定期清理旧实验

---

## 12. 未来规划

### 12.1 短期目标 (1-2 个月)

- [ ] 完善工具模块实现
- [ ] 实现配置解析器
- [ ] 实现实验跟踪器
- [ ] 添加更多测试

### 12.2 中期目标 (3-6 个月)

- [ ] 支持多模型优化
- [ ] 实现分布式训练
- [ ] 添加可视化界面
- [ ] 优化智能体推理

### 12.3 长期目标 (6-12 个月)

- [ ] 支持自动机器学习
- [ ] 集成更多评估指标
- [ ] 实现模型压缩和优化
- [ ] 构建模型版本控制系统

---

## 13. 总结

本架构设计基于完善的路径管理工具，提供了：

1. **清晰的目录结构**: 模块化设计，职责分明
2. **统一的路径管理**: 使用 `path_tool.py` 统一管理所有路径
3. **可扩展的架构**: 易于添加新功能和模块
4. **完整的实验管理**: 支持实验跟踪、记录和分析
5. **灵活的配置管理**: 支持多层配置和版本控制
6. **强大的智能体系统**: 基于 LangChain 的自然语言交互

通过遵循此架构，可以构建一个高效、可维护、可扩展的超参数优化智能体系统。