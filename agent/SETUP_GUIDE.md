# ECAPA-TDNN 超参数优化智能体系统 - 完整设置指南

本指南将帮助您在 conda 虚拟环境 `agent` 中设置和运行智能体系统。

## 📋 目录

1. [系统要求](#系统要求)
2. [环境设置](#环境设置)
3. [配置说明](#配置说明)
4. [测试系统](#测试系统)
5. [使用智能体](#使用智能体)
6. [故障排除](#故障排除)

## 🔧 系统要求

### 硬件要求
- **CPU**: 4核及以上
- **内存**: 8GB 及以上（推荐 16GB）
- **GPU**: NVIDIA GPU（推荐，用于训练）
- **存储**: 至少 20GB 可用空间（用于数据集和模型）

### 软件要求
- **操作系统**: Linux (Ubuntu 18.04+) 或 Windows 10/11
- **Python**: 3.8+
- **Conda**: Miniconda 或 Anaconda
- **CUDA**: 11.8 (如果使用 GPU)

## 🚀 环境设置

### 步骤 1: 创建 conda 虚拟环境

```bash
# 在项目根目录下执行
cd /home/lixh26/agent/SR-agent

# 创建 conda 环境（使用预配置的 environment.yml）
conda env create -f agent/environment.yml

# 或者手动创建环境
conda create -n agent python=3.9 -y
conda activate agent
```

### 步骤 2: 安装 PyTorch（如果手动创建环境）

```bash
# CUDA 11.8 版本
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 或 CPU 版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 步骤 3: 安装 LangChain 和相关依赖

```bash
# 激活环境
conda activate agent

# 安装核心依赖
pip install langchain>=0.1.0
pip install langchain-community>=0.0.10
pip install langchain-core>=0.1.0

# 安装 YAML 处理库
pip install pyyaml>=6.0
pip install ruamel.yaml>=0.17.0

# 安装环境变量管理
pip install python-dotenv>=1.0.0

# 安装 SpeechBrain 依赖
pip install hyperpyyaml>=0.0.1
pip install joblib>=1.3.0
pip install requests>=2.20.0
pip install sentencepiece>=0.1.91
pip install soundfile>=0.12.1
pip install huggingface_hub>=0.8.0
pip install tqdm>=4.42.0
pip install numpy>=1.17.0
pip install scipy>=1.4.1
```

### 步骤 4: 安装项目本身

```bash
# 在项目根目录
cd /home/lixh26/agent/SR-agent

# 安装 SpeechBrain（可编辑模式）
pip install -e .
```

### 步骤 5: 验证安装

```bash
# 检查 Python 版本
python --version  # 应该显示 Python 3.9.x

# 检查 PyTorch 安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查 LangChain 安装
python -c "import langchain; print(f'LangChain: {langchain.__version__}')"

# 检查 SpeechBrain 安装
python -c "import speechbrain; print(f'SpeechBrain: {speechbrain.__version__}')"
```

## ⚙️ 配置说明

### 环境变量配置

确保 `.env` 文件存在于项目根目录：

```bash
cd /home/lixh26/agent/SR-agent
ls -la .env  # 确认文件存在
```

`.env` 文件内容应包含：

```env
ZHIPUAI_API_KEY=your_api_key_here
ZHIU_API_BASE_URL=https://llmapi.paratera.com
```

**注意**: 
- 如果使用不同的 API 提供商，请相应修改 `ZHIU_API_BASE_URL`
- 确保 API 密钥有效且有足够的配额

### 路径配置

检查以下路径是否正确（在 `agent/hpo_agent.py` 中）：

```python
CONFIG_PATH = "../configs/train_ecapa_tdnn.yaml"
TRAIN_SCRIPT = "../recipes/voxceleb/train_speaker_embeddings.py"
EVAL_SCRIPT = "../recipes/voxceleb/speaker_verification_cosine.py"
```

这些路径是相对于 `agent/` 目录的相对路径。确保：

```bash
cd /home/lixh26/agent/SR-agent/agent

# 检查配置文件是否存在
ls -la ../configs/train_ecapa_tdnn.yaml

# 检查训练脚本是否存在
ls -la ../recipes/voxceleb/train_speaker_embeddings.py

# 检查评估脚本是否存在
ls -la ../recipes/voxceleb/speaker_verification_cosine.py
```

## 🧪 测试系统

### 运行测试脚本

```bash
# 激活 conda 环境
conda activate agent

# 进入 agent 目录
cd /home/lixh26/agent/SR-agent/agent

# 运行测试
python test_agent.py
```

测试脚本会验证以下功能：
1. ✅ 读取配置文件
2. ✅ 备份配置文件
3. ✅ 修改配置参数
4. ✅ 获取训练日志
5. ✅ 智能体初始化

**预期输出**: 所有测试应该通过，并显示 "🎉 所有测试通过！智能体系统已准备就绪。"

### 手动测试智能体

```bash
# 启动智能体
python hpo_agent.py

# 测试命令（在交互式界面中输入）：
# 1. 读取当前配置
# 2. 备份当前配置
# 3. 将学习率调整为 0.015（persist=False 测试）
# 4. 退出
```

## 🎮 使用智能体

### 基本使用流程

```bash
# 1. 激活环境
conda activate agent

# 2. 进入 agent 目录
cd /home/lixh26/agent/SR-agent/agent

# 3. 启动智能体
python hpo_agent.py

# 4. 输入自然语言指令
# 示例：
# - "读取当前配置"
# - "分析当前配置并给出优化建议"
# - "将学习率调整为 0.01"
# - "训练模型并评估性能"
```

### 高级使用示例

#### 示例 1: 渐进式优化

```
1. 备份当前配置
2. 读取当前配置
3. 将学习率调整为 0.015
4. 训练模型
5. 查看训练日志
6. 评估模型性能
```

#### 示例 2: 参数搜索

```
1. 备份配置
2. 将学习率设为 0.01，批次大小设为 64
3. 训练模型
4. 评估性能
5. 将学习率设为 0.02，批次大小设为 32
6. 训练模型
7. 评估性能
8. 比较两次结果
```

#### 示例 3: 智能优化

```
优化超参数以提升模型性能，最多尝试3次配置
```

## 🐛 故障排除

### 问题 1: conda 环境创建失败

**错误信息**: `CondaError: Download failed`

**解决方案**:
```bash
# 清理 conda 缓存
conda clean --all

# 使用镜像源（如果在中国）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --set show_channel_urls yes

# 重新创建环境
conda env create -f agent/environment.yml
```

### 问题 2: PyTorch 安装失败

**错误信息**: `CUDA out of memory` 或 `torch.cuda.is_available() returns False`

**解决方案**:
```bash
# 检查 CUDA 版本
nvidia-smi

# 安装匹配的 PyTorch 版本
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 或使用 CPU 版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 问题 3: LangChain API 调用失败

**错误信息**: `AuthenticationError` 或 `API connection error`

**解决方案**:
```bash
# 1. 检查 .env 文件是否存在
cat .env

# 2. 验证 API 密钥格式
# 确保 ZHIPUAI_API_KEY 以 "sk-" 开头

# 3. 测试网络连接
curl -I https://llmapi.paratera.com

# 4. 检查 API 配额
# 登录 API 提供商控制台查看配额使用情况
```

### 问题 4: 配置文件读取失败

**错误信息**: `FileNotFoundError` 或 `YAML parse error`

**解决方案**:
```bash
# 1. 检查文件路径
cd /home/lixh26/agent/SR-agent/agent
ls -la ../configs/train_ecapa_tdnn.yaml

# 2. 检查文件权限
chmod 644 ../configs/train_ecapa_tdnn.yaml

# 3. 验证 YAML 语法
python -c "import yaml; yaml.safe_load(open('../configs/train_ecapa_tdnn.yaml'))"
```

### 问题 5: 训练脚本执行失败

**错误信息**: `ModuleNotFoundError` 或 `ImportError`

**解决方案**:
```bash
# 1. 确保 SpeechBrain 已安装
pip install -e .

# 2. 检查 Python 路径
which python
python --version

# 3. 手动测试训练脚本
cd /home/lixh26/agent/SR-agent
python recipes/voxceleb/train_speaker_embeddings.py configs/train_ecapa_tdnn.yaml --help
```

### 问题 6: 智能体响应异常

**错误信息**: 智能体无法理解指令或工具调用失败

**解决方案**:
```bash
# 1. 增加日志详细程度
# 修改 hpo_agent.py 中的 verbose=True

# 2. 测试单个工具
python -c "from hpo_agent import read_config; print(read_config.invoke({}))"

# 3. 检查 LLM 响应
# 在 .env 中添加调试日志
```

## 📊 性能优化建议

### 1. GPU 加速

确保使用 GPU 进行训练：
```python
# 在训练前检查
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

### 2. 批次大小调整

根据 GPU 内存调整批次大小：
- 8GB GPU: batch_size=16-32
- 16GB GPU: batch_size=32-64
- 24GB+ GPU: batch_size=64-128

### 3. 数据加载优化

增加 `num_workers` 以加速数据加载：
```yaml
# 在配置文件中
num_workers: 8  # 根据 CPU 核心数调整
```

## 📚 参考资源

- [LangChain 官方文档](https://python.langchain.com/)
- [SpeechBrain 官方文档](https://speechbrain.readthedocs.io/)
- [ECAPA-TDNN 论文](https://arxiv.org/abs/2005.07143)
- [PyTorch 官方文档](https://pytorch.org/docs/)

## 🆘 获取帮助

如果遇到问题：

1. 查看 [README.md](README.md) 了解基本使用方法
2. 运行 `test_agent.py` 诊断问题
3. 检查错误日志和输出信息
4. 参考故障排除部分
5. 提交 Issue 描述问题和环境信息

## ✅ 检查清单

在开始使用前，请确认：

- [ ] Conda 环境 `agent` 已创建并激活
- [ ] 所有依赖已正确安装
- [ ] `.env` 文件存在且配置正确
- [ ] 配置文件路径正确
- [ ] 测试脚本运行通过
- [ ] 可以成功启动智能体
- [ ] 智能体能够响应基本指令

---

**祝您使用愉快！🎉**