# 实验管理文件结构说明

本文档详细说明了SR-agent中实验管理的文件组织结构，包括训练和评估的流程。

## 目录结构

```
agent/
├── experiments/                          # 实验根目录
│   ├── experiments_history.json           # 实验历史记录（所有实验的索引）
│   ├── configs/                         # 配置文件备份目录
│   │   ├── config_20260326_090000.yaml  # 配置备份（按时间戳命名）
│   │   └── ...
│   └── exp_{YYYYMMDD_HHMMSS}/          # 单个实验目录（实验ID为时间戳）
│       ├── experiment_record.json         # 该实验的详细记录
│       ├── train_config.yaml             # 训练配置文件（修改后的）
│       ├── config.yaml                   # 原始配置文件备份
│       ├── results/                     # 训练输出目录
│       │   ├── save/                    # 模型checkpoint目录
│       │   │   ├── CKPT+2026-03-25+02-36-45+00.ckpt
│       │   │   └── ...
│       │   ├── train_log.txt             # 训练日志
│       │   ├── CSV files/               # 数据集CSV文件
│       │   └── ...
│       └── evaluation/                  # 评估结果目录
│           ├── verification_config.yaml   # 评估配置文件
│           ├── evaluation_log.txt        # 评估日志
│           └── ...
```

## 核心特性

### 1. 独立的实验隔离
- 每次训练都使用独立的输出目录：`experiments/exp_{experiment_id}/results/`
- 不同实验之间完全隔离，不会相互干扰
- 所有实验结果都保存在对应的实验目录下

### 2. 完整的实验记录
- `experiments_history.json`: 所有实验的集中索引
- `exp_{experiment_id}/experiment_record.json`: 单个实验的详细记录
- 记录包括：配置、训练指标、评估结果、时间戳等

### 3. Checkpoint管理
- 训练完成后自动查找最新的checkpoint
- Checkpoint路径保存在实验记录中
- 评估时自动从实验记录读取checkpoint路径

## 使用流程

### 训练流程

```python
# 1. 运行训练
result = run_training(config_path="../configs/train_ecapa_tdnn.yaml")

# 系统会自动：
# - 生成实验ID（如：20260326_090000）
# - 创建实验目录：experiments/exp_20260326_090000/
# - 通过命令行参数指定输出目录（--output_folder），不修改配置文件
# - 备份原始配置文件到实验目录
# - 运行训练脚本
# - 训练完成后：
#   - 解析训练日志
#   - 查找最新checkpoint
#   - 保存实验记录到experiments_history.json
#   - 保存实验记录到exp_{experiment_id}/experiment_record.json
```

### 评估流程

```python
# 1. 评估最新实验
result = run_evaluation()

# 2. 评估指定实验
result = run_evaluation(experiment_id="20260326_090000")

# 3. 使用指定checkpoint评估
result = run_evaluation(checkpoint_path="/path/to/checkpoint.ckpt")

# 系统会自动：
# - 从实验记录读取checkpoint路径
# - 修改评估配置，使用指定的checkpoint
# - 保存修改后的配置到实验目录
# - 运行评估脚本
# - 评估完成后：
#   - 解析评估日志提取EER和minDCF
#   - 复制评估日志到实验目录
#   - 更新实验记录，添加评估结果
```

## 实验记录结构

### experiments_history.json
```json
[
  {
    "experiment_id": "20260326_090000",
    "timestamp": "2026-03-26T09:00:00",
    "duration_seconds": 1234.56,
    "status": "success",
    "config": {
      "lr": 0.0001,
      "batch_size": 32,
      "number_of_epochs": 10,
      "step_size": 5,
      "seed": "1234"
    },
    "training_log_path": "experiments/exp_20260326_090000/results/train_log.txt",
    "final_metrics": {
      "final_epoch": 10,
      "final_lr": 7.96e-05,
      "final_train_loss": 0.274,
      "final_valid_loss": 0.247,
      "final_valid_error_rate": 0.00489,
      "total_epochs": 10,
      "best_epoch": 8,
      "best_valid_loss": 0.234,
      "best_error_rate": 0.00456
    },
    "evaluation_results": {
      "timestamp": "2026-03-26T10:30:00",
      "duration_seconds": 300.0,
      "eer": 2.4886,
      "min_dcf": 0.2280,
      "evaluation_log_path": "experiments/exp_20260326_090000/evaluation/evaluation_log.txt",
      "checkpoint_used": "experiments/exp_20260326_090000/results/save/CKPT+2026-03-25+02-36-45+00.ckpt",
      "evaluation_dir": "experiments/exp_20260326_090000/evaluation"
    },
    "checkpoint_info": {
      "checkpoint_path": "experiments/exp_20260326_090000/results/save/CKPT+2026-03-25+02-36-45+00.ckpt",
      "checkpoint_filename": "CKPT+2026-03-25+02-36-45+00.ckpt",
      "checkpoint_backup_path": "experiments/exp_20260326_090000/results/save/CKPT+2026-03-25+02-36-45+00.ckpt"
    },
    "experiment_dir": "experiments/exp_20260326_090000",
    "output_folder": "experiments/exp_20260326_090000/results",
    "config_backup_path": "experiments/exp_20260326_090000/config.yaml"
  }
]
```

## 工具函数

### run_training
运行训练并自动管理实验记录

**参数：**
- `config_path`: 配置文件路径（默认：`../configs/train_ecapa_tdnn.yaml`）
- `experiment_id`: 实验ID（可选，默认自动生成）

**返回：**
- 训练结果摘要，包括：
  - 实验ID
  - 训练时长
  - 实验目录
  - 性能指标
  - Checkpoint信息

### run_evaluation
运行评估并更新实验记录

**参数：**
- `eval_config_path`: 评估配置文件路径（默认：`../configs/verification_ecapa.yaml`）
- `experiment_id`: 实验ID（可选，默认使用最新的成功实验）
- `checkpoint_path`: Checkpoint路径（可选，默认从实验记录读取）

**返回：**
- 评估结果摘要，包括：
  - 实验ID
  - 评估时长
  - 使用的Checkpoint
  - 性能指标（EER, minDCF）

### view_experiment_history
查看实验历史记录

**参数：**
- `n`: 显示最近n次实验（默认：10）

**返回：**
- 实验历史摘要

### get_experiment_details
获取特定实验的详细信息

**参数：**
- `experiment_id`: 实验ID

**返回：**
- 实验详细信息

### compare_experiments
比较多个实验的性能

**参数：**
- `experiment_ids`: 实验ID列表（逗号分隔）

**返回：**
- 实验比较结果

### get_best_experiment
找出最佳实验

**参数：**
- `metric`: 优化指标（默认：`best_error_rate`）
  - `best_error_rate`: 最佳验证错误率（越小越好）
  - `final_valid_error_rate`: 最终验证错误率（越小越好）
  - `eer`: 等错误率（越小越好）
  - `accuracy`: 准确率（越大越好）

**返回：**
- 最佳实验的详细信息

### analyze_training_trends
分析训练趋势

**参数：**
- `experiment_id`: 实验ID（可选，默认分析最新实验）

**返回：**
- 训练趋势分析报告

## 配置文件修改

### 训练配置修改
系统会自动修改以下配置项：
- `output_folder`: 设置为 `experiments/exp_{experiment_id}/results`
- `save_folder`: 设置为 `experiments/exp_{experiment_id}/results/save`

修改后的配置保存在 `experiments/exp_{experiment_id}/train_config.yaml`

### 评估配置修改
系统会自动修改评估配置中的pretrainer路径，使用训练时保存的checkpoint。

修改后的配置保存在 `experiments/exp_{experiment_id}/evaluation/verification_config.yaml`

## 注意事项

1. **目录隔离**: 每次训练使用独立的输出目录，不会相互干扰
2. **配置备份**: 原始配置文件会备份到实验目录，不会修改原始配置
3. **Checkpoint追踪**: Checkpoint路径保存在实验记录中，评估时自动读取
4. **日志解析**: 自动解析训练和评估日志，提取关键指标
5. **实验记录**: 所有实验记录保存在 `experiments_history.json` 和各自的 `experiment_record.json`

## 示例

### 示例1：运行训练
```python
from agent.hpo_agent import run_training

# 运行训练
result = run_training()
print(result)

# 输出示例：
# ✅ 训练完成！
# 实验ID: 20260326_090000
# 训练时长: 1234.56 秒
# 实验目录: /home/lixh26/agent/SR-agent/agent/experiments/exp_20260326_090000
# 训练输出目录: experiments/exp_20260326_090000/results
#
# 性能指标:
#   - 最终Epoch: 10
#   - 最终学习率: 7.96e-05
#   - 最终训练损失: 0.2740
#   - 最终验证损失: 0.2470
#   - 最终验证错误率: 0.0049
#   - 最佳Epoch: 8
#   - 最佳验证错误率: 0.0046
#
# 模型Checkpoints:
#   - 最新checkpoint: CKPT+2026-03-25+02-36-45+00.ckpt
#   - 完整路径: /home/lixh26/agent/SR-agent/agent/experiments/exp_20260326_090000/results/save/CKPT+2026-03-25+02-36-45+00.ckpt
#
# 配置和结果已保存到 experiments/exp_20260326_090000/
```

### 示例2：运行评估
```python
from agent.hpo_agent import run_evaluation

# 评估最新实验
result = run_evaluation()
print(result)

# 输出示例：
# ✅ 评估完成！
# 实验ID: 20260326_090000
# 评估时长: 300.00 秒
# 使用的Checkpoint: /home/lixh26/agent/SR-agent/agent/experiments/exp_20260326_090000/results/save/CKPT+2026-03-25+02-36-45+00.ckpt
# 评估结果目录: /home/lixh26/agent/SR-agent/agent/experiments/exp_20260326_090000/evaluation
#
# 性能指标:
#   - EER (等错误率): 2.4886%
#   - minDCF: 0.2280
#
# 评估结果已保存到实验记录中
# 评估日志: /home/lixh26/agent/SR-agent/agent/experiments/exp_20260326_090000/evaluation/evaluation_log.txt
```

### 示例3：查看实验历史
```python
from agent.hpo_agent import view_experiment_history

# 查看最近10次实验
result = view_experiment_history(n=10)
print(result)
```

### 示例4：比较实验
```python
from agent.hpo_agent import compare_experiments

# 比较多个实验
result = compare_experiments("20260326_090000,20260326_100000,20260326_110000")
print(result)
```

## 总结

这个实验管理系统提供了：

1. **完整的实验隔离**: 每次训练独立的输出目录
2. **自动化记录**: 自动保存配置、日志、checkpoint路径
3. **便捷的评估**: 自动从实验记录读取checkpoint
4. **灵活的查询**: 支持查看历史、比较实验、查找最佳实验
5. **趋势分析**: 自动分析训练趋势，识别过拟合等问题

通过这个系统，您可以轻松管理大量的超参数优化实验，快速找到最佳配置。