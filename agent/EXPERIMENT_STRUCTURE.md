# 实验存储结构说明

## 概述

本项目的实验管理系统采用集中化的存储结构，每个实验的所有相关文件都存储在一个独立的目录中，便于管理和查找。

## 目录结构

```
agent/experiments/
├── experiments_history.json          # 全局实验历史记录
├── 20260325_064906/                  # 实验目录（格式：YYYYMMDD_HHMMSS）
│   ├── config.yaml                   # 配置文件备份
│   ├── experiment_record.json        # 实验记录
│   ├── experiment.log                # 实验日志
│   ├── model.ckpt                    # 模型权重文件（可选）
│   ├── scores.txt                    # 评估结果（可选）
│   └── results/                      # 其他结果文件（可选）
│       ├── metrics.json
│       └── plots/
├── 20260325_123456/
│   ├── config.yaml
│   ├── experiment_record.json
│   └── experiment.log
└── ...
```

## 文件说明

### 1. experiments_history.json
全局实验历史记录文件，包含所有实验的简要信息：
- `experiment_id`: 实验唯一标识
- `timestamp`: 实验创建时间
- `description`: 实验描述
- `status`: 实验状态（created/running/success/failed）
- `config_file`: 配置文件路径
- `duration_seconds`: 训练时长
- `results`: 关键性能指标

### 2. {experiment_id}/config.yaml
实验使用的配置文件备份，完整保存了训练时的所有参数。

### 3. {experiment_id}/experiment_record.json
详细的实验记录，包含：
- 完整的配置信息（已解析后的字典形式）
- 训练时长
- 性能指标（EER, accuracy, loss, error_rate等）
- 错误信息（如果失败）
- 训练输出摘要

### 4. {experiment_id}/experiment.log
完整的训练日志，包括：
- 训练过程中的所有输出
- 错误和警告信息
- 性能指标变化

### 5. {experiment_id}/model.ckpt
训练好的模型权重文件（可选）。

### 6. {experiment_id}/scores.txt
模型评估结果文件（可选）。

### 7. {experiment_id}/results/
其他结果文件目录，可包含：
- `metrics.json`: 详细的性能指标
- `plots/`: 可视化图表
- 其他分析结果

## 实验 ID 命名规则

实验 ID 采用时间戳格式：`YYYYMMDD_HHMMSS`

例如：`20260325_064906` 表示 2026年3月25日 06:49:06 创建的实验。

## 使用示例

### 1. 查找实验目录
```python
from agent.utils import get_experiments_dir, get_experiment_log_path

exp_dir = get_experiments_dir() / "20260325_064906"
log_path = get_experiment_log_path("20260325_064906")
```

### 2. 读取实验记录
```python
import json
from pathlib import Path

exp_id = "20260325_064906"
record_path = Path("agent/experiments") / exp_id / "experiment_record.json"

with open(record_path, 'r') as f:
    record = json.load(f)
```

### 3. 列出所有实验
```python
from agent.utils import list_experiments

# 获取所有实验
all_experiments = list_experiments()

# 只获取成功的实验
successful_experiments = list_experiments(status="success")

# 获取最近的 5 个实验
recent_experiments = list_experiments(limit=5)
```

### 4. 查找最佳实验
```python
from agent.utils import find_best_experiment

# 找到 EER 最低的实验
best_eer = find_best_experiment(metric="eer", minimize=True)

# 找到准确率最高的实验
best_acc = find_best_experiment(metric="accuracy", minimize=False)
```

### 5. 比较多个实验
```python
from agent.utils import ExperimentTracker

tracker = ExperimentTracker()
comparison = tracker.compare_experiments([
    "20260325_064906",
    "20260325_123456",
    "20260325_180000"
])

print(comparison["metrics_comparison"])
print(comparison["best_by_metric"])
```

## 清理旧实验

### 1. 删除单个实验
```python
from agent.utils import ExperimentTracker

tracker = ExperimentTracker()
tracker.delete_experiment("20260325_064906")
```

### 2. 批量清理
```python
# 保留最近的 10 个成功的实验
deleted = tracker.cleanup_old_experiments(keep_n=10, status_filter="success")

# 保留所有实验中最新的 20 个
deleted = tracker.cleanup_old_experiments(keep_n=20)
```

## 优势

1. **结构清晰**：每个实验的所有文件集中在一个目录，便于查找和管理
2. **易于备份**：可以单独备份某个实验的整个目录
3. **便于分享**：复制实验目录即可分享完整实验结果
4. **版本控制友好**：实验目录结构简单，便于 .gitignore 配置
5. **扩展性强**：可以在实验目录下添加任意额外的结果文件

## 迁移说明

如果您有旧版本的实验记录（使用分离的 `configs/` 目录），系统会自动兼容：
- 旧实验可以继续正常读取
- 新实验将采用新的目录结构
- 建议逐步迁移旧实验到新结构

## 注意事项

1. **不要手动修改** `experiments_history.json`，应通过 `ExperimentTracker` 类操作
2. **实验 ID 冲突**：如果同一秒内创建多个实验，需要手动调整 ID
3. **磁盘空间**：定期清理失败的实验和旧实验以节省空间
4. **备份重要实验**：对于重要的实验结果，建议额外备份到安全位置