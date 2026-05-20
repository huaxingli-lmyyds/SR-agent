# 实验记录结构设计

本文档说明当前实验记录的统一基类和三个智能体的派生结构，便于后续把超参数智能体、数据处理智能体和统筹智能体接入同一套实验记录体系。

## 0. 目录映射

当前三类实验记录分别落到不同目录：

- 超参数智能体: `agent/experiments/hpo/`
- 数据处理智能体: `agent/experiments/dp/`
- 统筹智能体: `agent/experiments/manage/`

每个目录下都有自己的 `experiments_history.json` 和按 `experiment_id` 分隔的子目录。

## 1. 统一基类

所有实验记录都包含以下公共字段：

- `experiment_type`：实验类型，取值为 `hpo`、`data_processing`、`orchestration`
- `experiment_id`：实验 ID
- `timestamp`：创建时间
- `description`：实验描述
- `status`：实验状态，默认 `created`
- `duration_seconds`：实验耗时
- `config_path`：配置文件路径
- `results`：通用结果区
- `error`：错误信息

这个基类的作用是保证三类智能体的实验记录都能被统一检索、比较和导出。

## 2. 超参数智能体记录

对应类型：`hpo`

### 顶层结构

- 公共字段：来自基类
- `training`：训练阶段记录
- `evaluation`：评估阶段记录

### `training` 建议字段

- `config_backup_path`：配置备份路径
- `data_folder`：数据目录
- `output_folder`：输出目录
- `train_log_path`：训练日志路径
- `metrics`：训练指标
- `model_paths`：模型文件路径列表

### `evaluation` 建议字段

- `timestamp`：评估时间
- `duration_seconds`：评估耗时
- `status`：评估状态
- `log_path`：评估日志路径
- `evaluation_log_path`：评估过程日志路径
- `output_folder`：评估输出目录
- `model_path`：参与评估的模型路径
- `results`：评估指标

### 示例

```json
{
  "experiment_type": "hpo",
  "experiment_id": "20260520_120000_0",
  "status": "success",
  "training": {
    "config_backup_path": "/path/to/config.yaml",
    "data_folder": "../datasets/voxceleb1",
    "output_folder": "/path/to/output",
    "train_log_path": "/path/to/train_log.txt",
    "metrics": {
      "eer": 3.42
    },
    "model_paths": ["/path/to/model.ckpt"]
  },
  "evaluation": {
    "status": "success",
    "results": {
      "eer": 3.12,
      "min_dcf": 0.21
    }
  },
  "results": {
    "eer": 3.12,
    "min_dcf": 0.21
  }
}
```

## 3. 数据处理智能体记录

对应类型：`data_processing`

### 顶层结构

- 公共字段：来自基类
- `data_processing`：数据准备和统计信息

### `data_processing` 建议字段

- `config_backup_path`：配置备份路径
- `data_folder`：原始数据目录
- `output_folder`：CSV 或数据产物输出目录
- `save_folder`：保存目录
- `verification_file`：验证列表
- `split_ratio`：训练/验证拆分比例
- `sentence_len`：片段长度
- `skip_prep`：是否跳过准备
- `stats`：CSV 行数或数据统计
- `summary`：执行摘要

### 示例

```json
{
  "experiment_type": "data_processing",
  "experiment_id": "20260520_120000_1",
  "status": "success",
  "data_processing": {
    "config_backup_path": "/path/to/config.yaml",
    "data_folder": "../datasets/voxceleb1",
    "output_folder": "/path/to/csv",
    "save_folder": "/path/to/csv",
    "verification_file": "/path/to/veri.txt",
    "split_ratio": [90, 10],
    "sentence_len": 3.0,
    "skip_prep": false,
    "stats": {
      "train": 1234,
      "dev": 123,
      "test": 456,
      "enrol": 78
    },
    "summary": "..."
  },
  "results": {
    "data_processing_summary": {
      "timestamp": "2026-05-20T12:00:00",
      "objective": "提升数据质量并保持训练/验证分布稳定",
      "best_config": {},
      "summary": "..."
    }
  }
}
```

## 4. 统筹智能体记录

对应类型：`orchestration`

### 顶层结构

- 公共字段：来自基类
- `orchestration`：统筹过程本身的状态
- `linked_experiments`：关联实验
- `a2a_messages`：智能体间消息
- `data_processing_summary_history`：数据处理摘要历史
- `hpo_feedback_history`：超参数反馈历史

### `orchestration` 建议字段

- `config_backup_path`：配置备份路径
- `data_folder`：数据目录
- `output_folder`：输出目录
- `manager_decision_history`：管理智能体决策轨迹
- `rounds`：统筹轮数
- `final_state`：最终状态快照

### `linked_experiments` 建议字段

- `manage`：统筹实验 ID
- `data_processing`：数据处理实验 ID
- `hpo`：超参数实验 ID

### 示例

```json
{
  "experiment_type": "orchestration",
  "experiment_id": "20260520_120000_2",
  "status": "success",
  "orchestration": {
    "config_backup_path": "/path/to/config.yaml",
    "data_folder": "../datasets/voxceleb1",
    "output_folder": "/path/to/output",
    "manager_decision_history": ["data_processing", "hpo"],
    "rounds": 2,
    "final_state": {}
  },
  "linked_experiments": {
    "manage": "20260520_120000_2",
    "data_processing": "20260520_120000_1",
    "hpo": "20260520_120000_0"
  },
  "a2a_messages": [],
  "data_processing_summary_history": [],
  "hpo_feedback_history": []
}
```

## 5. 当前实现建议

- 如果是超参数智能体，优先使用 `create_hpo_experiment` / `update_hpo_experiment`
- 如果是数据处理智能体，优先使用 `create_data_processing_experiment` / `update_data_processing_experiment`
- 如果是统筹智能体，优先使用 `create_orchestration_experiment` / `update_orchestration_experiment`
- 现有 `create_experiment` / `update_experiment` 保持兼容，默认按 `hpo` 结构工作

## 6. 后续接入顺序

1. 先把数据处理智能体的记录写入 `data_processing`
2. 再把统筹智能体的记录写入 `orchestration`、`linked_experiments` 和 `a2a_messages`
3. 最后统一让查询工具按 `experiment_type` 展示对应摘要
