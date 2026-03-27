# 实验管理系统使用指南

## 概述

实验管理系统为 ECAPA-TDNN 声纹识别模型的超参数优化提供了完整的实验跟踪、比较和管理功能。

## 快速开始

### 1. 生成演示实验

```bash
# 生成 5 个演示实验
python agent/generate_demo_experiments.py generate --num 5

# 生成 10 个演示实验（默认）
python agent/generate_demo_experiments.py generate
```

### 2. 查看现有实验

```bash
python agent/generate_demo_experiments.py view
```

### 3. 清理所有实验

```bash
python agent/generate_demo_experiments.py cleanup
```

## 实验目录结构

```
agent/experiments/
├── 20260325_073427_0/              # 实验目录
│   ├── config.yaml                 # 配置文件备份
│   ├── experiment_record.json      # 实验记录
│   └── experiment.log              # 训练日志
├── 20260325_073427_1/
│   ├── config.yaml
│   ├── experiment_record.json
│   └── experiment.log
└── experiments_history.json        # 实验历史索引
```

### 实验目录说明

每个实验目录包含：

- **config.yaml**: 训练使用的完整配置文件备份
- **experiment_record.json**: 实验元数据和结果
  - experiment_id: 唯一实验ID（格式：YYYYMMDD_HHMMSS_counter）
  - timestamp: 创建时间
  - description: 实验描述
  - status: 状态（created/running/success/failed）
  - config: 使用的配置参数
  - results: 实验结果（eer, accuracy, loss等）
  - duration_seconds: 训练时长
- **experiment.log**: 详细的训练日志

## 核心功能

### 1. 创建实验

```python
from agent.utils.experiment_tracker import ExperimentTracker
from agent.utils import get_config_file, ConfigParser

# 初始化跟踪器
tracker = ExperimentTracker()

# 加载配置
config_path = get_config_file("train_ecapa_tdnn.yaml")
parser = ConfigParser(str(config_path))
config = parser.load_config()

# 修改超参数
config['lr'] = 0.001
config['batch_size'] = 32

# 创建实验
exp_id = tracker.create_experiment(
    config=config,
    config_path=str(config_path),
    data_folder="../datasets/voxceleb",
    description="实验描述"
)
```

### 2. 更新实验

```python
# 训练完成后更新结果
tracker.update_experiment(
    experiment_id=exp_id,
    results={
        'eer': 0.0275,
        'accuracy': 0.9655,
        'loss': 0.1401
    },
    status='success',
    duration=4035.78  # 秒
)
```

### 3. 查找最佳实验

```python
# 按 EER 最小化查找最佳实验
best_exps = tracker.find_best_experiment(metric='eer', minimize=True, top_n=3)

for exp in best_exps:
    print(f"{exp['experiment_id']}: EER={exp['results']['eer']:.4f}")
```

### 4. 比较实验

```python
# 比较多个实验
comparison = tracker.compare_experiments(['exp_id_1', 'exp_id_2', 'exp_id_3'])

print(f"最佳 EER: {comparison['best_by_metric']['eer']}")
print(f"最佳准确率: {comparison['best_by_metric']['accuracy']}")
```

### 5. 获取统计信息

```python
stats = tracker.get_statistics()
print(f"总实验数: {stats['total']}")
print(f"成功率: {stats['success_rate']}")
print(f"平均时长: {stats['average_duration_seconds']/60:.1f} 分钟")
```

### 6. 导出实验数据

```python
# 导出所有实验
export_path = Path("experiments_export.json")
tracker.export_experiments(export_path)

# 导出指定实验
tracker.export_experiments(export_path, experiment_ids=['exp_id_1', 'exp_id_2'])
```

### 7. 清理旧实验

```python
# 保留最近 5 个实验
deleted = tracker.cleanup_old_experiments(keep_n=5)

# 只清理成功的实验，保留最近 3 个
deleted = tracker.cleanup_old_experiments(keep_n=3, status_filter='success')
```

## 在智能体中使用

智能体系统可以通过工具调用实验管理功能：

### 示例：训练工具集成

```python
class TrainTool:
    def execute(self, config: dict) -> dict:
        """执行训练并记录实验"""
        # 1. 创建实验
        tracker = ExperimentTracker()
        exp_id = tracker.create_experiment(
            config=config,
            config_path=config['config_path'],
            data_folder=config['data_folder'],
            description=f"训练: lr={config['lr']}, batch_size={config['batch_size']}"
        )
        
        # 2. 创建日志记录器
        logger = ExperimentLogger(exp_id)
        logger.log_start("开始训练")
        
        # 3. 执行训练
        start_time = time.time()
        try:
            # 这里调用实际的训练代码
            results = self._run_training(config)
            duration = time.time() - start_time
            
            # 4. 更新实验记录
            tracker.update_experiment(
                experiment_id=exp_id,
                results=results,
                status='success',
                duration=duration
            )
            logger.log_end(success=True, duration=duration)
            
            return {
                'success': True,
                'experiment_id': exp_id,
                'results': results
            }
        except Exception as e:
            duration = time.time() - start_time
            tracker.update_experiment(
                experiment_id=exp_id,
                status='failed',
                error=str(e),
                duration=duration
            )
            logger.log_end(success=False, duration=duration, error=str(e))
            
            return {
                'success': False,
                'experiment_id': exp_id,
                'error': str(e)
            }
```

### 示例：超参数优化

```python
class HyperparameterOptimizer:
    def optimize(self, config: dict, num_trials: int = 10) -> dict:
        """执行超参数优化"""
        tracker = ExperimentTracker()
        
        best_experiment = None
        best_eer = float('inf')
        
        for trial in range(num_trials):
            # 生成新的超参数组合
            new_config = self._generate_hyperparameters(config)
            
            # 训练模型
            result = self.train_tool.execute(new_config)
            
            if result['success']:
                eer = result['results']['eer']
                if eer < best_eer:
                    best_eer = eer
                    best_experiment = result
        
        # 返回最佳实验
        return best_experiment
```

## 实验结果示例

以下是生成的演示实验结果：

```
最佳实验 (按 EER 最小化):
  1. 20260325_073427_1
     EER: 0.0275
     Accuracy: 0.9655
     Duration: 67.3 分钟
     Description: 实验 2: lr=0.001, batch_size=32, epochs=10, seed=5678
  
  2. 20260325_073427_4
     EER: 0.0538
     Accuracy: 0.9138
     Duration: 108.6 分钟
     Description: 实验 5: lr=0.0005, batch_size=16, epochs=30, seed=7890
```

## 运行测试

```bash
# 运行所有测试
python agent/tests/test_experiment_tracker.py
```

测试覆盖：
- ✅ 创建实验
- ✅ 获取实验记录
- ✅ 更新实验
- ✅ 列出实验
- ✅ 查找最佳实验
- ✅ 比较实验
- ✅ 获取统计信息
- ✅ 删除实验
- ✅ 清理旧实验
- ✅ 导出实验
- ✅ 完整工作流集成

## 注意事项

1. **实验ID格式**: `YYYYMMDD_HHMMSS_counter` 确保唯一性
2. **并发安全**: 每秒最多创建 1000 个实验
3. **备份**: 配置文件会自动备份到实验目录
4. **清理**: 使用 `cleanup_old_experiments` 定期清理旧实验
5. **导出**: 重要实验结果建议定期导出备份

## 常见问题

### Q: 如何查看某个实验的详细配置？
A: 进入实验目录查看 `config.yaml` 文件

```bash
cat agent/experiments/20260325_073427_1/config.yaml
```

### Q: 如何比较不同超参数的效果？
A: 使用 `compare_experiments` 方法

```python
comparison = tracker.compare_experiments(['exp_id_1', 'exp_id_2'])
print(comparison['metrics_comparison'])
```

### Q: 实验失败后如何查看错误信息？
A: 查看 `experiment_record.json` 中的 `error` 字段

```python
record = tracker.get_experiment('failed_exp_id')
print(record['error'])
```

### Q: 如何恢复使用特定配置重新训练？
A: 使用实验目录中的 `config.yaml` 文件

```bash
python train.py --config agent/experiments/20260325_073427_1/config.yaml
```

## 总结

实验管理系统提供了：
- ✅ 完整的实验生命周期管理
- ✅ 自动化的实验跟踪和记录
- ✅ 灵活的实验比较和分析
- ✅ 方便的实验查询和筛选
- ✅ 可靠的实验数据存储和导出

通过这个系统，您可以高效地进行超参数优化，追踪实验进展，并轻松找到最佳模型配置。