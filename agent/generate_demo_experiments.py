"""
生成演示实验记录脚本
用于创建模拟的试验记录来测试实验管理功能
"""

import sys
from pathlib import Path
import random
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.utils.experiment_tracker import ExperimentTracker
from agent.utils.logger import ExperimentLogger
from agent.utils import get_config_file, ConfigParser


def generate_demo_experiments(num_experiments=10):
    """
    生成演示实验记录
    
    参数:
        num_experiments: 要生成的实验数量
    """
    print("="*80)
    print("生成演示实验记录")
    print("="*80)
    
    # 初始化实验跟踪器
    tracker = ExperimentTracker()
    
    # 加载配置文件
    config_path = get_config_file("train_ecapa_tdnn.yaml")
    if not config_path.exists():
        print(f"错误：配置文件 {config_path} 不存在")
        return
    
    parser = ConfigParser(str(config_path))
    base_config = parser.load_config(resolve_references=False)
    
    print(f"\n使用配置文件: {config_path}")
    print(f"实验目录: {tracker.experiments_dir}")
    print(f"\n将生成 {num_experiments} 个演示实验...")
    
    # 定义一些超参数组合
    lr_values = [0.0001, 0.0005, 0.001, 0.002, 0.005]
    batch_sizes = [16, 32, 64, 128]
    max_epochs = [10, 20, 30]
    seeds = [1234, 5678, 9012, 3456, 7890]
    
    exp_ids = []
    
    for i in range(num_experiments):
        # 随机选择超参数
        lr = random.choice(lr_values)
        batch_size = random.choice(batch_sizes)
        max_epoch = random.choice(max_epochs)
        seed = random.choice(seeds)
        
        # 修改配置
        config = base_config.copy()
        config['lr'] = lr
        config['batch_size'] = batch_size
        config['max_epochs'] = max_epoch
        config['seed'] = seed
        
        # 创建实验描述
        description = f"实验 {i+1}: lr={lr}, batch_size={batch_size}, epochs={max_epoch}, seed={seed}"
        
        print(f"\n[{i+1}/{num_experiments}] 创建实验...")
        print(f"  配置: {description}")
        
        # 创建实验
        exp_id = tracker.create_experiment(
            config=config,
            config_path=str(config_path),
            data_folder="../datasets/voxceleb",
            description=description
        )
        
        print(f"  实验 ID: {exp_id}")
        
        # 创建日志记录器
        logger = ExperimentLogger(exp_id)
        logger.log_start(description)
        
        # 模拟训练（随机生成结果）
        # 模拟训练时长（30-120 分钟）
        duration_minutes = random.uniform(30, 120)
        duration_seconds = duration_minutes * 60
        
        # 根据超参数生成合理的模拟结果
        # 学习率适中、批次大小适中的通常效果较好
        lr_score = 1.0 - abs(lr - 0.001) / 0.005
        batch_score = 1.0 - abs(batch_size - 32) / 64
        
        # 基础 EER（0.03-0.08）
        base_eer = 0.03 + (1.0 - lr_score * batch_score) * 0.05
        eer = round(base_eer + random.uniform(-0.01, 0.01), 4)
        eer = max(0.01, min(0.15, eer))  # 限制在合理范围内
        
        # 准确率（与 EER 相关）
        accuracy = round(1.0 - eer * 1.5 + random.uniform(-0.02, 0.02), 4)
        accuracy = max(0.8, min(0.99, accuracy))
        
        # Loss
        loss = round(eer * 2.0 + random.uniform(0, 0.1), 4)
        
        # 验证集准确率
        val_accuracy = round(accuracy - random.uniform(0.01, 0.03), 4)
        
        # 训练集准确率
        train_accuracy = round(accuracy + random.uniform(0.01, 0.05), 4)
        
        # 随机决定是否成功（80% 成功率）
        is_success = random.random() < 0.8
        
        if is_success:
            status = "success"
            results = {
                'eer': eer,
                'accuracy': accuracy,
                'loss': loss,
                'val_accuracy': val_accuracy,
                'train_accuracy': train_accuracy,
                'error_rate': eer
            }
            print(f"  状态: 成功")
            print(f"  结果: EER={eer:.4f}, Accuracy={accuracy:.4f}, Loss={loss:.4f}")
            logger.log_end(success=True, duration=duration_seconds)
        else:
            status = "failed"
            results = None
            error_reason = random.choice([
                "训练过程中梯度爆炸",
                "内存不足",
                "验证损失不下降",
                "数据加载失败",
                "超参数导致发散"
            ])
            print(f"  状态: 失败 ({error_reason})")
            tracker.update_experiment(
                experiment_id=exp_id,
                status=status,
                error=error_reason,
                duration=duration_seconds
            )
            logger.log_end(success=False, duration=duration_seconds, error=error_reason)
        
        # 更新实验记录
        tracker.update_experiment(
            experiment_id=exp_id,
            results=results,
            status=status,
            duration=duration_seconds
        )
        
        exp_ids.append(exp_id)
    
    print("\n" + "="*80)
    print("演示实验生成完成！")
    print("="*80)
    
    # 显示统计信息
    stats = tracker.get_statistics()
    print(f"\n实验统计:")
    print(f"  总实验数: {stats['total']}")
    print(f"  成功: {stats['status_counts'].get('success', 0)}")
    print(f"  失败: {stats['status_counts'].get('failed', 0)}")
    print(f"  成功率: {stats['success_rate']}")
    print(f"  平均时长: {stats['average_duration_seconds']/60:.1f} 分钟")
    print(f"  总时长: {stats['total_duration_hours']:.2f} 小时")
    
    # 显示最佳实验
    print("\n最佳实验 (按 EER 最小化):")
    best = tracker.find_best_experiment(metric='eer', minimize=True, top_n=3)
    if best:
        for i, exp in enumerate(best, 1):
            print(f"  {i}. {exp['experiment_id']}")
            print(f"     EER: {exp['results']['eer']:.4f}")
            print(f"     Accuracy: {exp['results']['accuracy']:.4f}")
            print(f"     Duration: {exp['duration_seconds']/60:.1f} 分钟")
            print(f"     Description: {exp['description']}")
    
    # 导出所有实验
    export_path = Path(__file__).parent.parent / "experiments_demo_export.json"
    tracker.export_experiments(export_path)
    print(f"\n所有实验已导出到: {export_path}")
    
    # 列出所有实验目录
    print(f"\n实验目录: {tracker.experiments_dir}")
    print("\n所有实验:")
    all_exps = tracker.list_experiments(sort_by='timestamp', reverse=True)
    for exp in all_exps:
        print(f"  - {exp['experiment_id']}: {exp['status']}")
        if exp['results']:
            print(f"    EER: {exp['results']['eer']:.4f}, Accuracy: {exp['results']['accuracy']:.4f}")
    
    print("\n" + "="*80)
    print("提示: 您可以使用以下命令查看实验目录")
    print(f"  cd {tracker.experiments_dir}")
    print("  ls -la")
    print("="*80)
    
    return exp_ids


def cleanup_demo_experiments():
    """清理所有演示实验"""
    print("="*80)
    print("清理演示实验")
    print("="*80)
    
    tracker = ExperimentTracker()
    all_exps = tracker.list_experiments()
    
    if not all_exps:
        print("没有找到需要清理的实验")
        return
    
    print(f"找到 {len(all_exps)} 个实验")
    print(f"实验目录: {tracker.experiments_dir}")
    
    confirm = input("\n确认要删除所有实验吗？(yes/no): ")
    if confirm.lower() != 'yes':
        print("取消清理")
        return
    
    deleted = 0
    for exp in all_exps:
        if tracker.delete_experiment(exp['experiment_id']):
            deleted += 1
    
    print(f"\n已删除 {deleted} 个实验")


def view_experiments():
    """查看现有实验"""
    print("="*80)
    print("查看现有实验")
    print("="*80)
    
    tracker = ExperimentTracker()
    all_exps = tracker.list_experiments(sort_by='timestamp', reverse=True)
    
    if not all_exps:
        print("没有找到实验记录")
        return
    
    print(f"\n实验目录: {tracker.experiments_dir}")
    print(f"总实验数: {len(all_exps)}\n")
    
    for exp in all_exps:
        print(f"实验 ID: {exp['experiment_id']}")
        print(f"  时间: {exp['timestamp']}")
        print(f"  状态: {exp['status']}")
        print(f"  描述: {exp['description']}")
        if exp['results']:
            print(f"  结果:")
            for key, value in exp['results'].items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
        print(f"  时长: {exp['duration_seconds']/60:.1f} 分钟")
        print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='实验管理演示脚本')
    parser.add_argument('command', choices=['generate', 'view', 'cleanup'], 
                       help='命令: generate(生成), view(查看), cleanup(清理)')
    parser.add_argument('--num', type=int, default=10,
                       help='生成的实验数量 (默认: 10)')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        generate_demo_experiments(args.num)
    elif args.command == 'view':
        view_experiments()
    elif args.command == 'cleanup':
        cleanup_demo_experiments()