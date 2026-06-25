"""
性能指标分析模块
提供指标提取、计算、比较和可视化等功能
针对 SpeechBrain ECAPA-TDNN 的输出格式进行优化
"""

from pathlib import Path
from typing import Union, Dict, List, Optional, Any
import re


class MetricsExtractor:
    """性能指标提取器"""
    
    def __init__(self):
        """初始化指标提取器"""
        # SpeechBrain 输出格式的正则表达式模式
        self.epoch_pattern = re.compile(
            r'epoch:\s*(\d+).*?lr:\s*([\d.e+-]+).*?'
            r'train loss:\s*([\d.e+-]+).*?'
            r'valid loss:\s*([\d.e+-]+).*?'
            r'valid ErrorRate:\s*([\d.e+-]+)',
            re.IGNORECASE
        )
        
        # 日志中的最终指标
        self.eer_pattern = re.compile(r'EER\(%\)\s*=\s*([\d.e+-]+)', re.IGNORECASE)
        self.min_dcf_pattern = re.compile(r'minDCF\s*=\s*([\d.e+-]+)', re.IGNORECASE)
    
    def extract_from_log(self, log_path: Union[str, Path]) -> Dict[str, Any]:
        """
        从日志文件中提取性能指标
        
        参数:
            log_path: 日志文件路径（train_log.txt 或 log.txt）
        
        Returns:
            提取到的指标字典，包含训练历史和最终结果
        """
        log_path = Path(log_path)
        if not log_path.exists():
            return {"error": f"日志文件不存在: {log_path}"}
        
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        result = {
            "training_history": self._extract_training_history(content),
            "final_metrics": self._extract_final_metrics(content),
            "log_file": str(log_path)
        }
        
        return result
    
    def _extract_training_history(self, content: str) -> Dict[str, List[Any]]:
        """
        从日志内容中提取训练历史
        
        格式示例: epoch: 10, lr: 7.96e-05 - train loss: 2.74e-01 - valid loss: 2.47e-01, valid ErrorRate: 4.89e-03
        """
        history = {
            "epochs": [],
            "learning_rates": [],
            "train_losses": [],
            "valid_losses": [],
            "valid_error_rates": []
        }
        
        matches = self.epoch_pattern.findall(content)
        
        for match in matches:
            epoch, lr, train_loss, valid_loss, error_rate = match
            
            try:
                history["epochs"].append(int(epoch))
                history["learning_rates"].append(float(lr))
                history["train_losses"].append(float(train_loss))
                history["valid_losses"].append(float(valid_loss))
                history["valid_error_rates"].append(float(error_rate))
            except ValueError:
                continue
        
        return history
    
    def _extract_final_metrics(self, content: str) -> Dict[str, Optional[float]]:
        """
        从日志末尾提取最终性能指标
        
        格式示例:
        2026-03-06 00:37:53,208 - __main__ - INFO - EER(%)=2.488634
        2026-03-06 00:38:03,743 - __main__ - INFO - minDCF=0.228039
        """
        metrics = {
            "eer": None,
            "min_dcf": None
        }
        
        # 提取 EER
        eer_matches = self.eer_pattern.findall(content)
        if eer_matches:
            try:
                metrics["eer"] = float(eer_matches[-1])
            except ValueError:
                pass
        
        # 提取 minDCF
        min_dcf_matches = self.min_dcf_pattern.findall(content)
        if min_dcf_matches:
            try:
                metrics["min_dcf"] = float(min_dcf_matches[-1])
            except ValueError:
                pass
        
        return metrics
    
    def extract_from_scores(self, scores_path: Union[str, Path]) -> Dict[str, Any]:
        """
        从 scores.txt 文件中提取分数数据
        
        格式示例:
        id10270/x6uYqmx31kE/00001 id10270/8jEAjG6SegY/00008 1 5.364582
        id10270/x6uYqmx31kE/00001 id10300/ize_eiCFEg0/00003 0 -7.294191
        
        参数:
            scores_path: scores.txt 文件路径
        
        Returns:
            分数数据字典
        """
        scores_path = Path(scores_path)
        if not scores_path.exists():
            return {"error": f"scores 文件不存在: {scores_path}"}
        
        genuine_scores = []  # 标签为 1 的分数（真实说话人）
        impostor_scores = []  # 标签为 0 的分数（冒充者）
        raw_data = []
        
        with open(scores_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        label = int(parts[2])
                        score = float(parts[3])
                        
                        raw_data.append({
                            "enrol": parts[0],
                            "test": parts[1],
                            "label": label,
                            "score": score
                        })
                        
                        if label == 1:
                            genuine_scores.append(score)
                        elif label == 0:
                            impostor_scores.append(score)
                    except (ValueError, IndexError):
                        continue
        
        return {
            "genuine_scores": genuine_scores,
            "impostor_scores": impostor_scores,
            "num_genuine": len(genuine_scores),
            "num_impostor": len(impostor_scores),
            "total_pairs": len(raw_data),
            "raw_data": raw_data
        }


class MetricsCalculator:
    """性能指标计算器"""
    
    @staticmethod
    def calculate_eer(genuine_scores: List[float], 
                     impostor_scores: List[float]) -> float:
        """
        计算等错误率 (EER)
        
        参数:
            genuine_scores: 真实说话人匹配分数列表
            impostor_scores: 冒充者分数列表
        
        Returns:
            EER 值（百分比）
        """
        import numpy as np
        
        if not genuine_scores or not impostor_scores:
            raise ValueError("genuine_scores 和 impostor_scores 不能为空")
        
        # 合并所有分数并打标签
        genuine_arr = np.asarray(genuine_scores, dtype=float)
        impostor_arr = np.asarray(impostor_scores, dtype=float)
        all_scores = np.concatenate([genuine_arr, impostor_arr])
        
        # 排序
        sorted_indices = np.argsort(all_scores)
        sorted_scores = all_scores[sorted_indices]
        
        # 计算不同阈值下的 FAR 和 FRR
        far_list = []
        frr_list = []
        
        for threshold in sorted_scores:
            # FAR: 冒充者被错误接受的比例
            impostor_above = np.sum(impostor_arr >= threshold)
            far = impostor_above / len(impostor_arr)
            far_list.append(far)
            
            # FRR: 真实说话人被拒绝的比例
            genuine_below = np.sum(genuine_arr < threshold)
            frr = genuine_below / len(genuine_arr)
            frr_list.append(frr)
        
        # 找到 FAR 和 FRR 最接近的点
        far_arr = np.array(far_list)
        frr_arr = np.array(frr_list)
        
        eer_index = np.argmin(np.abs(far_arr - frr_arr))
        eer = (far_arr[eer_index] + frr_arr[eer_index]) / 2 * 100  # 转换为百分比
        
        return eer
    
    @staticmethod
    def calculate_min_dcf(genuine_scores: List[float],
                        impostor_scores: List[float],
                        p_target: float = 0.01,
                        c_miss: float = 1.0,
                        c_fa: float = 1.0) -> float:
        """
        计算最小检测代价 (minDCF)
        
        参数:
            genuine_scores: 真实说话人匹配分数列表
            impostor_scores: 冒充者分数列表
            p_target: 目标说话人先验概率
            c_miss: 错误拒绝代价
            c_fa: 错误接受代价
        
        Returns:
            minDCF 值
        """
        import numpy as np
        
        if not genuine_scores or not impostor_scores:
            raise ValueError("genuine_scores 和 impostor_scores 不能为空")
        
        genuine_arr = np.asarray(genuine_scores, dtype=float)
        impostor_arr = np.asarray(impostor_scores, dtype=float)
        all_scores = np.concatenate([genuine_arr, impostor_arr])
        thresholds = np.linspace(np.min(all_scores), np.max(all_scores), 1000)
        
        dcf_values = []
        
        for threshold in thresholds:
            # 计算 FAR 和 FRR
            impostor_above = np.sum(impostor_arr >= threshold)
            far = impostor_above / len(impostor_arr)
            
            genuine_below = np.sum(genuine_arr < threshold)
            frr = genuine_below / len(genuine_arr)
            
            # 计算 DCF
            dcf = p_target * c_miss * frr + (1 - p_target) * c_fa * far
            dcf_values.append(dcf)
        
        return min(dcf_values)
    
    @staticmethod
    def compute_all_metrics(genuine_scores: List[float],
                          impostor_scores: List[float]) -> Dict[str, float]:
        """
        计算所有常用性能指标
        
        参数:
            genuine_scores: 真实说话人匹配分数列表
            impostor_scores: 冒充者分数列表
        
        Returns:
            包含所有指标的字典
        """
        try:
            eer = MetricsCalculator.calculate_eer(genuine_scores, impostor_scores)
            min_dcf = MetricsCalculator.calculate_min_dcf(genuine_scores, impostor_scores)
            
            return {
                "eer": eer,
                "min_dcf": min_dcf
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def get_best_epoch(training_history: Dict) -> Dict[str, Any]:
        """
        从训练历史中找出最佳 epoch
        
        参数:
            training_history: 训练历史字典（包含 valid_losses 或 valid_error_rates）
        
        Returns:
            最佳 epoch 信息
        """
        valid_losses = training_history.get("valid_losses", [])
        valid_error_rates = training_history.get("valid_error_rates", [])
        epochs = training_history.get("epochs", [])
        
        if not valid_losses and not valid_error_rates:
            return {"error": "没有找到验证集数据"}
        
        result = {}
        
        # 根据 valid loss 找出最佳 epoch
        if valid_losses:
            best_loss_idx = min(range(len(valid_losses)), key=valid_losses.__getitem__)
            result["best_loss_epoch"] = epochs[best_loss_idx]
            result["best_valid_loss"] = valid_losses[best_loss_idx]
        
        # 根据 valid error rate 找出最佳 epoch
        if valid_error_rates:
            best_error_idx = min(range(len(valid_error_rates)), key=valid_error_rates.__getitem__)
            result["best_error_epoch"] = epochs[best_error_idx]
            result["best_valid_error_rate"] = valid_error_rates[best_error_idx]
        
        return result


class MetricsComparator:
    """性能指标比较器"""

    @staticmethod
    def _flat_metrics(experiment: Dict) -> Dict[str, Any]:
        flattened: Dict[str, Any] = {}
        for split_metrics in (experiment.get("metrics") or {}).values():
            if isinstance(split_metrics, dict):
                flattened.update(split_metrics)
        return flattened

    @staticmethod
    def compare_experiments(
        experiments: List[Dict],
        primary_metric: str = "eer",
        metric_modes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        比较多个实验的性能指标
        
        参数:
            experiments: 实验记录列表，每个实验应包含 'results' 字段
            primary_metric: 主要比较指标（eer 或 min_dcf）
        
        Returns:
            比较结果字典
        """
        if not experiments:
            return {"error": "没有实验记录"}
        
        # 收集所有有效实验
        valid_experiments = []
        for exp in experiments:
            if exp.get("metrics") and exp.get('status') == 'success':
                valid_experiments.append(exp)
        
        if not valid_experiments:
            return {"error": "没有找到成功的实验"}
        
        metric_modes = dict(metric_modes or {})
        comparison = {
            "experiments": {},
            "metrics_summary": {},
            "best_by_metric": {},
            "ranking": {}
        }
        
        # 添加实验基本信息
        for exp in valid_experiments:
            exp_id = exp.get('experiment_id', 'unknown')
            comparison["experiments"][exp_id] = {
                "timestamp": exp.get('timestamp'),
                "duration_seconds": exp.get('duration_seconds'),
                "metrics": exp.get('metrics', {})
            }
        
        # 收集所有指标
        all_metrics = set()
        for exp in valid_experiments:
            results = MetricsComparator._flat_metrics(exp)
            for key, value in results.items():
                if value is not None and isinstance(value, (int, float)):
                    all_metrics.add(key)
        
        # 比较每个指标
        for metric in sorted(all_metrics):
            values = []
            for exp in valid_experiments:
                results = MetricsComparator._flat_metrics(exp)
                if metric in results and results[metric] is not None:
                    exp_id = exp.get('experiment_id', 'unknown')
                    values.append((exp_id, float(results[metric])))
            
            if values:
                mode = metric_modes.get(metric, _default_metric_mode(metric))
                sorted_values = sorted(values, key=lambda x: x[1], reverse=mode == "max")
                
                # 计算统计信息
                numeric_values = [v[1] for v in values]
                
                comparison["metrics_summary"][metric] = {
                    "best_experiment": sorted_values[0][0],
                    "best_value": sorted_values[0][1],
                    "worst_experiment": sorted_values[-1][0],
                    "worst_value": sorted_values[-1][1],
                    "average": sum(numeric_values) / len(numeric_values),
                    "std": MetricsComparator._std_dev(numeric_values),
                    "all_values": {eid: val for eid, val in values},
                    "mode": mode,
                }
                
                # 找出最佳实验
                comparison["best_by_metric"][metric] = sorted_values[0][0]
        
        # 根据主要指标排名
        if primary_metric in comparison["metrics_summary"]:
            metric_data = comparison["metrics_summary"][primary_metric]
            sorted_experiments = sorted(
                metric_data["all_values"].items(),
                key=lambda x: x[1],
                reverse=metric_data.get("mode") == "max",
            )
            comparison["ranking"] = {
                eid: rank + 1 
                for rank, (eid, _) in enumerate(sorted_experiments)
            }
        
        return comparison
    
    @staticmethod
    def _std_dev(values: List[float]) -> float:
        """计算标准差"""
        import numpy as np
        return float(np.std(values))
    
    @staticmethod
    def compare_with_baseline(baseline: Dict,
                             current: Dict,
                             metrics: List[str]) -> Dict[str, Any]:
        """
        将当前实验与基准实验进行比较
        
        参数:
            baseline: 基准实验
            current: 当前实验
            metrics: 要比较的指标列表
        
        Returns:
            比较结果
        """
        comparison = {
            "baseline_id": baseline.get('experiment_id', 'baseline'),
            "current_id": current.get('experiment_id', 'current'),
            "metrics": {}
        }
        
        baseline_results = MetricsComparator._flat_metrics(baseline)
        current_results = MetricsComparator._flat_metrics(current)
        
        for metric in metrics:
            baseline_value = baseline_results.get(metric)
            current_value = current_results.get(metric)
            
            if baseline_value is not None and current_value is not None:
                try:
                    baseline_val = float(baseline_value)
                    current_val = float(current_value)
                    
                    # 计算改进（假设越小越好，如 EER）
                    mode = _default_metric_mode(metric)
                    delta = current_val - baseline_val
                    signed_gain = delta if mode == "max" else -delta
                    improvement = (
                        signed_gain / abs(baseline_val) * 100
                        if baseline_val != 0 else None
                    )
                    
                    comparison["metrics"][metric] = {
                        "baseline": baseline_val,
                        "current": current_val,
                        "difference": current_val - baseline_val,
                        "improvement_percent": improvement,
                        "mode": mode,
                        "is_better": current_val > baseline_val if mode == "max" else current_val < baseline_val,
                    }
                except (ValueError, TypeError):
                    comparison["metrics"][metric] = {"error": "无法比较数值"}
            else:
                comparison["metrics"][metric] = {"error": "指标缺失"}
        
        # 计算平均改进
        valid_improvements = [
            m["improvement_percent"] 
            for m in comparison["metrics"].values() 
            if m.get("improvement_percent") is not None
        ]
        
        if valid_improvements:
            comparison["average_improvement"] = sum(valid_improvements) / len(valid_improvements)
        
        return comparison


def _default_metric_mode(metric: str) -> str:
    normalized = metric.lower()
    maximize_tokens = ("accuracy", "precision", "recall", "f1", "auc", "map", "reward")
    return "max" if any(token in normalized for token in maximize_tokens) else "min"


class MetricsVisualizer:
    """性能指标可视化工具"""
    
    @staticmethod
    def plot_training_curves(training_history: Dict,
                           output_path: Optional[Union[str, Path]] = None):
        """
        绘制训练曲线（loss 和 error rate）
        
        参数:
            training_history: 训练历史字典
            output_path: 输出文件路径，如果为 None 则显示图形
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("需要安装 matplotlib: pip install matplotlib")
            return
        
        epochs = training_history.get("epochs", [])
        train_losses = training_history.get("train_losses", [])
        valid_losses = training_history.get("valid_losses", [])
        valid_error_rates = training_history.get("valid_error_rates", [])
        
        if not epochs:
            print("没有训练历史数据")
            return
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 绘制 loss 曲线
        if train_losses:
            ax1.plot(epochs, train_losses, 'b-', label='Train Loss', marker='o')
        if valid_losses:
            ax1.plot(epochs, valid_losses, 'r-', label='Valid Loss', marker='s')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制 error rate 曲线
        if valid_error_rates:
            ax2.plot(epochs, valid_error_rates, 'g-', label='Valid Error Rate', marker='^')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Error Rate')
            ax2.set_title('Validation Error Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存或显示
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_metrics_comparison(comparison: Dict,
                               metric: str = "eer",
                               output_path: Optional[Union[str, Path]] = None):
        """
        绘制实验比较的柱状图
        
        参数:
            comparison: compare_experiments 的返回结果
            metric: 要绘制的指标
            output_path: 输出文件路径
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("需要安装 matplotlib: pip install matplotlib")
            return
        
        if metric not in comparison.get("metrics_summary", {}):
            print(f"指标 {metric} 不存在")
            return
        
        metric_data = comparison["metrics_summary"][metric]
        experiments = list(metric_data["all_values"].keys())
        values = list(metric_data["all_values"].values())
        
        # 创建条形图
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(experiments, values)
        
        # 标记最佳和最差
        best_exp = metric_data["best_experiment"]
        worst_exp = metric_data["worst_experiment"]
        
        for i, (exp, val) in enumerate(zip(experiments, values)):
            if exp == best_exp:
                bars[i].set_color('green')
            elif exp == worst_exp:
                bars[i].set_color('red')
        
        # 设置标签
        ax.set_xlabel('实验 ID')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} 比较 (越低越好)')
        plt.xticks(rotation=45, ha='right')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{value:.4f}', ha='center', va='bottom')
        
        # 添加平均线
        ax.axhline(y=metric_data["average"], color='gray', 
                  linestyle='--', label=f'平均: {metric_data["average"]:.4f}')
        ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"比较图已保存到: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_roc_curve(genuine_scores: List[float],
                      impostor_scores: List[float],
                      output_path: Optional[Union[str, Path]] = None):
        """
        绘制 ROC 曲线
        
        参数:
            genuine_scores: 真实说话人分数列表
            impostor_scores: 冒充者分数列表
            output_path: 输出文件路径
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from sklearn.metrics import roc_curve, auc
        except ImportError:
            print("需要安装 sklearn 和 matplotlib: pip install scikit-learn matplotlib")
            return
        
        # 合并分数和标签
        y_true = np.concatenate([
            np.ones(len(genuine_scores)),
            np.zeros(len(impostor_scores))
        ])
        y_scores = np.concatenate([genuine_scores, impostor_scores])
        
        # 计算 ROC 曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # 绘图
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FAR)')
        plt.ylabel('True Positive Rate (1 - FRR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"ROC 曲线已保存到: {output_path}")
        else:
            plt.show()
        
        plt.close()


# 便捷函数
def extract_log_metrics(log_path: Union[str, Path]) -> Dict[str, Any]:
    """快速从日志提取指标的便捷函数"""
    extractor = MetricsExtractor()
    return extractor.extract_from_log(log_path)


def extract_scores_data(scores_path: Union[str, Path]) -> Dict[str, Any]:
    """快速从 scores.txt 提取数据的便捷函数"""
    extractor = MetricsExtractor()
    return extractor.extract_from_scores(scores_path)


def compute_metrics_from_scores(scores_path: Union[str, Path]) -> Dict[str, float]:
    """从 scores.txt 计算所有指标的便捷函数"""
    extractor = MetricsExtractor()
    scores_data = extractor.extract_from_scores(scores_path)
    
    if "error" in scores_data:
        return scores_data
    
    calculator = MetricsCalculator()
    return calculator.compute_all_metrics(
        scores_data["genuine_scores"],
        scores_data["impostor_scores"]
    )


def compare_experiments(experiments: List[Dict],
                      primary_metric: str = "eer",
                      metric_modes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """快速比较实验的便捷函数"""
    comparator = MetricsComparator()
    return comparator.compare_experiments(experiments, primary_metric, metric_modes)
