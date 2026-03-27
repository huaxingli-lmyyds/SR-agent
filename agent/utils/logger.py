"""
日志管理模块
提供结构化日志记录、日志轮转、日志查询等功能
"""

from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
import json
import os


class Logger:
    """日志记录器类"""
    
    def __init__(self, 
                 log_path: Union[str, Path],
                 name: str = "agent",
                 level: str = "INFO",
                 console: bool = True,
                 file: bool = True,
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """
        初始化日志记录器
        
        参数:
            log_path: 日志文件路径
            name: 日志记录器名称
            level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
            console: 是否输出到控制台
            file: 是否输出到文件
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的备份文件数量
        """
        self.log_path = Path(log_path)
        self.name = name
        
        # 确保日志目录存在
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # 避免重复添加处理器
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 添加控制台处理器
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 添加文件处理器（带轮转）
        if file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, extra: Optional[Dict] = None):
        """记录 DEBUG 级别日志"""
        self._log(logging.DEBUG, message, extra)
    
    def info(self, message: str, extra: Optional[Dict] = None):
        """记录 INFO 级别日志"""
        self._log(logging.INFO, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict] = None):
        """记录 WARNING 级别日志"""
        self._log(logging.WARNING, message, extra)
    
    def error(self, message: str, extra: Optional[Dict] = None, exc_info: bool = False):
        """记录 ERROR 级别日志"""
        self._log(logging.ERROR, message, extra, exc_info)
    
    def critical(self, message: str, extra: Optional[Dict] = None, exc_info: bool = False):
        """记录 CRITICAL 级别日志"""
        self._log(logging.CRITICAL, message, extra, exc_info)
    
    def _log(self, level: int, message: str, extra: Optional[Dict] = None, exc_info: bool = False):
        """内部日志记录方法"""
        if extra:
            extra_str = json.dumps(extra, ensure_ascii=False)
            message = f"{message} | Extra: {extra_str}"
        self.logger.log(level, message, exc_info=exc_info)
    
    def get_logs(self, 
                 start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None,
                 level: Optional[str] = None,
                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取日志记录
        
        参数:
            start_time: 开始时间
            end_time: 结束时间
            level: 日志级别过滤
            limit: 限制返回数量
        
        Returns:
            日志记录列表
        """
        logs = []
        
        # 主日志文件
        log_files = [self.log_path]
        
        # 添加备份日志文件
        for i in range(1, 10):  # 最多检查 9 个备份
            backup_path = self.log_path.with_suffix(f'.log.{i}')
            if backup_path.exists():
                log_files.append(backup_path)
            else:
                break
        
        # 解析日志文件
        for log_file in log_files:
            if not log_file.exists():
                continue
            
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = self._parse_log_line(line)
                        if log_entry:
                            # 时间过滤
                            if start_time and log_entry['timestamp'] < start_time:
                                continue
                            if end_time and log_entry['timestamp'] > end_time:
                                continue
                            
                            # 级别过滤
                            if level and log_entry['level'] != level.upper():
                                continue
                            
                            logs.append(log_entry)
                    except Exception:
                        continue
        
        # 按时间排序
        logs.sort(key=lambda x: x['timestamp'])
        
        # 限制数量
        if limit:
            logs = logs[-limit:]
        
        return logs
    
    def _parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """解析日志行"""
        try:
            # 格式: 2024-03-25 14:30:22 - agent - INFO - message
            parts = line.split(' - ', 3)
            if len(parts) < 4:
                return None
            
            timestamp_str, name, level, message = parts
            
            # 解析时间戳
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            
            # 解析 extra 信息
            extra = None
            if ' | Extra: ' in message:
                message, extra_str = message.split(' | Extra: ', 1)
                try:
                    extra = json.loads(extra_str)
                except json.JSONDecodeError:
                    pass
            
            return {
                'timestamp': timestamp,
                'name': name,
                'level': level,
                'message': message,
                'extra': extra
            }
        except Exception:
            return None
    
    def clear_logs(self):
        """清空日志文件"""
        if self.log_path.exists():
            self.log_path.write_text('', encoding='utf-8')
        
        # 删除备份文件
        for i in range(1, 10):
            backup_path = self.log_path.with_suffix(f'.log.{i}')
            if backup_path.exists():
                backup_path.unlink()
    
    def get_log_stats(self) -> Dict[str, Any]:
        """
        获取日志统计信息
        
        Returns:
            统计信息字典
        """
        logs = self.get_logs()
        
        if not logs:
            return {"total": 0, "message": "暂无日志"}
        
        # 统计各级别数量
        level_counts = {}
        for log in logs:
            level = log['level']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # 时间范围
        timestamps = [log['timestamp'] for log in logs]
        time_range = {
            "first": min(timestamps).isoformat(),
            "last": max(timestamps).isoformat()
        }
        
        # 文件大小
        file_size = self.log_path.stat().st_size if self.log_path.exists() else 0
        
        return {
            "total": len(logs),
            "level_counts": level_counts,
            "time_range": time_range,
            "file_size_bytes": file_size,
            "log_file": str(self.log_path)
        }
    
    def export_logs(self, 
                   output_path: Union[str, Path],
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   level: Optional[str] = None,
                   format: str = 'json') -> Path:
        """
        导出日志
        
        参数:
            output_path: 输出文件路径
            start_time: 开始时间
            end_time: 结束时间
            level: 日志级别过滤
            format: 导出格式（json, txt, csv）
        
        Returns:
            导出文件路径
        """
        logs = self.get_logs(start_time, end_time, level)
        output_path = Path(output_path)
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False, default=str)
        
        elif format == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for log in logs:
                    f.write(f"{log['timestamp']} - {log['level']} - {log['message']}\n")
        
        elif format == 'csv':
            import csv
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                if logs:
                    writer = csv.DictWriter(f, fieldnames=['timestamp', 'name', 'level', 'message'])
                    writer.writeheader()
                    writer.writerows([
                        {
                            'timestamp': log['timestamp'].isoformat(),
                            'name': log['name'],
                            'level': log['level'],
                            'message': log['message']
                        }
                        for log in logs
                    ])
        
        return output_path


class ExperimentLogger(Logger):
    """实验专用日志记录器"""
    
    def __init__(self, experiment_id: str, experiments_dir: Optional[Union[str, Path]] = None):
        """
        初始化实验日志记录器
        
        参数:
            experiment_id: 实验 ID
            experiments_dir: 实验目录
        """
        from .path_tool import get_experiments_dir, ensure_dir
        
        if experiments_dir:
            exp_dir = Path(experiments_dir) / experiment_id
        else:
            exp_dir = get_experiments_dir() / experiment_id
        
        ensure_dir(exp_dir)
        
        log_path = exp_dir / "experiment.log"
        
        super().__init__(
            log_path=log_path,
            name=f"experiment_{experiment_id}",
            level="INFO",
            console=False,
            file=True
        )
        
        self.experiment_id = experiment_id
    
    def log_config(self, config: Dict):
        """记录配置信息"""
        self.info("实验配置", extra={"config": config})
    
    def log_start(self, description: str = ""):
        """记录实验开始"""
        self.info(f"实验开始: {description}", extra={"experiment_id": self.experiment_id})
    
    def log_end(self, success: bool = True, duration: Optional[float] = None, error: Optional[str] = None):
        """记录实验结束"""
        status = "成功" if success else "失败"
        message = f"实验结束: {status}"
        extra = {"experiment_id": self.experiment_id, "success": success}
        if duration:
            extra["duration_seconds"] = duration
        if error:
            extra["error"] = error
        self.info(message, extra=extra)
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """记录性能指标"""
        self.info("性能指标", extra={"metrics": metrics})
    
    def log_error(self, error: str, exc_info: bool = False):
        """记录错误"""
        self.error(error, extra={"experiment_id": self.experiment_id}, exc_info=exc_info)


# 便捷函数
def get_logger(log_path: Union[str, Path], name: str = "agent") -> Logger:
    """快速获取日志记录器的便捷函数"""
    return Logger(log_path, name)


def get_experiment_logger(experiment_id: str) -> ExperimentLogger:
    """快速获取实验日志记录器的便捷函数"""
    return ExperimentLogger(experiment_id)


def setup_logging(log_dir: Union[str, Path], 
                  level: str = "INFO",
                  console: bool = True) -> Dict[str, Logger]:
    """
    设置多个日志记录器
    
    参数:
        log_dir: 日志目录
        level: 日志级别
        console: 是否输出到控制台
    
    Returns:
        日志记录器字典
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    loggers = {
        'main': Logger(log_dir / 'main.log', 'main', level, console),
        'agent': Logger(log_dir / 'agent.log', 'agent', level, console),
        'training': Logger(log_dir / 'training.log', 'training', level, console),
        'evaluation': Logger(log_dir / 'evaluation.log', 'evaluation', level, console),
    }
    
    return loggers