"""
智能体日志和中间件模块

该模块提供智能体交互的日志记录和工具调用流程追踪功能。
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from functools import wraps
from langchain_core.runnables import RunnableConfig
from langchain.agents.middleware import before_model, after_model


class AgentLogger:
    """智能体日志记录器"""
    
    def __init__(self, log_dir: Path = None):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录路径，默认为 agent/logs
        """
        if log_dir is None:
            log_dir = Path(__file__).parent / "logs"
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建当前会话的日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"agent_session_{timestamp}.log"
        
        # 初始化Python日志记录器
        self.logger = logging.getLogger("AgentLogger")
        self.logger.setLevel(logging.DEBUG)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 记录会话开始
        self.log_session_start()
        
        # 保存工具调用历史
        self.tool_call_history = []
    
    def log_session_start(self):
        """记录会话开始"""
        separator = "=" * 80
        self.logger.info(separator)
        self.logger.info("智能体会话开始")
        self.logger.info(f"日志文件: {self.log_file}")
        self.logger.info(separator)
    
    def log_user_message(self, message: str):
        """
        记录用户消息
        
        Args:
            message: 用户消息内容
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("👤 用户输入:")
        self.logger.info(message)
    
    def log_tool_call(self, tool_name: str, tool_input: Dict[str, Any]):
        """
        记录工具调用
        
        Args:
            tool_name: 工具名称
            tool_input: 工具输入参数
        """
        timestamp = datetime.now().isoformat()
        
        # 记录到日志文件
        self.logger.info("\n" + "-" * 80)
        self.logger.info(f"🔧 调用工具: {tool_name}")
        self.logger.info(f"时间: {timestamp}")
        
        # 格式化显示参数
        if tool_input:
            self.logger.debug("输入参数:")
            for key, value in tool_input.items():
                if isinstance(value, (str, int, float, bool)):
                    self.logger.debug(f"  {key}: {value}")
                elif isinstance(value, dict):
                    self.logger.debug(f"  {key}: <dict> (keys: {list(value.keys())})")
                elif isinstance(value, list):
                    self.logger.debug(f"  {key}: <list> (length: {len(value)})")
                else:
                    self.logger.debug(f"  {key}: <{type(value).__name__}>")
        
        # 保存到历史记录
        call_record = {
            "timestamp": timestamp,
            "tool_name": tool_name,
            "input": tool_input
        }
        self.tool_call_history.append(call_record)
    
    def log_tool_result(self, tool_name: str, result: str, success: bool = True):
        """
        记录工具结果
        
        Args:
            tool_name: 工具名称
            result: 工具返回结果
            success: 是否成功
        """
        status = "✅ 成功" if success else "❌ 失败"
        
        self.logger.info(f"{status}: {tool_name}")
        
        # 截断过长的结果
        if result:
            if len(result) > 500:
                self.logger.debug(f"结果 (前500字符):\n{result[:500]}...")
            else:
                self.logger.debug(f"结果:\n{result}")
        
        # 更新历史记录
        if self.tool_call_history:
            self.tool_call_history[-1]["success"] = success
            self.tool_call_history[-1]["result_preview"] = result[:200] if result else ""
    
    def log_model_thinking(self, thinking: str):
        """
        记录模型思考过程
        
        Args:
            thinking: 思考内容
        """
        self.logger.debug("\n" + "-" * 80)
        self.logger.debug("🧠 模型思考:")
        self.logger.debug(thinking)
    
    def log_model_response(self, response: str):
        """
        记录模型响应
        
        Args:
            response: 模型响应内容
        """
        self.logger.info("\n" + "-" * 80)
        self.logger.info("🤖 智能体回复:")
        self.logger.info(response)
    
    def log_error(self, error: str):
        """
        记录错误
        
        Args:
            error: 错误信息
        """
        self.logger.error("\n" + "=" * 80)
        self.logger.error(f"❌ 错误: {error}")
    
    def log_summary(self):
        """记录会话摘要"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 会话摘要")
        self.logger.info(f"工具调用次数: {len(self.tool_call_history)}")
        
        # 统计各工具调用次数
        tool_counts = {}
        for call in self.tool_call_history:
            tool_name = call["tool_name"]
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        
        if tool_counts:
            self.logger.info("工具调用统计:")
            for tool_name, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  {tool_name}: {count} 次")
        
        # 保存完整的工具调用历史到JSON文件
        history_file = self.log_file.with_suffix('.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump({
                "session_info": {
                    "start_time": datetime.now().isoformat(),
                    "log_file": str(self.log_file),
                    "total_tool_calls": len(self.tool_call_history)
                },
                "tool_call_history": self.tool_call_history
            }, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n工具调用历史已保存到: {history_file}")
        self.logger.info("=" * 80 + "\n")
    
    def get_log_file_path(self) -> Path:
        """获取当前日志文件路径"""
        return self.log_file
    
    def get_tool_call_history(self) -> list:
        """获取工具调用历史"""
        return self.tool_call_history


# 全局日志记录器实例
_global_logger: Optional[AgentLogger] = None


def get_global_logger() -> AgentLogger:
    """获取全局日志记录器实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = AgentLogger()
    return _global_logger


def create_tool_call_middleware(logger: AgentLogger):
    """
    创建工具调用中间件
    
    这个中间件会在工具调用前后记录日志
    
    Args:
        logger: AgentLogger实例
    """
    
    def tool_call_decorator(func):
        """工具调用装饰器"""
        
        @wraps(func)
        def wrapped(*args, **kwargs):
            # 获取工具名称
            tool_name = func.__name__
            
            # 记录工具调用
            logger.log_tool_call(tool_name, kwargs)
            
            try:
                # 执行工具函数
                result = func(*args, **kwargs)
                
                # 记录工具结果（成功）
                logger.log_tool_result(tool_name, str(result), success=True)
                
                return result
            except Exception as e:
                # 记录工具结果（失败）
                logger.log_tool_result(tool_name, str(e), success=False)
                logger.log_error(f"工具 {tool_name} 执行失败: {str(e)}")
                raise
        
        return wrapped
    
    return tool_call_decorator


def create_before_model_middleware(logger: AgentLogger):
    """
    创建模型调用前的中间件
    
    Args:
        logger: AgentLogger实例
    """
    
    @before_model
    async def log_before_model(inputs: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        """在模型调用前记录日志"""
        # 检查是否有消息输入
        if "messages" in inputs:
            messages = inputs["messages"]
            if messages:
                # 获取最后一条消息（通常是用户输入）
                last_message = messages[-1]
                if isinstance(last_message, dict):
                    content = last_message.get("content", "")
                    if content:
                        logger.log_user_message(content)
        
        return inputs
    
    return log_before_model


def create_after_model_middleware(logger: AgentLogger):
    """
    创建模型调用后的中间件
    
    Args:
        logger: AgentLogger实例
    """
    
    @after_model
    async def log_after_model(outputs: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        """在模型调用后记录日志"""
        # 检查是否有输出消息
        if "messages" in outputs:
            messages = outputs["messages"]
            if messages:
                # 获取最后一条消息（通常是模型回复）
                last_message = messages[-1]
                if isinstance(last_message, dict):
                    content = last_message.get("content", "")
                    if content:
                        logger.log_model_response(content)
        
        return outputs
    
    return log_after_model


def apply_tool_logging(tools: list, logger: AgentLogger) -> list:
    """
    为所有工具应用日志记录
    
    Args:
        tools: 工具列表
        logger: AgentLogger实例
    
    Returns:
        应用日志记录后的工具列表
    """
    decorated_tools = []
    
    for tool in tools:
        # 获取工具的实际函数
        if hasattr(tool, 'func'):
            original_func = tool.func
            # 创建装饰器并应用
            decorator = create_tool_call_middleware(logger)
            tool.func = decorator(original_func)
        elif hasattr(tool, 'run'):
            original_func = tool.run
            decorator = create_tool_call_middleware(logger)
            tool.run = decorator(original_func)
        
        decorated_tools.append(tool)
    
    return decorated_tools


def log_agent_execution(logger: AgentLogger):
    """
    智能体执行装饰器，用于包装整个智能体的执行过程
    
    Args:
        logger: AgentLogger实例
    """
    
    def decorator(func):
        """装饰器函数"""
        
        @wraps(func)
        async def wrapped(*args, **kwargs):
            """包装函数"""
            logger.log_session_start()
            
            try:
                result = await func(*args, **kwargs)
                logger.log_summary()
                return result
            except Exception as e:
                logger.log_error(f"智能体执行失败: {str(e)}")
                logger.log_summary()
                raise
        
        return wrapped
    
    return decorator


if __name__ == "__main__":
    # 测试代码
    logger = AgentLogger()
    
    logger.log_user_message("测试用户消息")
    logger.log_tool_call("test_tool", {"param1": "value1", "param2": 123})
    logger.log_tool_result("test_tool", "操作成功", success=True)
    logger.log_model_response("这是智能体的回复")
    logger.log_summary()
    
    print(f"日志已保存到: {logger.get_log_file_path()}")