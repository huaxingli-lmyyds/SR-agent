"""
智能体中间件构建
"""

from typing import Any, List
import json

from langchain.agents.middleware import before_agent, after_model, wrap_tool_call, AgentState


def build_agent_logging_middleware(logger: Any, truncate_limit: int = 800) -> List[Any]:
	"""构建智能体日志中间件列表。"""

	def _truncate(text: str) -> str:
		text = text.strip()
		if len(text) <= truncate_limit:
			return text
		return f"{text[:truncate_limit]}..."

	@before_agent
	def _log_start(state: AgentState, runtime: Any) -> None:
		logger.append("agent_run_start")

	@after_model
	def _log_model_output(state: AgentState, runtime: Any) -> None:
		if not state.get("messages"):
			return
		last_msg = state["messages"][-1]
		content = getattr(last_msg, "content", "")
		if content:
			logger.append(f"model_response={_truncate(str(content))}")

	@wrap_tool_call
	def _log_tool_call(request: Any, handler: Any) -> Any:
		tool_name = "unknown"
		tool_args = {}
		if hasattr(request, "tool_call"):
			tool_name = request.tool_call.get("name", "unknown")
			tool_args = request.tool_call.get("args", {})
		logger.append(
			"tool_call name="
			f"{tool_name} args={_truncate(json.dumps(tool_args, ensure_ascii=False))}"
		)
		result = handler(request)
		result_text = getattr(result, "content", str(result))
		logger.append(
			f"tool_result name={tool_name} content={_truncate(str(result_text))}"
		)
		return result

	return [_log_start, _log_model_output, _log_tool_call]
