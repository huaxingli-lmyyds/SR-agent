import ast
from pathlib import Path


def test_all_tool_functions_have_docstrings() -> None:
    missing = []
    tools_dir = Path(__file__).resolve().parents[2] / "agent" / "tools"
    for path in tools_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorator_names = {
                getattr(decorator, "id", None) or getattr(decorator, "attr", None)
                for decorator in node.decorator_list
            }
            if "tool" in decorator_names and ast.get_docstring(node) is None:
                missing.append(f"{path.name}:{node.lineno}:{node.name}")

    assert not missing, f"@tool functions missing docstrings: {missing}"
