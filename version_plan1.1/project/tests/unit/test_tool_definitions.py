import ast
from pathlib import Path


def _tool_source() -> Path:
    return Path(__file__).resolve().parents[2] / "agent" / "tools"


def test_all_tool_functions_have_docstrings() -> None:
    missing = []
    tools_dir = _tool_source()
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


def test_tool_package_history_exports_exist() -> None:
    tools_dir = _tool_source()
    history_tree = ast.parse(
        (tools_dir / "experiment_history_tools.py").read_text(encoding="utf-8")
    )
    available = {
        node.name
        for node in history_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    available.update(
        target.id
        for node in history_tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    )

    package_tree = ast.parse((tools_dir / "__init__.py").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in package_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "experiment_history_tools"
        for alias in node.names
    }

    assert imported <= available, f"missing experiment history exports: {imported - available}"
