"""Guardrails for the Redis + Qdrant-only online dependency graph."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ONLINE_ROOTS = [ROOT / "api" / "main.py", ROOT / "feedback" / "__init__.py"]
FORBIDDEN_MODULES = {"database", "feedback.storage", "trending.storage"}
SQL_MARKERS = (
    "select ", "insert into ", "update ", "delete from ",
    "create table", "alter table", "drop table",
)


def _local_module_path(module: str) -> Path | None:
    parts = module.split(".")
    module_file = ROOT.joinpath(*parts).with_suffix(".py")
    if module_file.exists():
        return module_file
    package_file = ROOT.joinpath(*parts, "__init__.py")
    return package_file if package_file.exists() else None


def _module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_from(current: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    package = current.split(".")[:-1]
    keep = max(0, len(package) - (node.level - 1))
    prefix = package[:keep]
    if node.module:
        prefix.extend(node.module.split("."))
    return ".".join(prefix)


def _reachable_files() -> set[Path]:
    pending = list(ONLINE_ROOTS)
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        current = _module_name(path)
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_from(current, node)
                modules.append(base)
                modules.extend(f"{base}.{alias.name}" for alias in node.names if base)
            for module in modules:
                candidate = _local_module_path(module)
                if candidate and candidate not in visited:
                    pending.append(candidate)
    return visited


def test_online_import_graph_has_no_database_dependency():
    reachable = _reachable_files()
    violations = []
    for path in reachable:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        current = _module_name(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [_resolve_from(current, node)]
            else:
                continue
            for name in names:
                if any(name == forbidden or name.startswith(f"{forbidden}.") for forbidden in FORBIDDEN_MODULES):
                    violations.append(f"{path.relative_to(ROOT)} imports {name}")
    assert not violations, "\n".join(violations)


def test_reachable_online_modules_contain_no_sql_or_feedback_store():
    violations = []
    for path in _reachable_files():
        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        if "feedbackstore" in lowered:
            violations.append(f"{path.relative_to(ROOT)} references FeedbackStore")
        for marker in SQL_MARKERS:
            if marker in lowered:
                violations.append(f"{path.relative_to(ROOT)} contains SQL marker {marker!r}")
    assert not violations, "\n".join(violations)


def test_api_import_does_not_require_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    module = importlib.import_module("api.main")
    importlib.reload(module)
    assert module.app is not None
