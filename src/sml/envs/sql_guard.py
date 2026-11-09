"""Read-only DuckDB SQL guardrail and AST feature extractor powered by `sqlglot`."""

from __future__ import annotations

from typing import Any

import sqlglot
from sqlglot import exp

FORBIDDEN_KEYWORDS: frozenset[str] = frozenset(
    {
        "drop",
        "delete",
        "update",
        "insert",
        "alter",
        "create",
        "truncate",
        "replace",
        "copy",
        "attach",
        "detach",
        "pragma",
        "install",
        "load",
        "call",
        "export",
        "import",
        "vacuum",
        "merge",
    }
)

_FORBIDDEN_NODE_NAMES: frozenset[str] = frozenset(
    {
        "Drop",
        "Delete",
        "Update",
        "Insert",
        "Alter",
        "Create",
        "TruncateTable",
        "Copy",
        "Attach",
        "Detach",
        "Pragma",
        "Install",
        "Load",
        "Call",
        "Export",
        "Command",
        "Merge",
    }
)


def validate_readonly_sql(sql: str) -> tuple[bool, str]:
    """Validate that `sql` is a single read-only `SELECT` / `UNION` query for DuckDB."""
    if not sql or not sql.strip():
        return False, "Empty SQL query."

    stripped = sql.strip()
    if stripped.count(";") > 1 or (stripped.count(";") == 1 and not stripped.endswith(";")):
        return False, "Multiple statements detected; only a single SELECT query is permitted."

    try:
        parsed = sqlglot.parse(stripped, read="duckdb")
    except Exception as exc:
        return False, f"SQL parse error: {exc}"

    if not parsed or len(parsed) != 1 or parsed[0] is None:
        return False, "Failed to parse a single SQL statement."

    root = parsed[0]
    if not isinstance(root, (exp.Select, exp.Union)):
        return (
            False,
            f"Forbidden root statement '{type(root).__name__}'; "
            "only SELECT/UNION queries are allowed.",
        )

    for node in root.walk():
        node_cls_name = type(node).__name__
        if node_cls_name in _FORBIDDEN_NODE_NAMES:
            return False, f"Forbidden SQL AST operation detected: {node_cls_name}"
        if isinstance(node, exp.Anonymous) and node.name.lower() in FORBIDDEN_KEYWORDS:
            return False, f"Forbidden SQL function/command: {node.name}"

    return True, "ok"


def extract_sql_ast_features(sql: str) -> dict[str, Any]:
    """Extract referenced tables, columns, and filter/group flags from a DuckDB SQL query."""
    try:
        parsed = sqlglot.parse_one(sql, read="duckdb")
    except Exception:
        return {"tables": set(), "columns": set(), "has_filter_or_group": False}

    cte_names: set[str] = set()
    for cte in parsed.find_all(exp.CTE):
        alias = cte.alias_or_name
        if alias:
            cte_names.add(alias.lower())

    tables: set[str] = set()
    for tbl in parsed.find_all(exp.Table):
        tbl_name = tbl.name.lower()
        if tbl_name and tbl_name not in cte_names:
            tables.add(tbl_name)

    columns: set[str] = set()
    for col in parsed.find_all(exp.Column):
        col_name = col.name.lower()
        if col_name:
            columns.add(col_name)

    has_filter_or_group = any(
        isinstance(node, (exp.Where, exp.Group, exp.Having, exp.Join)) for node in parsed.walk()
    )

    return {
        "tables": tables,
        "columns": columns,
        "has_filter_or_group": has_filter_or_group,
    }
