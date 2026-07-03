"""Сравнительный анализ технологий/материалов — «вариант А vs Б» (§5 плана).

Строит таблицу параметр × вариант из измеренных Property/Condition двух узлов
графа (обычно Process или Material). Не про пробелы (klubok/graph/gaps.py),
а про сопоставление того, что уже известно по обеим сторонам — прямое
требование ТЗ («сравнительный анализ технологий по параметрам: эффективность,
CAPEX, применимость в холодном климате, экологические ограничения»).
"""
from __future__ import annotations

from klubok.graph.neo4j_client import Neo4jClient

_PARAMS_QUERY = """
MATCH (n {canonical_id: $cid})-[:EXHIBITS|MEASURES|RESULTS_IN|OPERATES_AT_CONDITION|PRODUCES_OUTPUT]->(p)
RETURN p.name AS param, p.value AS value, p.unit AS unit
"""


def _params_for(client: Neo4jClient, canonical_id: str) -> dict[str, dict]:
    rows = client.run(_PARAMS_QUERY, cid=canonical_id)
    return {r["param"]: {"value": r["value"], "unit": r["unit"]} for r in rows}


def _fmt(entry: dict | None) -> str:
    if not entry or entry.get("value") is None:
        return "—"
    unit = f" {entry['unit']}" if entry.get("unit") else ""
    return f"{entry['value']}{unit}"


def compare(client: Neo4jClient, cid_a: str, cid_b: str,
           label_a: str = "Вариант А", label_b: str = "Вариант Б") -> list[dict]:
    """Таблица сравнения: строка на параметр, столбцы — значения по вариантам.

    `cid_a`/`cid_b` — canonical_id узлов (обычно 'Process:...' или
    'Material:...'), см. klubok/extraction/resolver.py::canonical_id.
    """
    a = _params_for(client, cid_a)
    b = _params_for(client, cid_b)
    rows = []
    for param in sorted(set(a) | set(b)):
        rows.append({"parameter": param, label_a: _fmt(a.get(param)), label_b: _fmt(b.get(param))})
    return rows
