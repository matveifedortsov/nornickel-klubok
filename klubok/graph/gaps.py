"""Поиск пробелов в данных — прямое требование трека и главный «вау» на демо.

Каждая функция — это аналитический Cypher-запрос по структуре графа.
Запросы оформлены как константы, чтобы их можно было ревьюить/показывать
жюри отдельно от кода.
"""
from __future__ import annotations

from klubok.graph.neo4j_client import Neo4jClient


# Материалы, для которых применяли процессы, но НЕ измеряли ни одного свойства.
MATERIALS_WITHOUT_PROPERTIES = """
MATCH (m:Material)
WHERE NOT (m)-[:EXHIBITS]->(:Property)
  AND NOT (:Experiment)-[:USES]->(m)<-[:USES]-(:Experiment)-[:MEASURES]->(:Property)
RETURN m.canonical_id AS material, m.name AS name
ORDER BY name
LIMIT $limit
"""

# Пары (Материал, тип процесса), для которых нет данных об эффекте/свойстве.
MATERIAL_PROCESS_NO_EFFECT = """
MATCH (e:Experiment)-[:USES]->(m:Material)
MATCH (e)-[:APPLIES]->(p:Process)
WHERE NOT (e)-[:MEASURES]->(:Property)
  AND NOT (e)-[:RESULTS_IN]->(:Property)
RETURN m.name AS material, p.name AS process, count(*) AS experiments
ORDER BY experiments DESC
LIMIT $limit
"""

# Свойства с разреженным покрытием: измерены у очень малого числа материалов.
SPARSE_PROPERTIES = """
MATCH (prop:Property)<-[:MEASURES|EXHIBITS]-(x)
WITH prop, count(DISTINCT x) AS coverage
WHERE coverage <= $threshold
RETURN prop.name AS property, coverage
ORDER BY coverage ASC
LIMIT $limit
"""

# «Структурные дыры»: материалы, похожие по составу (общий элемент), у которых
# один обладает свойством, а у второго оно не измерено — кандидат на эксперимент.
ANALOGY_GAPS = """
MATCH (m1:Material)-[:HAS_COMPOSITION]->(el:Element)<-[:HAS_COMPOSITION]-(m2:Material)
WHERE m1 <> m2
MATCH (m1)-[:EXHIBITS]->(prop:Property)
WHERE NOT (m2)-[:EXHIBITS]->(prop)
RETURN m2.name AS material, prop.name AS missing_property,
       m1.name AS analog, el.name AS shared_element
LIMIT $limit
"""

# Изолированные узлы — индикатор проблем извлечения/резолвинга (для отладки качества).
ORPHAN_NODES = """
MATCH (n)
WHERE NOT (n)--() AND NOT n:Publication
RETURN labels(n)[0] AS type, n.name AS name
LIMIT $limit
"""

# Технологии/процессы, описанные ТОЛЬКО в отечественных ИЛИ ТОЛЬКО в
# зарубежных источниках — прямое требование ТЗ (гео-фильтр отечественная vs
# мировая практика).
ONLY_DOMESTIC_OR_FOREIGN = """
MATCH (p:Process)<-[:APPLIES|USES_MATERIAL]-(x)
WHERE x.is_domestic IS NOT NULL
WITH p, collect(DISTINCT x.is_domestic) AS flags
WHERE size(flags) = 1
RETURN p.name AS process, flags[0] AS is_domestic
ORDER BY process
LIMIT $limit
"""

# Комбинации материал-режим-условие без данных — обобщение материал/процесс
# на произвольную комбинацию через Condition.
UNSTUDIED_COMBINATIONS = """
MATCH (m:Material)<-[:USES]-(e:Experiment)-[:APPLIES]->(p:Process)
WHERE NOT (p)-[:OPERATES_AT_CONDITION]->(:Condition)
RETURN m.name AS material, p.name AS process, count(*) AS experiments
ORDER BY experiments DESC
LIMIT $limit
"""

# Явные противоречия (ребро CONTRADICTS) — для ручного разбора аналитиком.
CONTRADICTIONS = """
MATCH (a)-[r:CONTRADICTS]-(b)
RETURN labels(a)[0] AS type_a, a.name AS a, labels(b)[0] AS type_b, b.name AS b,
       r.evidence AS evidence
LIMIT $limit
"""

# Узлы с расходящимися версиями числового значения (см. graph/ingest.py:
# _merge_entity — value_history/has_conflicting_versions при повторном ингесте).
CONFLICTING_VERSIONS = """
MATCH (n)
WHERE n.has_conflicting_versions = true
RETURN labels(n)[0] AS type, n.name AS name, n.value AS current_value,
       n.unit AS unit, n.value_history AS history
LIMIT $limit
"""


def materials_without_properties(client: Neo4jClient, limit: int = 50) -> list[dict]:
    return [dict(r) for r in client.run(MATERIALS_WITHOUT_PROPERTIES, limit=limit)]


def material_process_no_effect(client: Neo4jClient, limit: int = 50) -> list[dict]:
    return [dict(r) for r in client.run(MATERIAL_PROCESS_NO_EFFECT, limit=limit)]


def sparse_properties(client: Neo4jClient, threshold: int = 2, limit: int = 50) -> list[dict]:
    return [dict(r) for r in client.run(SPARSE_PROPERTIES, threshold=threshold, limit=limit)]


def analogy_gaps(client: Neo4jClient, limit: int = 50) -> list[dict]:
    return [dict(r) for r in client.run(ANALOGY_GAPS, limit=limit)]


def orphan_nodes(client: Neo4jClient, limit: int = 50) -> list[dict]:
    return [dict(r) for r in client.run(ORPHAN_NODES, limit=limit)]


def only_domestic_or_foreign(client: Neo4jClient, limit: int = 50) -> list[dict]:
    return [dict(r) for r in client.run(ONLY_DOMESTIC_OR_FOREIGN, limit=limit)]


def unstudied_combinations(client: Neo4jClient, limit: int = 50) -> list[dict]:
    return [dict(r) for r in client.run(UNSTUDIED_COMBINATIONS, limit=limit)]


def contradictions(client: Neo4jClient, limit: int = 50) -> list[dict]:
    return [dict(r) for r in client.run(CONTRADICTIONS, limit=limit)]


def conflicting_versions(client: Neo4jClient, limit: int = 50) -> list[dict]:
    return [dict(r) for r in client.run(CONFLICTING_VERSIONS, limit=limit)]


def gap_report(client: Neo4jClient) -> dict[str, list[dict]]:
    """Сводный отчёт для дашборда UI."""
    return {
        "materials_without_properties": materials_without_properties(client),
        "material_process_no_effect": material_process_no_effect(client),
        "sparse_properties": sparse_properties(client),
        "analogy_gaps": analogy_gaps(client),
        "only_domestic_or_foreign": only_domestic_or_foreign(client),
        "unstudied_combinations": unstudied_combinations(client),
        "contradictions": contradictions(client),
        "conflicting_versions": conflicting_versions(client),
        "orphan_nodes": orphan_nodes(client, limit=20),
    }
