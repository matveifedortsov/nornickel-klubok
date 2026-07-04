"""Рекомендации: похожие кейсы и эксперты по теме (§5 плана).

Прямое требование ТЗ: «похожие кейсы и потенциально применимые решения из
смежных областей», «эксперты и команды, которые работали с аналогичными
задачами». Переиспользует паттерн klubok/graph/gaps.py::ANALOGY_GAPS
(структурная аналогия через общий признак), обобщённый на произвольный узел.
"""
from __future__ import annotations

from klubok.graph.neo4j_client import Neo4jClient

# Похожие материалы: общий элемент состава — обобщение ANALOGY_GAPS без
# требования "у соседа есть свойство, которого нет у нас" (это для gaps.py),
# здесь просто «похожие по составу кейсы» для раздела рекомендаций.
SIMILAR_MATERIALS = """
MATCH (m1 {canonical_id: $cid})-[:HAS_COMPOSITION]->(el:Element)<-[:HAS_COMPOSITION]-(m2)
WHERE m1 <> m2
RETURN DISTINCT m2.name AS similar, m2.canonical_id AS canonical_id, el.name AS shared_element
LIMIT $limit
"""

# Похожие процессы: применялись в экспериментах с тем же материалом или
# похожим условием (Condition) — «смежная область» через общий контекст.
SIMILAR_PROCESSES = """
MATCH (p1 {canonical_id: $cid})<-[:APPLIES]-(:Experiment)-[:USES]->(m:Material)
MATCH (m)<-[:USES]-(:Experiment)-[:APPLIES]->(p2:Process)
WHERE p1 <> p2
RETURN DISTINCT p2.name AS similar, p2.canonical_id AS canonical_id, m.name AS shared_material
LIMIT $limit
"""

EXPERTS_BY_TOPIC = """
CALL db.index.fulltext.queryNodes('entity_names', $topic) YIELD node AS topic_node, score
MATCH (topic_node)<-[:EXPERT_IN]-(ex:Expert)
OPTIONAL MATCH (ex)<-[:AUTHORED_BY]-(pub:Publication)
WITH ex, topic_node, count(DISTINCT pub) AS publications
RETURN ex.name AS expert, topic_node.name AS topic, publications
ORDER BY publications DESC
LIMIT $limit
"""


def similar_materials(client: Neo4jClient, canonical_id: str, limit: int = 20) -> list[dict]:
    return [dict(r) for r in client.run(SIMILAR_MATERIALS, cid=canonical_id, limit=limit)]


def similar_processes(client: Neo4jClient, canonical_id: str, limit: int = 20) -> list[dict]:
    return [dict(r) for r in client.run(SIMILAR_PROCESSES, cid=canonical_id, limit=limit)]


# Фолбэк: эксперты по активности (числу публикаций). Нужен, когда контент ещё
# не извлечён (нет EXPERT_IN-связей), но узлы Expert созданы из имён файлов —
# показываем самых публикующихся авторов, а не пустой список.
EXPERTS_BY_ACTIVITY = """
MATCH (ex:Expert)<-[:AUTHORED_BY]-(pub:Publication)
WITH ex, count(DISTINCT pub) AS publications
OPTIONAL MATCH (ex)-[:AFFILIATED_WITH]->(f:Facility)
RETURN ex.name AS expert, null AS topic, publications,
       collect(DISTINCT f.name) AS facilities
ORDER BY publications DESC
LIMIT $limit
"""


def experts_by_topic(client: Neo4jClient, topic: str, limit: int = 20) -> list[dict]:
    try:
        rows = [dict(r) for r in client.run(EXPERTS_BY_TOPIC, topic=topic, limit=limit)]
    except Exception:                                     # noqa: BLE001 — индекса может не быть
        rows = []
    if rows:
        return rows
    # по теме ничего (нет EXPERT_IN) -> отдаём экспертов по активности
    try:
        rows = [dict(r) for r in client.run(EXPERTS_BY_ACTIVITY, limit=limit)]
        for r in rows:
            r["by_activity"] = True          # пометка для UI: не тематический матч
        return rows
    except Exception:                                     # noqa: BLE001
        return []
