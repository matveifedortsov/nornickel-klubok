"""Дашборд-метрики для руководителя проекта (§5 плана).

Прямое требование ТЗ: «метрики покрытия знаний по направлениям, активность
команд, зоны риска (темы с малым числом источников или противоречивыми
данными)». «Зоны риска» переиспользует паттерн
klubok/graph/gaps.py::SPARSE_PROPERTIES, обобщённый на Property и Process.
"""
from __future__ import annotations

from klubok.graph.neo4j_client import Neo4jClient

COVERAGE_BY_DOMAIN = """
MATCH (n)
WHERE n.domain IS NOT NULL AND (n:Publication OR n:Experiment)
RETURN n.domain AS domain, labels(n)[0] AS type, count(*) AS count
ORDER BY domain, type
"""

FACILITY_ACTIVITY = """
MATCH (f:Facility)<-[:AFFILIATED_WITH]-(ex:Expert)<-[:AUTHORED_BY]-(pub:Publication)
RETURN f.name AS facility, count(DISTINCT ex) AS experts, count(DISTINCT pub) AS publications
ORDER BY publications DESC
LIMIT $limit
"""

RISK_ZONES = """
MATCH (topic)<-[:MEASURES|EXHIBITS|APPLIES]-(x)
WHERE topic:Property OR topic:Process
WITH topic, count(DISTINCT x) AS coverage
WHERE coverage <= $threshold
RETURN labels(topic)[0] AS type, topic.name AS name, coverage
ORDER BY coverage ASC
LIMIT $limit
"""


def coverage_by_domain(client: Neo4jClient) -> list[dict]:
    return [dict(r) for r in client.run(COVERAGE_BY_DOMAIN)]


def facility_activity(client: Neo4jClient, limit: int = 20) -> list[dict]:
    return [dict(r) for r in client.run(FACILITY_ACTIVITY, limit=limit)]


def risk_zones(client: Neo4jClient, threshold: int = 2, limit: int = 30) -> list[dict]:
    return [dict(r) for r in client.run(RISK_ZONES, threshold=threshold, limit=limit)]


def dashboard_report(client: Neo4jClient) -> dict:
    return {
        "coverage_by_domain": coverage_by_domain(client),
        "facility_activity": facility_activity(client),
        "risk_zones": risk_zones(client),
    }
