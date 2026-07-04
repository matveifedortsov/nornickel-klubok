"""Тонкая обёртка над официальным neo4j-драйвером.

Драйвер импортируется лениво, чтобы модуль можно было импортировать без БД
(например, в юнит-тестах чистой логики). Сам Neo4j поднимается в docker
без GPU — этот код можно запускать уже сейчас.
"""
from __future__ import annotations

from pathlib import Path

from config import settings


class Neo4jClient:
    def __init__(self, uri: str | None = None, user: str | None = None,
                 password: str | None = None) -> None:
        from neo4j import GraphDatabase
        # Заглушаем серверные нотификации (schema "already exists", "missing
        # property" при первом insert) — это ожидаемый шум, заваливающий логи
        # ингеста. Параметр появился в драйвере 5.7; на старых — тихий fallback.
        kwargs = dict(auth=(user or settings.neo4j_user, password or settings.neo4j_password))
        try:
            self._driver = GraphDatabase.driver(
                uri or settings.neo4j_uri,
                notifications_min_severity="OFF", **kwargs,
            )
        except (TypeError, ValueError):
            self._driver = GraphDatabase.driver(uri or settings.neo4j_uri, **kwargs)

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def run(self, query: str, **params):
        with self._driver.session() as session:
            return list(session.run(query, **params))

    def apply_schema(self, schema_path: str | Path | None = None) -> None:
        """Прогнать schema.cypher (по одному стейтменту)."""
        path = Path(schema_path) if schema_path else Path(__file__).with_name("schema.cypher")
        text = path.read_text(encoding="utf-8")
        for stmt in _split_statements(text):
            self.run(stmt)

    def wipe(self) -> None:
        """Очистить граф (удобно при повторных прогонах ingestion)."""
        self.run("MATCH (n) DETACH DELETE n")

    def count_nodes(self) -> int:
        return self.run("MATCH (n) RETURN count(n) AS c")[0]["c"]


def _split_statements(cypher_text: str) -> list[str]:
    """Разбить файл на отдельные стейтменты по ';', игнорируя // комментарии."""
    lines = [ln for ln in cypher_text.splitlines() if not ln.strip().startswith("//")]
    joined = "\n".join(lines)
    return [s.strip() for s in joined.split(";") if s.strip()]
