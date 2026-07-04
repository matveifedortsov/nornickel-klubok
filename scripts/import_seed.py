"""Загрузка seed (граф + векторы) в пустые Neo4j/Qdrant — для деплоя.

Идемпотентно (MERGE по canonical_id). Вызывается вручную ИЛИ автоматически при
старте API, если граф пуст (klubok/pipeline.py::seed_if_empty).

Запуск:
    python scripts/import_seed.py
"""
from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401,E402

from klubok.graph.neo4j_client import Neo4jClient
from klubok.vectorstore.store import QdrantStore
from klubok.vectorstore.embeddings import get_embedder
from klubok.graph.seed import import_seed_graph, import_seed_vectors, SEED_DIR


def main() -> None:
    if not (SEED_DIR / "nodes.jsonl").exists():
        print(f"seed не найден в {SEED_DIR}/ — нечего импортировать")
        return
    client = Neo4jClient()
    client.apply_schema()
    store = QdrantStore(embedder=get_embedder())
    try:
        n, e = import_seed_graph(client)
        v = import_seed_vectors(store)
        print(f"seed загружен: узлов={n} рёбер={e} векторов={v}")
    finally:
        store.close()
        client.close()


if __name__ == "__main__":
    main()
