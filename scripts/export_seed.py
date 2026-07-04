"""Экспорт наполненного графа + векторов в переносимый seed (для деплоя).

Проблема деплоя: у жюри `docker compose up` поднимает ПУСТЫЕ Neo4j/Qdrant —
демо без данных. Решение: коммитим компактный дамп текущего графа и векторов в
`seed/`, а при старте API он автоматически загружается в пустую БД
(см. klubok/pipeline.py::seed_if_empty). Так `docker compose up` сразу даёт
рабочее демо, без ингеста и без Yandex-квоты.

Форматы (JSONL, построчно — устойчиво к обрыву):
  seed/nodes.jsonl   {"labels": [...], "props": {...}}
  seed/edges.jsonl   {"src": cid, "stype": label, "rel": type, "dst": cid, "dtype": label, "props": {...}}
  seed/vectors.jsonl {"id": ..., "vector": [...], "payload": {...}}

Запуск (при поднятых Neo4j/Qdrant с данными):
    python scripts/export_seed.py
"""
from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401,E402

from config import settings
from klubok.graph.neo4j_client import Neo4jClient
from klubok.vectorstore.store import QdrantStore
from klubok.vectorstore.embeddings import get_embedder

SEED = Path("seed")


def export_graph(client: Neo4jClient) -> tuple[int, int]:
    SEED.mkdir(exist_ok=True)
    # узлы
    n = 0
    with (SEED / "nodes.jsonl").open("w", encoding="utf-8") as f:
        for r in client.run("MATCH (n) RETURN labels(n) AS labels, properties(n) AS props"):
            f.write(json.dumps({"labels": r["labels"], "props": r["props"]},
                               ensure_ascii=False) + "\n")
            n += 1
    # рёбра (по canonical_id концов)
    e = 0
    with (SEED / "edges.jsonl").open("w", encoding="utf-8") as f:
        for r in client.run(
            "MATCH (a)-[rel]->(b) WHERE a.canonical_id IS NOT NULL AND b.canonical_id IS NOT NULL "
            "RETURN a.canonical_id AS src, labels(a)[0] AS stype, type(rel) AS rel, "
            "b.canonical_id AS dst, labels(b)[0] AS dtype, properties(rel) AS props"
        ):
            f.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
            e += 1
    return n, e


def export_vectors(store: QdrantStore) -> int:
    cnt = 0
    with (SEED / "vectors.jsonl").open("w", encoding="utf-8") as f:
        offset = None
        while True:
            points, offset = store._client.scroll(
                store.collection, limit=256, offset=offset,
                with_vectors=True, with_payload=True)
            for p in points:
                f.write(json.dumps({"id": str(p.id), "vector": list(p.vector),
                                    "payload": p.payload}, ensure_ascii=False) + "\n")
                cnt += 1
            if offset is None:
                break
    # запомним размерность для sanity-check при импорте
    (SEED / "meta.json").write_text(
        json.dumps({"embedding_dim": store.embedder.dim,
                    "collection": store.collection}, ensure_ascii=False),
        encoding="utf-8")
    return cnt


def main() -> None:
    client = Neo4jClient()
    store = QdrantStore(embedder=get_embedder())
    try:
        n, e = export_graph(client)
        v = export_vectors(store)
        print(f"seed экспортирован: узлов={n} рёбер={e} векторов={v} -> {SEED}/")
    finally:
        store.close()
        client.close()


if __name__ == "__main__":
    main()
