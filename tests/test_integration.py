"""Интеграционный smoke-тест: реальные Neo4j + Qdrant, но Mock LLM/эмбеддер.

Проверяет сквозной путь ingest -> граф+вектор -> ask на живых хранилищах, БЕЗ
сети/квоты (Mock-бэкенды). Полностью изолирован от демо-данных:
  * Qdrant — во ВРЕМЕННОЙ папке (tmp_path), отдельная коллекция;
  * Neo4j — уникальный тестовый документ, всё созданное удаляется в finally.

Пропускается по умолчанию (нужен запущенный Neo4j). Запуск:
    RUN_INTEGRATION=1 pytest -m integration -q
"""
from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.integration

_SKIP = os.environ.get("RUN_INTEGRATION") != "1"


@pytest.mark.skipif(_SKIP, reason="нужен живой Neo4j+Qdrant; задайте RUN_INTEGRATION=1")
def test_ingest_and_ask_end_to_end(tmp_path, monkeypatch):
    from config import settings
    # изоляция: Mock-бэкенды + временный Qdrant + отдельная коллекция
    monkeypatch.setattr(settings, "llm_backend", "mock")
    monkeypatch.setattr(settings, "embedder_backend", "mock")
    monkeypatch.setattr(settings, "qdrant_mode", "local")
    monkeypatch.setattr(settings, "qdrant_path", tmp_path / "qd")
    monkeypatch.setattr(settings, "qdrant_collection", "klubok_itest")

    from klubok.graph.neo4j_client import Neo4jClient
    from klubok.vectorstore.store import QdrantStore
    from klubok.vectorstore.embeddings import get_embedder
    from klubok.ontology import Document, Chunk
    from klubok.pipeline import ingest_document, answer_question

    doc_id = f"ITEST_{uuid.uuid4().hex[:8]}"
    doc = Document(
        doc_id=doc_id, title="Интеграционный тест",
        chunks=[Chunk(chunk_id=f"{doc_id}:c0", doc_id=doc_id,
                      text="Образцы медно-никелевого сплава CuNi отжигали при 800 C в "
                           "течение двух часов. После термообработки микротвёрдость по "
                           "Виккерсу выросла на 20 процентов относительно исходного "
                           "состояния, что подтверждено измерениями.")],
    )

    client = Neo4jClient()
    store = QdrantStore(embedder=get_embedder())
    store.ensure_collection(recreate=True)
    try:
        res = ingest_document(doc, client, store)
        assert res["chunks_indexed"] >= 1
        # Publication записана
        n = client.run("MATCH (p:Publication {canonical_id:$c}) RETURN count(p) AS n",
                       c=f"Publication:{doc_id}")[0]["n"]
        assert n == 1
        # сквозной ответ строится (Mock LLM, но путь ретривала реальный)
        ans = answer_question("что за материал?", client, store)
        assert ans.text and isinstance(ans.sources, list)
    finally:
        # полная очистка тестовых данных, чтобы не засорять граф
        client.run("MATCH (p:Publication {canonical_id:$c}) DETACH DELETE p",
                   c=f"Publication:{doc_id}")
        client.run("MATCH ()-[r]->() WHERE r.doc_id=$d DELETE r", d=doc_id)
        store.close()
        client.close()
