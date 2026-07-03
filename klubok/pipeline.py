"""Высокоуровневый фасад: связывает все слои в два сценария — ingest и query.

Бэкенды (LLM/эмбеддер) выбираются конфигом, поэтому один и тот же код
работает и на Mock (без GPU), и на MetalGPT+BGE (с железом).
"""
from __future__ import annotations

import logging

from klubok.ontology import Document
from klubok.parsing.pdf_parser import parse_pdf
from klubok.parsing.docx_parser import parse_docx
from klubok.parsing.pptx_parser import parse_pptx
from klubok.parsing.filename_meta import parse_filename_meta
from klubok.extraction.llm_client import get_llm
from klubok.extraction.extractor import extract_from_chunks
from klubok.graph.neo4j_client import Neo4jClient
from klubok.graph.ingest import ingest_all, upsert_document, ingest_authorship
from klubok.vectorstore.embeddings import get_embedder
from klubok.vectorstore.store import QdrantStore
from klubok.retrieval.graphrag import retrieve
from klubok.qa.answer import generate_answer, Answer

log = logging.getLogger(__name__)

# Расширение файла -> функция парсинга с контрактом (path, max_chars) -> Document
PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".pptx": parse_pptx,
}


def get_parser(path: str):
    """Парсер по расширению файла или None, если формат не поддержан."""
    from pathlib import Path
    return PARSERS.get(Path(path).suffix.lower())


def build_stores(recreate: bool = False) -> tuple[Neo4jClient, QdrantStore]:
    client = Neo4jClient()
    client.apply_schema()
    store = QdrantStore(embedder=get_embedder())
    store.ensure_collection(recreate=recreate)
    return client, store


def ingest_document(doc: Document, client: Neo4jClient, store: QdrantStore,
                    watch_store=None) -> dict:
    """Полный ingestion одного документа: граф + вектор.

    `watch_store` (klubok.notify.watchlist.WatchStore) — если передан, новые
    сущности матчатся против подписок и порождают уведомления (§Y7). Опционален,
    чтобы оффлайн-демо/тесты не тянули SQLite.
    """
    llm = get_llm()
    upsert_document(client, doc)

    # авторство/лаборатория из имени файла — дешёвый точный сигнал, не через LLM
    if doc.source_path:
        ingest_authorship(client, doc, parse_filename_meta(doc.source_path))

    results = extract_from_chunks(doc.chunks, llm)
    graph_counts = ingest_all(client, results, doc=doc)
    indexed = store.index_chunks(doc.chunks)

    notified = 0
    if watch_store is not None:
        names = sorted({e.name for r in results for e in r.entities})
        notified = watch_store.notify_new_document(doc.doc_id, doc.title or doc.doc_id, names)

    log.info("doc=%s entities=%d relations=%d chunks=%d notify=%d",
             doc.doc_id, graph_counts["entities"], graph_counts["relations"], indexed, notified)
    return {"doc_id": doc.doc_id, **graph_counts, "chunks_indexed": indexed,
            "notifications": notified}


def parse_path(path: str) -> list[Document]:
    """Распарсить один файл или папку — расширение определяет парсер (§2)."""
    from pathlib import Path
    p = Path(path)
    if p.is_file():
        parser = get_parser(str(p))
        if parser is None:
            raise ValueError(f"Нет парсера для расширения {p.suffix!r}: {p}")
        return [parser(p)]
    docs: list[Document] = []
    for ext, parser in PARSERS.items():
        for f in sorted(p.rglob(f"*{ext}")):
            docs.append(parser(f))
    return docs


def ingest_path(path: str, client: Neo4jClient, store: QdrantStore,
                watch_store=None) -> list[dict]:
    """Заингестить один файл (PDF/DOCX/PPTX) или папку с ними."""
    docs = parse_path(path)
    return [ingest_document(d, client, store, watch_store=watch_store) for d in docs]


def answer_question(question: str, client: Neo4jClient, store: QdrantStore,
                    geography: bool | None = "auto", domain: str | None = None) -> Answer:
    ctx = retrieve(question, store, client, geography=geography, domain=domain)
    return generate_answer(ctx, get_llm())
