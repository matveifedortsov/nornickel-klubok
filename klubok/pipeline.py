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


def seed_if_empty(client: Neo4jClient, store: QdrantStore) -> bool:
    """Если граф пуст, а в репозитории есть seed — загрузить его.

    Делает `docker compose up` самодостаточным для жюри: демо с данными без
    ингеста и без Yandex-квоты. Вызывается из API lifespan при старте.
    Возвращает True, если посев выполнен.
    """
    from klubok.graph.seed import seed_exists, import_seed_graph, import_seed_vectors
    if not seed_exists():
        return False

    # Граф и векторы проверяются/сеются НЕЗАВИСИМО: раньше один try на оба шага
    # оставлял полузасеянное состояние (граф есть, векторов нет), которое
    # count_nodes()>0 навсегда исключало из повторного посева.
    seeded = False
    try:
        if client.count_nodes() == 0:
            log.info("граф пуст + найден seed -> загружаю seed-граф…")
            import_seed_graph(client)
            seeded = True
    except Exception as exc:                              # noqa: BLE001
        log.warning("seed-граф не загружен: %s", exc)

    try:
        if store.count() == 0:
            log.info("векторная коллекция пуста -> загружаю seed-векторы…")
            import_seed_vectors(store)
            seeded = True
    except Exception as exc:                              # noqa: BLE001
        log.warning("seed-векторы не загружены: %s", exc)

    if seeded:
        log.info("seed загружен, узлов теперь: %d", client.count_nodes())
    return seeded


def ingest_document(doc: Document, client: Neo4jClient, store: QdrantStore,
                    watch_store=None, extract_cache=None) -> dict:
    """Полный ingestion одного документа: граф + вектор.

    `watch_store` (klubok.notify.watchlist.WatchStore) — если передан, новые
    сущности матчатся против подписок и порождают уведомления (§Y7). Опционален,
    чтобы оффлайн-демо/тесты не тянули SQLite.
    `extract_cache` (klubok.extraction.extract_cache.ExtractCache) — если передан,
    сырые ответы LLM кэшируются: ретрай документа не переизвлекает уже готовые
    чанки (критично при малой квоте Yandex).
    """
    # required=True: при недоступном YandexGPT ингест обязан упасть, а не молча
    # заполнить граф заглушками MockLLM (см. get_llm).
    llm = get_llm(required=True)

    # доменно/гео эвристики по тексту документа (не через LLM) — заполняют
    # geography/is_domestic/domain, когда метаданные/LLM молчат (иначе гео-фильтр
    # и «модель верификации» из ТЗ пустые на реальном корпусе).
    from klubok.extraction.heuristics import detect_is_domestic, detect_domain
    full_text = " ".join(c.text for c in doc.chunks)[:20000]

    # скан/пустой документ (нет текстового слоя) — не гоним через LLM/эмбеддер,
    # только регистрируем узел Publication. Экономит квоту и не засоряет граф.
    if len(full_text.strip()) < 100:
        upsert_document(client, doc)
        if doc.source_path:
            ingest_authorship(client, doc, parse_filename_meta(doc.source_path))
        log.warning("doc=%s пропущен: нет текстового слоя (%d симв.)",
                    doc.doc_id, len(full_text.strip()))
        return {"doc_id": doc.doc_id, "entities": 0, "relations": 0,
                "chunks_indexed": 0, "notifications": 0, "skipped": "no_text"}

    if doc.is_domestic is None:
        doc.is_domestic, doc.geography = detect_is_domestic(full_text)
    if doc.domain is None:
        doc.domain = detect_domain(full_text)

    upsert_document(client, doc)

    # авторство/лаборатория из имени файла — дешёвый точный сигнал, не через LLM
    if doc.source_path:
        ingest_authorship(client, doc, parse_filename_meta(doc.source_path))

    results = extract_from_chunks(doc.chunks, llm, cache=extract_cache)

    # backfill связей/сущностей гео/доменом документа, где LLM оставил пусто —
    # чтобы структурный фильтр (graphrag) реально срабатывал на рёбрах/узлах.
    for res in results:
        for rel in res.relations:
            if rel.is_domestic is None:
                rel.is_domestic = doc.is_domestic
            if rel.geography is None:
                rel.geography = doc.geography
        for e in res.entities:
            if e.domain is None:
                e.domain = doc.domain
    graph_counts = ingest_all(client, results, doc=doc)

    # Векторная индексация НЕ фатальна: граф (ядро решения) уже записан выше.
    # Сбой эмбеддера (исчерпана квота, сеть) не должен обнулять работу по графу —
    # деградируем до «граф есть, векторов нет», НО ошибку возвращаем вызывающему
    # в поле index_error: батч-ингест обязан считать такой файл неуспешным (иначе
    # checkpoint навсегда исключит его из повторной индексации).
    index_error: str | None = None
    try:
        indexed = store.index_chunks(doc.chunks)
    except Exception as exc:                              # noqa: BLE001
        log.warning("doc=%s: векторная индексация пропущена (%s)", doc.doc_id, exc)
        indexed, index_error = 0, str(exc)

    notified = 0
    if watch_store is not None:
        names = sorted({e.name for r in results for e in r.entities})
        notified = watch_store.notify_new_document(doc.doc_id, doc.title or doc.doc_id, names)

    log.info("doc=%s entities=%d relations=%d chunks=%d notify=%d",
             doc.doc_id, graph_counts["entities"], graph_counts["relations"], indexed, notified)
    result = {"doc_id": doc.doc_id, **graph_counts, "chunks_indexed": indexed,
              "notifications": notified}
    if index_error:
        result["index_error"] = index_error
    return result


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
                    geography: bool | None = "auto", domain: str | None = None,
                    year_from: int | None = None, year_to: int | None = None) -> Answer:
    ctx = retrieve(question, store, client, geography=geography, domain=domain,
                   year_from=year_from, year_to=year_to)
    return generate_answer(ctx, get_llm())
