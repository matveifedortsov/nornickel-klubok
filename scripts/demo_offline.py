"""End-to-end демо БЕЗ GPU и БЕЗ баз данных.

Показывает, что вся логика пайплайна (кроме слоя Neo4j) работает на Mock'ах:
    парсинг текста -> извлечение -> резолвинг -> вектор-поиск -> ответ.

Запуск:
    python scripts/demo_offline.py

Когда появится железо: меняете в .env LLM_BACKEND=metalgpt, EMBEDDER_BACKEND=bge,
поднимаете docker compose — и тот же код работает на реальной модели и графе.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401,E402  (путь к пакету + UTF-8 вывод)

from klubok.ontology import Document
from klubok.parsing.pdf_parser import chunk_text
from klubok.extraction.llm_client import MockLLM
from klubok.extraction.extractor import extract_from_chunks
from klubok.extraction.resolver import resolve, group_aliases
from klubok.vectorstore.embeddings import MockEmbedder
from klubok.vectorstore.store import InMemoryVectorStore
from klubok.retrieval.graphrag import RetrievalContext
from klubok.qa.answer import generate_answer


SAMPLE = """
Образцы медно-никелевого сплава (CuNi) подвергали отжигу при 800 °C в течение 2 часов.
После термообработки твёрдость по Виккерсу выросла до 145 HV. Микроструктуру
исследовали методом SEM. Для сравнения сплав Cu-Ni того же состава при 1073 K
показал снижение твёрдости. Фазовый состав определяли методом XRD.
"""


def main() -> None:
    print("=== 1. Парсинг и чанкинг ===")
    doc = Document(doc_id="demo_doc", title="Демо-статья")
    doc.chunks = chunk_text(SAMPLE, doc_id="demo_doc", page=1, max_chars=400)
    print(f"чанков: {len(doc.chunks)}")

    print("\n=== 2. Извлечение триплетов (MockLLM) ===")
    results = extract_from_chunks(doc.chunks, MockLLM())
    all_entities = [e for r in results for e in r.entities]
    all_relations = [rel for r in results for rel in r.relations]
    for e in all_entities:
        print(f"  • {e.type.value}: {e.name}  {e.attributes or ''}")
    for rel in all_relations:
        print(f"  → {rel.src_name} -{rel.rel.value}-> {rel.dst_name}")

    print("\n=== 3. Резолвинг сущностей (дедуп) ===")
    resolved, _ = resolve(all_entities)
    print(f"до: {len(all_entities)}  ->  после дедупа: {len(resolved)}")
    aliases = group_aliases(all_entities)
    if aliases:
        print("схлопнутые алиасы:")
        for cid, names in aliases.items():
            print(f"  {cid}  <-  {names}")

    print("\n=== 4. Векторный поиск (MockEmbedder, в памяти) ===")
    store = InMemoryVectorStore(MockEmbedder(dim=512))
    store.index_chunks(doc.chunks)
    question = "какой эффект у отжига сплава CuNi на твёрдость"
    hits = store.search(question, top_k=2)
    for h in hits:
        print(f"  score={h['score']:.3f}  [{h['doc_id']}] {h['text'][:70]}…")

    print("\n=== 5. Генерация ответа (MockLLM, с цитатами) ===")
    ctx = RetrievalContext(question=question, passages=hits, subgraph_edges=[])
    ans = generate_answer(ctx, MockLLM())
    print("ВОПРОС:", ans.question)
    print("ОТВЕТ :", ans.text)
    print("ИСТОЧНИКИ:", ans.sources)

    print("\nОК — пайплайн проходит end-to-end без GPU и без БД.")


if __name__ == "__main__":
    main()
