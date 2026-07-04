"""Ингест авторства из имён файлов — Expert/Facility БЕЗ LLM и эмбеддингов.

Папка «Статьи» кодирует автора и лабораторию прямо в имени файла
(«26 Статья - Великая Т.И. (ИАЦ).docx»). Создаём узлы Publication + Expert +
Facility и связи AUTHORED_BY/AFFILIATED_WITH напрямую — нулевая квота, наполняет
фичу ТЗ «эксперты/лаборатории по теме» (/experts) без разбора содержимого.

Контент этих файлов можно доизвлечь позже, когда будет квота генерации.

Запуск:
    python scripts/ingest_authorship.py --root "data/.../Статьи"
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401,E402  (путь к пакету + UTF-8 вывод)

from klubok.ontology import Document
from klubok.parsing.pdf_parser import _doc_id
from klubok.parsing.filename_meta import parse_filename_meta
from klubok.graph.neo4j_client import Neo4jClient
from klubok.graph.ingest import upsert_document, ingest_authorship

log = logging.getLogger("ingest_authorship")
SUPPORTED = {".pdf", ".docx", ".pptx"}


def run(root: Path, is_domestic: bool = True) -> dict:
    files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in SUPPORTED]
    client = Neo4jClient()
    experts, facilities, docs = set(), set(), 0
    try:
        for path in files:
            meta = parse_filename_meta(str(path))
            doc = Document(doc_id=_doc_id(path), title=path.stem, source_path=str(path),
                           publication_type="статья", is_domestic=is_domestic,
                           geography="Россия" if is_domestic else None)
            upsert_document(client, doc)
            docs += 1
            if meta.author_name:
                ingest_authorship(client, doc, meta)
                experts.add(meta.author_name)
                if meta.lab_abbr:
                    facilities.add(meta.lab_abbr)
            log.info("OK %s | автор=%s лаб=%s", path.name[:40], meta.author_name, meta.lab_abbr)
    finally:
        client.close()
    summary = {"publications": docs, "experts": len(experts), "facilities": len(facilities)}
    log.info("Готово: %s", summary)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Папка со «Статьями»")
    args = ap.parse_args()
    run(Path(args.root))


if __name__ == "__main__":
    main()
