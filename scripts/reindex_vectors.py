"""Переиндексация векторного слоя Qdrant — БЕЗ LLM и БЕЗ квоты.

Граф (Neo4j) уже наполнен; здесь только заново считаем эмбеддинги чанков и
кладём в Qdrant текущим эмбеддером (get_embedder — fastembed/локально). Не
трогает граф, не вызывает генерацию. Нужен, когда сменили эмбеддер (напр. с
Yandex на fastembed) или квота эмбеддингов недоступна.

chunk_id детерминированы (parse_path), поэтому совпадают с провенансом в графе.

Запуск:
    python scripts/reindex_vectors.py --list runtime/pilot.txt
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import _bootstrap  # noqa: F401,E402  (путь к пакету + UTF-8 вывод)

from klubok.pipeline import parse_path
from klubok.vectorstore.embeddings import get_embedder
from klubok.vectorstore.store import QdrantStore

log = logging.getLogger("reindex_vectors")


def reindex(file_list: Path, recreate: bool = True) -> dict:
    files = [l for l in file_list.read_text(encoding="utf-8").splitlines() if l.strip()]
    store = QdrantStore(embedder=get_embedder())
    # смена эмбеддера меняет размерность -> пересоздаём коллекцию
    store.ensure_collection(recreate=recreate)

    total_chunks, ok, failed = 0, 0, 0
    try:
        for path in files:
            try:
                for doc in parse_path(path):
                    if not doc.chunks:
                        continue
                    n = store.index_chunks(doc.chunks)
                    total_chunks += n
                    log.info("OK  %s  чанков=%d", Path(path).name, n)
                ok += 1
            except Exception as exc:                      # noqa: BLE001
                failed += 1
                log.error("FAIL %s: %s", path, exc)
    finally:
        store.close()

    summary = {"files_ok": ok, "files_failed": failed, "chunks_indexed": total_chunks}
    log.info("Готово: %s", summary)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True, help="Файл со списком путей")
    ap.add_argument("--keep", action="store_true", help="не пересоздавать коллекцию")
    args = ap.parse_args()
    t0 = time.time()
    reindex(Path(args.list), recreate=not args.keep)
    log.info("время: %.1fс", time.time() - t0)


if __name__ == "__main__":
    main()
