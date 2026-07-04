"""Батч-ингест отобранного сабсета корпуса (§3 плана) через реальный пайплайн.

Читает список файлов (по одному пути на строку, см. scripts/select_corpus.py),
парсит, извлекает через LLM/эмбеддер и пишет в Neo4j+Qdrant. Один битый файл
(скан без текстового слоя, повреждённый docx и т.п.) НЕ должен ронять весь
прогон — обязательное требование ТЗ «надёжность». Ошибки логируются и
пропускаются с ретраями; прогресс чекпоинтится построчно, чтобы длинный
прогон можно было прервать (Ctrl+C) и продолжить с того же места.

Запуск:
    python scripts/ingest_corpus.py --list scripts/corpus_subset.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import _bootstrap  # noqa: F401,E402  (путь к пакету + UTF-8 вывод)

from klubok.pipeline import build_stores, ingest_document, get_parser

log = logging.getLogger("ingest_corpus")


def _load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _append_checkpoint(path: Path, file_path: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(file_path + "\n")


def ingest_corpus(file_list: Path, checkpoint: Path, report: Path,
                  retries: int = 2, retry_delay: float = 5.0) -> dict:
    files = [Path(l) for l in file_list.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = _load_checkpoint(checkpoint)

    from klubok.extraction.extract_cache import ExtractCache
    extract_cache = ExtractCache()      # кэш сырых LLM-ответов -> ретраи не жгут квоту

    client, store = build_stores()
    summary = {"ok": 0, "failed": 0, "skipped": 0, "entities": 0, "relations": 0, "errors": []}

    try:
        for path in files:
            key = str(path)
            if key in done:
                summary["skipped"] += 1
                continue

            parser = get_parser(key)
            if parser is None:
                log.warning("Нет парсера для %s, пропуск", path)
                summary["failed"] += 1
                summary["errors"].append({"file": key, "error": "unsupported extension"})
                continue

            last_exc: Exception | None = None
            for attempt in range(1, retries + 2):
                try:
                    doc = parser(path)
                    result = ingest_document(doc, client, store, extract_cache=extract_cache)
                    # сбой векторной индексации = файл НЕ готов: без этого он попадал
                    # в checkpoint и навсегда оставался без векторов (0 failed в отчёте)
                    if result.get("index_error"):
                        raise RuntimeError(f"векторная индексация: {result['index_error']}")
                    summary["ok"] += 1
                    summary["entities"] += result.get("entities", 0)
                    summary["relations"] += result.get("relations", 0)
                    _append_checkpoint(checkpoint, key)
                    log.info("OK  %s  entities=%s relations=%s", key,
                             result.get("entities"), result.get("relations"))
                    last_exc = None
                    break
                except Exception as exc:                          # noqa: BLE001
                    last_exc = exc
                    log.warning("Попытка %d/%d не удалась для %s: %s",
                                attempt, retries + 1, key, exc)
                    if attempt <= retries:
                        time.sleep(retry_delay)

            if last_exc is not None:
                summary["failed"] += 1
                summary["errors"].append({"file": key, "error": str(last_exc)})
                log.error("FAIL %s: %s", key, last_exc)
    finally:
        client.close()
        store.close()
        extract_cache.close()
        report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Готово: ok=%d failed=%d skipped=%d -> %s",
                  summary["ok"], summary["failed"], summary["skipped"], report)

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True, help="Файл со списком путей (см. select_corpus.py)")
    ap.add_argument("--checkpoint", default="scripts/ingest_checkpoint.txt")
    ap.add_argument("--report", default="scripts/ingest_report.json")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry-delay", type=float, default=5.0)
    args = ap.parse_args()

    ingest_corpus(Path(args.list), Path(args.checkpoint), Path(args.report),
                 retries=args.retries, retry_delay=args.retry_delay)


if __name__ == "__main__":
    main()
