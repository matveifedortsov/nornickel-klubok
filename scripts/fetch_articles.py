"""Сбор дополнительных статей из ОТКРЫТЫХ источников под темы экспертов.

Зачем: качество ответов упирается в наполнение графа. Скрипт скачивает
open-access PDF по 10 темам экспертных вопросов (обессоливание, шахтные воды,
электроэкстракция, техногенный гипс, SO2, штейн/шлак, Pb-Zn, ...) и готовит
список файлов для scripts/ingest_corpus.py.

Источники (только легальные, open access):
  * CyberLeninka  — русскоязычные научные статьи (CC BY), есть JSON API.
  * DOAJ          — агрегатор open-access журналов (MDPI, Springer OA, Wiley OA);
                    отдаёт прямые ссылки на полные тексты.

Запуск:
    python scripts/fetch_articles.py                      # всё, по 5 статей на запрос
    python scripts/fetch_articles.py --per-query 8 --topics шахтные,гипс
    python scripts/ingest_corpus.py --list data/openaccess/files.txt

Вежливость: 1 запрос/сек, свой User-Agent, повторный запуск не перекачивает
уже скачанное (skip по имени файла).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import _bootstrap  # noqa: F401,E402
import requests

log = logging.getLogger("fetch_articles")

OUT_DIR = Path("data/openaccess")
# браузерный UA: DOAJ (Cloudflare) отдаёт 403 на непохожие на браузер клиенты
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept": "application/json, application/pdf, */*",
}
PAUSE_S = 1.0          # вежливый rate-limit на все внешние запросы

# Темы = 10 экспертных вопросов. RU-запросы идут в CyberLeninka, EN — в DOAJ.
TOPICS: dict[str, dict[str, list[str]]] = {
    "обессоливание": {
        "ru": ["обессоливание воды обогатительная фабрика",
               "обратный осмос шахтная вода"],
        "en": ["mine water desalination reverse osmosis"],
    },
    "шахтные-воды": {
        "ru": ["очистка шахтных вод цветная металлургия"],
        "en": ["mine water treatment non-ferrous metallurgy"],
    },
    "католит": {
        "ru": ["электроэкстракция никеля катод электролит"],
        "en": ["nickel electrowinning catholyte circulation"],
    },
    "электролиз-ванны": {
        "ru": ["электролитическое рафинирование никеля меди ванна"],
        "en": ["electrolyte flow electrowinning cell design"],
    },
    "техногенный-гипс": {
        "ru": ["техногенный гипс переработка фосфогипс"],
        "en": ["phosphogypsum processing utilization"],
    },
    "закачка-вод": {
        "ru": ["закачка шахтных вод глубокие горизонты"],
        "en": ["deep well injection mine water"],
    },
    "закладка": {
        "ru": ["закладка выработанного пространства отходы угольной промышленности"],
        "en": ["coal waste backfill mined-out area"],
    },
    "SO2": {
        "ru": ["удаление диоксида серы отходящие газы металлургия"],
        "en": ["SO2 removal smelter off-gas desulfurization"],
    },
    "штейн-шлак": {
        "ru": ["распределение платиновых металлов штейн шлак"],
        "en": ["precious metals distribution matte slag nickel copper"],
    },
    "свинец-цинк": {
        "ru": ["переработка свинцово-цинкового сырья"],
        "en": ["lead zinc concentrate processing hydrometallurgy"],
    },
}


def _slug(s: str, maxlen: int = 80) -> str:
    s = re.sub(r"<[^>]+>", "", s)          # выдача CyberLeninka подсвечивает <b>…</b>
    s = re.sub(r"[^\w\s-]", "", s.lower()).strip()
    return re.sub(r"[\s_-]+", "-", s)[:maxlen] or "article"


def _download_pdf(url: str, dest: Path) -> bool:
    """Скачать PDF; False — если не PDF/ошибка. Существующий файл не перекачиваем."""
    if dest.exists() and dest.stat().st_size > 0:
        log.info("skip (есть): %s", dest.name)
        return True
    try:
        time.sleep(PAUSE_S)
        resp = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)
        ctype = resp.headers.get("Content-Type", "")
        if resp.status_code != 200 or ("pdf" not in ctype and not resp.content[:4] == b"%PDF"):
            log.warning("не PDF (%s, %s): %s", resp.status_code, ctype[:40], url)
            return False
        dest.write_bytes(resp.content)
        log.info("OK  %s (%.0f КБ)", dest.name, len(resp.content) / 1024)
        return True
    except requests.RequestException as exc:
        log.warning("сбой скачивания %s: %s", url, exc)
        return False


def fetch_cyberleninka(query: str, topic_dir: Path, limit: int) -> int:
    """Поиск в CyberLeninka (open access, CC BY) + скачивание PDF."""
    try:
        time.sleep(PAUSE_S)
        resp = requests.post("https://cyberleninka.ru/api/search",
                             json={"mode": "articles", "q": query, "size": limit, "from": 0},
                             headers=HEADERS, timeout=30)
        resp.raise_for_status()
        arts = resp.json().get("articles", [])
    except (requests.RequestException, ValueError) as exc:
        log.warning("CyberLeninka недоступна для %r: %s", query, exc)
        return 0
    n = 0
    for a in arts[:limit]:
        link = a.get("link")            # вида /article/n/<slug>
        name = a.get("name") or "article"
        if not link:
            continue
        pdf_url = f"https://cyberleninka.ru{link}/pdf"
        if _download_pdf(pdf_url, topic_dir / f"{_slug(name)}.pdf"):
            n += 1
    return n


def fetch_doaj(query: str, topic_dir: Path, limit: int) -> int:
    """Поиск в DOAJ (агрегатор open access: MDPI, Springer OA и др.)."""
    try:
        time.sleep(PAUSE_S)
        resp = requests.get(
            f"https://doaj.org/api/v3/search/articles/{requests.utils.quote(query)}",
            params={"pageSize": limit}, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except (requests.RequestException, ValueError) as exc:
        log.warning("DOAJ недоступен для %r: %s", query, exc)
        return 0
    n = 0
    for r in results[:limit]:
        bib = r.get("bibjson", {})
        title = bib.get("title") or "article"
        pdf_links = [l.get("url") for l in bib.get("link", [])
                     if l.get("url") and (l.get("content_type") == "PDF"
                                          or str(l.get("url")).lower().endswith(".pdf"))]
        # fallback: любая fulltext-ссылка (издатели часто отдают PDF по ней)
        links = pdf_links or [l.get("url") for l in bib.get("link", [])
                              if l.get("type") == "fulltext" and l.get("url")]
        for url in links[:1]:
            if _download_pdf(url, topic_dir / f"{_slug(title)}.pdf"):
                n += 1
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-query", type=int, default=5, help="статей на один запрос")
    ap.add_argument("--topics", default="", help="фильтр тем через запятую (подстроки)")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out_root = Path(args.out)
    wanted = [t.strip() for t in args.topics.split(",") if t.strip()]
    total = 0
    for topic, queries in TOPICS.items():
        if wanted and not any(w in topic for w in wanted):
            continue
        topic_dir = out_root / topic
        topic_dir.mkdir(parents=True, exist_ok=True)
        for q in queries.get("ru", []):
            total += fetch_cyberleninka(q, topic_dir, args.per_query)
        for q in queries.get("en", []):
            total += fetch_doaj(q, topic_dir, args.per_query)

    # список файлов для батч-ингеста (формат scripts/ingest_corpus.py)
    files = sorted(str(p) for p in out_root.rglob("*.pdf") if p.stat().st_size > 0)
    list_path = out_root / "files.txt"
    list_path.write_text("\n".join(files) + "\n", encoding="utf-8")
    log.info("Итого скачано/имеется: %d PDF; список для ингеста: %s (%d файлов)",
             total, list_path, len(files))
    log.info("Дальше: python scripts/ingest_corpus.py --list %s", list_path)


if __name__ == "__main__":
    main()
