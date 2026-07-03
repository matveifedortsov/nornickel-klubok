"""Отбор файлов реального корпуса под первый прогон MetalGPT-1 (§3 плана).

Гнать весь корпус (1453 файла, 4.9 ГБ) через LLM нереально по времени/деньгам.
Берём компактный, но содержательный сабсет:
  - «Статьи» (внутренние, авторство закодировано в имени файла — сигнал для
    Expert/Facility без LLM, см. klubok/parsing/filename_meta.py)
  - «Обзоры» (литературные обзоры по конкретным темам)
  - «Доклады» (презентации сотрудников, их немного — 16 файлов)
  - по N последних номеров каждого журнала (не весь архив 2003-2026)

Явно ИСКЛЮЧАЕМ «Материалы конференций/Источники данных о *» (рыночная
аналитика CRU/Brook Hunt/ICSG/Metal Bulletin — цены/производство металлов,
не технологии) и всё, что не PDF/DOCX/PPTX (архивы zip/rar вне контракта
парсеров — распаковка и разбор архивов не входит в эту версию пайплайна).

Запуск:
    python scripts/select_corpus.py \
        --root "Задача 2. Научный клубок/Задача 2. Научный клубок/Источники информации" \
        --out scripts/corpus_subset.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401,E402  (путь к пакету + UTF-8 вывод)

SUPPORTED_EXT = {".pdf", ".docx", ".pptx"}
FULL_INCLUDE_DIRS = ("Статьи", "Обзоры", "Доклады")
EXCLUDE_MARKERS = ("Источники данных о",)


def _latest_journal_issues(journals_dir: Path, n_per_journal: int) -> list[Path]:
    """По каждому журналу (папка внутри «Журналы») — N самых свежих файлов.

    Сортировка по (имя папки-года, имя файла) — устойчиво для схемы корпуса
    «Журналы/<название>/<год>/<номер>.pdf», не зависит от дат mtime файлов.
    """
    out: list[Path] = []
    if not journals_dir.is_dir():
        return out
    for journal_dir in sorted(p for p in journals_dir.iterdir() if p.is_dir()):
        files: list[Path] = []
        for ext in SUPPORTED_EXT:
            files.extend(journal_dir.rglob(f"*{ext}"))
        files.sort(key=lambda p: (p.parent.name, p.name))
        out.extend(files[-n_per_journal:] if n_per_journal > 0 else files)
    return out


def select_corpus(root: Path, n_per_journal: int = 2) -> list[Path]:
    selected: list[Path] = []

    for name in FULL_INCLUDE_DIRS:
        d = root / name
        if not d.is_dir():
            continue
        for ext in SUPPORTED_EXT:
            selected.extend(d.rglob(f"*{ext}"))

    selected.extend(_latest_journal_issues(root / "Журналы", n_per_journal))

    # страховка: только поддержанные форматы и явно не рыночная аналитика
    selected = [
        p for p in selected
        if p.suffix.lower() in SUPPORTED_EXT and not any(m in str(p) for m in EXCLUDE_MARKERS)
    ]
    return sorted(set(selected))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Папка «Источники информации» корпуса")
    ap.add_argument("--out", default="scripts/corpus_subset.txt")
    ap.add_argument("--n-per-journal", type=int, default=2,
                     help="Сколько последних номеров брать с каждого журнала (0 = все)")
    args = ap.parse_args()

    files = select_corpus(Path(args.root), args.n_per_journal)
    Path(args.out).write_text("\n".join(str(p) for p in files), encoding="utf-8")
    print(f"Отобрано {len(files)} файлов -> {args.out}")


if __name__ == "__main__":
    main()
