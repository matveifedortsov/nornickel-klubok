"""PDF -> Document(chunks). Гибкий парсер: текст по страницам + таблицы.

Корпус неизвестен до старта хакатона, поэтому НЕ хардкодим структуру —
бьём на чанки по абзацам с ограничением длины и тащим метаданные
(doc_id, страница) для будущих цитат.

Тяжёлые библиотеки импортируются лениво, чтобы модуль грузился без них.
Функция `chunk_text` — чистая, тестируется без PDF.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from klubok.ontology import Chunk, Document


def _doc_id(path: Path) -> str:
    h = hashlib.sha1(str(path).encode()).hexdigest()[:8]
    return f"{path.stem[:40]}_{h}"


def chunk_text(text: str, doc_id: str, page: int | None = None,
               max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    """Разбить текст на перекрывающиеся чанки по границам абзацев/предложений."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []

    # сперва по абзацам, затем добираем по длине
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 1 <= max_chars:
            buf = f"{buf}\n{p}".strip()
        else:
            if buf:
                chunks.append(buf)
            # абзац длиннее лимита — режем по предложениям
            if len(p) > max_chars:
                for piece in _split_long(p, max_chars, overlap):
                    chunks.append(piece)
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)

    out: list[Chunk] = []
    for i, c in enumerate(chunks):
        cid = f"{doc_id}:p{page if page is not None else 0}:c{i}"
        out.append(Chunk(chunk_id=cid, doc_id=doc_id, page=page, text=c))
    return out


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text)
    pieces, buf = [], ""
    for s in sents:
        if len(buf) + len(s) + 1 <= max_chars:
            buf = f"{buf} {s}".strip()
        else:
            if buf:
                pieces.append(buf)
            buf = (buf[-overlap:] + " " + s).strip() if overlap and buf else s
    if buf:
        pieces.append(buf)
    return pieces


def parse_pdf(path: str | Path, max_chars: int = 1200) -> Document:
    """Распарсить один PDF в Document. Требует pymupdf (ленивый импорт).

    Устойчив к битым страницам: сбой одной страницы не роняет весь файл
    (в реальном корпусе есть повреждённые/сканированные PDF). Скан без
    текстового слоя вернёт Document с пустыми/скудными chunks — дальше
    pipeline.ingest_document пропустит его до LLM (экономия квоты).
    """
    import logging
    import fitz  # PyMuPDF

    path = Path(path)
    doc_id = _doc_id(path)
    document = Document(doc_id=doc_id, title=path.stem, source_path=str(path))

    with fitz.open(path) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            try:
                text = page.get_text("text")
            except Exception as exc:                      # noqa: BLE001 — битая страница
                logging.getLogger(__name__).warning(
                    "PDF %s: пропуск страницы %d (%s)", path.name, page_no, exc)
                continue
            document.chunks.extend(chunk_text(text, doc_id, page=page_no, max_chars=max_chars))

    return document


def parse_dir(folder: str | Path, pattern: str = "*.pdf") -> list[Document]:
    folder = Path(folder)
    return [parse_pdf(p) for p in sorted(folder.glob(pattern))]


def extract_tables(path: str | Path) -> list[list[list[str]]]:
    """Таблицы свойств — отдельная ветка (часто там численные значения).

    Возвращает список таблиц, каждая — список строк, строка — список ячеек.
    Требует pdfplumber (ленивый импорт).
    """
    import pdfplumber

    tables: list[list[list[str]]] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            for tbl in page.extract_tables():
                tables.append([[(c or "").strip() for c in row] for row in tbl])
    return tables
