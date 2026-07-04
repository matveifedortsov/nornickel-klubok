"""Экспорт ответов в PDF/Markdown/JSON-LD (§7 плана).

Формат ответов уже структурирован (Answer/LiteratureReview из
klubok/qa/answer.py) — экспорт это чистое форматирование поверх них, никакой
новой бизнес-логики. Принимает любой объект с полями text/sources и
question ИЛИ topic (duck typing — не завязываемся на конкретный класс).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_FONTS_DIR = Path(__file__).with_name("fonts")
_FONT_REGISTERED: str | None = None       # кэш имени зарегистрированного шрифта


def _ensure_cyrillic_font() -> tuple[str, str]:
    """Зарегистрировать кириллический шрифт в reportlab (идемпотентно).

    Возвращает (regular_name, bold_name). Дефолтные шрифты reportlab (Helvetica)
    НЕ содержат кириллицы — русский текст в PDF выходил бы пустым. Бандлим
    DejaVuSans в klubok/export/fonts, чтобы работало и локально, и в Docker.
    При отсутствии файла — тихий откат на Helvetica (лучше латиница, чем краш).
    """
    global _FONT_REGISTERED
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if _FONT_REGISTERED == "dejavu":
        return "DejaVuSans", "DejaVuSans-Bold"
    if _FONT_REGISTERED == "helvetica":
        return "Helvetica", "Helvetica-Bold"

    reg, bold = _FONTS_DIR / "DejaVuSans.ttf", _FONTS_DIR / "DejaVuSans-Bold.ttf"
    try:
        pdfmetrics.registerFont(TTFont("DejaVuSans", str(reg)))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(bold if bold.exists() else reg)))
        pdfmetrics.registerFontFamily("DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold")
        _FONT_REGISTERED = "dejavu"
        return "DejaVuSans", "DejaVuSans-Bold"
    except Exception:                      # noqa: BLE001 — шрифта нет/битый
        _FONT_REGISTERED = "helvetica"
        return "Helvetica", "Helvetica-Bold"


def _title(result: Any) -> str:
    return getattr(result, "question", None) or getattr(result, "topic", None) or "Ответ"


def to_markdown(result: Any) -> str:
    lines = [f"# {_title(result)}", "", result.text, ""]
    if result.sources:
        lines.append("## Источники")
        lines.extend(f"- {s}" for s in result.sources)
    return "\n".join(lines)


def to_json_ld(result: Any) -> dict:
    return {
        "@context": {
            "@vocab": "https://schema.org/",
            "klubok": "https://nornickel-hackathon.local/klubok#",
        },
        "@type": "klubok:Answer",
        "name": _title(result),
        "text": result.text,
        "dateCreated": datetime.now(timezone.utc).isoformat(),
        "citation": [{"@type": "CreativeWork", "identifier": s} for s in result.sources],
        "klubok:edgesUsed": getattr(result, "edges_used", None),
        "klubok:passagesUsed": getattr(result, "passages_used", None),
    }


def to_pdf(result: Any) -> bytes:
    """PDF из текста ответа (reportlab — чистый Python, без системных зависимостей)."""
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    regular, bold = _ensure_cyrillic_font()
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    # переопределяем шрифт стилей на кириллический
    for name, font in (("Title", bold), ("Heading2", bold), ("BodyText", regular)):
        styles[name].fontName = font

    story = [Paragraph(_title(result), styles["Title"]), Spacer(1, 12)]
    for para in result.text.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.replace("\n", "<br/>"), styles["BodyText"]))
            story.append(Spacer(1, 8))
    if result.sources:
        story.append(Paragraph("Источники", styles["Heading2"]))
        for s in result.sources:
            story.append(Paragraph(f"- {s}", styles["BodyText"]))

    doc.build(story)
    return buf.getvalue()
