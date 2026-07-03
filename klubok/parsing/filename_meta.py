"""Метаданные автора/лаборатории из имени файла — без LLM.

В части корпуса («Статьи», «Доклады») автор и аббревиатура лаборатории
закодированы прямо в имени файла (напр. «26 Статья - Великая Т.И. (ИАЦ).docx»,
«Доклад_Вострикова Н.М.pdf») — дешёвый и точный сигнал, на который LLM обычно
тратит токены и чаще ошибается, чем regex по структурированному имени.

Эвристика, не полный NER: покрывает частый паттерн «Фамилия И.О.» (кириллица)
и аббревиатуру лаборатории в скобках. Имена вида "KorzhakovAA" (латиница, без
пробелов) не разбираются — для них авторство остаётся на совести LLM-извлечения
из текста статьи.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# "Фамилия И.О." / "Фамилия И. О." — кириллица. Последняя точка опциональна:
# Path.stem режет по ПОСЛЕДНЕЙ точке в имени файла, поэтому в "...Н.М.pdf" её
# съедает разделитель расширения и до regex доходит уже "...Н.М" без хвоста.
_AUTHOR_RE = re.compile(r"([А-ЯЁ][а-яё]+)\s+([А-ЯЁ])\.\s?([А-ЯЁ])\.?")

# аббревиатура лаборатории/подразделения в скобках: (ИАЦ), (ЛПМ), (ЛГМ) ...
_LAB_RE = re.compile(r"\(([А-ЯA-Z]{2,8})\)")


@dataclass(frozen=True)
class FilenameMeta:
    author_name: Optional[str] = None
    lab_abbr: Optional[str] = None


def parse_filename_meta(path: str | Path) -> FilenameMeta:
    """Разобрать имя файла на (автор, лаборатория). Оба поля опциональны."""
    stem = Path(path).stem

    lab = None
    m = _LAB_RE.search(stem)
    if m:
        lab = m.group(1)

    author = None
    m = _AUTHOR_RE.search(stem)
    if m:
        author = f"{m.group(1)} {m.group(2)}.{m.group(3)}."

    return FilenameMeta(author_name=author, lab_abbr=lab)
