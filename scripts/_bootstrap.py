"""Общий bootstrap для CLI-скриптов: путь к пакету + UTF-8 вывод.

Импортировать ПЕРВОЙ строкой (до импортов klubok/config):

    import _bootstrap  # noqa: F401

Делает две вещи:
1. Добавляет корень репозитория в sys.path, чтобы `python scripts/xxx.py`
   находил пакет `klubok` (иначе sys.path[0] — папка scripts/).
2. Переводит stdout/stderr в UTF-8 — на Windows консоль по умолчанию cp1251
   и падает на кириллице/эмодзи (UnicodeEncodeError). Нужно для демо жюри.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")   # type: ignore[union-attr]
    except Exception:                            # noqa: BLE001 — старый Python/перенаправление
        pass
