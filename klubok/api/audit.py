"""Аудит запросов к API (§7 плана): кто/что/когда.

Минимальная реализация — файл-лог, а не отдельная БД. Граф не место для
логов доступа (это не факт предметной области), полноценная система аудита —
overkill для объёма хакатона. Формат — по строке JSON на запись (легко
парсить/грепать при разборе инцидента).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import settings

log = logging.getLogger("audit")


def log_request(role: str, method: str, path: str, params: dict) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "method": method,
        "path": path,
        "params": params,
    }
    try:
        p = settings.audit_log_path
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning("Не удалось записать аудит-лог: %s", exc)
