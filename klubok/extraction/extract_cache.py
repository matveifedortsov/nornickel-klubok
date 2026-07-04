"""Дисковый кэш результатов извлечения (сырой JSON от LLM) по чанкам.

Мотивация: квота Yandex AI Studio мала (~10 запросов/окно), а батч-ингест
делает десятки LLM-вызовов на документ. При ретрае документа (падение на
середине из-за 429) или повторном прогоне НЕ хотим переплачивать квотой за
уже извлечённые чанки. Ключ = sha1(prompt_version + '\\x00' + chunk_text);
prompt_version меняем при правке промпта, чтобы кэш инвалидировался.

Тот же паттерн, что vectorstore/emb_cache.py. Потокобезопасно.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

from config import settings

# Версия промпта — хэш фактического содержимого prompts.py: любая правка
# системного промпта или шаблона автоматически инвалидирует кэш (ручной бамп
# константы забывался — кэш молча подсовывал извлечения старым промптом).
from klubok.extraction.prompts import EXTRACTION_SYSTEM, build_extraction_prompt

PROMPT_VERSION = hashlib.sha1(
    (EXTRACTION_SYSTEM + "\x00" + build_extraction_prompt("")).encode("utf-8")
).hexdigest()[:12]


def _key(chunk_text: str, prompt_version: str) -> str:
    return hashlib.sha1(f"{prompt_version}\x00{chunk_text}".encode("utf-8")).hexdigest()


class ExtractCache:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path or settings.extract_cache_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS extractions (k TEXT PRIMARY KEY, raw TEXT)")
        self._conn.commit()

    def get(self, chunk_text: str, prompt_version: str = PROMPT_VERSION) -> str | None:
        cur = self._conn.execute(
            "SELECT raw FROM extractions WHERE k=?", (_key(chunk_text, prompt_version),))
        row = cur.fetchone()
        return row[0] if row else None

    def put(self, chunk_text: str, raw: str, prompt_version: str = PROMPT_VERSION) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO extractions (k, raw) VALUES (?, ?)",
                (_key(chunk_text, prompt_version), raw))
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
