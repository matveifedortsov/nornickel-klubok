"""Дисковый кэш эмбеддингов поверх любого Embedder.

Мотивация: Yandex textEmbedding платный и медленный (1 текст/запрос). Ингест
корпуса — тысячи вызовов; при повторном/прерванном прогоне не хотим платить и
ждать снова. Ключ кэша = sha1(kind + '\\x00' + text), значение = float32-байты.

Реализует тот же интерфейс Embedder, поэтому прозрачно подставляется в
QdrantStore/InMemoryVectorStore. Потокобезопасен (SQLite + короткие транзакции,
проверка same-thread отключена, запись под локом).
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import numpy as np

from config import settings
from klubok.vectorstore.embeddings import Embedder


def _key(text: str, kind: str) -> str:
    return hashlib.sha1(f"{kind}\x00{text}".encode("utf-8")).hexdigest()


class CachedEmbedder:
    def __init__(self, inner: Embedder, path: str | Path | None = None) -> None:
        self.inner = inner
        self.dim = inner.dim
        self.remote = getattr(inner, "remote", False)   # прозрачно для потребителей
        self._path = Path(path or settings.emb_cache_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, dim INT, v BLOB)")
        self._conn.commit()

    def _get(self, k: str) -> np.ndarray | None:
        cur = self._conn.execute("SELECT dim, v FROM emb WHERE k=?", (k,))
        row = cur.fetchone()
        if row is None:
            return None
        dim, blob = row
        return np.frombuffer(blob, dtype=np.float32).reshape(dim)

    def _put(self, k: str, vec: np.ndarray) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO emb (k, dim, v) VALUES (?, ?, ?)",
                (k, int(vec.shape[0]), vec.astype(np.float32).tobytes()))
            self._conn.commit()

    def encode(self, texts: list[str], kind: str = "doc") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[np.ndarray | None] = [None] * len(texts)
        missing_idx, missing_txt = [], []
        for i, t in enumerate(texts):
            cached = self._get(_key(t, kind))
            if cached is None:
                missing_idx.append(i)
                missing_txt.append(t)
            else:
                out[i] = cached
        if missing_txt:
            fresh = self.inner.encode(missing_txt, kind=kind)
            for j, i in enumerate(missing_idx):
                out[i] = fresh[j]
                self._put(_key(texts[i], kind), fresh[j])
        return np.vstack(out)

    def encode_query(self, text: str) -> np.ndarray:
        k = _key(text, "query")
        cached = self._get(k)
        if cached is not None:
            return cached
        vec = self.inner.encode_query(text)
        self._put(k, vec)
        return vec

    def close(self) -> None:
        self._conn.close()
