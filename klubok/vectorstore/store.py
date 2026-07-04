"""Qdrant-обёртка для семантического поиска по чанкам.

Qdrant поднимается в docker без GPU. Эмбеддинги может считать MockEmbedder,
поэтому весь путь «чанк -> вектор -> поиск» проверяется до выделения железа.

Есть также InMemoryVectorStore — для тестов без поднятого Qdrant.
"""
from __future__ import annotations

import uuid

import numpy as np

from config import settings
from klubok.ontology import Chunk
from klubok.vectorstore.embeddings import Embedder


class QdrantStore:
    def __init__(self, embedder: Embedder, collection: str | None = None,
                 url: str | None = None) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.embedder = embedder
        self.collection = collection or settings.qdrant_collection
        # local -> встроенный Qdrant в файле (без сервера/докера); server -> внешний
        if settings.qdrant_mode == "local":
            self._client = self._open_local(QdrantClient)
        else:
            self._client = QdrantClient(url=url or settings.qdrant_url)
        self._VectorParams = VectorParams
        self._Distance = Distance

    @staticmethod
    def _open_local(QdrantClient):
        """Открыть встроенный Qdrant, сняв stale `.lock` от жёстко убитого процесса.

        Встроенный режим — строго однопроцессный (API ЛИБО ingest, никогда
        вместе). Поэтому если при старте видим lock — предыдущий владелец мёртв,
        и lock устарел: пробуем открыть, при ошибке блокировки удаляем `.lock` и
        повторяем один раз. Для server-режима это неактуально.
        """
        from pathlib import Path
        path = Path(settings.qdrant_path)
        try:
            return QdrantClient(path=str(path))
        except Exception as exc:                          # noqa: BLE001
            if "already accessed" not in str(exc) and "lock" not in str(exc).lower():
                raise
            lock = path / ".lock"
            if lock.exists():
                import warnings
                warnings.warn(f"Снимаю stale Qdrant-lock {lock} (владелец мёртв).")
                lock.unlink(missing_ok=True)
            return QdrantClient(path=str(path))

    def close(self) -> None:
        """Корректно закрыть клиент (в local-режиме снимает файловый lock)."""
        try:
            self._client.close()
        except Exception:                                 # noqa: BLE001
            pass

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self._client.collection_exists(self.collection)
        # Смена эмбеддера меняет размерность (bge=1024, yandex=256, fastembed=384).
        # Несовпадение — ГРОМКАЯ ошибка, а не авто-пересоздание: раньше здесь
        # молча удалялась вся коллекция (часы ингеста и квоты эмбеддера) из-за
        # одной строчки EMBEDDER_BACKEND. Пересоздание — только явным recreate=True.
        if exists and not recreate:
            info = self._client.get_collection(self.collection)
            existing_dim = info.config.params.vectors.size
            if existing_dim != self.embedder.dim:
                raise RuntimeError(
                    f"Коллекция '{self.collection}' имеет dim={existing_dim}, а текущий "
                    f"эмбеддер даёт dim={self.embedder.dim}. Вероятно, сменился "
                    f"EMBEDDER_BACKEND. Верните прежний бэкенд или пересоздайте коллекцию "
                    f"явно: python scripts/reindex_vectors.py (старые векторы удалятся).")
        if exists and recreate:
            self._client.delete_collection(self.collection)
            exists = False
        if not exists:
            self._client.create_collection(
                self.collection,
                vectors_config=self._VectorParams(
                    size=self.embedder.dim, distance=self._Distance.COSINE),
            )

    def index_chunks(self, chunks: list[Chunk], batch: int = 128) -> int:
        from qdrant_client.models import PointStruct

        n = 0
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            vecs = self.embedder.encode([c.text for c in part], kind="doc")
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vecs[j].tolist(),
                    payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id,
                             "page": c.page, "text": c.text},
                )
                for j, c in enumerate(part)
            ]
            self._client.upsert(self.collection, points=points)
            n += len(points)
        return n

    def count(self) -> int:
        """Число точек в коллекции (0, если коллекции ещё нет)."""
        if not self._client.collection_exists(self.collection):
            return 0
        return int(self._client.count(self.collection, exact=True).count)

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        # qdrant-client >=1.10 убрал .search() в пользу .query_points()
        qv = self.embedder.encode_query(query).tolist()
        res = self._client.query_points(self.collection, query=qv, limit=top_k)
        return [{"score": p.score, **(p.payload or {})} for p in res.points]


class InMemoryVectorStore:
    """Лёгкая замена Qdrant для юнит-тестов: косинус по матрице в памяти."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._payloads: list[dict] = []
        self._matrix: np.ndarray | None = None

    def index_chunks(self, chunks: list[Chunk]) -> int:
        self._payloads = [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "page": c.page, "text": c.text}
            for c in chunks
        ]
        self._matrix = self.embedder.encode([c.text for c in chunks], kind="doc")
        return len(chunks)

    def count(self) -> int:
        return len(self._payloads)

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        if self._matrix is None or len(self._payloads) == 0:
            return []
        qv = self.embedder.encode_query(query)
        scores = self._matrix @ qv
        order = np.argsort(-scores)[:top_k]
        return [{"score": float(scores[i]), **self._payloads[i]} for i in order]
