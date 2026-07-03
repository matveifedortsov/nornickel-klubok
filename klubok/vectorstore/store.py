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
        self._client = QdrantClient(url=url or settings.qdrant_url)
        self._VectorParams = VectorParams
        self._Distance = Distance

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self._client.collection_exists(self.collection)
        # смена эмбеддера меняет размерность (bge=1024, yandex=256) — иначе upsert
        # упадёт с несовпадением. При несоответствии пересоздаём коллекцию.
        if exists and not recreate:
            info = self._client.get_collection(self.collection)
            existing_dim = info.config.params.vectors.size
            if existing_dim != self.embedder.dim:
                import warnings
                warnings.warn(
                    f"Коллекция '{self.collection}' имеет dim={existing_dim}, "
                    f"эмбеддер даёт dim={self.embedder.dim}. Пересоздаю коллекцию "
                    f"(старые векторы удаляются — переиндексируйте корпус).")
                recreate = True
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

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        qv = self.embedder.encode_query(query).tolist()
        hits = self._client.search(self.collection, query_vector=qv, limit=top_k)
        return [{"score": h.score, **h.payload} for h in hits]


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

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        if self._matrix is None or len(self._payloads) == 0:
            return []
        qv = self.embedder.encode_query(query)
        scores = self._matrix @ qv
        order = np.argsort(-scores)[:top_k]
        return [{"score": float(scores[i]), **self._payloads[i]} for i in order]
