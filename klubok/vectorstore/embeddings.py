"""Эмбеддеры с общим интерфейсом.

Контракт (важно для асимметричных моделей Yandex):
    encode(texts, kind="doc"|"query") -> np.ndarray[N, dim]
    encode_query(text) -> np.ndarray[dim]     # 1D-вектор запроса

  * MockEmbedder   — детерминированные векторы из хэша n-грамм (без сети).
  * BGEEmbedder    — BAAI/bge-m3 (симметричная, kind игнорируется).
  * YandexEmbedder — Yandex AI Studio textEmbedding: РАЗНЫЕ модели для документов
                     (text-search-doc) и запросов (text-search-query), dim=256,
                     один текст на HTTP-запрос. Оборачивать в CachedEmbedder.

Переключение — settings.embedder_backend (mock | bge | yandex).
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from config import settings
from klubok.extraction.llm_client import RateLimiter

# Троттлинг эмбеддингов Yandex (общий на все экземпляры). Квота эмбеддингов
# обычно выше генерации, поэтому отдельный, более быстрый rps.
_emb_rate_limiter = RateLimiter(settings.yandex_emb_rps)


class Embedder(Protocol):
    dim: int
    def encode(self, texts: list[str], kind: str = "doc") -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norm, 1e-9, None)


class MockEmbedder:
    """Хэш-эмбеддер на character n-grams. Не для качества — для проводки пайплайна."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.embedding_dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"\w+", text.lower())
        grams = tokens + [a + "_" + b for a, b in zip(tokens, tokens[1:])]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return v

    def encode(self, texts: list[str], kind: str = "doc") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalize(np.vstack([self._vec(t) for t in texts]))

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text], kind="query")[0]


class BGEEmbedder:
    """BAAI/bge-m3 — двуязычный, симметричный (kind не влияет)."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name or settings.embedder_model)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], kind: str = "doc") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
            dtype=np.float32,
        )

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text], kind="query")[0]


class YandexEmbedder:
    """Yandex AI Studio textEmbedding (dim=256, асимметричные doc/query).

    Особенности API: один текст на запрос, нет батча -> для массового ингеста
    ОБЯЗАТЕЛЬНО оборачивать в CachedEmbedder + ThreadPoolExecutor на стороне
    ingest-скрипта. Авторизация — 'Authorization: Api-Key <key>'.
    """

    remote = True   # 1 текст = 1 троттлированный HTTP-вызов: массовые encode
                    # (реранк рёбер и т.п.) на таком эмбеддере недопустимы

    def __init__(self) -> None:
        import requests
        if not settings.yandex_api_key or not settings.yandex_folder_id:
            raise RuntimeError(
                "Не заданы YANDEX_API_KEY / YANDEX_FOLDER_ID в .env — "
                "YandexEmbedder недоступен. См. PLAN_FINAL.md §Y3.")
        self._requests = requests
        self.dim = settings.yandex_embedding_dim
        self._url = settings.yandex_emb_url
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {settings.yandex_api_key}",
            "x-folder-id": settings.yandex_folder_id,
        }
        folder = settings.yandex_folder_id
        self._doc_uri = f"emb://{folder}/{settings.yandex_emb_doc_model}"
        self._query_uri = f"emb://{folder}/{settings.yandex_emb_query_model}"

    def _embed_one(self, text: str, kind: str) -> np.ndarray:
        import random
        import time
        model_uri = self._query_uri if kind == "query" else self._doc_uri
        payload = {"modelUri": model_uri, "text": text[:8000]}
        last_exc: Exception | None = None
        for attempt in range(settings.yandex_emb_max_retries + 1):
            try:
                _emb_rate_limiter.wait()      # троттлинг под квоту (burst index_chunks)
                resp = self._requests.post(self._url, headers=self._headers,
                                           json=payload, timeout=settings.yandex_timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"retryable HTTP {resp.status_code}: {resp.text[:150]}")
                resp.raise_for_status()
                vec = np.asarray(resp.json()["embedding"], dtype=np.float32)
                return _l2_normalize(vec[None, :])[0]
            except Exception as exc:                          # noqa: BLE001
                last_exc = exc
                if attempt < settings.yandex_emb_max_retries:
                    time.sleep(min(2 ** attempt, 8) + random.uniform(0, 0.5))
        raise RuntimeError(f"Yandex textEmbedding не удался: {last_exc}")

    def encode(self, texts: list[str], kind: str = "doc") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(t, kind) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed_one(text, "query")


class FastEmbedEmbedder:
    """Локальный эмбеддер на ONNX-runtime (fastembed) — БЕЗ torch и БЕЗ квоты.

    Нужен там, где sentence-transformers/torch падает (segfault на новых Python)
    или недоступна облачная квота. Мультиязычная модель (RU/EN). Модель ~120МБ
    качается один раз в кэш fastembed.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from fastembed import TextEmbedding
        kw = {"cache_dir": settings.fastembed_cache} if settings.fastembed_cache else {}
        self._model = TextEmbedding(model_name or settings.fastembed_model, **kw)
        # определяем размерность пробным эмбеддингом
        self.dim = int(np.asarray(next(iter(self._model.embed(["probe"])))).shape[0])

    def encode(self, texts: list[str], kind: str = "doc") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        gen = self._model.query_embed(texts) if kind == "query" else self._model.embed(texts)
        arr = np.asarray(list(gen), dtype=np.float32)
        return _l2_normalize(arr)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text], kind="query")[0]


def get_embedder() -> Embedder:
    backend = settings.embedder_backend
    if backend == "bge":
        try:
            return BGEEmbedder()
        except ImportError as exc:
            # sentence-transformers вынесен в опциональные ML-зависимости —
            # без подсказки голый ModuleNotFoundError на старте API нечитаем
            raise RuntimeError(
                "EMBEDDER_BACKEND=bge требует sentence-transformers: "
                "pip install -r requirements-ml.txt (или используйте fastembed)") from exc
    if backend == "fastembed":
        return FastEmbedEmbedder()
    if backend == "yandex":
        from klubok.vectorstore.emb_cache import CachedEmbedder
        return CachedEmbedder(YandexEmbedder())
    return MockEmbedder()
