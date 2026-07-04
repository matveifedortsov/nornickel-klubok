"""Реранкинг результатов графового обхода по релевантности к вопросу.

Проблема: рёбра подграфа приходят в порядке обхода (APOC) и обрезаются по
LIMIT — без учёта того, насколько ребро relevantно вопросу. Векторные пассажи
уже ранжированы эмбеддером, а рёбра — нет.

Решение: пере-ранжируем рёбра по косинусной близости их текстового
представления к вопросу тем же локальным эмбеддером (bi-encoder rerank, без
torch/квоты). Это позволяет тянуть обходом ШИРЕ (recall), а в контекст LLM
оставлять только top-K самых релевантных рёбер (precision).

Полноценный cross-encoder — апгрейд на будущее (нужен ONNX-реранкер); bi-encoder
уже заметно лучше «порядка обхода».
"""
from __future__ import annotations

import hashlib
import numpy as np

# In-memory кэш эмбеддингов текстов рёбер. Граф статичен между запросами, рёбра
# повторяются -> эмбеддим каждое ребро один раз (fastembed на CPU медленный).
# Ключ = sha1(текст ребра). Простая ограниченная map (демо-масштаб).
_EDGE_EMB: dict[str, np.ndarray] = {}
_EDGE_EMB_MAX = 20000


def _edge_text(e: dict) -> str:
    return f"{e.get('src','')} {e.get('rel','')} {e.get('dst','')} {e.get('evidence') or ''}".strip()


def _embed_cached(texts: list[str], embedder) -> np.ndarray:
    """Эмбеддинг с in-memory кэшем: считаем только новые тексты."""
    # Сброс переполненного кэша — ДО вычисления промахов: clear() между
    # miss_idx и финальной сборкой стирал ключи-хиты текущего батча -> KeyError
    # (и реранк молча отключался через blanket except в rerank_edges).
    if len(_EDGE_EMB) > _EDGE_EMB_MAX:
        _EDGE_EMB.clear()
    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    miss_idx = [i for i, k in enumerate(keys) if k not in _EDGE_EMB]
    if miss_idx:
        fresh = embedder.encode([texts[i] for i in miss_idx], kind="doc")
        for j, i in enumerate(miss_idx):
            _EDGE_EMB[keys[i]] = fresh[j]
    return np.vstack([_EDGE_EMB[k] for k in keys])


def rerank_edges(query: str, edges: list[dict], embedder, top_k: int = 40) -> list[dict]:
    """Отсортировать рёбра по релевантности вопросу, оставить top_k.

    Не фатально: при сбое эмбеддера возвращаем исходный порядок (обрезанный).
    Эмбеддер переиспользуется из vector_store (уже загружен) — без новых моделей.
    Эмбеддинги рёбер кэшируются в памяти (повторные запросы почти бесплатны).
    """
    if not edges or len(edges) == 1:
        return edges
    try:
        mat = _embed_cached([_edge_text(e) for e in edges], embedder)       # (N, d), L2-норм
        q = embedder.encode_query(query)                                    # (d,)
        scores = mat @ q
        order = np.argsort(-scores)
        ranked = []
        for i in order[:top_k]:
            e = dict(edges[int(i)])
            e["rerank_score"] = round(float(scores[int(i)]), 4)
            ranked.append(e)
        return ranked
    except Exception:                                     # noqa: BLE001 — эмбеддер недоступен
        return edges[:top_k]
