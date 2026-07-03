"""Тесты Yandex-слоя без сети: кэш эмбеддингов, ретраи/троттлинг LLM,
корректные ошибки при отсутствии ключей. HTTP замокан.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from klubok.vectorstore.embeddings import MockEmbedder
from klubok.vectorstore.emb_cache import CachedEmbedder
from klubok.extraction.llm_client import RateLimiter, OpenAICompatClient


# --------------------------------------------------------------------------
# CachedEmbedder — кэширование без повторных вызовов inner
# --------------------------------------------------------------------------
class _CountingEmbedder:
    """MockEmbedder + счётчик реально посчитанных текстов."""
    def __init__(self, dim: int = 64) -> None:
        self._inner = MockEmbedder(dim=dim)
        self.dim = dim
        self.calls: list[str] = []

    def encode(self, texts, kind="doc"):
        self.calls.extend(texts)
        return self._inner.encode(texts, kind=kind)

    def encode_query(self, text):
        self.calls.append(text)
        return self._inner.encode_query(text)


def test_cached_embedder_hits_cache(tmp_path):
    inner = _CountingEmbedder(dim=64)
    cache = CachedEmbedder(inner, path=tmp_path / "emb.sqlite")

    v1 = cache.encode(["alpha", "beta"], kind="doc")
    assert v1.shape == (2, 64)
    assert inner.calls == ["alpha", "beta"]      # оба посчитаны

    # второй запрос: "alpha" из кэша, только "gamma" считается заново
    inner.calls.clear()
    v2 = cache.encode(["alpha", "gamma"], kind="doc")
    assert v2.shape == (2, 64)
    assert inner.calls == ["gamma"]
    # закэшированный вектор идентичен исходному
    assert np.allclose(v1[0], v2[0])
    cache.close()


def test_cached_embedder_separates_doc_and_query(tmp_path):
    inner = _CountingEmbedder(dim=32)
    cache = CachedEmbedder(inner, path=tmp_path / "emb.sqlite")
    cache.encode(["text"], kind="doc")
    inner.calls.clear()
    # тот же текст, но kind=query -> другой ключ -> считается заново
    cache.encode_query("text")
    assert inner.calls == ["text"]
    cache.close()


# --------------------------------------------------------------------------
# RateLimiter — соблюдение минимального интервала
# --------------------------------------------------------------------------
def test_rate_limiter_enforces_interval():
    rl = RateLimiter(rps=50)          # 20 мс между вызовами
    t0 = time.monotonic()
    for _ in range(3):
        rl.wait()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.02 * 2 * 0.8  # 2 интервала с запасом на планировщик


def test_rate_limiter_zero_rps_noop():
    rl = RateLimiter(rps=0)
    t0 = time.monotonic()
    rl.wait(); rl.wait()
    assert time.monotonic() - t0 < 0.05


# --------------------------------------------------------------------------
# OpenAICompatClient — ретраи на 429 и проброс ответа
# --------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, status: int, content: str = "") -> None:
        self.status_code = status
        self.text = content
        self._content = content

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeRequests:
    def __init__(self, responses: list[_FakeResp]) -> None:
        self._responses = responses
        self.calls = 0

    def post(self, *a, **kw) -> _FakeResp:
        r = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return r


def _make_client(fake: _FakeRequests, monkeypatch) -> OpenAICompatClient:
    # без sleep, чтобы тест был быстрым
    monkeypatch.setattr("klubok.extraction.llm_client.time.sleep", lambda *_: None)
    c = OpenAICompatClient(base_url="http://x/v1", model="m",
                           auth_header={"Authorization": "Api-Key k"}, max_retries=3)
    c._requests = fake
    return c


def test_retry_then_success(monkeypatch):
    fake = _FakeRequests([_FakeResp(429, "rate"), _FakeResp(200, "готово")])
    c = _make_client(fake, monkeypatch)
    out = c.complete("привет")
    assert out == "готово"
    assert fake.calls == 2          # один ретрай


def test_retry_exhausted_raises(monkeypatch):
    fake = _FakeRequests([_FakeResp(503, "down")])
    c = _make_client(fake, monkeypatch)
    with pytest.raises(RuntimeError):
        c.complete("привет")
    assert fake.calls == 4          # 1 + 3 ретрая


def test_non_retryable_4xx_raises_immediately(monkeypatch):
    fake = _FakeRequests([_FakeResp(400, "bad request")])
    c = _make_client(fake, monkeypatch)
    with pytest.raises(Exception):
        c.complete("привет")
    assert fake.calls == 1          # 400 не ретраится


# --------------------------------------------------------------------------
# Отсутствие ключей -> понятная ошибка (не падение где-то в сети)
# --------------------------------------------------------------------------
def test_yandex_llm_without_keys_raises(monkeypatch):
    from klubok.extraction import llm_client
    monkeypatch.setattr(llm_client.settings, "yandex_api_key", "")
    monkeypatch.setattr(llm_client.settings, "yandex_folder_id", "")
    with pytest.raises(RuntimeError, match="YANDEX"):
        llm_client.YandexLLMClient()


def test_yandex_embedder_without_keys_raises(monkeypatch):
    from klubok.vectorstore import embeddings
    monkeypatch.setattr(embeddings.settings, "yandex_api_key", "")
    monkeypatch.setattr(embeddings.settings, "yandex_folder_id", "")
    with pytest.raises(RuntimeError, match="YANDEX"):
        embeddings.YandexEmbedder()
