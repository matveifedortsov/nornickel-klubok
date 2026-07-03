"""Абстракция над LLM + реализации.

  * MockLLM         — детерминированные ответы без сети. Отладка пайплайна.
  * OpenAICompatClient — база для любого OpenAI-совместимого /chat/completions
                    эндпоинта (ретраи 429/5xx + троттлинг под RPS-квоту).
  * MetalGPTClient  — MetalGPT-1 через локальный vLLM (on-prem вариант для прода).
  * YandexLLMClient — YandexGPT через Yandex AI Studio (боевой бэкенд хакатона).

Переключение — через settings.llm_backend (mock | metalgpt | yandex).
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from typing import Protocol

from config import settings


class LLMClient(Protocol):
    def complete(self, prompt: str, system: str = "") -> str:
        """Вернуть текстовый ответ модели."""
        ...


# --------------------------------------------------------------------------
# Троттлинг: не превышать N запросов/сек (общий для всех потоков ингеста)
# --------------------------------------------------------------------------
class RateLimiter:
    """Простой лимитер: гарантирует минимальный интервал между вызовами."""

    def __init__(self, rps: float) -> None:
        self._min_interval = 1.0 / rps if rps > 0 else 0.0
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = self._next_at - now
            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.monotonic()
            self._next_at = now + self._min_interval


# --------------------------------------------------------------------------
# Mock — работает без сети
# --------------------------------------------------------------------------
class MockLLM:
    """Эвристический заглушечный LLM.

    Для промпта извлечения возвращает простые триплеты, найденные регулярками.
    Достаточно, чтобы end-to-end прогнать ingestion -> граф -> поиск без модели.
    """

    _ALLOY_RE = re.compile(r"\b((?:[A-Z][a-z]?){2,})\b")        # CuNi, FeCr...
    _TEMP_RE = re.compile(r"(\d+)\s*°?\s*[CК]\b")

    def complete(self, prompt: str, system: str = "") -> str:
        if '"entities"' in prompt or "Извлеки сущности" in prompt:
            return self._fake_extraction(prompt)
        if "Дай связный ответ" in prompt or prompt.startswith("Вопрос:"):
            return self._fake_answer(prompt)
        return "[MOCK] нет обработчика для данного промпта"

    def _fake_extraction(self, prompt: str) -> str:
        body = prompt.split('"""')[-2] if '"""' in prompt else prompt
        entities, relations = [], []
        seen = set()

        for m in self._ALLOY_RE.finditer(body):
            name = m.group(1)
            if name in seen or len(name) < 3:
                continue
            seen.add(name)
            entities.append({"name": name, "type": "Material", "attributes": {}})

        temp = self._TEMP_RE.search(body)
        if temp and entities:
            entities.append({
                "name": "отжиг", "type": "Process",
                "attributes": {"temperature": int(temp.group(1)), "unit": "°C"},
            })
            relations.append({
                "src_name": "Эксперимент", "src_type": "Experiment",
                "rel": "USES", "dst_name": entities[0]["name"], "dst_type": "Material",
                "evidence": temp.group(0), "confidence": 0.5,
            })
        return json.dumps({"entities": entities, "relations": relations}, ensure_ascii=False)

    def _fake_answer(self, prompt: str) -> str:
        return (
            "[MOCK-ОТВЕТ] На основе предоставленного контекста сформирован "
            "демонстрационный ответ. Подставьте реальный LLM для боевого вывода. "
            "Источники: [doc_mock]."
        )


# --------------------------------------------------------------------------
# База для OpenAI-совместимых эндпоинтов
# --------------------------------------------------------------------------
class OpenAICompatClient:
    """Общая логика вызова /chat/completions с ретраями и троттлингом.

    Наследники задают base_url, model, заголовок авторизации и параметры.
    """

    RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}

    def __init__(self, base_url: str, model: str, auth_header: dict[str, str],
                 temperature: float = 0.1, max_tokens: int | None = None,
                 timeout: int = 120, max_retries: int = 3,
                 rate_limiter: RateLimiter | None = None,
                 json_mode: bool = False) -> None:
        import requests
        self._requests = requests
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.auth_header = auth_header
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limiter = rate_limiter
        self.json_mode = json_mode

    def _payload(self, messages: list[dict]) -> dict:
        body: dict = {"model": self.model, "messages": messages,
                      "temperature": self.temperature}
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", **self.auth_header}

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if self.rate_limiter:
                self.rate_limiter.wait()
            try:
                resp = self._requests.post(url, headers=headers,
                                           json=self._payload(messages), timeout=self.timeout)
                if resp.status_code in self.RETRYABLE_STATUS:
                    raise _RetryableHTTP(resp.status_code, resp.text[:200])
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except _RetryableHTTP as exc:
                last_exc = exc
            except Exception as exc:                          # noqa: BLE001
                # сетевые таймауты/сбросы тоже ретраим
                if _is_network_error(exc) and attempt < self.max_retries:
                    last_exc = exc
                else:
                    raise
            # backoff с джиттером
            if attempt < self.max_retries:
                delay = min(2 ** attempt, 30) + random.uniform(0, 0.5)
                time.sleep(delay)
        raise RuntimeError(f"LLM-запрос не удался после {self.max_retries + 1} попыток: {last_exc}")


class _RetryableHTTP(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status


def _is_network_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connection", "connect"))


# --------------------------------------------------------------------------
# MetalGPT-1 — локальный vLLM (on-prem вариант для прода)
# --------------------------------------------------------------------------
class MetalGPTClient(OpenAICompatClient):
    """Клиент к MetalGPT-1.

    Запуск модели (на машине с GPU):
        python -m vllm.entrypoints.openai.api_server \\
            --model Nornickel/MetalGPT-1 --quantization awq --port 8000
    """

    def __init__(self) -> None:
        super().__init__(
            base_url=settings.metalgpt_base_url,
            model=settings.metalgpt_model,
            auth_header={"Authorization": f"Bearer {settings.metalgpt_api_key}"},
            temperature=0.1,
            timeout=120,
            max_retries=3,
        )


# --------------------------------------------------------------------------
# YandexGPT — Yandex AI Studio (боевой бэкенд)
# --------------------------------------------------------------------------
_yandex_rate_limiter = RateLimiter(settings.yandex_rps)


class YandexLLMClient(OpenAICompatClient):
    """YandexGPT через OpenAI-совместимый эндпоинт Yandex AI Studio.

    Модель адресуется как gpt://<folder_id>/<model>. Авторизация — заголовок
    'Authorization: Api-Key <key>' (не Bearer). Общий RateLimiter на все
    экземпляры, чтобы многопоточный ингест не превышал RPS-квоту.
    """

    def __init__(self) -> None:
        if not settings.yandex_api_key or not settings.yandex_folder_id:
            raise RuntimeError(
                "Не заданы YANDEX_API_KEY / YANDEX_FOLDER_ID в .env — "
                "YandexGPT недоступен. См. PLAN_FINAL.md §Y3.")
        model = f"gpt://{settings.yandex_folder_id}/{settings.yandex_llm_model}"
        super().__init__(
            base_url=settings.yandex_llm_base_url,
            model=model,
            auth_header={"Authorization": f"Api-Key {settings.yandex_api_key}"},
            temperature=settings.yandex_temperature,
            max_tokens=settings.yandex_max_tokens,
            timeout=settings.yandex_timeout,
            max_retries=settings.yandex_max_retries,
            rate_limiter=_yandex_rate_limiter,
            json_mode=False,      # включить True, если слой стабильно принимает response_format
        )


# --------------------------------------------------------------------------
# Фабрика
# --------------------------------------------------------------------------
def get_llm() -> LLMClient:
    backend = settings.llm_backend
    if backend == "metalgpt":
        return MetalGPTClient()
    if backend == "yandex":
        return YandexLLMClient()
    return MockLLM()
