"""Центральная конфигурация. Читает .env через pydantic-settings.

Импортируйте готовый объект:  from config import settings
"""
from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "klubok_dev_password"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "klubok_chunks"

    # Эмбеддинги
    embedder_backend: str = "mock"          # mock | bge | yandex
    embedder_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024               # mock/bge; для yandex фактическая — 256

    # LLM
    llm_backend: str = "mock"               # mock | metalgpt | yandex
    metalgpt_model: str = "Nornickel/MetalGPT-1"
    metalgpt_base_url: str = "http://localhost:8000/v1"
    metalgpt_api_key: str = "EMPTY"

    # --- Yandex AI Studio ---
    # Ключ и folder_id класть ТОЛЬКО в .env (не коммитить). Пусто => клиент не поднимется.
    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    # LLM — OpenAI-совместимый эндпоинт
    yandex_llm_base_url: str = "https://llm.api.cloud.yandex.net/v1"
    yandex_llm_model: str = "yandexgpt/latest"      # или yandexgpt-lite/latest (дешевле)
    yandex_temperature: float = 0.1
    yandex_max_tokens: int = 4000
    # Эмбеддинги — нативный REST (dim=256, асимметричные doc/query)
    yandex_emb_url: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
    yandex_emb_doc_model: str = "text-search-doc/latest"
    yandex_emb_query_model: str = "text-search-query/latest"
    yandex_embedding_dim: int = 256
    # Квоты/надёжность
    yandex_rps: float = 5.0                 # ограничение запросов/сек (троттлинг)
    yandex_max_retries: int = 5
    yandex_timeout: int = 180
    emb_cache_path: Path = Path("./data/cache/emb_cache.sqlite")

    # Прочее
    data_dir: Path = Path("./data")
    log_level: str = "INFO"

    # RBAC: API-key -> роль (researcher | analyst | project_lead | admin | external_partner).
    # Переопределяется в .env как JSON-строка: API_KEYS={"prod-key-1":"analyst",...}
    # Дефолт — только для локальной разработки/демо, не для боевого использования.
    api_keys: dict[str, str] = Field(default_factory=lambda: {
        "dev-admin": "admin",
        "dev-lead": "project_lead",
        "dev-analyst": "analyst",
        "dev-researcher": "researcher",
        "dev-partner": "external_partner",
    })
    audit_log_path: Path = Path("./data/audit.log")
    watchlist_path: Path = Path("./data/watchlist.sqlite")


settings = Settings()
