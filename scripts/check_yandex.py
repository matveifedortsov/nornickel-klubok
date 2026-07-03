"""Смоук-тест Yandex AI Studio — запускать ПЕРВЫМ после настройки .env.

Проверяет за ~10 секунд, что ключ/folder_id/сеть/квоты работают, до любого
дорогого ингеста. Ничего не пишет в БД.

Запуск:
    # в .env: YANDEX_API_KEY=..., YANDEX_FOLDER_ID=...
    python scripts/check_yandex.py

Выход 0 — всё ок; ненулевой — печатает, что именно сломалось.
"""
from __future__ import annotations

import sys
import time

import _bootstrap  # noqa: F401,E402  (путь к пакету + UTF-8 вывод)

from config import settings


def _fail(msg: str) -> None:
    print(f"❌ {msg}")
    sys.exit(1)


def main() -> None:
    print("=== Смоук-тест Yandex AI Studio ===\n")

    # 0. Конфигурация
    if not settings.yandex_api_key or not settings.yandex_folder_id:
        _fail("YANDEX_API_KEY / YANDEX_FOLDER_ID не заданы в .env")
    print(f"folder_id: {settings.yandex_folder_id[:6]}…  "
          f"api_key: {'*' * 6}{settings.yandex_api_key[-4:]}")
    print(f"LLM модель: {settings.yandex_llm_model} | эмбеддер dim={settings.yandex_embedding_dim}\n")

    # 1. LLM
    print("1. YandexGPT complete() …")
    from klubok.extraction.llm_client import YandexLLMClient
    try:
        t0 = time.monotonic()
        out = YandexLLMClient().complete("Ответь ровно одним словом: работает")
        dt = time.monotonic() - t0
        print(f"   ✅ ответ ({dt:.1f}s): {out.strip()[:80]}\n")
    except Exception as exc:                                # noqa: BLE001
        _fail(f"LLM недоступен: {exc}")

    # 2. Эмбеддинги (doc + query, проверяем размерность)
    print("2. textEmbedding doc + query …")
    from klubok.vectorstore.embeddings import YandexEmbedder
    try:
        emb = YandexEmbedder()
        dv = emb.encode(["обессоливание воды обратным осмосом"], kind="doc")
        qv = emb.encode_query("методы деминерализации воды")
        print(f"   ✅ doc dim={dv.shape[1]}, query dim={qv.shape[0]}")
        if dv.shape[1] != settings.yandex_embedding_dim:
            print(f"   ⚠️  фактическая dim={dv.shape[1]} ≠ YANDEX_EMBEDDING_DIM="
                  f"{settings.yandex_embedding_dim} — поправьте .env")
        # косинус похожих текстов должен быть заметно > 0
        import numpy as np
        sim = float(dv[0] @ qv)
        print(f"   косинус(похожие doc/query) = {sim:.3f}\n")
    except Exception as exc:                                # noqa: BLE001
        _fail(f"Эмбеддинги недоступны: {exc}")

    # 3. Реальное извлечение триплетов на одном gold-примере
    print("3. Извлечение триплетов на gold-примере …")
    from klubok.extraction.prompts import build_extraction_prompt, EXTRACTION_SYSTEM
    from klubok.extraction.extractor import parse_extraction
    from klubok.eval.gold_set import GOLD_SET
    try:
        text = GOLD_SET[0]["text"]
        raw = YandexLLMClient().complete(build_extraction_prompt(text),
                                         system=EXTRACTION_SYSTEM)
        res = parse_extraction(raw, doc_id="smoke", chunk_id=None)
        print(f"   ✅ сущностей: {len(res.entities)}, связей: {len(res.relations)}")
        for e in res.entities[:6]:
            print(f"      • {e.type.value}: {e.name}")
    except Exception as exc:                                # noqa: BLE001
        _fail(f"Извлечение не удалось: {exc}")

    print("\n✅ Все проверки пройдены. Можно запускать eval_extraction.py и ingest_corpus.py.")


if __name__ == "__main__":
    main()
