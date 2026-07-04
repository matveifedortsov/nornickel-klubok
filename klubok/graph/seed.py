"""Загрузка seed-дампа графа и векторов (для деплоя без ингеста).

Используется и скриптом scripts/import_seed.py, и автозагрузкой при старте API
(pipeline.seed_if_empty). Идемпотентно (MERGE по canonical_id).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

SEED_DIR = Path(__file__).resolve().parents[2] / "seed"

_SAFE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_label(s: str) -> str:
    """Метки/типы приходят из НАШЕГО же экспорта, но валидируем от инъекций."""
    if not s or not _SAFE.match(s):
        raise ValueError(f"недопустимая метка/тип: {s!r}")
    return s


def seed_exists() -> bool:
    return (SEED_DIR / "nodes.jsonl").exists()


_BATCH = 1000


def import_seed_graph(client) -> tuple[int, int]:
    # Сначала ПОЛНОСТЬЮ парсим оба файла, и только потом пишем в граф: битая
    # JSONL-строка обнаруживается до первого MERGE. Иначе остаётся полузасеянный
    # граф, который count_nodes()>0 навсегда исключит из повторного посева.
    nodes_by_label: dict[str, list[dict]] = {}
    for line in (SEED_DIR / "nodes.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        labels, props = row.get("labels", []), row.get("props", {})
        cid = props.get("canonical_id")
        if not labels or not cid:
            continue
        label = _safe_label(labels[0])
        nodes_by_label.setdefault(label, []).append({"cid": cid, "props": props})

    edges_by_type: dict[tuple[str, str, str], list[dict]] = {}
    edges_path = SEED_DIR / "edges.jsonl"
    if edges_path.exists():
        for line in edges_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            try:
                key = (_safe_label(r["stype"]), _safe_label(r["rel"]), _safe_label(r["dtype"]))
            except (KeyError, ValueError):
                continue
            edges_by_type.setdefault(key, []).append(
                {"src": r["src"], "dst": r["dst"], "props": r.get("props", {}) or {}})

    # Запись UNWIND-батчами: тысячи строк за несколько запросов вместо
    # round-trip'а на строку (импорт идёт синхронно в lifespan при старте API).
    n = 0
    for label, rows in nodes_by_label.items():
        for i in range(0, len(rows), _BATCH):
            batch = rows[i:i + _BATCH]
            client.run(
                f"UNWIND $rows AS r MERGE (x:{label} {{canonical_id: r.cid}}) SET x += r.props",
                rows=batch)
            n += len(batch)

    e = 0
    for (st, rel, dt), rows in edges_by_type.items():
        for i in range(0, len(rows), _BATCH):
            batch = rows[i:i + _BATCH]
            client.run(
                f"UNWIND $rows AS r "
                f"MATCH (a:{st} {{canonical_id: r.src}}) MATCH (b:{dt} {{canonical_id: r.dst}}) "
                f"MERGE (a)-[x:{rel}]->(b) SET x += r.props",
                rows=batch)
            e += len(batch)
    log.info("seed граф загружен: узлов=%d рёбер=%d", n, e)
    return n, e


def import_seed_vectors(store) -> int:
    path = SEED_DIR / "vectors.jsonl"
    if not path.exists():
        return 0

    # sanity-check размерности из meta.json (её пишет export_seed.py): сид с
    # fastembed-векторами (384) нельзя молча заливать в коллекцию под другой
    # эмбеддер (mock=1024, yandex=256) — упадёт на upsert после загрузки графа.
    meta_path = SEED_DIR / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        seed_dim = meta.get("embedding_dim")
        if seed_dim and seed_dim != store.embedder.dim:
            raise RuntimeError(
                f"seed-векторы имеют dim={seed_dim}, а текущий эмбеддер — "
                f"dim={store.embedder.dim}. Установите EMBEDDER_BACKEND, которым "
                f"делался export_seed (обычно fastembed), либо переиндексируйте корпус.")

    from qdrant_client.models import PointStruct
    store.ensure_collection()
    points, cnt = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        points.append(PointStruct(id=p["id"], vector=p["vector"], payload=p.get("payload", {})))
        if len(points) >= 256:
            store._client.upsert(store.collection, points=points)
            cnt += len(points)
            points = []
    if points:
        store._client.upsert(store.collection, points=points)
        cnt += len(points)
    log.info("seed векторы загружены: %d", cnt)
    return cnt
