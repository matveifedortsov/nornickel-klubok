"""FastAPI поверх пайплайна.

Запуск (когда подняты Neo4j/Qdrant):
    uvicorn klubok.api.app:app --reload

Эндпоинты:
    POST /ingest    {"path": "./data/sample"}                -> заингестить PDF/DOCX/PPTX/папку
    POST /ask       {"question": "...", "geography": bool|null, "domain": str|null} -> ответ с цитатами
    POST /review    {"topic": "..."}                          -> структурированный литературный обзор
    GET  /gaps                                                -> отчёт о пробелах (не для external_partner)
    GET  /dashboard                                           -> метрики руководителя (не для external_partner)
    GET  /subgraph?q=...&geography=&domain=                   -> подграф для визуализации
    POST /compare   {"cid_a":"Process:...", "cid_b":"Process:..."} -> таблица сравнения
    GET  /experts?topic=...                                   -> эксперты/лаборатории по теме
    GET  /facilities                                          -> активность лабораторий
    POST /export    {"kind":"answer"|"review", ..., "format":"markdown"|"json-ld"|"pdf"}
    POST /graph/edge {"src_type":..., "src_cid":..., "rel":..., "dst_type":..., "dst_cid":...,
                      "editor_name":..., "comment":...}       -> ручная корректировка (project_lead/admin)
    GET  /health

Заголовок X-API-Key определяет роль (RBAC, см. klubok/api/auth.py). Без ключа
роль — external_partner (самый ограниченный доступ). Каждый запрос
логируется в аудит (klubok/api/audit.py).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from klubok.pipeline import build_stores, ingest_path, answer_question, seed_if_empty
from klubok.retrieval.graphrag import retrieve
from klubok.graph import gaps
from klubok.graph.ingest import upsert_manual_edge
from klubok.analytics import compare as compare_mod, recommend, dashboard as dashboard_mod
from klubok.qa.answer import generate_literature_review
from klubok.export import to_markdown, to_json_ld, to_pdf
from klubok.extraction.llm_client import get_llm
from klubok.ontology import NodeType, RelType
from klubok.api.auth import get_role, require_full_access, require_editor
from klubok.api.audit import log_request
from klubok.notify.watchlist import WatchStore

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    client, store = build_stores()
    seed_if_empty(client, store)      # пустой граф + есть seed -> демо-данные для жюри
    _state["client"], _state["store"] = client, store
    _state["watch"] = WatchStore()
    _warmup(client, store)
    try:
        yield
    finally:
        client.close()
        store.close()
        _state["watch"].close()


def _warmup(client, store) -> None:
    """Прогреть медленный fulltext-индекс Neo4j фиктивным seed-запросом.

    Именно первый seed-запрос к fulltext «холодный» (~5с) и упирается в
    требование ТЗ «3-5с»; векторный поиск и так ~9мс. Поэтому греем ТОЛЬКО
    fulltext (find_seed_nodes), НЕ полный retrieve — иначе при исчерпанной квоте
    эмбеддингов прогрев ушёл бы в долгие ретраи и заблокировал старт API.
    Ошибки глушим (пустой граф/нет индекса не должны ронять старт).
    """
    import logging
    from klubok.retrieval.graphrag import find_seed_nodes
    try:
        find_seed_nodes(client, "прогрев индекса", limit=3)
        logging.getLogger("klubok.api").info("warm-up fulltext-индекса выполнен")
    except Exception as exc:                              # noqa: BLE001
        logging.getLogger("klubok.api").warning("warm-up пропущен: %s", exc)


app = FastAPI(title="Научный клубок API", version="0.2.0", lifespan=lifespan)


@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    response = await call_next(request)
    from config import settings
    api_key = request.headers.get("x-api-key")
    role = settings.api_keys.get(api_key, "external_partner") if api_key else "external_partner"
    log_request(role, request.method, request.url.path, dict(request.query_params))
    return response


class IngestRequest(BaseModel):
    path: str


class AskRequest(BaseModel):
    question: str
    geography: bool | None = None      # None=авто из текста вопроса; передайте явно, чтобы переопределить
    domain: str | None = None
    year_from: int | None = None       # временной диапазон (ТЗ); None=авто из вопроса
    year_to: int | None = None


class ReviewRequest(BaseModel):
    topic: str


class CompareRequest(BaseModel):
    cid_a: str
    cid_b: str
    label_a: str = "Вариант А"
    label_b: str = "Вариант Б"


class ExportRequest(BaseModel):
    kind: Literal["answer", "review"] = "answer"
    question: str | None = None
    topic: str | None = None
    format: Literal["markdown", "json-ld", "pdf"] = "markdown"


class GraphEditRequest(BaseModel):
    src_type: str
    src_cid: str
    rel: str
    dst_type: str
    dst_cid: str
    editor_name: str
    comment: str | None = None


class WatchRequest(BaseModel):
    topic: str


def _visible_sources(client, role: str, sources: list[str]) -> list[str]:
    """Скрыть источники с sensitivity=internal от external_partner (§7 RBAC)."""
    if role != "external_partner" or not sources:
        return sources
    doc_ids = [s.split(":", 1)[-1] if ":" in s else s for s in sources]
    rows = client.run(
        "MATCH (p:Publication) WHERE p.canonical_id IN $cids RETURN p.canonical_id AS cid, p.sensitivity AS sens",
        cids=[f"Publication:{d}" for d in doc_ids],
    )
    internal = {r["cid"].split(":", 1)[-1] for r in rows if r["sens"] == "internal"}
    return [s for s in sources if s not in internal]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "nodes": _state["client"].count_nodes()}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    results = ingest_path(req.path, _state["client"], _state["store"],
                          watch_store=_state["watch"])
    return {"ingested": results}


@app.post("/ask")
def ask(req: AskRequest, role: str = Depends(get_role)) -> dict:
    ans = answer_question(req.question, _state["client"], _state["store"],
                          geography=req.geography if req.geography is not None else "auto",
                          domain=req.domain, year_from=req.year_from, year_to=req.year_to)
    sources = _visible_sources(_state["client"], role, ans.sources)
    return {
        "question": ans.question, "answer": ans.text, "sources": sources,
        "edges_used": ans.edges_used, "passages_used": ans.passages_used,
        "constraints": [c.model_dump() for c in (ans.constraints or [])],
        "geography_filter": ans.geography_filter,
        "year_from": ans.year_from, "year_to": ans.year_to,
        "timings_ms": ans.timings_ms or {},
        # подграф уже найден этим же retrieve — отдаём сразу, чтобы UI не гонял
        # весь гибридный поиск второй раз через GET /subgraph
        "subgraph": {"seeds": ans.seed_nodes or [], "edges": ans.subgraph_edges or []},
    }


@app.post("/review")
def review(req: ReviewRequest) -> dict:
    rev = generate_literature_review(req.topic, _state["client"], _state["store"], get_llm())
    return {"topic": rev.topic, "text": rev.text, "sources": rev.sources,
            "edges_used": rev.edges_used, "passages_used": rev.passages_used}


@app.get("/subgraph")
def subgraph(q: str, geography: bool | None = None, domain: str | None = None,
             year_from: int | None = None, year_to: int | None = None) -> dict:
    ctx = retrieve(q, _state["store"], _state["client"],
                   geography=geography if geography is not None else "auto", domain=domain,
                   year_from=year_from, year_to=year_to)
    return {"seeds": ctx.seed_nodes, "edges": ctx.subgraph_edges,
            "constraints": [c.model_dump() for c in ctx.constraints],
            "geography_filter": ctx.geography_filter,
            "year_from": ctx.year_from, "year_to": ctx.year_to,
            "timings_ms": ctx.timings_ms}


@app.get("/gaps")
def gap_report(role: str = Depends(require_full_access)) -> dict:
    return gaps.gap_report(_state["client"])


@app.get("/dashboard")
def dashboard(role: str = Depends(require_full_access)) -> dict:
    return dashboard_mod.dashboard_report(_state["client"])


@app.get("/entities")
def entities(type: str, q: str = "", limit: int = 50) -> dict:
    """Список сущностей заданного типа (для автодополнения в UI «Сравнение»)."""
    try:
        label = NodeType(type).value           # валидация против инъекции
    except ValueError:
        raise HTTPException(status_code=400, detail=f"неизвестный тип: {type}")
    rows = _state["client"].run(
        f"MATCH (n:{label}) WHERE $q = '' OR toLower(n.name) CONTAINS toLower($q) "
        "RETURN n.canonical_id AS cid, n.name AS name ORDER BY name LIMIT $limit",
        q=q, limit=limit)
    return {"entities": [dict(r) for r in rows]}


@app.post("/compare")
def compare_endpoint(req: CompareRequest) -> dict:
    rows = compare_mod.compare(_state["client"], req.cid_a, req.cid_b, req.label_a, req.label_b)
    return {"rows": rows}


@app.get("/experts")
def experts(topic: str) -> dict:
    return {"experts": recommend.experts_by_topic(_state["client"], topic)}


@app.get("/facilities")
def facilities() -> dict:
    return {"facilities": dashboard_mod.facility_activity(_state["client"])}


@app.post("/export")
def export(req: ExportRequest) -> Response:
    if req.kind == "answer":
        if not req.question:
            raise HTTPException(status_code=400, detail="'question' обязателен при kind='answer'")
        result = answer_question(req.question, _state["client"], _state["store"])
    else:
        if not req.topic:
            raise HTTPException(status_code=400, detail="'topic' обязателен при kind='review'")
        result = generate_literature_review(req.topic, _state["client"], _state["store"], get_llm())

    if req.format == "markdown":
        return Response(content=to_markdown(result), media_type="text/markdown")
    if req.format == "json-ld":
        import json
        return Response(content=json.dumps(to_json_ld(result), ensure_ascii=False, indent=2),
                        media_type="application/ld+json")
    return Response(content=to_pdf(result), media_type="application/pdf")


@app.post("/graph/edge")
def graph_edge(req: GraphEditRequest, role: str = Depends(require_editor)) -> dict:
    try:
        src_type = NodeType(req.src_type)
        dst_type = NodeType(req.dst_type)
        rel = RelType(req.rel)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        upsert_manual_edge(_state["client"], src_type, req.src_cid, rel, dst_type, req.dst_cid,
                           edited_by=req.editor_name, comment=req.comment)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok"}


# --- Уведомления (§Y7): подписка на темы + лента ---
@app.post("/watch")
def watch(req: WatchRequest, role: str = Depends(get_role)) -> dict:
    _state["watch"].subscribe(role, req.topic)
    return {"status": "ok", "subscriptions": _state["watch"].subscriptions_of(role)}


@app.delete("/watch")
def unwatch(req: WatchRequest, role: str = Depends(get_role)) -> dict:
    _state["watch"].unsubscribe(role, req.topic)
    return {"status": "ok", "subscriptions": _state["watch"].subscriptions_of(role)}


@app.get("/notifications")
def notifications(unseen_only: bool = False, mark_seen: bool = False,
                  role: str = Depends(get_role)) -> dict:
    feed = _state["watch"].feed(role, unseen_only=unseen_only)
    if mark_seen:
        _state["watch"].mark_seen(role)
    return {"subscriber": role, "subscriptions": _state["watch"].subscriptions_of(role),
            "notifications": feed}
