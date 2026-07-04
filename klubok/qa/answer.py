"""Генерация ответа с цитатами поверх RetrievalContext.

Ответ строго заземлён в подграф + пассажи (анти-галлюцинационный аргумент
для жюри). Источники возвращаются отдельным полем для подсветки в UI.
"""
from __future__ import annotations

from dataclasses import dataclass

from klubok.extraction.prompts import (
    ANSWER_SYSTEM, build_answer_prompt, REVIEW_SYSTEM, build_review_prompt,
)
from klubok.extraction.llm_client import LLMClient
from klubok.retrieval.graphrag import RetrievalContext, retrieve


@dataclass
class Answer:
    question: str
    text: str
    sources: list[str]
    edges_used: int
    passages_used: int
    constraints: list = None            # NumericConstraint из вопроса, см. graphrag.py
    geography_filter: bool | None = None
    year_from: int | None = None        # временной диапазон (ТЗ), см. graphrag.py
    year_to: int | None = None
    timings_ms: dict = None             # retrieval (vector/seed/graph) + llm_ms (§Y8)
    subgraph_edges: list = None         # рёбра подграфа для визуализации: ретривал
    seed_nodes: list = None             # уже их нашёл — не гонять /subgraph повторно


def generate_answer(ctx: RetrievalContext, llm: LLMClient) -> Answer:
    import time
    prompt = build_answer_prompt(
        question=ctx.question,
        graph_context=ctx.graph_context_text(),
        passages=ctx.passages_text(),
    )
    t0 = time.perf_counter()
    try:
        text = llm.complete(prompt, system=ANSWER_SYSTEM)
    except Exception as exc:                              # noqa: BLE001 — квота/сеть LLM
        import logging
        logging.getLogger(__name__).warning("генерация ответа не удалась (%s)", exc)
        text = ""
    llm_ms = round((time.perf_counter() - t0) * 1000, 1)
    # пустой/сбойный ответ LLM не должен выглядеть как «пустой ответ» — даём
    # честный фолбэк, но сохраняем найденный подграф/источники (польза для жюри).
    if not text or not text.strip():
        text = ("Не удалось сгенерировать связный ответ (LLM недоступен или пуст). "
                "Ниже — найденные связи графа и источники по запросу.")
    return Answer(
        question=ctx.question,
        text=text,
        sources=ctx.cited_docs(),
        edges_used=len(ctx.subgraph_edges),
        passages_used=len(ctx.passages),
        constraints=ctx.constraints,
        geography_filter=ctx.geography_filter,
        year_from=ctx.year_from,
        year_to=ctx.year_to,
        timings_ms={**ctx.timings_ms, "llm_ms": llm_ms},
        subgraph_edges=ctx.subgraph_edges,
        seed_nodes=ctx.seed_nodes,
    )


@dataclass
class LiteratureReview:
    """Структурированный синтез по теме (не просто Q&A) — §6 плана.

    Группировка по методу/году/географии, консенсус vs разногласия и степень
    уверенности формируются самим LLM внутри `text` по инструкции
    REVIEW_INSTRUCTION — здесь фиксируется только provenance-обвязка,
    аналогично Answer.
    """
    topic: str
    text: str
    sources: list[str]
    edges_used: int
    passages_used: int


def generate_literature_review(topic: str, client, store, llm: LLMClient,
                               top_k_passages: int = 20, top_k_seeds: int = 15) -> LiteratureReview:
    """Литературный обзор по теме: шире ретривал, чем у обычного ответа —
    цель не «найти точный факт», а «собрать все релевантные источники»."""
    ctx = retrieve(topic, store, client, top_k_passages=top_k_passages, top_k_seeds=top_k_seeds)
    prompt = build_review_prompt(
        topic=topic,
        graph_context=ctx.graph_context_text(),
        passages=ctx.passages_text(),
    )
    text = llm.complete(prompt, system=REVIEW_SYSTEM)
    return LiteratureReview(
        topic=topic, text=text, sources=ctx.cited_docs(),
        edges_used=len(ctx.subgraph_edges), passages_used=len(ctx.passages),
    )
