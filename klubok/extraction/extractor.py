"""Оркестрация извлечения триплетов из чанков.

Поток:  chunk -> промпт -> LLM -> JSON -> Pydantic -> фильтр по онтологии
        -> прокидывание chunk_id/evidence -> ExtractionResult.

Устойчив к «грязному» JSON от LLM (markdown-обёртки, болтовня вокруг).
"""
from __future__ import annotations

import json
import re
import logging

from klubok.ontology import (
    Chunk, Entity, Relation, ExtractionResult, NodeType, RelType, NumericConstraint,
)
from klubok.extraction.prompts import EXTRACTION_SYSTEM, build_extraction_prompt
from klubok.extraction.llm_client import LLMClient
from klubok.extraction.heuristics import verification_from_evidence

log = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _try_loads(raw: str) -> dict | None:
    """JSON-объект из ответа LLM (терпит markdown-обёртку) или None, если битый."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(raw)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def _loads_lenient(raw: str) -> dict:
    """Достать JSON-объект из ответа LLM, даже если он обёрнут в текст/```json."""
    data = _try_loads(raw)
    if data is None:
        log.warning("Не удалось распарсить JSON из ответа LLM: %.120s", raw)
        return {"entities": [], "relations": []}
    return data


def _coerce_node_type(v: str) -> NodeType | None:
    try:
        return NodeType(v)
    except ValueError:
        return None


def _coerce_rel_type(v: str) -> RelType | None:
    try:
        return RelType(v)
    except ValueError:
        return None


def _coerce_constraints(raw: list | None) -> list[NumericConstraint]:
    """Мягко распарсить attrs['constraints'] от LLM — пропускаем битые записи."""
    out: list[NumericConstraint] = []
    for c in raw or []:
        try:
            out.append(NumericConstraint(**c))
        except Exception as exc:                        # noqa: BLE001
            log.debug("skip constraint %s: %s", c, exc)
    return out


def parse_extraction(raw: str, doc_id: str, chunk_id: str | None) -> ExtractionResult:
    """Превратить сырой ответ LLM в валидированный ExtractionResult."""
    data = _loads_lenient(raw)
    entities: list[Entity] = []
    for e in data.get("entities", []):
        nt = _coerce_node_type(e.get("type", ""))
        if nt is None or not e.get("name"):
            continue
        try:
            entities.append(Entity(
                name=e["name"], type=nt, attributes=e.get("attributes", {}) or {},
                constraints=_coerce_constraints(e.get("constraints")),
                geography=e.get("geography"), domain=e.get("domain"),
                is_domestic=e.get("is_domestic"),
            ))
        except Exception as exc:                       # noqa: BLE001
            log.debug("skip entity %s: %s", e, exc)

    relations: list[Relation] = []
    for r in data.get("relations", []):
        st = _coerce_node_type(r.get("src_type", ""))
        dt = _coerce_node_type(r.get("dst_type", ""))
        rt = _coerce_rel_type(r.get("rel", ""))
        if not all([st, dt, rt]) or not r.get("src_name") or not r.get("dst_name"):
            continue
        # уровень верификации: берём у LLM, а если тот смолчал ("unverified") —
        # добираем эвристикой по глаголам в цитате (получен/показал → confirmed).
        vlevel = r.get("verification_level") or "unverified"
        if vlevel == "unverified":
            vlevel = verification_from_evidence(r.get("evidence"))
        rel = Relation(
            src_name=r["src_name"], src_type=st, rel=rt,
            dst_name=r["dst_name"], dst_type=dt,
            evidence=r.get("evidence"), chunk_id=chunk_id,
            confidence=float(r.get("confidence", 1.0)),
            source_type=r.get("source_type"),
            verification_level=vlevel,
            actualized_at=r.get("actualized_at"),
            geography=r.get("geography"), is_domestic=r.get("is_domestic"),
        )
        relations.append(rel)

    result = ExtractionResult(doc_id=doc_id, chunk_id=chunk_id,
                              entities=entities, relations=relations)
    # фильтр галлюцинаций: оставляем только связи, разрешённые онтологией
    result.relations = result.schema_valid_relations()
    return result


def extract_from_chunk(chunk: Chunk, llm: LLMClient, cache=None) -> ExtractionResult:
    """Извлечь триплеты из чанка. `cache` (ExtractCache) — опциональный кэш сырого
    ответа LLM: экономит квоту при ретраях/повторных прогонах (см. extract_cache.py)."""
    raw = cache.get(chunk.text) if cache is not None else None
    if raw is None:
        raw = llm.complete(build_extraction_prompt(chunk.text), system=EXTRACTION_SYSTEM)
        # В кэш кладём только распарсиваемый ответ и только от реального LLM:
        # обрезанный по max_tokens / битый JSON (или заглушка MockLLM) иначе
        # «отравил» бы кэш навсегда — чанк переставал бы извлекаться даже после
        # починки бэкенда, т.к. ключ кэша не зависит от бэкенда.
        from klubok.extraction.llm_client import MockLLM
        if cache is not None and not isinstance(llm, MockLLM) and _try_loads(raw) is not None:
            cache.put(chunk.text, raw)
    return parse_extraction(raw, doc_id=chunk.doc_id, chunk_id=chunk.chunk_id)


def extract_from_chunks(chunks: list[Chunk], llm: LLMClient, cache=None) -> list[ExtractionResult]:
    return [extract_from_chunk(c, llm, cache=cache) for c in chunks]
