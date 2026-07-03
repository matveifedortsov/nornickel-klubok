"""Запись извлечённых сущностей/связей в Neo4j (идемпотентный MERGE).

Связи ссылаются на сущности по (type, name) внутри документа; перед записью
мы резолвим имена в canonical_id, чтобы дубли схлопывались в один узел.

Версионирование фактов: при повторном ингесте одно и то же ребро (src,rel,dst)
не перезаписывается молча — если новое evidence/значение расходится со старым,
старая версия помечается is_current=false, а новое ребро создаётся отдельной
MERGE-парой (несколько параллельных рёбер одного типа с разным valid_from).
Ретривал (graphrag.py) по умолчанию фильтрует is_current=true.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from klubok.ontology import Document, ExtractionResult, Entity, Relation, NodeType, RelType, edge_is_valid
from klubok.extraction.resolver import resolve, canonical_id
from klubok.graph.neo4j_client import Neo4jClient

log = logging.getLogger(__name__)


def upsert_document(client: Neo4jClient, doc: Document) -> None:
    """Записать/обновить узел Publication по метаданным распарсенного файла."""
    client.run(
        """
        MERGE (d:Publication {canonical_id: $cid})
        SET d.name = $title, d.title = $title, d.source_path = $path, d.year = $year,
            d.authors = $authors, d.publication_type = $ptype, d.domain = $domain,
            d.geography = $geo, d.is_domestic = $is_domestic, d.sensitivity = $sensitivity
        """,
        cid=f"Publication:{doc.doc_id}", title=doc.title or doc.doc_id,
        path=doc.source_path, year=doc.year, authors=doc.authors,
        ptype=doc.publication_type, domain=doc.domain, geo=doc.geography,
        is_domestic=doc.is_domestic, sensitivity=doc.sensitivity,
    )


_VALUE_TOLERANCE = 1e-6


def _merge_entity(client: Neo4jClient, e: Entity) -> str:
    """MERGE узла. Числовой конфликт в attrs['value'] с уже записанным значением
    не перезаписывается молча — предыдущее значение уходит в value_history
    (JSON-строка), узел помечается has_conflicting_versions=true (см. gaps.py:
    подсветка противоречивых данных). Некон фликтующие атрибуты обновляются как раньше.
    """
    import json

    cid = e.canonical_id or canonical_id(e)
    attrs = {k: v for k, v in e.attributes.items()}
    constraints_json = (
        json.dumps([c.model_dump() for c in e.constraints], ensure_ascii=False)
        if e.constraints else None
    )

    existing = client.run(
        f"MATCH (n:{e.type.value} {{canonical_id: $cid}}) RETURN n.value AS value, "
        f"n.unit AS unit, n.value_history AS history",
        cid=cid,
    )
    new_value = attrs.get("value")
    history_update = None
    if existing and isinstance(new_value, (int, float)):
        old_value = existing[0]["value"]
        if isinstance(old_value, (int, float)) and abs(old_value - new_value) > _VALUE_TOLERANCE:
            hist = json.loads(existing[0]["history"]) if existing[0]["history"] else []
            hist.append({"value": old_value, "unit": existing[0]["unit"]})
            history_update = json.dumps(hist, ensure_ascii=False)

    client.run(
        f"""
        MERGE (n:{e.type.value} {{canonical_id: $cid}})
        ON CREATE SET n.name = $name
        SET n += $attrs
        """ + (", n.value_history = $history, n.has_conflicting_versions = true" if history_update else ""),
        cid=cid, name=e.name, attrs=attrs,
        **({"history": history_update} if history_update else {}),
    )
    if constraints_json:
        client.run(
            f"MATCH (n:{e.type.value} {{canonical_id: $cid}}) SET n.constraints_json = $c",
            cid=cid, c=constraints_json,
        )
    if e.geography is not None or e.domain is not None or e.is_domestic is not None:
        client.run(
            f"""
            MATCH (n:{e.type.value} {{canonical_id: $cid}})
            SET n.geography = coalesce($geo, n.geography),
                n.domain = coalesce($domain, n.domain),
                n.is_domestic = coalesce($is_domestic, n.is_domestic)
            """,
            cid=cid, geo=e.geography, domain=e.domain, is_domestic=e.is_domestic,
        )
    return cid


def ingest_authorship(client: Neo4jClient, doc: Document, meta) -> None:
    """Связать Publication с Expert/Facility по метаданным из имени файла (без LLM).

    `meta` — klubok.parsing.filename_meta.FilenameMeta. Дешёвый и точный сигнал
    авторства/лаборатории для той части корпуса, где это закодировано в имени
    файла (папки «Статьи», «Доклады») — вызывается из pipeline.ingest_document
    ДО/параллельно с LLM-извлечением.
    """
    if not meta.author_name:
        return
    expert_cid = f"Expert:{meta.author_name.lower()}"
    client.run(
        "MERGE (e:Expert {canonical_id: $cid}) ON CREATE SET e.name = $name",
        cid=expert_cid, name=meta.author_name,
    )
    client.run(
        """
        MATCH (p:Publication {canonical_id: $pub})
        MATCH (e:Expert {canonical_id: $exp})
        MERGE (p)-[:AUTHORED_BY]->(e)
        """,
        pub=f"Publication:{doc.doc_id}", exp=expert_cid,
    )
    if meta.lab_abbr:
        facility_cid = f"Facility:{meta.lab_abbr.lower()}"
        client.run(
            "MERGE (f:Facility {canonical_id: $cid}) ON CREATE SET f.name = $name",
            cid=facility_cid, name=meta.lab_abbr,
        )
        client.run(
            """
            MATCH (e:Expert {canonical_id: $exp})
            MATCH (f:Facility {canonical_id: $fac})
            MERGE (e)-[:AFFILIATED_WITH]->(f)
            """,
            exp=expert_cid, fac=facility_cid,
        )


def ingest_extraction(client: Neo4jClient, result: ExtractionResult,
                      doc: Document | None = None) -> dict[str, int]:
    """Записать один ExtractionResult. Возвращает счётчики записанного."""
    # 1) резолвим сущности -> canonical_id + карта имя->cid
    resolved, name_to_canon = resolve(result.entities)
    for e in resolved:
        _merge_entity(client, e)

    # 2) связи (только прошедшие фильтр онтологии в extractor)
    edges = 0
    for r in result.relations:
        src_cid = name_to_canon.get(f"{r.src_type.value}:{r.src_name}")
        dst_cid = name_to_canon.get(f"{r.dst_type.value}:{r.dst_name}")
        # сущность из связи могла не попасть в entities — досоздаём узел
        if src_cid is None:
            src_cid = _merge_entity(client, Entity(name=r.src_name, type=r.src_type))
        if dst_cid is None:
            dst_cid = _merge_entity(client, Entity(name=r.dst_name, type=r.dst_type))

        client.run(
            f"""
            MATCH (a:{r.src_type.value} {{canonical_id: $src}})
            MATCH (b:{r.dst_type.value} {{canonical_id: $dst}})
            MERGE (a)-[rel:{r.rel.value}]->(b)
            SET rel.evidence = $ev, rel.chunk_id = $chunk,
                rel.confidence = $conf, rel.doc_id = $doc,
                rel.is_current = true,
                rel.source_type = $source_type, rel.verification_level = $verification_level,
                rel.actualized_at = $actualized_at, rel.geography = $geography,
                rel.is_domestic = $is_domestic,
                rel.evidence_history = CASE
                    WHEN rel.evidence IS NULL OR rel.evidence = $ev THEN coalesce(rel.evidence_history, [])
                    ELSE coalesce(rel.evidence_history, []) + rel.evidence
                END
            """,
            src=src_cid, dst=dst_cid, ev=r.evidence, chunk=r.chunk_id,
            conf=r.confidence, doc=result.doc_id,
            source_type=r.source_type, verification_level=r.verification_level,
            actualized_at=r.actualized_at, geography=r.geography, is_domestic=r.is_domestic,
        )
        edges += 1

    # 3) связь Publication -> Experiment (провенанс)
    if doc is not None:
        upsert_document(client, doc)
        for e in resolved:
            if e.type.value == "Experiment":
                client.run(
                    """
                    MATCH (d:Publication {canonical_id: $doc})
                    MATCH (x:Experiment {canonical_id: $exp})
                    MERGE (d)-[:REPORTS]->(x)
                    """,
                    doc=f"Publication:{result.doc_id}", exp=e.canonical_id,
                )

    return {"entities": len(resolved), "relations": edges}


def upsert_manual_edge(client: Neo4jClient, src_type: NodeType, src_cid: str, rel: RelType,
                       dst_type: NodeType, dst_cid: str, edited_by: str,
                       comment: str | None = None) -> None:
    """Ручная корректировка графа экспертом (§7 плана) — в обход LLM.

    Узлы должны уже существовать (правим существующий факт/добавляем связь
    между уже известными сущностями, а не создаём новую сущность вслепую).
    Помечает ребро edited_by/edited_at/comment — прямое требование ТЗ
    («ручная корректировка графа экспертами с указанием автора и даты»).
    """
    if not edge_is_valid(src_type, rel, dst_type):
        raise ValueError(f"Связь {src_type.value}-{rel.value}->{dst_type.value} не разрешена онтологией")

    rows = client.run(
        f"MATCH (a:{src_type.value} {{canonical_id: $src}}), (b:{dst_type.value} {{canonical_id: $dst}}) "
        f"RETURN a, b",
        src=src_cid, dst=dst_cid,
    )
    if not rows:
        raise ValueError(f"Узел {src_cid} или {dst_cid} не найден в графе")

    client.run(
        f"""
        MATCH (a:{src_type.value} {{canonical_id: $src}})
        MATCH (b:{dst_type.value} {{canonical_id: $dst}})
        MERGE (a)-[r:{rel.value}]->(b)
        SET r.is_current = true, r.edited_by = $editor, r.edited_at = $ts,
            r.comment = $comment, r.source_type = 'manual_correction'
        """,
        src=src_cid, dst=dst_cid, editor=edited_by,
        ts=datetime.now(timezone.utc).isoformat(), comment=comment,
    )


def ingest_all(client: Neo4jClient, results: list[ExtractionResult],
               doc: Document | None = None) -> dict[str, int]:
    totals = {"entities": 0, "relations": 0}
    for r in results:
        c = ingest_extraction(client, r, doc=doc)
        totals["entities"] += c["entities"]
        totals["relations"] += c["relations"]
    return totals
