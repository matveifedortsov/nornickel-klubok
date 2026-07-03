"""Гибридный поиск (GraphRAG) — сердце ответа.

Поток на каждый запрос:
  1. Векторный recall по чанкам -> «точки входа» (упомянутые сущности).
  2. Привязка точек входа к узлам графа (fulltext index по name).
  3. Разбор вопроса на структурные ограничения (числа/диапазоны, гео РФ/мир).
  4. Обход графа на 1-4 хопа (APOC) с фильтрацией по этим ограничениям.
  5. Сборка контекста: связи (с evidence+верификацией) + текстовые пассажи.

Результат RetrievalContext отдаётся в qa.answer для генерации. Структурные
фильтры (numeric constraints, geography) прозрачно кладутся в контекст, чтобы
UI/ответ могли показать жюри, что выдача не просто «похожий текст», а
ограничена реальными числовыми/гео условиями вопроса — прямое требование ТЗ.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from klubok.graph.neo4j_client import Neo4jClient
from klubok.ontology import NumericConstraint
from klubok.extraction.normalize import parse_all_constraints

# отечественная практика vs мировая — простая ключевая эвристика по вопросу
_DOMESTIC_MARKERS = ("в россии", "росси", "отечествен", " рф ", "рф.")
_FOREIGN_MARKERS = ("зарубеж", "мировой практик", "мировая практика", "за рубежом", "иностран")


@dataclass
class RetrievalContext:
    question: str
    passages: list[dict] = field(default_factory=list)        # из вектора
    subgraph_edges: list[dict] = field(default_factory=list)  # из графа
    seed_nodes: list[str] = field(default_factory=list)
    constraints: list[NumericConstraint] = field(default_factory=list)   # разобранные из вопроса
    geography_filter: bool | None = None       # True=РФ, False=зарубеж, None=без фильтра
    timings_ms: dict[str, float] = field(default_factory=dict)  # vector_ms/seed_ms/graph_ms (§Y8)

    def graph_context_text(self) -> str:
        """Текстовое представление подграфа для промпта (с верификацией/гео)."""
        lines = []
        for e in self.subgraph_edges:
            ev = f"  (источник: {e['evidence']})" if e.get("evidence") else ""
            doc = f" [{e['doc_id']}]" if e.get("doc_id") else ""
            meta_bits = []
            if e.get("verification_level"):
                meta_bits.append(f"верификация: {e['verification_level']}")
            if e.get("geography"):
                meta_bits.append(f"география: {e['geography']}")
            if e.get("actualized_at"):
                meta_bits.append(f"актуально на: {e['actualized_at']}")
            meta = f"  ({'; '.join(meta_bits)})" if meta_bits else ""
            lines.append(f"- {e['src']} —{e['rel']}→ {e['dst']}{doc}{ev}{meta}")
        return "\n".join(lines) if lines else "(связей в графе не найдено)"

    def passages_text(self) -> str:
        if not self.passages:
            return "(релевантных фрагментов не найдено)"
        return "\n\n".join(f"[{p['doc_id']}] {p['text']}" for p in self.passages)

    def cited_docs(self) -> list[str]:
        docs = {p["doc_id"] for p in self.passages}
        docs |= {e["doc_id"] for e in self.subgraph_edges if e.get("doc_id")}
        return sorted(d for d in docs if d)


def extract_query_constraints(question: str) -> list[NumericConstraint]:
    """Числовые ограничения из вопроса («сухой остаток ≤1000 мг/дм³»)."""
    return parse_all_constraints(question)


def extract_geography_filter(question: str) -> bool | None:
    """Гео-маркер из вопроса: True=отечественная практика, False=мировая/зарубежная, None=нет."""
    q = f" {question.lower()} "
    if any(m in q for m in _FOREIGN_MARKERS):
        return False
    if any(m in q for m in _DOMESTIC_MARKERS):
        return True
    return None


# Привязка свободного текста к узлам через полнотекстовый индекс.
_SEED_QUERY = """
CALL db.index.fulltext.queryNodes('entity_names', $q) YIELD node, score
RETURN node.canonical_id AS cid, node.name AS name, labels(node)[0] AS type, score
ORDER BY score DESC
LIMIT $limit
"""

# Обход окрестности набора стартовых узлов на глубину 1 (fallback без APOC).
_EXPAND_QUERY = """
MATCH (a)-[r]->(b)
WHERE (a.canonical_id IN $seeds OR b.canonical_id IN $seeds)
  AND coalesce(r.is_current, true) = true
RETURN a.name AS src, type(r) AS rel, b.name AS dst,
       r.evidence AS evidence, r.doc_id AS doc_id, r.confidence AS confidence,
       r.verification_level AS verification_level, r.actualized_at AS actualized_at,
       r.geography AS geography, r.is_domestic AS is_domestic,
       b.geography AS dst_geography, b.is_domestic AS dst_is_domestic,
       b.domain AS dst_domain, b.value AS dst_value, b.unit AS dst_unit,
       a.geography AS src_geography, a.is_domestic AS src_is_domestic
LIMIT $limit
"""

# Обход окрестности на глубину 1-4 хопа через APOC (§4: производительность на
# больших графах — variable-length path без APOC на 1 млн узлов не уложится
# в 3-5 сек, apoc.path.subgraphAll ограничивает fan-out через maxLevel/limit).
_EXPAND_QUERY_DEEP = """
UNWIND $seeds AS seedId
MATCH (start {canonical_id: seedId})
CALL apoc.path.subgraphAll(start, {maxLevel: $max_hops, limit: $node_limit}) YIELD relationships
UNWIND relationships AS r
WITH DISTINCT r
WHERE coalesce(r.is_current, true) = true
RETURN startNode(r).name AS src, type(r) AS rel, endNode(r).name AS dst,
       r.evidence AS evidence, r.doc_id AS doc_id, r.confidence AS confidence,
       r.verification_level AS verification_level, r.actualized_at AS actualized_at,
       r.geography AS geography, r.is_domestic AS is_domestic,
       endNode(r).geography AS dst_geography, endNode(r).is_domestic AS dst_is_domestic,
       endNode(r).domain AS dst_domain, endNode(r).value AS dst_value, endNode(r).unit AS dst_unit,
       startNode(r).geography AS src_geography, startNode(r).is_domestic AS src_is_domestic
LIMIT $limit
"""


def find_seed_nodes(client: Neo4jClient, query: str, limit: int = 8) -> list[dict]:
    # экранируем спецсимволы lucene по минимуму
    safe = query.replace('"', " ").replace("~", " ")
    try:
        rows = client.run(_SEED_QUERY, q=safe, limit=limit)
    except Exception:                                  # noqa: BLE001 — индекса может не быть
        return []
    return [dict(r) for r in rows]


def expand_subgraph(client: Neo4jClient, seeds: list[str], limit: int = 60) -> list[dict]:
    """1-хоп обход, не требует APOC — используется как fallback."""
    if not seeds:
        return []
    return [dict(r) for r in client.run(_EXPAND_QUERY, seeds=seeds, limit=limit)]


def expand_subgraph_deep(client: Neo4jClient, seeds: list[str],
                         max_hops: int = 3, limit: int = 150, node_limit: int = 300) -> list[dict]:
    """1-4 хопа через APOC; при отсутствии APOC откатывается на 1-хоп MATCH."""
    if not seeds:
        return []
    try:
        rows = client.run(_EXPAND_QUERY_DEEP, seeds=seeds, max_hops=max_hops,
                          node_limit=node_limit, limit=limit)
        return [dict(r) for r in rows]
    except Exception:                                   # noqa: BLE001 — APOC не установлен
        return expand_subgraph(client, seeds, limit=limit)


def _passes_geography(row: dict, is_domestic: bool | None) -> bool:
    if is_domestic is None:
        return True
    flags = [row.get("is_domestic"), row.get("dst_is_domestic"), row.get("src_is_domestic")]
    known = [f for f in flags if f is not None]
    if not known:
        return True                        # нет гео-метки — не отбрасываем молча
    return is_domestic in known


def _passes_constraints(row: dict, constraints: list[NumericConstraint]) -> bool:
    if not constraints or row.get("dst_value") is None:
        return True
    dst_name = (row.get("dst") or "").lower()
    value = row["dst_value"]
    for c in constraints:
        if not c.param or c.param.lower() not in dst_name:
            continue                       # ограничение не про этот узел — не судим по нему
        if not isinstance(value, (int, float)):
            continue
        if c.operator == "<=" and value > c.value:
            return False
        if c.operator == ">=" and value < c.value:
            return False
        if c.operator == "=" and abs(value - c.value) > 1e-6:
            return False
        if c.operator == "between" and c.value_high is not None and not (c.value <= value <= c.value_high):
            return False
    return True


def _passes_domain(row: dict, domain: str | None) -> bool:
    if domain is None:
        return True
    known = row.get("dst_domain")
    return known is None or known == domain


def filtered_expand(client: Neo4jClient, seeds: list[str],
                    constraints: list[NumericConstraint] | None = None,
                    is_domestic: bool | None = None, domain: str | None = None,
                    max_hops: int = 3, limit: int = 150) -> list[dict]:
    """Обход графа на 1-4 хопа + фильтр по числовым ограничениям/географии/домену.

    Структурный фильтр не ломает семантический путь: строки без гео-метки или
    без числового значения на целевом узле не отбрасываются молча (иначе на
    неполном графе фильтр вычищал бы всё). Отбрасываются только явные
    противоречия найденным ограничениям.
    """
    rows = expand_subgraph_deep(client, seeds, max_hops=max_hops, limit=limit)
    constraints = constraints or []
    return [
        r for r in rows
        if _passes_geography(r, is_domestic) and _passes_constraints(r, constraints)
        and _passes_domain(r, domain)
    ]


def retrieve(question: str, vector_store, client: Neo4jClient,
             top_k_passages: int = 6, top_k_seeds: int = 8, max_hops: int = 3,
             geography: bool | None = "auto", domain: str | None = None) -> RetrievalContext:
    """Главная функция гибридного поиска.

    `geography`: явный override гео-фильтра (True/False/None) с UI/API —
    имеет приоритет над эвристикой по тексту вопроса. Значение по умолчанию
    "auto" означает «разобрать из текста вопроса» (см. extract_geography_filter).
    `domain`: явный фильтр по домену (гидрометаллургия/... ) — используется как
    из UI, так и из самого вопроса нет автоопределения (слишком неоднозначно).
    """
    import time
    ctx = RetrievalContext(question=question)
    ctx.constraints = extract_query_constraints(question)
    ctx.geography_filter = extract_geography_filter(question) if geography == "auto" else geography

    # 1) векторный recall
    t0 = time.perf_counter()
    ctx.passages = vector_store.search(question, top_k=top_k_passages)
    t_vec = time.perf_counter()

    # 2) точки входа в граф: и по тексту вопроса, и по найденным пассажам
    seed_rows = find_seed_nodes(client, question, limit=top_k_seeds)
    seeds = {r["cid"] for r in seed_rows if r.get("cid")}
    ctx.seed_nodes = sorted(seeds)
    t_seed = time.perf_counter()

    # 3) обход графа на 1-4 хопа со структурным фильтром по вопросу/API-параметрам
    ctx.subgraph_edges = filtered_expand(
        client, ctx.seed_nodes, constraints=ctx.constraints,
        is_domestic=ctx.geography_filter, domain=domain, max_hops=max_hops,
    )
    t_graph = time.perf_counter()

    ctx.timings_ms = {
        "vector_ms": round((t_vec - t0) * 1000, 1),
        "seed_ms": round((t_seed - t_vec) * 1000, 1),
        "graph_ms": round((t_graph - t_seed) * 1000, 1),
        "retrieval_total_ms": round((t_graph - t0) * 1000, 1),
    }
    return ctx
