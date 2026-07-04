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
    geography_relaxed: bool = False            # гео-фильтр снят, т.к. опустошал подграф
    year_from: int | None = None               # временной диапазон (ТЗ: «за 5 лет»)
    year_to: int | None = None
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
        body = "\n".join(lines) if lines else "(связей в графе не найдено)"
        if self.geography_relaxed:
            note = ("(!) Источников запрошенной географии (напр. зарубежная практика) "
                    "в базе не найдено — ниже приведены доступные связи иной географии; "
                    "укажи это в ответе.\n")
            body = note + body
        return body

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


import re as _re
_YEAR = r"(19\d{2}|20\d{2})"
# Предлог не должен быть хвостом слова: «процесС 2010», «фаЗА 2 года» — иначе
# однобуквенные «с»/«за» матчатся внутри слов и включают фиктивный фильтр.
_NOT_WORD_TAIL = r"(?<![а-яёa-z])"
# Число 1900–2099 с единицей измерения после — величина, а не год («с 2000
# м³/сут», «до 2000 кПа»). «г» намеренно не в списке: «до 2020 г.» — это год.
_NO_UNIT = (r"(?!\d)"
            # после единицы — не буква (не \b: «³» в «м³» — словесный символ,
            # и \b между «м» и «³» не срабатывает)
            r"(?!\s*(?:мк?г|кг|км|мм|см|дм|мл|л|т|м|к?па|мпа|гпа|атм|бар|"
            r"м?вт|квт|к?дж|шт|об|ч|мин|сек|°|%)(?![а-яёa-z]))"
            r"(?!\s*[²³/])")


def extract_year_filter(question: str, now_year: int | None = None) -> tuple[int | None, int | None]:
    """Временной диапазон из вопроса (ТЗ: «за последние 5 лет», «с 2019», «за 2018–2022»).

    Возвращает (year_from, year_to); None означает открытую границу. Чистая
    функция — тестируется без БД. now_year для детерминизма в тестах.
    """
    from datetime import datetime
    now_year = now_year or datetime.now().year
    q = question.lower()

    # «за последние N лет» / «за N лет» / «последних N лет»
    m = _re.search(rf"{_NOT_WORD_TAIL}(?:за|последн\w*)\s+(?:последн\w*\s+)?(\d{{1,2}})\s+(?:год|лет|года)", q)
    if m:
        return now_year - int(m.group(1)), None

    # диапазон «2018–2022» / «2018-2022» / «с 2018 по 2022»
    m = _re.search(rf"{_YEAR}\s*(?:[-–—]|по|до)\s*{_YEAR}{_NO_UNIT}", q)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        return min(y1, y2), max(y1, y2)

    # «с 2019» / «после 2019» / «начиная с 2019»
    m = _re.search(rf"{_NOT_WORD_TAIL}(?:с|после|начиная с|from|since)\s+{_YEAR}{_NO_UNIT}", q)
    if m:
        return int(m.group(1)), None

    # «до 2020» / «по 2020» / «before 2020»
    m = _re.search(rf"{_NOT_WORD_TAIL}(?:до|по|before)\s+{_YEAR}{_NO_UNIT}", q)
    if m:
        return None, int(m.group(1))

    return None, None


# Привязка свободного текста к узлам через полнотекстовый индекс.
_SEED_QUERY = """
CALL db.index.fulltext.queryNodes('entity_names', $q) YIELD node, score
RETURN node.canonical_id AS cid, node.name AS name, labels(node)[0] AS type, score
ORDER BY score DESC
LIMIT $limit
"""

# Обход окрестности набора стартовых узлов на глубину 1 (fallback без APOC).
# $excluded_docs — doc_id публикаций вне запрошенного диапазона лет: фильтр по
# году применяется ДО LIMIT (пост-фильтрация обрезанной выборки схлопывала recall).
_EXPAND_QUERY = """
MATCH (a)-[r]->(b)
WHERE (a.canonical_id IN $seeds OR b.canonical_id IN $seeds)
  AND coalesce(r.is_current, true) = true
  AND (r.doc_id IS NULL OR NOT r.doc_id IN $excluded_docs)
RETURN a.name AS src, type(r) AS rel, b.name AS dst,
       labels(a)[0] AS src_type, labels(b)[0] AS dst_type,
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
  AND (r.doc_id IS NULL OR NOT r.doc_id IN $excluded_docs)
RETURN startNode(r).name AS src, type(r) AS rel, endNode(r).name AS dst,
       labels(startNode(r))[0] AS src_type, labels(endNode(r))[0] AS dst_type,
       r.evidence AS evidence, r.doc_id AS doc_id, r.confidence AS confidence,
       r.verification_level AS verification_level, r.actualized_at AS actualized_at,
       r.geography AS geography, r.is_domestic AS is_domestic,
       endNode(r).geography AS dst_geography, endNode(r).is_domestic AS dst_is_domestic,
       endNode(r).domain AS dst_domain, endNode(r).value AS dst_value, endNode(r).unit AS dst_unit,
       startNode(r).geography AS src_geography, startNode(r).is_domestic AS src_is_domestic
LIMIT $limit
"""


# Стоп-слова вопроса: не несут сущности, но раздувают wildcard-запрос.
_SEED_STOPWORDS = {
    "какие", "какая", "какой", "каких", "что", "как", "где", "когда", "чем",
    "для", "при", "про", "под", "над", "без", "или", "если", "это", "эти",
    "описаны", "описан", "существуют", "применялись", "применяют", "используют",
    "используются", "считается", "считаются", "покажите", "перечислите", "обзор",
    "литобзор", "сравнение", "анализ", "решения", "решений", "методы", "методов",
    "способы", "способов", "практике", "практика", "мировой", "мировая", "россии",
    "рубежом", "зарубежных", "отечественной", "последних", "которые", "также",
    "which", "what", "how", "the", "and", "for", "with", "are", "were",
}
# слово вопроса: кириллица/латиница/дефис, длиной ≥4
_SEED_WORD_RE = _re.compile(r"[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z\-]{3,}")


def _build_seed_lucene(query: str) -> str:
    """Собрать префиксный wildcard-запрос из значимых слов вопроса.

    Стандартный анализатор fulltext Neo4j НЕ стеммит русский: «электроэкстракции»
    (родительный) не матчит индексированное «электроэкстракция». Отбрасываем
    окончание и добавляем `*` — `электроэкстракци*` ловит все падежи. Повышает
    recall seed-узлов; downstream реранкинг/фильтры отсекают лишнее.
    """
    terms: list[str] = []
    for w in _SEED_WORD_RE.findall(query):
        low = w.lower()
        if low in _SEED_STOPWORDS:
            continue
        stem = low[:-1] if len(low) > 5 else low       # срезаем 1 символ окончания
        terms.append(stem + "*")
    return " ".join(dict.fromkeys(terms))              # уник, сохраняя порядок


def find_seed_nodes(client: Neo4jClient, query: str, limit: int = 8) -> list[dict]:
    """Seed-узлы графа по вопросу. Сначала морфология-толерантный wildcard-запрос
    (падежи русского), при пустом результате — исходный текст как fallback."""
    seen: dict[str, dict] = {}
    for q in (_build_seed_lucene(query), query.replace('"', " ").replace("~", " ")):
        if not q.strip():
            continue
        try:
            rows = client.run(_SEED_QUERY, q=q, limit=limit)
        except Exception:                              # noqa: BLE001 — индекс/синтаксис
            continue
        for r in rows:
            d = dict(r)
            if d.get("cid") and d["cid"] not in seen:
                seen[d["cid"]] = d
        if len(seen) >= limit:
            break
    return list(seen.values())[:limit]


def expand_subgraph(client: Neo4jClient, seeds: list[str], limit: int = 60,
                    excluded_docs: list[str] | None = None) -> list[dict]:
    """1-хоп обход, не требует APOC — используется как fallback."""
    if not seeds:
        return []
    return [dict(r) for r in client.run(_EXPAND_QUERY, seeds=seeds, limit=limit,
                                        excluded_docs=excluded_docs or [])]


def expand_subgraph_deep(client: Neo4jClient, seeds: list[str],
                         max_hops: int = 3, limit: int = 150, node_limit: int = 300,
                         excluded_docs: list[str] | None = None) -> list[dict]:
    """1-4 хопа через APOC; при отсутствии APOC откатывается на 1-хоп MATCH."""
    if not seeds:
        return []
    try:
        rows = client.run(_EXPAND_QUERY_DEEP, seeds=seeds, max_hops=max_hops,
                          node_limit=node_limit, limit=limit,
                          excluded_docs=excluded_docs or [])
        return [dict(r) for r in rows]
    except Exception:                                   # noqa: BLE001 — APOC не установлен
        return expand_subgraph(client, seeds, limit=limit, excluded_docs=excluded_docs)


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
                    max_hops: int = 3, limit: int = 150,
                    excluded_docs: list[str] | None = None) -> list[dict]:
    """Обход графа на 1-4 хопа + фильтр по числовым ограничениям/географии/домену.

    Структурный фильтр не ломает семантический путь: строки без гео-метки или
    без числового значения на целевом узле не отбрасываются молча (иначе на
    неполном графе фильтр вычищал бы всё). Отбрасываются только явные
    противоречия найденным ограничениям. `excluded_docs` (фильтр по годам)
    применяется внутри Cypher — до LIMIT.
    """
    rows = expand_subgraph_deep(client, seeds, max_hops=max_hops, limit=limit,
                                excluded_docs=excluded_docs)
    constraints = constraints or []
    return [
        r for r in rows
        if _passes_geography(r, is_domestic) and _passes_constraints(r, constraints)
        and _passes_domain(r, domain)
    ]


# Годы публикаций по их doc_id — для временного фильтра (год лежит на Publication,
# а рёбра/пассажи ссылаются на источник через doc_id). Выборка полная, а не по
# списку cid: годы нужны ДО обхода графа, чтобы фильтр применился до LIMIT.
_PUB_YEARS_QUERY = """
MATCH (p:Publication) WHERE p.year IS NOT NULL
RETURN p.canonical_id AS cid, p.year AS year
"""


def _publication_years(client: Neo4jClient) -> dict[str, int]:
    try:
        rows = client.run(_PUB_YEARS_QUERY)
    except Exception as exc:                            # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "годы публикаций недоступны — фильтр по годам не применён (%s)", exc)
        return {}
    out: dict[str, int] = {}
    for r in rows:
        cid, year = r.get("cid") or "", r.get("year")
        # canonical_id имеет вид 'Publication:<doc_id>' — парсим в Python, чтобы
        # смена формата дала пустой результат с warning, а не тихий no-op в Cypher
        if year is not None and ":" in cid:
            out[cid.split(":", 1)[1]] = year
    return out


def _passes_year(year: int | None, yf: int | None, yt: int | None) -> bool:
    if year is None:
        return True                    # неизвестный год не отбрасываем молча (неполный граф)
    if yf is not None and year < yf:
        return False
    if yt is not None and year > yt:
        return False
    return True


def retrieve(question: str, vector_store, client: Neo4jClient,
             top_k_passages: int = 6, top_k_seeds: int = 8, max_hops: int = 3,
             geography: bool | None = "auto", domain: str | None = None,
             year_from: int | None = None, year_to: int | None = None,
             rerank_top_k: int = 40, graph_limit: int = 60) -> RetrievalContext:
    """Главная функция гибридного поиска.

    `geography`: явный override гео-фильтра (True/False/None) с UI/API —
    имеет приоритет над эвристикой по тексту вопроса. Значение по умолчанию
    "auto" означает «разобрать из текста вопроса» (см. extract_geography_filter).
    `domain`: явный фильтр по домену (гидрометаллургия/... ).
    `year_from`/`year_to`: временной диапазон (ТЗ). Если оба None — разбираем из
    текста вопроса («за последние 5 лет»). Фильтрует пассажи и рёбра по году
    публикации-источника; узлы с неизвестным годом не отбрасываются.
    """
    import time
    ctx = RetrievalContext(question=question)
    ctx.constraints = extract_query_constraints(question)
    ctx.geography_filter = extract_geography_filter(question) if geography == "auto" else geography
    if year_from is None and year_to is None:
        year_from, year_to = extract_year_filter(question)
    ctx.year_from, ctx.year_to = year_from, year_to

    # 1) векторный recall (НЕ фатально: при сбое эмбеддера/квоты деградируем в
    # graph-only — seed-узлы берутся из fulltext Neo4j, обход графа не требует
    # векторов, так что ответ всё равно строится по подграфу).
    t0 = time.perf_counter()
    try:
        ctx.passages = vector_store.search(question, top_k=top_k_passages)
    except Exception as exc:                              # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning("векторный recall пропущен (%s)", exc)
        ctx.passages = []
    t_vec = time.perf_counter()

    # 2) точки входа в граф: и по тексту вопроса, и по найденным пассажам
    seed_rows = find_seed_nodes(client, question, limit=top_k_seeds)
    seeds = {r["cid"] for r in seed_rows if r.get("cid")}
    ctx.seed_nodes = sorted(seeds)
    t_seed = time.perf_counter()

    # 3) временной фильтр по году публикации-источника: вычисляем исключённые
    # doc_id ЗАРАНЕЕ и передаём в Cypher-обход — фильтрация после LIMIT молча
    # схлопывала recall (свежие рёбра за пределами первых LIMIT терялись).
    excluded_docs: list[str] = []
    if ctx.year_from is not None or ctx.year_to is not None:
        years = _publication_years(client)
        excluded_docs = [d for d, y in years.items()
                         if not _passes_year(y, ctx.year_from, ctx.year_to)]
        if excluded_docs:
            excluded_set = set(excluded_docs)
            ctx.passages = [p for p in ctx.passages if p.get("doc_id") not in excluded_set]

    # обход графа на 1-4 хопа со структурным фильтром по вопросу/API-параметрам
    ctx.subgraph_edges = filtered_expand(
        client, ctx.seed_nodes, constraints=ctx.constraints,
        is_domestic=ctx.geography_filter, domain=domain, max_hops=max_hops,
        limit=graph_limit, excluded_docs=excluded_docs,
    )
    # Мягкий откат гео-фильтра: если строгий фильтр опустошил подграф (в корпусе
    # нет источников запрошенной практики — типично для «мировой практики» на
    # преимущественно отечественном корпусе), повторяем без гео и помечаем —
    # честнее ответить с оговоркой «зарубежных источников нет, вот доступные»,
    # чем «нет информации». Числовые ограничения при этом сохраняем.
    if not ctx.subgraph_edges and ctx.geography_filter is not None and ctx.seed_nodes:
        relaxed = filtered_expand(
            client, ctx.seed_nodes, constraints=ctx.constraints,
            is_domestic=None, domain=domain, max_hops=max_hops,
            limit=graph_limit, excluded_docs=excluded_docs,
        )
        if relaxed:
            ctx.subgraph_edges = relaxed
            ctx.geography_relaxed = True
    t_graph = time.perf_counter()

    # 4) реранкинг рёбер по релевантности вопросу (bi-encoder) — обход даёт рёбра
    # «в порядке графа», реранк выносит вперёд самые relevantные и режет до
    # rerank_top_k (precision контекста для LLM). Только ЛОКАЛЬНЫЙ эмбеддер:
    # удалённый (Yandex, 1 текст = 1 HTTP-вызов с троттлингом) стоил бы до
    # graph_limit вызовов квоты и ~30с на первый вопрос.
    emb = getattr(vector_store, "embedder", None)
    if emb is not None and not getattr(emb, "remote", False) and ctx.subgraph_edges:
        from klubok.retrieval.rerank import rerank_edges
        ctx.subgraph_edges = rerank_edges(question, ctx.subgraph_edges, emb, top_k=rerank_top_k)
    t_rerank = time.perf_counter()

    ctx.timings_ms = {
        "vector_ms": round((t_vec - t0) * 1000, 1),
        "seed_ms": round((t_seed - t_vec) * 1000, 1),
        "graph_ms": round((t_graph - t_seed) * 1000, 1),
        "rerank_ms": round((t_rerank - t_graph) * 1000, 1),
        "retrieval_total_ms": round((t_rerank - t0) * 1000, 1),
    }
    return ctx
