"""Юнит-тесты слоёв, не требующих GPU/БД.

Запуск:  pytest -q
Покрывают: онтологию, нормализацию величин, резолвинг сущностей,
парсинг ответа LLM, чанкинг, mock-эмбеддер + поиск в памяти, mock-LLM.
"""
from __future__ import annotations

from klubok.ontology import (
    Entity, Relation, NodeType, RelType, edge_is_valid, ExtractionResult,
)
from klubok.extraction import normalize as nz
from klubok.extraction.resolver import (
    canonical_material, canonical_element, resolve, canonical_id,
)
from klubok.extraction.extractor import parse_extraction
from klubok.extraction.llm_client import MockLLM
from klubok.parsing.pdf_parser import chunk_text
from klubok.vectorstore.embeddings import MockEmbedder
from klubok.vectorstore.store import InMemoryVectorStore


# --- онтология ---
def test_allowed_edges():
    assert edge_is_valid(NodeType.EXPERIMENT, RelType.USES, NodeType.MATERIAL)
    assert not edge_is_valid(NodeType.MATERIAL, RelType.USES, NodeType.PROPERTY)


def test_schema_filters_bad_relations():
    good = Relation(src_name="E1", src_type=NodeType.EXPERIMENT, rel=RelType.USES,
                    dst_name="CuNi", dst_type=NodeType.MATERIAL)
    bad = Relation(src_name="CuNi", src_type=NodeType.MATERIAL, rel=RelType.USES,
                   dst_name="X", dst_type=NodeType.PROPERTY)
    res = ExtractionResult(doc_id="d", entities=[], relations=[good, bad])
    assert res.schema_valid_relations() == [good]


# --- нормализация величин ---
def test_parse_quantity_temp():
    q = nz.parse_quantity("отжиг при 800 °C в течение часа")
    assert q.value == 800.0 and q.unit == "°C"


def test_kelvin_to_celsius():
    q = nz.to_celsius(nz.parse_quantity("1073 K"))
    assert abs(q.value - 799.85) < 1e-6 and q.unit == "°C"


def test_time_to_hours():
    assert nz.to_hours(nz.parse_quantity("120 min")).value == 2.0


def test_parse_all_quantities_and_dimension():
    qs = nz.parse_all_quantities("твёрдость 120 HV при 500 MPa")
    units = {q.unit for q in qs}
    assert "HV" in units and "MPa" in units
    assert nz.dimension_of(nz.parse_quantity("120 HV")) == "hardness"


# --- числовые ограничения (§4 гибридного поиска, §2 извлечения) ---
def test_parse_constraint_le():
    c = nz.parse_constraint("сульфаты не более 300 мг/л")
    assert c.operator == "<=" and c.value == 300.0 and c.unit == "mg/L"


def test_parse_constraint_ge():
    c = nz.parse_constraint("производительность от 100 т/сут")
    assert c.operator == ">=" and c.value == 100.0 and c.unit == "t/day"


def test_parse_constraint_between():
    c = nz.parse_constraint("200-300 мг/л")
    assert c.operator == "between" and c.value == 200.0 and c.value_high == 300.0


def test_parse_constraint_equality_fallback():
    c = nz.parse_constraint("расход 0.5 м3/ч")
    assert c.operator == "=" and c.value == 0.5 and c.unit == "m3/h"


def test_parse_constraint_no_match_returns_none():
    assert nz.parse_constraint("без каких-либо чисел тут") is None


def test_parse_all_constraints_multiple():
    cs = nz.parse_all_constraints("сульфаты ≤300 мг/л, а хлориды ≥50 мг/л")
    ops = {c.operator for c in cs}
    assert ops == {"<=", ">="}


# --- резолвинг ---
def test_canonical_material_variants_collapse():
    assert canonical_material("Cu-Ni") == "CuNi"
    assert canonical_material("CuNi") == "CuNi"
    assert canonical_material("Ni-Cu") == "CuNi"          # порядок не важен
    assert canonical_material("медно-никелевый") == "CuNi"


def test_canonical_element_ru():
    assert canonical_element("медь") == "Cu"
    assert canonical_element("Ni") == "Ni"


def test_resolve_dedup():
    ents = [
        Entity(name="Cu-Ni", type=NodeType.MATERIAL),
        Entity(name="CuNi", type=NodeType.MATERIAL, attributes={"note": "x"}),
        Entity(name="твёрдость", type=NodeType.PROPERTY),
    ]
    resolved, mapping = resolve(ents)
    cids = {e.canonical_id for e in resolved}
    assert "Material:CuNi" in cids
    # два варианта сплава схлопнулись в один узел
    assert len([e for e in resolved if e.type == NodeType.MATERIAL]) == 1
    assert mapping["Material:Cu-Ni"] == "Material:CuNi"


def test_canonical_id_format():
    assert canonical_id(Entity(name="SEM", type=NodeType.METHOD)) == "Method:sem"


# --- новые типы узлов из ТЗ (Equipment/Expert/Facility/Publication/Condition) ---
def test_new_node_types_resolve():
    for name, ntype in [
        ("ванна электроэкстракции", NodeType.EQUIPMENT),
        ("Иванов И.И.", NodeType.EXPERT),
        ("ИАЦ", NodeType.FACILITY),
        ("расход католита", NodeType.CONDITION),
    ]:
        cid = canonical_id(Entity(name=name, type=ntype))
        assert cid.startswith(f"{ntype.value}:")


def test_glossary_synonyms_collapse_ru_en():
    assert canonical_id(Entity(name="electrowinning", type=NodeType.PROCESS)) == \
        canonical_id(Entity(name="электролиз", type=NodeType.PROCESS))
    assert canonical_id(Entity(name="ПВП", type=NodeType.EQUIPMENT)) == \
        canonical_id(Entity(name="fluidized bed furnace", type=NodeType.EQUIPMENT))


def test_new_relation_types_schema_valid():
    rel = Relation(src_name="электроэкстракция никеля", src_type=NodeType.PROCESS,
                   rel=RelType.OPERATES_AT_CONDITION, dst_name="расход католита",
                   dst_type=NodeType.CONDITION)
    assert rel.is_schema_valid()
    rel2 = Relation(src_name="Публикация 1", src_type=NodeType.PUBLICATION,
                    rel=RelType.AUTHORED_BY, dst_name="Иванов И.И.", dst_type=NodeType.EXPERT)
    assert rel2.is_schema_valid()


# --- метаданные из имени файла (§2 плана) ---
def test_filename_meta_author_and_lab():
    from klubok.parsing.filename_meta import parse_filename_meta
    m = parse_filename_meta("26 Статья - Великая Т.И. (ИАЦ).docx")
    assert m.author_name == "Великая Т.И." and m.lab_abbr == "ИАЦ"


def test_filename_meta_author_without_lab():
    from klubok.parsing.filename_meta import parse_filename_meta
    m = parse_filename_meta("Доклад_Вострикова Н.М.pdf")
    assert m.author_name == "Вострикова Н.М." and m.lab_abbr is None


def test_filename_meta_no_match_is_safe():
    from klubok.parsing.filename_meta import parse_filename_meta
    m = parse_filename_meta("11 KorzhakovAA 811.docx")
    assert m.author_name is None and m.lab_abbr is None


# --- уведомления: чистая логика матчинга подписок (§Y7, без БД) ---
def test_match_topics_substring_both_directions():
    from klubok.notify.watchlist import match_topics
    ents = ["циркуляция католита", "выход по току"]
    hits = match_topics(ents, ["католит", "электроэкстракция никеля"])
    matched_topics = {t for t, _ in hits}
    assert "католит" in matched_topics                 # тема ⊆ сущность
    assert "электроэкстракция никеля" not in matched_topics


def test_match_topics_uses_ru_en_glossary():
    from klubok.notify.watchlist import match_topics
    # новая статья на английском, подписка на русском — глоссарий должен связать
    hits = match_topics(["electrowinning"], ["электроэкстракция"])
    assert hits and hits[0][0] == "электроэкстракция"


def test_match_topics_no_false_positive():
    from klubok.notify.watchlist import match_topics
    assert match_topics(["флотация медной руды"], ["обессоливание воды"]) == []


def test_watchstore_end_to_end(tmp_path):
    from klubok.notify.watchlist import WatchStore
    ws = WatchStore(path=tmp_path / "wl.sqlite")
    try:
        ws.subscribe("analyst", "католит")
        n = ws.notify_new_document("doc1", "Циркуляция католита в ваннах",
                                   ["циркуляция католита", "выход по току"])
        assert n == 1
        feed = ws.feed("analyst")
        assert len(feed) == 1 and feed[0]["doc_id"] == "doc1"
        # чужие подписки не задеты
        assert ws.feed("researcher") == []
    finally:
        ws.close()


# --- парсинг ответа LLM (устойчивость к грязному JSON) ---
def test_parse_extraction_strips_markdown():
    raw = '```json\n{"entities": [{"name": "CuNi", "type": "Material"}], "relations": []}\n```'
    res = parse_extraction(raw, doc_id="d1", chunk_id="c1")
    assert len(res.entities) == 1 and res.entities[0].type == NodeType.MATERIAL


def test_parse_extraction_drops_invalid_relation():
    raw = ('{"entities": [], "relations": ['
           '{"src_name":"CuNi","src_type":"Material","rel":"USES",'
           '"dst_name":"X","dst_type":"Property"}]}')   # запрещено онтологией
    res = parse_extraction(raw, doc_id="d", chunk_id=None)
    assert res.relations == []


def test_parse_extraction_garbage_is_safe():
    res = parse_extraction("модель что-то наговорила без json", doc_id="d", chunk_id=None)
    assert res.entities == [] and res.relations == []


# --- чанкинг ---
def test_chunk_text_empty_returns_nothing():
    # скан/пустой документ -> нет чанков -> pipeline пропустит его до LLM
    assert chunk_text("", doc_id="d") == []
    assert chunk_text("   \n\n\t  ", doc_id="d") == []


def test_chunk_text_respects_max_chars():
    text = "Абзац один.\n\n" + ("Длинное предложение. " * 200)
    chunks = chunk_text(text, doc_id="d", page=1, max_chars=500)
    assert chunks and all(len(c.text) <= 700 for c in chunks)  # с запасом на overlap
    assert all(c.doc_id == "d" for c in chunks)


# --- эмбеддер + поиск в памяти (без Qdrant) ---
def test_mock_embedder_shapes_and_norm():
    emb = MockEmbedder(dim=256)
    v = emb.encode(["сплав CuNi", "отжиг 800 C"])
    assert v.shape == (2, 256)
    import numpy as np
    assert abs(np.linalg.norm(v[0]) - 1.0) < 1e-5


def test_inmemory_search_ranks_relevant_first():
    from klubok.ontology import Chunk
    emb = MockEmbedder(dim=512)
    store = InMemoryVectorStore(emb)
    store.index_chunks([
        Chunk(chunk_id="c1", doc_id="d1", text="отжиг сплава CuNi при 800 C повышает твёрдость"),
        Chunk(chunk_id="c2", doc_id="d2", text="коррозия титановых сплавов в морской воде"),
    ])
    hits = store.search("твёрдость CuNi после отжига", top_k=2)
    assert hits[0]["chunk_id"] == "c1"


# --- консистентность few-shot примеров (ловит опечатки в эталонах) ---
def test_fewshot_examples_are_self_consistent():
    from klubok.extraction.prompts import FEW_SHOT_EXAMPLES
    for ex in FEW_SHOT_EXAMPLES:
        out = ex["output"]
        defined = {(e["name"], e["type"]) for e in out["entities"]}
        for r in out["relations"]:
            # каждая сущность из связи должна быть определена в entities
            assert (r["src_name"], r["src_type"]) in defined, r
            assert (r["dst_name"], r["dst_type"]) in defined, r
            # связь должна быть валидна по онтологии
            rel = Relation(
                src_name=r["src_name"], src_type=NodeType(r["src_type"]),
                rel=RelType(r["rel"]), dst_name=r["dst_name"],
                dst_type=NodeType(r["dst_type"]),
            )
            assert rel.is_schema_valid(), r


def test_fewshot_examples_parse_through_extractor():
    """Эталонный JSON проходит реальный парсер без потерь связей."""
    import json
    from klubok.extraction.prompts import FEW_SHOT_EXAMPLES
    for ex in FEW_SHOT_EXAMPLES:
        raw = json.dumps(ex["output"], ensure_ascii=False)
        res = parse_extraction(raw, doc_id="d", chunk_id="c")
        assert len(res.relations) == len(ex["output"]["relations"])


# --- эвристики верификации/гео/домена ---
def test_verification_from_evidence():
    from klubok.extraction.heuristics import verification_from_evidence
    assert verification_from_evidence("твёрдость составила 168 HV") == "confirmed"
    assert verification_from_evidence("предполагается рост извлечения") == "preliminary"
    assert verification_from_evidence("образцы сплава CuNi") == "unverified"
    assert verification_from_evidence(None) == "unverified"


def test_detect_is_domestic():
    from klubok.extraction.heuristics import detect_is_domestic
    ru = "Обеднение шлаков медеплавильного производства методом электротермии " * 8
    en = "Copper matte converting and slag cleaning in flash smelting furnace " * 8
    assert detect_is_domestic(ru) == (True, "Россия")
    assert detect_is_domestic(en)[0] is False
    assert detect_is_domestic("короткий текст") == (None, None)   # мало текста


def test_detect_domain():
    from klubok.extraction.heuristics import detect_domain
    assert detect_domain("электроэкстракция никеля, циркуляция католита, раствор") == "гидрометаллургия"
    assert detect_domain("плавка в печи взвешенной плавки, штейн и шлак") == "пирометаллургия"
    assert detect_domain("флотация руды и обогащение концентрата") == "обогащение"
    assert detect_domain("нейтральный текст без ключевых слов") is None


# --- временной фильтр из вопроса ---
def test_extract_year_filter_last_n_years():
    from klubok.retrieval.graphrag import extract_year_filter
    assert extract_year_filter("публикации за последние 5 лет", now_year=2026) == (2021, None)


def test_extract_year_filter_range():
    from klubok.retrieval.graphrag import extract_year_filter
    assert extract_year_filter("эксперименты за 2018–2022", now_year=2026) == (2018, 2022)
    assert extract_year_filter("с 2018 по 2022", now_year=2026) == (2018, 2022)


def test_extract_year_filter_since_and_before():
    from klubok.retrieval.graphrag import extract_year_filter
    assert extract_year_filter("работы с 2019 года", now_year=2026) == (2019, None)
    assert extract_year_filter("статьи до 2015", now_year=2026) == (None, 2015)


def test_extract_year_filter_none_and_no_false_positive():
    from klubok.retrieval.graphrag import extract_year_filter
    assert extract_year_filter("методы обессоливания воды", now_year=2026) == (None, None)
    # концентрации/температуры не должны попадать в годы
    assert extract_year_filter("сульфаты 300 мг/л при 80 C", now_year=2026) == (None, None)


def test_passes_year_unknown_kept():
    from klubok.retrieval.graphrag import _passes_year
    assert _passes_year(None, 2020, None) is True       # неизвестный год не отбрасываем
    assert _passes_year(2019, 2020, None) is False
    assert _passes_year(2021, 2020, 2022) is True
    assert _passes_year(2023, None, 2022) is False


# --- реранкинг рёбер по релевантности вопросу ---
def test_rerank_edges_orders_by_relevance():
    from klubok.retrieval.rerank import rerank_edges
    from klubok.vectorstore.embeddings import MockEmbedder
    edges = [
        {"src": "вода", "rel": "APPLIES", "dst": "закачка шахтных вод", "evidence": ""},
        {"src": "шлак", "rel": "RESULTS_IN", "dst": "обеднение шлаков", "evidence": "обеднение"},
    ]
    ranked = rerank_edges("обеднение шлаков", edges, MockEmbedder(dim=512), top_k=40)
    assert ranked[0]["dst"] == "обеднение шлаков"        # релевантное ребро — первым
    assert "rerank_score" in ranked[0]


def test_rerank_edges_respects_top_k_and_empty():
    from klubok.retrieval.rerank import rerank_edges
    from klubok.vectorstore.embeddings import MockEmbedder
    emb = MockEmbedder(dim=128)
    many = [{"src": f"m{i}", "rel": "USES", "dst": f"d{i}", "evidence": ""} for i in range(50)]
    assert len(rerank_edges("запрос", many, emb, top_k=10)) == 10
    assert rerank_edges("q", [], emb) == []


def test_rerank_edges_embedder_failure_is_safe():
    from klubok.retrieval.rerank import rerank_edges
    class _BadEmb:
        def encode(self, *a, **k): raise RuntimeError("down")
        def encode_query(self, *a, **k): raise RuntimeError("down")
    edges = [{"src": "a", "rel": "USES", "dst": "b"}, {"src": "c", "rel": "USES", "dst": "d"}]
    assert rerank_edges("q", edges, _BadEmb(), top_k=1) == edges[:1]   # фолбэк без краша


# --- метрики оценки извлечения ---
def test_prf_perfect_and_partial():
    from klubok.eval.metrics import prf_from_sets
    perfect = prf_from_sets({1, 2, 3}, {1, 2, 3})
    assert perfect.precision == 1.0 and perfect.recall == 1.0 and perfect.f1 == 1.0
    partial = prf_from_sets({1, 2, 4}, {1, 2, 3})   # tp=2, fp=1, fn=1
    assert partial.tp == 2 and partial.fp == 1 and partial.fn == 1
    assert abs(partial.f1 - (2 * (2/3) * (2/3)) / ((2/3) + (2/3))) < 1e-9


def test_score_extraction_matches_canonical_variants():
    from klubok.eval.metrics import score_extraction, gold_from_dict
    gold = gold_from_dict({
        "entities": [{"name": "CuNi", "type": "Material"}],
        "relations": [{"src_name": "E1", "src_type": "Experiment", "rel": "USES",
                       "dst_name": "CuNi", "dst_type": "Material"}],
    })
    # предсказание пишет сплав иначе — канонизация должна засчитать совпадение
    pred = gold_from_dict({
        "entities": [{"name": "Cu-Ni", "type": "Material"}],
        "relations": [{"src_name": "E1", "src_type": "Experiment", "rel": "USES",
                       "dst_name": "Cu-Ni", "dst_type": "Material"}],
    })
    s = score_extraction(pred, gold)
    assert s.relations.f1 == 1.0


def test_gold_from_dict_perfect_self_score():
    from klubok.eval.metrics import score_extraction, gold_from_dict
    from klubok.eval.gold_set import GOLD_SET
    for ex in GOLD_SET:
        g = gold_from_dict(ex["output"])
        s = score_extraction(g, g)              # сам с собой -> идеально
        assert s.relations.f1 == 1.0 and s.entities.f1 == 1.0


def test_gold_set_is_held_out():
    """Held-out набор не должен пересекаться с few-shot примерами (защита от утечки)."""
    from klubok.extraction.prompts import FEW_SHOT_EXAMPLES
    from klubok.eval.gold_set import GOLD_SET
    fewshot_texts = {ex["text"] for ex in FEW_SHOT_EXAMPLES}
    gold_texts = {ex["text"] for ex in GOLD_SET}
    assert fewshot_texts.isdisjoint(gold_texts)


def test_gold_set_relations_schema_valid():
    from klubok.eval.metrics import gold_from_dict
    from klubok.eval.gold_set import GOLD_SET
    for ex in GOLD_SET:
        g = gold_from_dict(ex["output"])
        defined = {(e.name, e.type) for e in g.entities}
        for r in g.relations:
            assert r.is_schema_valid()
            assert (r.src_name, r.src_type) in defined
            assert (r.dst_name, r.dst_type) in defined


# --- устойчивость генерации ответа к пустому/падающему LLM ---
def test_generate_answer_empty_llm_fallback():
    from klubok.retrieval.graphrag import RetrievalContext
    from klubok.qa.answer import generate_answer

    class _EmptyLLM:
        def complete(self, prompt, system=""): return "   "

    class _RaisingLLM:
        def complete(self, prompt, system=""): raise RuntimeError("HTTP 429 quota")

    ctx = RetrievalContext(question="вопрос",
                           passages=[{"doc_id": "d1", "text": "t"}], subgraph_edges=[])
    for llm in (_EmptyLLM(), _RaisingLLM()):
        ans = generate_answer(ctx, llm)
        assert "Не удалось сгенерировать" in ans.text     # фолбэк, не краш
        assert ans.sources == ["d1"]                       # источники сохранены


# --- mock LLM прогоняет извлечение без GPU ---
def test_mock_llm_extraction_roundtrip():
    from klubok.extraction.prompts import build_extraction_prompt
    llm = MockLLM()
    raw = llm.complete(build_extraction_prompt("Образцы сплава CuNi отжигали при 800 C."))
    res = parse_extraction(raw, doc_id="d", chunk_id="c")
    names = {e.name for e in res.entities}
    assert "CuNi" in names
