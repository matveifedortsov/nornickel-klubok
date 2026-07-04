"""Демо-интерфейс на Streamlit — R&D карта знаний горно-металлургической отрасли.

Запуск:  streamlit run ui/streamlit_app.py
Ожидает поднятый API:  uvicorn klubok.api.app:app

Ключевые UX-решения:
- результат вопроса хранится в st.session_state и переживает rerun (клики по
  экспорту/фильтрам не стирают ответ);
- экспорт (MD/JSON-LD/PDF) строится ЛОКАЛЬНО из полученного ответа
  (klubok.export) — без повторного вызова API/LLM, работает и при недоступной
  генерации;
- все обращения к API обёрнуты в обработку ошибок (API недоступен -> баннер).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import requests
import streamlit as st

# корень репозитория в путь -> локальный экспорт без обращения к API
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from klubok.export import to_markdown, to_json_ld, to_pdf
    _LOCAL_EXPORT = True
except Exception:                                        # noqa: BLE001
    _LOCAL_EXPORT = False

API = os.environ.get("KLUBOK_API_URL", "http://localhost:8000")

ROLE_KEYS = {
    "researcher": "dev-researcher",
    "analyst": "dev-analyst",
    "project_lead": "dev-lead",
    "admin": "dev-admin",
    "external_partner": "dev-partner",
}

# 4 эталонных вопроса ТЗ — быстрые кнопки для демо
EXAMPLE_QUESTIONS = [
    ("💧 Обессоливание воды",
     "Какие методы обессоливания воды подходят для обогатительной фабрики, если "
     "исходная вода содержит сульфаты, хлориды, Ca, Mg, Na по 200–300 мг/л, а "
     "требуемый сухой остаток — ≤1000 мг/дм³?"),
    ("⚡ Циркуляция католита",
     "Какие технические решения организации циркуляции католита при электроэкстракции "
     "никеля описаны в мировой практике, и какая скорость потока считается оптимальной?"),
    ("🥇 Au/Ag/МПГ штейн-шлак",
     "Покажите эксперименты и публикации по распределению Au, Ag и МПГ между медным/"
     "никелевым штейном и шлаком за последние 5 лет."),
    ("🕳 Закачка шахтных вод",
     "Какие способы закачки шахтных вод в глубокие горизонты применялись в России и "
     "за рубежом, и каковы их технико-экономические показатели?"),
    ("🇬🇧 EN: catholyte flow",
     "What technical solutions for catholyte circulation in nickel electrowinning are "
     "described in world practice, and what flow rate is considered optimal?"),
]

st.set_page_config(page_title="Научный клубок", layout="wide", page_icon="🧶")

st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; }
  [data-testid="stMetricValue"] { color: #0072CE; }
  .stTabs [data-baseweb="tab"] { font-weight: 600; }
  .klubok-hero {
    background: linear-gradient(90deg, #0A2540 0%, #0072CE 100%);
    color: #FFFFFF; border-radius: 12px; padding: 18px 24px; margin-bottom: 12px;
  }
  .klubok-hero h1 { color: #FFFFFF; font-size: 1.7rem; margin: 0; }
  .klubok-hero p  { color: #CFE3F5; margin: 4px 0 0 0; font-size: 0.95rem; }
</style>
<div class="klubok-hero">
  <h1>🧶 Научный клубок</h1>
  <p>R&D карта знаний горно-металлургической отрасли · граф + гибридный поиск · Норникель AI Science Hack</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# HTTP-обёртки с обработкой ошибок
# --------------------------------------------------------------------------
def _headers() -> dict:
    return {"X-API-Key": ROLE_KEYS[st.session_state.get("role", "researcher")]}


def _api(method: str, path: str, **kw):
    """Вызов API. Возвращает (ok, data|Response|None). Ошибки не роняют UI."""
    try:
        resp = requests.request(method, f"{API}{path}", headers=_headers(), timeout=180, **kw)
        return True, resp
    except requests.exceptions.RequestException as exc:
        st.session_state["_api_error"] = str(exc)
        return False, None


@st.cache_data(ttl=15, show_spinner=False)
def _cached_get_json(path: str, role: str, params: tuple = ()) -> dict | None:
    """GET с коротким кэшем: сайдбар/списки не дёргают API на каждый rerun.

    `role` в ключе кэша — ответы зависят от X-API-Key. TTL малый, чтобы после
    ингеста/подписки данные обновлялись без перезапуска UI.
    """
    ok, resp = _api("GET", path, params=dict(params))
    if ok and resp.status_code == 200:
        return resp.json()
    return None


def _health() -> dict | None:
    return _cached_get_json("/health", st.session_state.get("role", "researcher"))


# --------------------------------------------------------------------------
# Сайдбар: роль + статус API
# --------------------------------------------------------------------------
with st.sidebar:
    st.session_state["role"] = st.selectbox("Роль", list(ROLE_KEYS.keys()), index=0)
    st.caption("Определяет видимость внутренних данных и доступ к «Пробелам»/«Дашборду»")
    st.divider()
    health = _health()
    if health:
        st.success(f"API online · узлов в графе: {health.get('nodes', '—')}")
    else:
        st.error("API недоступен. Запустите: `uvicorn klubok.api.app:app`")
    # бейдж непрочитанных уведомлений (кэш — не дёргаем API на каждый rerun)
    notif = _cached_get_json("/notifications", st.session_state["role"],
                             params=(("unseen_only", True),))
    if notif:
        unseen = len(notif.get("notifications", []))
        if unseen:
            st.warning(f"🔔 Новое по вашим темам: {unseen}")

if not health:
    st.stop()      # без API остальное бессмысленно


# --------------------------------------------------------------------------
# Визуализация подграфа
# --------------------------------------------------------------------------
_VERIF_COLOR = {"disputed": "#e74c3c", "preliminary": "#f39c12", "confirmed": "#27ae60"}
_VERIF_DEFAULT = "#95a5a6"

# Цвета узлов по типу сущности
_TYPE_COLOR = {
    "Material": "#0072CE", "Process": "#E8833A", "Experiment": "#6A4C93",
    "Property": "#1D9E75", "Publication": "#7F8C9B", "Expert": "#D4537E",
    "Facility": "#C0392B", "Equipment": "#16A085", "Condition": "#C9A227",
    "Method": "#2C82C9", "Phase": "#8E44AD", "Element": "#5DADE2",
}
_TYPE_DEFAULT = "#34495e"


def _render_graph(sg: dict) -> None:
    edges = sg.get("edges", [])
    if not edges:
        st.info("Связей в графе по этому запросу не найдено.")
        return

    try:
        from streamlit_agraph import agraph, Node, Edge, Config

        # тип и степень для каждого узла (из src_type/dst_type ребра)
        ntype: dict[str, str] = {}
        degree: dict[str, int] = {}
        for e in edges:
            ntype.setdefault(e["src"], e.get("src_type", ""))
            ntype.setdefault(e["dst"], e.get("dst_type", ""))
            degree[e["src"]] = degree.get(e["src"], 0) + 1
            degree[e["dst"]] = degree.get(e["dst"], 0) + 1

        nodes = {}
        for name, deg in degree.items():
            label = name if len(name) <= 26 else name[:24] + "…"
            nodes[name] = Node(id=name, label=label, size=12 + min(deg, 8) * 3,
                               color=_TYPE_COLOR.get(ntype.get(name), _TYPE_DEFAULT))
        graph_edges = [
            Edge(source=e["src"], target=e["dst"], label=e["rel"],
                 color=_VERIF_COLOR.get(e.get("verification_level"), _VERIF_DEFAULT))
            for e in edges
        ]
        # легенда типов узлов (только присутствующие)
        present = [t for t in _TYPE_COLOR if t in set(ntype.values())]
        legend = "  ".join(f"<span style='color:{_TYPE_COLOR[t]}'>●</span> {t}" for t in present)
        st.markdown(f"<div style='font-size:0.85em'>Узлы: {legend}</div>", unsafe_allow_html=True)
        st.caption("Цвет связи — достоверность: 🟢 подтверждено · 🟠 предварительно · "
                   "🔴 спорно · ⚪ не верифицировано.  Размер узла — число связей.")
        agraph(nodes=list(nodes.values()), edges=graph_edges,
               config=Config(width=1000, height=520, directed=True, physics=True,
                             hierarchical=False))
    except ImportError:
        st.warning("Для графа установите: `pip install streamlit-agraph`")

    with st.expander(f"Список связей с провенансом ({len(edges)})"):
        for e in edges:
            meta = []
            if e.get("verification_level"):
                meta.append(f"достоверность: {e['verification_level']}")
            if e.get("geography"):
                meta.append(f"география: {e['geography']}")
            if e.get("actualized_at"):
                meta.append(f"на: {e['actualized_at']}")
            meta_str = f"  _({'; '.join(meta)})_" if meta else ""
            st.write(f"`{e['src']}` —**{e['rel']}**→ `{e['dst']}`{meta_str}"
                     + (f"  · _{e['evidence']}_" if e.get("evidence") else ""))


@st.cache_data(show_spinner=False)
def _export_payloads(q: str, answer: str, sources: tuple, edges: int | None,
                     passages: int | None):
    """Экспортные артефакты (MD/JSON-LD/PDF) один раз на ответ: reportlab-PDF
    не пересобирается на каждый rerun (клики по фильтрам/вкладкам)."""
    obj = SimpleNamespace(question=q, text=answer, sources=list(sources),
                          edges_used=edges, passages_used=passages)
    md = to_markdown(obj)
    jld = json.dumps(to_json_ld(obj), ensure_ascii=False, indent=2)
    try:
        pdf = to_pdf(obj)
    except Exception:                                    # noqa: BLE001 — нет reportlab
        pdf = None
    return md, jld, pdf


def _render_answer(res: dict) -> None:
    """Отрисовать сохранённый результат вопроса (переживает rerun)."""
    r, q = res["ans"], res["q"]
    answer = r.get("answer", "")
    # структурный флаг из API; подстрока — только для старых сохранённых ответов
    is_fallback = (r.get("llm_ok") is False
                   or (r.get("llm_ok") is None and "Не удалось сгенерировать" in answer))

    st.markdown("### Ответ" if not res.get("review") else "### Литобзор")
    if is_fallback:
        st.warning("⚠️ Генерация ответа сейчас недоступна (LLM). Ниже — найденные "
                   "связи графа и источники: ретривал полностью рабочий.")
    else:
        # гео-фильтр был снят, т.к. источников запрошенной практики нет в базе —
        # честная оговорка (иначе «мировая практика» на отеч. корпусе вводит в заблуждение)
        if r.get("geography_relaxed"):
            st.info("🌍 Источников запрошенной географии (напр. зарубежная практика) в "
                    "базе не найдено — ответ построен по доступным источникам иной "
                    "географии. Для полноты нужен ингест зарубежных публикаций.")
        st.write(answer)

    is_review = res.get("review", False)
    cols = st.columns(3 if is_review else 5)
    cols[0].metric("Связей графа", r.get("edges_used", 0))
    cols[1].metric("Фрагментов", r.get("passages_used", 0))
    cols[2].metric("Источников", len(r.get("sources", [])))
    if not is_review:
        geo_label = {"True": "РФ", "False": "мир"}.get(str(r.get("geography_filter")), "—")
        cols[3].metric("Гео-фильтр", geo_label)
        tm = r.get("timings_ms") or {}
        if tm.get("retrieval_total_ms") is not None:
            cols[4].metric("Поиск, мс", tm["retrieval_total_ms"],
                           help=f"вектор {tm.get('vector_ms')} · seed {tm.get('seed_ms')} · "
                                f"граф {tm.get('graph_ms')} · реранк {tm.get('rerank_ms')} · "
                                f"LLM {tm.get('llm_ms')} мс")

    # применённые структурные фильтры — чипами
    chips = []
    if r.get("year_from") or r.get("year_to"):
        chips.append(f"🗓 годы: {r.get('year_from') or '…'}–{r.get('year_to') or '…'}")
    for c in r.get("constraints", []):
        rng = f"–{c['value_high']}" if c.get("value_high") is not None else ""
        chips.append(f"📏 {c['param']} {c['operator']} {c['value']}{rng} {c['unit']}")
    if chips:
        st.caption("Распознанные ограничения запроса:  " + "   ".join(chips))
    if r.get("sources"):
        st.caption("Источники: " + ", ".join(r["sources"]))

    # локальный экспорт — без повторного вызова API/LLM
    if _LOCAL_EXPORT:
        md, jld, pdf = _export_payloads(q, answer, tuple(r.get("sources", [])),
                                        r.get("edges_used"), r.get("passages_used"))
        e1, e2, e3 = st.columns(3)
        e1.download_button("⬇ Markdown", md, file_name="answer.md")
        e2.download_button("⬇ JSON-LD", jld, file_name="answer.jsonld")
        if pdf is not None:
            e3.download_button("⬇ PDF", pdf, file_name="answer.pdf", mime="application/pdf")
        else:
            e3.caption("PDF недоступен")

    if not is_review:
        st.markdown("### Подграф (доказательная база)")
        _render_graph(res["sg"])


# --------------------------------------------------------------------------
# Вкладки
# --------------------------------------------------------------------------
tab_titles = ["Вопрос", "Пробелы в данных", "Сравнение", "Эксперты", "Уведомления", "Загрузка"]
if st.session_state["role"] in ("project_lead", "admin"):
    tab_titles.insert(2, "Дашборд")
tabs = dict(zip(tab_titles, st.tabs(tab_titles)))


def _set_question(text: str) -> None:
    st.session_state["q_input"] = text


with tabs["Вопрос"]:
    col_filters, col_main = st.columns([1, 3])
    with col_filters:
        st.subheader("Фильтры")
        geo_choice = st.radio("География",
                              ["Авто (из вопроса)", "Отечественная (РФ)", "Мировая практика", "Все"],
                              index=0)
        domain = st.text_input("Домен", placeholder="гидрометаллургия")
        st.caption("Годы публикаций (0 = не фильтровать)")
        yc1, yc2 = st.columns(2)
        year_from = yc1.number_input("с", min_value=0, max_value=2100, value=0, step=1)
        year_to = yc2.number_input("по", min_value=0, max_value=2100, value=0, step=1)
    geo_map = {"Авто (из вопроса)": None, "Отечественная (РФ)": True,
               "Мировая практика": False, "Все": "__all__"}
    geo_value = geo_map[geo_choice]

    with col_main:
        st.caption("Примеры вопросов ТЗ:")
        ex_cols = st.columns(len(EXAMPLE_QUESTIONS))
        for col, (label, text) in zip(ex_cols, EXAMPLE_QUESTIONS):
            col.button(label, on_click=_set_question, args=(text,), use_container_width=True)

        st.session_state.setdefault("q_input", "")
        q = st.text_area("Вопрос на естественном языке", key="q_input", height=80,
                         placeholder="например: какая скорость циркуляции католита оптимальна…")

        bc1, bc2 = st.columns([1, 2])
        mode = bc2.radio("Режим", ["Ответ", "Литобзор"], horizontal=True,
                         label_visibility="collapsed",
                         help="«Литобзор» — структурированный синтез: группировка "
                              "источников, консенсус vs разногласия, степень уверенности.")
        if bc1.button("Спросить", type="primary", use_container_width=True) and q.strip():
            if mode == "Литобзор":
                with st.spinner("Сбор публикаций по теме + структурированный синтез…"):
                    ok, resp = _api("POST", "/review", json={"topic": q})
                    if ok and resp.status_code == 200:
                        ans = resp.json()
                        st.session_state["ask_result"] = {
                            "q": q, "ans": ans, "sg": {"edges": []}, "review": True}
                    else:
                        st.error("Не удалось получить литобзор от API.")
            else:
                payload: dict = {"question": q, "domain": domain or None}
                if geo_value != "__all__":
                    payload["geography"] = geo_value
                if year_from:
                    payload["year_from"] = int(year_from)
                if year_to:
                    payload["year_to"] = int(year_to)
                with st.spinner("Гибридный поиск (граф + вектор) + реранк + генерация…"):
                    ok, resp = _api("POST", "/ask", json=payload)
                    if ok and resp.status_code == 200:
                        ans = resp.json()
                        # подграф приходит в ответе /ask — второй прогон ретривала
                        # через GET /subgraph не нужен
                        sg = ans.get("subgraph") or {"edges": []}
                        st.session_state["ask_result"] = {"q": q, "ans": ans, "sg": sg,
                                                          "review": False}
                    else:
                        st.error("Не удалось получить ответ от API.")

        if st.session_state.get("ask_result"):
            st.divider()
            _render_answer(st.session_state["ask_result"])


def _guarded_json(path: str, params: dict | None = None):
    ok, resp = _api("GET", path, params=params or {})
    if not ok:
        st.error("API недоступен.")
        return None
    if resp.status_code == 403:
        st.error("Роль не имеет доступа к этому разделу.")
        return None
    return resp.json()


with tabs["Пробелы в данных"]:
    st.subheader("Где в данных дыры")
    st.caption("Материалы без свойств, неизученные комбинации, противоречия, гео-перекос.")
    if st.button("Построить отчёт"):
        rep = _guarded_json("/gaps")
        if rep:
            for title, rows in rep.items():
                st.markdown(f"**{title}** — {len(rows)} шт.")
                if rows:
                    st.dataframe(rows, use_container_width=True)

if "Дашборд" in tabs:
    with tabs["Дашборд"]:
        st.subheader("Метрики покрытия знаний и активности команд")
        if st.button("Обновить дашборд"):
            d = _guarded_json("/dashboard")
            if d:
                st.markdown("**Покрытие по доменам**")
                if d.get("coverage_by_domain"):
                    st.bar_chart({r.get("domain", "?"): r.get("count", 0)
                                  for r in d["coverage_by_domain"]})
                    st.dataframe(d["coverage_by_domain"], use_container_width=True)
                else:
                    st.caption("Нет данных — доменные теги не проставлены при извлечении.")
                st.markdown("**Активность лабораторий**")
                st.dataframe(d.get("facility_activity", []), use_container_width=True)
                st.markdown("**Зоны риска (мало подтверждающих источников)**")
                st.dataframe(d.get("risk_zones", []), use_container_width=True)

        # Ручная корректировка графа экспертом (ТЗ) — добавить/уточнить связь с
        # фиксацией автора и даты. Узлы должны существовать; связь проверяется
        # по онтологии на бэкенде (klubok.graph.ingest.upsert_manual_edge).
        with st.expander("✏️ Ручная корректировка графа (эксперт)"):
            from klubok.ontology import NodeType as _NT, RelType as _RT
            gc1, gc2, gc3 = st.columns(3)
            g_st = gc1.selectbox("Тип источника", [t.value for t in _NT], key="ge_st")
            g_rel = gc2.selectbox("Связь", [r.value for r in _RT], key="ge_rel")
            g_dt = gc3.selectbox("Тип цели", [t.value for t in _NT], key="ge_dt")
            g_src = st.text_input("canonical_id источника", placeholder=f"{g_st}:...")
            g_dst = st.text_input("canonical_id цели", placeholder=f"{g_dt}:...")
            g_comment = st.text_input("Комментарий эксперта")
            if st.button("Записать связь") and g_src and g_dst:
                ok, resp = _api("POST", "/graph/edge", json={
                    "src_type": g_st, "src_cid": g_src, "rel": g_rel,
                    "dst_type": g_dt, "dst_cid": g_dst,
                    "editor_name": st.session_state["role"], "comment": g_comment or None})
                if ok and resp.status_code == 200:
                    st.success("Связь записана (автор/дата зафиксированы на ребре).")
                elif ok:
                    st.error(resp.json().get("detail", "Ошибка (проверьте онтологию/узлы)."))
                else:
                    st.error("API недоступен.")

with tabs["Сравнение"]:
    st.subheader("Сравнение технологий/материалов по параметрам")
    st.caption("Типы можно выбирать разные — например, Process против Method.")
    # типы — из онтологии, а не захардкоженный список: новый NodeType сразу
    # появится в UI; тип выбирается отдельно для А и Б (кросс-типовые сравнения)
    from klubok.ontology import NodeType
    _types = [t.value for t in NodeType]

    def _entities_of(etype: str) -> dict[str, str]:
        data = _cached_get_json("/entities", st.session_state["role"],
                                params=(("type", etype), ("limit", 300))) or {}
        return {e["name"]: e["cid"] for e in data.get("entities", [])}

    col1, col2 = st.columns(2)
    ta = col1.selectbox("Тип А", _types, index=_types.index("Process"))
    tb = col2.selectbox("Тип Б", _types, index=_types.index("Process"))
    ents_a, ents_b = _entities_of(ta), _entities_of(tb)
    if not ents_a or not ents_b:
        empty = ta if not ents_a else tb
        st.info(f"В графе нет сущностей типа «{empty}».")
    else:
        a = col1.selectbox("Вариант А", list(ents_a), index=0)
        b = col2.selectbox("Вариант Б", list(ents_b),
                           index=min(1, len(ents_b) - 1) if ta == tb else 0)
        if st.button("Сравнить", type="primary") and ents_a[a] != ents_b[b]:
            ok, resp = _api("POST", "/compare", json={
                "cid_a": ents_a[a], "cid_b": ents_b[b], "label_a": a, "label_b": b})
            if ok and resp.status_code == 200:
                rows = resp.json().get("rows", [])
                (st.dataframe(rows, use_container_width=True) if rows
                 else st.info("Нет общих параметров для сравнения этих сущностей."))
            else:
                st.error("Ошибка сравнения.")

with tabs["Эксперты"]:
    st.subheader("Эксперты по теме")
    topic = st.text_input("Тема", placeholder="электроэкстракция никеля")
    if st.button("Найти экспертов") and topic:
        data = _guarded_json("/experts", {"topic": topic})
        experts = (data or {}).get("experts", [])
        if experts:
            if experts and experts[0].get("by_activity"):
                st.caption("Тематических связей пока нет — показаны эксперты по активности (числу публикаций).")
            st.dataframe(experts, use_container_width=True)
        else:
            st.info("Экспертов по теме не найдено.")

    st.divider()
    st.subheader("Активность лабораторий")
    if st.button("Показать лаборатории"):
        data = _guarded_json("/facilities")
        st.dataframe((data or {}).get("facilities", []), use_container_width=True)

with tabs["Уведомления"]:
    st.subheader("Подписки на темы")
    new_topic = st.text_input("Подписаться на тему", placeholder="электроэкстракция никеля")
    c1, c2 = st.columns(2)
    if c1.button("Подписаться") and new_topic:
        _api("POST", "/watch", json={"topic": new_topic})
        st.success(f"Подписка на «{new_topic}» добавлена.")
    if c2.button("Обновить ленту"):
        st.rerun()

    ok, resp = _api("GET", "/notifications", params={"mark_seen": True})
    data = resp.json() if (ok and resp.status_code == 200) else {}
    subs = data.get("subscriptions", [])
    if subs:
        st.caption("Ваши темы: " + ", ".join(subs))
        for t in subs:
            if st.button(f"Отписаться от «{t}»", key=f"unsub_{t}"):
                _api("DELETE", "/watch", json={"topic": t})
                st.rerun()
    else:
        st.info("Нет подписок. Добавьте тему — при ингесте новых документов по ней придут уведомления.")

    st.divider()
    st.subheader("Лента")
    notifs = data.get("notifications", [])
    st.dataframe(notifs, use_container_width=True) if notifs else st.caption("Пока нет событий по вашим темам.")

with tabs["Загрузка"]:
    st.subheader("Ингест документов")
    st.caption("PDF/DOCX/PPTX или папка. Парсинг → извлечение (LLM) → граф + вектор.")
    path = st.text_input("Путь", value="./data/sample")
    if st.button("Заингестить"):
        with st.spinner("Обработка…"):
            ok, resp = _api("POST", "/ingest", json={"path": path})
        if ok and resp.status_code == 200:
            st.success("Готово")
            st.json(resp.json())
        else:
            st.error("Ошибка ингеста (см. логи API).")
