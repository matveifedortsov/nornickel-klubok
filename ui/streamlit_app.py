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


def _health() -> dict | None:
    ok, resp = _api("GET", "/health")
    if ok and resp.status_code == 200:
        return resp.json()
    return None


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
    # бейдж непрочитанных уведомлений
    ok, resp = _api("GET", "/notifications", params={"unseen_only": True})
    if ok and resp.status_code == 200:
        unseen = len(resp.json().get("notifications", []))
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


def _render_answer(res: dict) -> None:
    """Отрисовать сохранённый результат вопроса (переживает rerun)."""
    r, q = res["ans"], res["q"]
    answer = r.get("answer", "")
    is_fallback = "Не удалось сгенерировать" in answer

    st.markdown("### Ответ")
    if is_fallback:
        st.warning("⚠️ Генерация ответа сейчас недоступна (LLM). Ниже — найденные "
                   "связи графа и источники: ретривал полностью рабочий.")
    else:
        st.write(answer)

    cols = st.columns(5)
    cols[0].metric("Связей графа", r.get("edges_used", 0))
    cols[1].metric("Фрагментов", r.get("passages_used", 0))
    cols[2].metric("Источников", len(r.get("sources", [])))
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
        obj = SimpleNamespace(question=q, text=answer, sources=r.get("sources", []),
                              edges_used=r.get("edges_used"), passages_used=r.get("passages_used"))
        e1, e2, e3 = st.columns(3)
        e1.download_button("⬇ Markdown", to_markdown(obj), file_name="answer.md")
        e2.download_button("⬇ JSON-LD",
                           json.dumps(to_json_ld(obj), ensure_ascii=False, indent=2),
                           file_name="answer.jsonld")
        try:
            e3.download_button("⬇ PDF", to_pdf(obj), file_name="answer.pdf", mime="application/pdf")
        except Exception:                                # noqa: BLE001
            e3.caption("PDF недоступен")

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

        if st.button("Спросить", type="primary") and q.strip():
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
                    st.session_state["ask_result"] = {"q": q, "ans": ans, "sg": sg}
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

with tabs["Сравнение"]:
    st.subheader("Сравнение технологий/материалов по параметрам")
    etype = st.selectbox("Тип сущностей", ["Process", "Material", "Equipment"], index=0)
    ents = (_guarded_json("/entities", {"type": etype, "limit": 300}) or {}).get("entities", [])
    if not ents:
        st.info(f"В графе нет сущностей типа «{etype}».")
    else:
        name2cid = {e["name"]: e["cid"] for e in ents}
        names = list(name2cid.keys())
        col1, col2 = st.columns(2)
        a = col1.selectbox("Вариант А", names, index=0)
        b = col2.selectbox("Вариант Б", names, index=min(1, len(names) - 1))
        if st.button("Сравнить", type="primary") and a != b:
            ok, resp = _api("POST", "/compare", json={
                "cid_a": name2cid[a], "cid_b": name2cid[b], "label_a": a, "label_b": b})
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
