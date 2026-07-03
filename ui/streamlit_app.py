"""Демо-интерфейс на Streamlit (быстрый путь к показу жюри).

Запуск:  streamlit run ui/streamlit_app.py
Ожидает поднятый API:  uvicorn klubok.api.app:app
(или переключите USE_API=False, чтобы дёргать пайплайн напрямую в процессе).
"""
from __future__ import annotations

import os

import requests
import streamlit as st

# В docker-compose UI ходит к API по имени сервиса (KLUBOK_API_URL=http://api:8000);
# локально — по localhost.
API = os.environ.get("KLUBOK_API_URL", "http://localhost:8000")

# Роль в UI -> dev-API-key (см. config.py Settings.api_keys — дефолт для
# локальной демки; в бою ключи переопределяются через .env). Роль определяет
# видимость внутренних данных и доступ к «Пробелам»/«Дашборду» (RBAC, §7).
ROLE_KEYS = {
    "researcher": "dev-researcher",
    "analyst": "dev-analyst",
    "project_lead": "dev-lead",
    "admin": "dev-admin",
    "external_partner": "dev-partner",
}

st.set_page_config(page_title="Научный клубок", layout="wide")
st.title("🧶 Научный клубок — R&D карта знаний горно-металлургической отрасли")

with st.sidebar:
    role = st.selectbox("Роль", list(ROLE_KEYS.keys()), index=0)
    st.caption("Определяет видимость внутренних данных и доступ к «Пробелам»/«Дашборду»")


def _headers() -> dict:
    return {"X-API-Key": ROLE_KEYS[role]}


# бейдж непрочитанных уведомлений по подпискам текущей роли (§Y7)
try:
    _notif = requests.get(f"{API}/notifications", params={"unseen_only": True},
                          headers=_headers(), timeout=10).json()
    _unseen = len(_notif.get("notifications", []))
    if _unseen:
        st.sidebar.warning(f"🔔 Новое по вашим темам: {_unseen}")
except Exception:
    pass


def _render_graph(sg: dict) -> None:
    """Визуализация подграфа графом (не текстовым списком) — §8 плана."""
    edges = sg.get("edges", [])
    if not edges:
        st.info("Связей в графе не найдено")
        return
    try:
        from streamlit_agraph import agraph, Node, Edge, Config
    except ImportError:
        st.warning("Для визуализации графа установите streamlit-agraph (`pip install streamlit-agraph`)")
        edges = None

    if edges is not None:
        nodes_seen: dict[str, "Node"] = {}
        graph_edges = []
        for e in sg["edges"]:
            for name in (e["src"], e["dst"]):
                if name not in nodes_seen:
                    nodes_seen[name] = Node(id=name, label=name, size=18)
            # спорные/предварительные факты подсвечиваем — требование ТЗ
            # «подсветка противоречивых данных»
            color = {"disputed": "#e74c3c", "preliminary": "#f39c12"}.get(e.get("verification_level"), "#3498db")
            graph_edges.append(Edge(source=e["src"], target=e["dst"], label=e["rel"], color=color))
        config = Config(width=900, height=500, directed=True, physics=True, hierarchical=False)
        agraph(nodes=list(nodes_seen.values()), edges=graph_edges, config=config)

    with st.expander("Список связей с провенансом", expanded=(edges is None)):
        for e in sg["edges"]:
            meta = []
            if e.get("verification_level"):
                meta.append(f"верификация: {e['verification_level']}")
            if e.get("geography"):
                meta.append(f"география: {e['geography']}")
            if e.get("actualized_at"):
                meta.append(f"на: {e['actualized_at']}")
            meta_str = f"  _({'; '.join(meta)})_" if meta else ""
            st.write(f"`{e['src']}` —**{e['rel']}**→ `{e['dst']}`{meta_str}"
                     + (f"  · _{e['evidence']}_" if e.get("evidence") else ""))


tab_titles = ["Вопрос", "Пробелы в данных", "Сравнение", "Эксперты", "Уведомления", "Загрузка"]
if role in ("project_lead", "admin"):
    tab_titles.insert(2, "Дашборд")
tabs = dict(zip(tab_titles, st.tabs(tab_titles)))

with tabs["Вопрос"]:
    col_filters, col_main = st.columns([1, 3])
    with col_filters:
        st.subheader("Фильтры")
        geo_choice = st.radio(
            "География", ["Авто (из вопроса)", "Отечественная (РФ)", "Мировая практика", "Все"], index=0,
        )
        domain = st.text_input("Домен", placeholder="гидрометаллургия")
    geo_map = {"Авто (из вопроса)": None, "Отечественная (РФ)": True, "Мировая практика": False, "Все": "__all__"}
    geo_value = geo_map[geo_choice]

    with col_main:
        q = st.text_input(
            "Вопрос",
            placeholder="какие технические решения циркуляции католита при электроэкстракции никеля "
                        "описаны в мировой практике, и какая скорость потока считается оптимальной?",
        )
        if st.button("Спросить", type="primary") and q:
            payload: dict = {"question": q, "domain": domain or None}
            if geo_value != "__all__":
                payload["geography"] = geo_value
            with st.spinner("Гибридный поиск + генерация…"):
                r = requests.post(f"{API}/ask", json=payload, headers=_headers(), timeout=180).json()

            st.markdown("### Ответ")
            st.write(r["answer"])

            cols = st.columns(5)
            cols[0].metric("Связей из графа", r["edges_used"])
            cols[1].metric("Фрагментов", r["passages_used"])
            cols[2].metric("Источников", len(r["sources"]))
            geo_label = {"True": "РФ", "False": "мир"}.get(str(r.get("geography_filter")), "—")
            cols[3].metric("Гео-фильтр", geo_label)
            tm = r.get("timings_ms") or {}
            if tm.get("retrieval_total_ms") is not None:
                cols[4].metric("Поиск, мс", tm["retrieval_total_ms"],
                               help=f"вектор {tm.get('vector_ms')} · seed {tm.get('seed_ms')} · "
                                    f"граф {tm.get('graph_ms')} · LLM {tm.get('llm_ms')} мс")

            if r.get("constraints"):
                cstr = ", ".join(
                    f"{c['param']} {c['operator']} {c['value']}"
                    + (f"–{c['value_high']}" if c.get("value_high") is not None else "")
                    + f" {c['unit']}"
                    for c in r["constraints"]
                )
                st.caption(f"Числовые ограничения из вопроса: {cstr}")
            if r["sources"]:
                st.caption("Источники: " + ", ".join(r["sources"]))

            exp1, exp2, exp3 = st.columns(3)
            if exp1.button("⬇ Markdown"):
                md = requests.post(f"{API}/export",
                                   json={"kind": "answer", "question": q, "format": "markdown"}).text
                st.download_button("Скачать .md", md, file_name="answer.md", key="dl_md")
            if exp2.button("⬇ JSON-LD"):
                jl = requests.post(f"{API}/export",
                                   json={"kind": "answer", "question": q, "format": "json-ld"}).text
                st.download_button("Скачать .jsonld", jl, file_name="answer.jsonld", key="dl_jsonld")
            if exp3.button("⬇ PDF"):
                pdf = requests.post(f"{API}/export",
                                    json={"kind": "answer", "question": q, "format": "pdf"}).content
                st.download_button("Скачать .pdf", pdf, file_name="answer.pdf", key="dl_pdf")

            st.markdown("### Подграф (доказательная база)")
            sg_params: dict = {"q": q}
            if geo_value != "__all__":
                sg_params["geography"] = geo_value
            if domain:
                sg_params["domain"] = domain
            sg = requests.get(f"{API}/subgraph", params=sg_params, headers=_headers(), timeout=60).json()
            _render_graph(sg)

with tabs["Пробелы в данных"]:
    st.subheader("Где в данных дыры")
    if st.button("Построить отчёт"):
        resp = requests.get(f"{API}/gaps", headers=_headers(), timeout=120)
        if resp.status_code == 403:
            st.error("Роль не имеет доступа к этому разделу")
        else:
            rep = resp.json()
            for title, rows in rep.items():
                st.markdown(f"**{title}** — {len(rows)} шт.")
                if rows:
                    st.dataframe(rows, use_container_width=True)

if "Дашборд" in tabs:
    with tabs["Дашборд"]:
        st.subheader("Метрики покрытия знаний и активности команд")
        if st.button("Обновить дашборд"):
            resp = requests.get(f"{API}/dashboard", headers=_headers(), timeout=60)
            if resp.status_code == 403:
                st.error("Роль не имеет доступа к этому разделу")
            else:
                d = resp.json()
                st.markdown("**Покрытие по доменам**")
                if d["coverage_by_domain"]:
                    st.dataframe(d["coverage_by_domain"], use_container_width=True)
                else:
                    st.caption("Нет данных — доменные теги не проставлены при извлечении")
                st.markdown("**Активность лабораторий**")
                st.dataframe(d["facility_activity"], use_container_width=True)
                st.markdown("**Зоны риска (мало подтверждающих источников)**")
                st.dataframe(d["risk_zones"], use_container_width=True)

with tabs["Сравнение"]:
    st.subheader("Сравнение технологий/материалов по параметрам")
    col1, col2 = st.columns(2)
    cid_a = col1.text_input("ID варианта А", placeholder="Process:электроэкстракция никеля")
    cid_b = col2.text_input("ID варианта Б", placeholder="Process:обратный осмос")
    label_a = col1.text_input("Название А", value="Вариант А")
    label_b = col2.text_input("Название Б", value="Вариант Б")
    if st.button("Сравнить") and cid_a and cid_b:
        r = requests.post(f"{API}/compare", json={
            "cid_a": cid_a, "cid_b": cid_b, "label_a": label_a, "label_b": label_b,
        }).json()
        st.dataframe(r["rows"], use_container_width=True)

with tabs["Эксперты"]:
    st.subheader("Эксперты по теме")
    topic = st.text_input("Тема", placeholder="электроэкстракция никеля")
    if st.button("Найти экспертов") and topic:
        r = requests.get(f"{API}/experts", params={"topic": topic}).json()
        st.dataframe(r["experts"], use_container_width=True)

    st.markdown("---")
    st.subheader("Активность лабораторий")
    if st.button("Показать лаборатории"):
        r = requests.get(f"{API}/facilities").json()
        st.dataframe(r["facilities"], use_container_width=True)

with tabs["Уведомления"]:
    st.subheader("Подписки на темы")
    new_topic = st.text_input("Подписаться на тему",
                              placeholder="электроэкстракция никеля")
    c1, c2 = st.columns(2)
    if c1.button("Подписаться") and new_topic:
        requests.post(f"{API}/watch", json={"topic": new_topic}, headers=_headers()).json()
        st.success(f"Подписка на «{new_topic}» добавлена")
    if c2.button("Обновить ленту"):
        pass

    data = requests.get(f"{API}/notifications", params={"mark_seen": True},
                        headers=_headers()).json()
    subs = data.get("subscriptions", [])
    if subs:
        st.caption("Ваши темы: " + ", ".join(subs))
        for t in subs:
            if st.button(f"Отписаться от «{t}»", key=f"unsub_{t}"):
                requests.delete(f"{API}/watch", json={"topic": t}, headers=_headers())
                st.rerun()
    else:
        st.info("Нет подписок. Добавьте тему выше, и при ингесте новых документов "
                "по ней появятся уведомления.")

    st.markdown("---")
    st.subheader("Лента")
    notifs = data.get("notifications", [])
    if notifs:
        st.dataframe(notifs, use_container_width=True)
    else:
        st.caption("Пока нет событий по вашим темам.")

with tabs["Загрузка"]:
    path = st.text_input("Путь к PDF/DOCX/PPTX или папке", value="./data/sample")
    if st.button("Заингестить"):
        with st.spinner("Парсинг → извлечение → граф + вектор…"):
            r = requests.post(f"{API}/ingest", json={"path": path}, timeout=600).json()
        st.success("Готово")
        st.json(r)
