"""Прогон 10 реальных экспертных вопросов по живому API -> docs/DEMO_ANSWERS.md.

Шпаргалка к защите + доказательство работы: показывает ответы системы на
эталонные вопросы Норникеля с провенансом (источники), числом рёбер графа и
таймингами ретривала. Требует поднятого API (uvicorn klubok.api.app:app) с
загруженным seed-графом.

Запуск:
    python scripts/gen_demo_answers.py
"""
from __future__ import annotations

import time
from pathlib import Path

import _bootstrap  # noqa: F401,E402  (путь к пакету + UTF-8 вывод)

import requests

API = "http://localhost:8000"

# 10 реальных вопросов экспертов Норникеля (см. память expert-eval-questions).
QUESTIONS = [
    "Технико-экономическое сравнение вариантов обессоливания воды для обогатительной "
    "фабрики, если сульфаты, хлориды, Ca, Mg, Na по 200–300 мг/л, а требуемый сухой "
    "остаток ≤1000 мг/дм3.",
    "Литобзор методов очистки шахтных вод горно-рудных предприятий цветной металлургии "
    "в России и за рубежом.",
    "Какие технические решения организации циркуляции католита при электроэкстракции "
    "никеля описаны в мировой практике, и какая скорость циркуляции считается оптимальной?",
    "Обзор решений электролитического производства никеля, меди и кобальта: подача "
    "электролита в ванны, циркуляция, конструкции диафрагменных ячеек.",
    "Литобзор источников техногенного гипса и способов его переработки.",
    "Анализ технологий закачки шахтных вод в глубокие горизонты в России и мире.",
    "Обзор практик использования угля и отходов угольной промышленности для закладки "
    "выработанного пространства.",
    "Обзор способов удаления SO2 из отходящих газов металлургических предприятий мира.",
    "Распределение Au, Ag и МПГ между медным и никелевым штейном и шлаком по зарубежным "
    "источникам за последние 5 лет.",
    "Обзор современных способов переработки свинцово-цинкового сырья в мировой практике.",
]


def _ask(q: str) -> dict:
    t0 = time.time()
    r = requests.post(f"{API}/ask", json={"question": q}, timeout=180).json()
    r["_elapsed_s"] = round(time.time() - t0, 1)
    return r


def _fmt_filters(r: dict) -> str:
    bits = []
    geo = r.get("geography_filter")
    if geo is True:
        bits.append("гео: Россия")
    elif geo is False:
        bits.append("гео: зарубеж/мир")
    if r.get("year_from") or r.get("year_to"):
        bits.append(f"годы: {r.get('year_from') or '…'}–{r.get('year_to') or '…'}")
    for c in r.get("constraints", []):
        hi = f"–{c['value_high']}" if c.get("value_high") is not None else ""
        bits.append(f"{c['param']} {c['operator']} {c['value']}{hi} {c['unit']}".strip())
    return "; ".join(bits) if bits else "—"


def main() -> None:
    out = ["# Демо-ответы на 10 экспертных вопросов Норникеля",
           "",
           "Прогон по живому API (`/ask`) на seed-графе. Для каждого вопроса: "
           "распознанные фильтры (числа/гео/годы), число рёбер графа в контексте, "
           "тайминги ретривала и сам ответ с источниками. Сгенерировано "
           "`scripts/gen_demo_answers.py`.",
           ""]
    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i}/10] {q[:60]}…")
        try:
            r = _ask(q)
        except Exception as exc:                          # noqa: BLE001
            out.append(f"## {i}. {q}\n\n**Ошибка:** {exc}\n")
            continue
        tm = r.get("timings_ms", {}) or {}
        out += [
            f"## {i}. {q}",
            "",
            f"- **Распознанные фильтры:** {_fmt_filters(r)}",
            f"- **Рёбер графа в контексте:** {r.get('edges_used')} · "
            f"**фрагментов:** {r.get('passages_used')} · **источников:** {len(r.get('sources', []))}",
            f"- **Тайминги:** ретривал {tm.get('retrieval_total_ms')} мс "
            f"(вектор {tm.get('vector_ms')} · seed {tm.get('seed_ms')} · граф {tm.get('graph_ms')} · "
            f"реранк {tm.get('rerank_ms')}) · генерация {tm.get('llm_ms')} мс · всего {r.get('_elapsed_s')} с",
            "",
            "**Ответ:**",
            "",
            r.get("answer", "(пусто)").strip(),
            "",
        ]
        if r.get("sources"):
            out.append("**Источники:** " + ", ".join(r["sources"]))
            out.append("")

    Path("docs/DEMO_ANSWERS.md").write_text("\n".join(out), encoding="utf-8")
    print("Готово -> docs/DEMO_ANSWERS.md")


if __name__ == "__main__":
    main()
