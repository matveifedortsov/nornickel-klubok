"""Held-out эталонный набор для оценки извлечения.

ВАЖНО: эти пары НЕ входят в few-shot промпт (FEW_SHOT_EXAMPLES), поэтому оценка
на них честная — модель их «не видела». Формат идентичен few-shot, чтобы
переиспользовать gold_from_dict().

Расширяйте набор по мере появления корпуса: 15-30 размеченных пассажей дают
устойчивую оценку precision/recall по рёбрам.
"""
from __future__ import annotations

GOLD_SET: list[dict] = [
    {
        "text": (
            "Титановый сплав ВТ6 подвергали горячей прокатке при 950 °C, после чего "
            "выполняли отжиг при 750 °C в течение 2 ч. Предел текучести достиг 950 МПа. "
            "Методом просвечивающей электронной микроскопии (ПЭМ) наблюдали "
            "пластинчатую α-фазу в β-матрице."
        ),
        "output": {
            "entities": [
                {"name": "ВТ6", "type": "Material", "attributes": {}},
                {"name": "Ti", "type": "Element", "attributes": {}},
                {"name": "горячая прокатка", "type": "Process",
                 "attributes": {"temperature": 950, "temperature_unit": "°C"}},
                {"name": "отжиг", "type": "Process",
                 "attributes": {"temperature": 750, "temperature_unit": "°C", "time": 2, "time_unit": "h"}},
                {"name": "предел текучести", "type": "Property",
                 "attributes": {"value": 950, "unit": "MPa"}},
                {"name": "ПЭМ", "type": "Method", "attributes": {}},
                {"name": "α-фаза", "type": "Phase", "attributes": {}},
                {"name": "Эксперимент 1", "type": "Experiment", "attributes": {}},
            ],
            "relations": [
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "USES",
                 "dst_name": "ВТ6", "dst_type": "Material", "evidence": "Титановый сплав ВТ6"},
                {"src_name": "ВТ6", "src_type": "Material", "rel": "HAS_COMPOSITION",
                 "dst_name": "Ti", "dst_type": "Element", "evidence": "Титановый сплав ВТ6"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "горячая прокатка", "dst_type": "Process",
                 "evidence": "горячей прокатке при 950 °C"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "отжиг", "dst_type": "Process",
                 "evidence": "отжиг при 750 °C в течение 2 ч"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "MEASURES",
                 "dst_name": "предел текучести", "dst_type": "Property",
                 "evidence": "Предел текучести достиг 950 МПа"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "CONTAINS_PHASE",
                 "dst_name": "α-фаза", "dst_type": "Phase",
                 "evidence": "наблюдали пластинчатую α-фазу"},
                {"src_name": "α-фаза", "src_type": "Phase", "rel": "OBSERVED_BY",
                 "dst_name": "ПЭМ", "dst_type": "Method",
                 "evidence": "Методом просвечивающей электронной микроскопии (ПЭМ) наблюдали"},
            ],
        },
    },
    {
        "text": (
            "Алюминиевый сплав АМг6 после холодной деформации показал увеличение "
            "предела прочности до 340 МПа при снижении пластичности. Энергодисперсионная "
            "рентгеновская спектроскопия (ЭДС) подтвердила содержание магния около 6 wt%."
        ),
        "output": {
            "entities": [
                {"name": "АМг6", "type": "Material", "attributes": {}},
                {"name": "Al", "type": "Element", "attributes": {}},
                {"name": "Mg", "type": "Element", "attributes": {"content": 6, "unit": "wt%"}},
                {"name": "холодная деформация", "type": "Process", "attributes": {}},
                {"name": "предел прочности", "type": "Property",
                 "attributes": {"value": 340, "unit": "MPa"}},
                {"name": "Эксперимент 1", "type": "Experiment", "attributes": {}},
            ],
            "relations": [
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "USES",
                 "dst_name": "АМг6", "dst_type": "Material", "evidence": "Алюминиевый сплав АМг6"},
                {"src_name": "АМг6", "src_type": "Material", "rel": "HAS_COMPOSITION",
                 "dst_name": "Al", "dst_type": "Element", "evidence": "Алюминиевый сплав АМг6"},
                {"src_name": "АМг6", "src_type": "Material", "rel": "HAS_COMPOSITION",
                 "dst_name": "Mg", "dst_type": "Element", "evidence": "содержание магния около 6 wt%"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "холодная деформация", "dst_type": "Process",
                 "evidence": "после холодной деформации"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "RESULTS_IN",
                 "dst_name": "предел прочности", "dst_type": "Property",
                 "evidence": "увеличение предела прочности до 340 МПа при снижении пластичности"},
            ],
        },
    },
    # --- Металлургический домен ТЗ: распределение Au/Ag/МПГ штейн-шлак ---
    {
        "text": (
            "При конвертировании медного штейна благородные металлы — золото, "
            "серебро и металлы платиновой группы — распределяются преимущественно "
            "в штейн, а не в шлак: содержание золота в шлаке не превышает 0.5 г/т. "
            "Эксперимент проводился на комбинате в Норильске."
        ),
        "output": {
            "entities": [
                {"name": "медный штейн", "type": "Material", "attributes": {}},
                {"name": "шлак", "type": "Material", "attributes": {}},
                {"name": "содержание золота в шлаке", "type": "Property",
                 "attributes": {"value": 0.5, "unit": "g/t"}},
                {"name": "Эксперимент 1", "type": "Experiment", "attributes": {}},
            ],
            "relations": [
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "USES",
                 "dst_name": "медный штейн", "dst_type": "Material",
                 "evidence": "При конвертировании медного штейна"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "USES",
                 "dst_name": "шлак", "dst_type": "Material",
                 "evidence": "распределяются преимущественно в штейн, а не в шлак"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "RESULTS_IN",
                 "dst_name": "содержание золота в шлаке", "dst_type": "Property",
                 "evidence": "содержание золота в шлаке не превышает 0.5 г/т"},
            ],
        },
    },
    # --- Металлургический домен ТЗ: закачка шахтных вод ---
    {
        "text": (
            "Закачка шахтных вод в глубокие водоносные горизонты применяется как "
            "способ утилизации на рудниках России и Канады. Производительность "
            "системы закачки составила 250 т/сут при устойчивой работе более пяти лет."
        ),
        "output": {
            "entities": [
                {"name": "закачка шахтных вод в глубокие горизонты", "type": "Process",
                 "attributes": {}},
                {"name": "производительность системы закачки", "type": "Property",
                 "attributes": {"value": 250, "unit": "t/day"}},
                {"name": "Эксперимент 1", "type": "Experiment", "attributes": {}},
            ],
            "relations": [
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "закачка шахтных вод в глубокие горизонты", "dst_type": "Process",
                 "evidence": "Закачка шахтных вод в глубокие водоносные горизонты применяется"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "MEASURES",
                 "dst_name": "производительность системы закачки", "dst_type": "Property",
                 "evidence": "Производительность системы закачки составила 250 т/сут"},
            ],
        },
    },
]
