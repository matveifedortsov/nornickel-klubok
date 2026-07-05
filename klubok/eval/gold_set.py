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

    # --- Обессоливание воды (вопрос ТЗ №1) ---
    {
        "text": (
            "Для обессоливания воды обогатительной фабрики применяли обратный осмос. "
            "Исходная вода содержала сульфаты около 280 мг/л. После очистки сухой "
            "остаток снизился до 850 мг/дм3."
        ),
        "output": {
            "entities": [
                {"name": "вода обогатительной фабрики", "type": "Material", "attributes": {}},
                {"name": "обратный осмос", "type": "Process", "attributes": {}},
                {"name": "сухой остаток", "type": "Property",
                 "attributes": {"value": 850, "unit": "мг/дм3"}},
                {"name": "Эксперимент 1", "type": "Experiment", "attributes": {}},
            ],
            "relations": [
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "USES",
                 "dst_name": "вода обогатительной фабрики", "dst_type": "Material",
                 "evidence": "обессоливания воды обогатительной фабрики"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "обратный осмос", "dst_type": "Process",
                 "evidence": "применяли обратный осмос"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "RESULTS_IN",
                 "dst_name": "сухой остаток", "dst_type": "Property",
                 "evidence": "сухой остаток снизился до 850 мг/дм3"},
            ],
        },
    },

    # --- Английский пример (мультиязычность RU/EN, вопрос ТЗ про electrowinning) ---
    {
        "text": (
            "Nickel electrowinning was carried out at a current density of 250 A/m2 with "
            "catholyte circulation. The cathode current efficiency reached 95%. X-ray "
            "diffraction (XRD) confirmed the deposition of pure nickel."
        ),
        "output": {
            "entities": [
                {"name": "nickel", "type": "Material", "attributes": {}},
                {"name": "electrowinning", "type": "Process",
                 "attributes": {"current_density": 250, "unit": "A/m2"}},
                {"name": "cathode current efficiency", "type": "Property",
                 "attributes": {"value": 95, "unit": "%"}},
                {"name": "XRD", "type": "Method", "attributes": {}},
                {"name": "pure nickel", "type": "Phase", "attributes": {}},
                {"name": "Experiment 1", "type": "Experiment", "attributes": {}},
            ],
            "relations": [
                {"src_name": "Experiment 1", "src_type": "Experiment", "rel": "USES",
                 "dst_name": "nickel", "dst_type": "Material",
                 "evidence": "Nickel electrowinning was carried out"},
                {"src_name": "Experiment 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "electrowinning", "dst_type": "Process",
                 "evidence": "at a current density of 250 A/m2 with catholyte circulation"},
                {"src_name": "Experiment 1", "src_type": "Experiment", "rel": "MEASURES",
                 "dst_name": "cathode current efficiency", "dst_type": "Property",
                 "evidence": "The cathode current efficiency reached 95%"},
                {"src_name": "Experiment 1", "src_type": "Experiment", "rel": "CONTAINS_PHASE",
                 "dst_name": "pure nickel", "dst_type": "Phase",
                 "evidence": "confirmed the deposition of pure nickel"},
                {"src_name": "pure nickel", "src_type": "Phase", "rel": "OBSERVED_BY",
                 "dst_name": "XRD", "dst_type": "Method",
                 "evidence": "X-ray diffraction (XRD) confirmed the deposition"},
            ],
        },
    },

    # --- Техногенный гипс (вопрос ТЗ №5) ---
    {
        "text": (
            "Техногенный гипс, образующийся при нейтрализации сернокислых стоков "
            "известняком, перерабатывали обжигом при 160 °C с получением строительного "
            "гипса. Прочность на сжатие полученного вяжущего составила 12 МПа."
        ),
        "output": {
            "entities": [
                {"name": "техногенный гипс", "type": "Material", "attributes": {}},
                {"name": "обжиг", "type": "Process",
                 "attributes": {"temperature": 160, "temperature_unit": "°C"}},
                {"name": "строительный гипс", "type": "Material", "attributes": {}},
                {"name": "прочность на сжатие", "type": "Property",
                 "attributes": {"value": 12, "unit": "MPa"}},
                {"name": "Эксперимент 1", "type": "Experiment", "attributes": {}},
            ],
            "relations": [
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "USES",
                 "dst_name": "техногенный гипс", "dst_type": "Material",
                 "evidence": "Техногенный гипс, образующийся при нейтрализации сернокислых стоков"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "обжиг", "dst_type": "Process",
                 "evidence": "перерабатывали обжигом при 160 °C"},
                {"src_name": "обжиг", "src_type": "Process", "rel": "PRODUCES_OUTPUT",
                 "dst_name": "строительный гипс", "dst_type": "Material",
                 "evidence": "с получением строительного гипса"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "MEASURES",
                 "dst_name": "прочность на сжатие", "dst_type": "Property",
                 "evidence": "Прочность на сжатие полученного вяжущего составила 12 МПа"},
            ],
        },
    },

    # --- Удаление SO2 из отходящих газов (вопрос ТЗ №8) ---
    {
        "text": (
            "Для очистки отходящих газов от диоксида серы применяли известняковый "
            "скруббер. Степень улавливания SO2 достигла 95 %. Метод обеспечил "
            "снижение концентрации SO2 в газе до 200 мг/м3."
        ),
        "output": {
            "entities": [
                {"name": "отходящие газы", "type": "Material", "attributes": {}},
                {"name": "известняковый скруббер", "type": "Equipment", "attributes": {}},
                {"name": "мокрая сероочистка", "type": "Process", "attributes": {}},
                {"name": "степень улавливания SO2", "type": "Property",
                 "attributes": {"value": 95, "unit": "%"}},
                {"name": "Эксперимент 1", "type": "Experiment", "attributes": {}},
            ],
            "relations": [
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "мокрая сероочистка", "dst_type": "Process",
                 "evidence": "применяли известняковый скруббер"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "известняковый скруббер", "dst_type": "Equipment",
                 "evidence": "применяли известняковый скруббер"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "USES",
                 "dst_name": "отходящие газы", "dst_type": "Material",
                 "evidence": "очистки отходящих газов от диоксида серы"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "MEASURES",
                 "dst_name": "степень улавливания SO2", "dst_type": "Property",
                 "evidence": "Степень улавливания SO2 достигла 95 %"},
            ],
        },
    },

    # --- Переработка свинцово-цинкового сырья (вопрос ТЗ №10) ---
    {
        "text": (
            "Свинцово-цинковый концентрат перерабатывали методом вельцевания во "
            "вращающейся печи при 1100 °C. Извлечение цинка в возгоны составило 92 %. "
            "Полученный вельц-оксид содержал 65 % цинка."
        ),
        "output": {
            "entities": [
                {"name": "свинцово-цинковый концентрат", "type": "Material", "attributes": {}},
                {"name": "вельцевание", "type": "Process",
                 "attributes": {"temperature": 1100, "temperature_unit": "°C"}},
                {"name": "вращающаяся печь", "type": "Equipment", "attributes": {}},
                {"name": "извлечение цинка", "type": "Property",
                 "attributes": {"value": 92, "unit": "%"}},
                {"name": "Эксперимент 1", "type": "Experiment", "attributes": {}},
            ],
            "relations": [
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "USES",
                 "dst_name": "свинцово-цинковый концентрат", "dst_type": "Material",
                 "evidence": "Свинцово-цинковый концентрат перерабатывали"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "вельцевание", "dst_type": "Process",
                 "evidence": "методом вельцевания во вращающейся печи при 1100 °C"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "вращающаяся печь", "dst_type": "Equipment",
                 "evidence": "во вращающейся печи при 1100 °C"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "MEASURES",
                 "dst_name": "извлечение цинка", "dst_type": "Property",
                 "evidence": "Извлечение цинка в возгоны составило 92 %"},
            ],
        },
    },

    # --- Английский пример №2: closure/backfill (вопрос ТЗ №7) ---
    {
        "text": (
            "Coal fly ash was used as a binder component for backfilling of mined-out "
            "stopes. The cemented paste backfill reached an unconfined compressive "
            "strength of 4.5 MPa after 28 days of curing."
        ),
        "output": {
            "entities": [
                {"name": "coal fly ash", "type": "Material", "attributes": {}},
                {"name": "cemented paste backfill", "type": "Process", "attributes": {}},
                {"name": "unconfined compressive strength", "type": "Property",
                 "attributes": {"value": 4.5, "unit": "MPa"}},
                {"name": "Experiment 1", "type": "Experiment", "attributes": {}},
            ],
            "relations": [
                {"src_name": "Experiment 1", "src_type": "Experiment", "rel": "USES",
                 "dst_name": "coal fly ash", "dst_type": "Material",
                 "evidence": "Coal fly ash was used as a binder component for backfilling"},
                {"src_name": "Experiment 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "cemented paste backfill", "dst_type": "Process",
                 "evidence": "The cemented paste backfill"},
                {"src_name": "Experiment 1", "src_type": "Experiment", "rel": "MEASURES",
                 "dst_name": "unconfined compressive strength", "dst_type": "Property",
                 "evidence": "reached an unconfined compressive strength of 4.5 MPa"},
            ],
        },
    },
]
