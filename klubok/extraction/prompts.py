"""Промпты для MetalGPT. Вынесены отдельно — их удобно править без кода.

Главный приём против галлюцинаций:
  * жёсткая JSON-схема + закрытый список типов узлов/связей в инструкции;
  * few-shot на реальном материаловедческом тексте (см. FEW_SHOT_EXAMPLES);
  * требование дословной цитаты (`evidence`) для каждой связи.

Few-shot примеры заданы как Python-структуры, а не как сырой текст, чтобы:
  - гарантировать валидный JSON в промпте (json.dumps);
  - переиспользовать те же пары (текст -> ожидаемые триплеты) как мини-датасет
    для оценки качества извлечения (precision/recall) на этапе с железом.
"""
from __future__ import annotations

import json

from klubok.ontology import NodeType, RelType

_NODE_TYPES = ", ".join(t.value for t in NodeType)
_REL_TYPES = ", ".join(r.value for r in RelType)


# --------------------------------------------------------------------------
# Few-shot датасет: (входной фрагмент, эталонный JSON извлечения)
# Тексты — реалистичные материаловедческие пассажи (RU), типичные для статей,
# которые встретятся в корпусе трека.
# --------------------------------------------------------------------------
FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "text": (
            "Образцы медно-никелевого сплава Cu-30Ni отжигали при 850 °C в течение 1 ч "
            "с последующим охлаждением на воздухе. Микротвёрдость по Виккерсу выросла "
            "до 168 HV, что на 20 % выше исходного состояния. Микроструктуру изучали "
            "методом сканирующей электронной микроскопии (СЭМ); выявлены равноосные "
            "зёрна α-твёрдого раствора."
        ),
        "output": {
            "entities": [
                {"name": "Cu-30Ni", "type": "Material", "attributes": {}},
                {"name": "Cu", "type": "Element", "attributes": {}},
                {"name": "Ni", "type": "Element", "attributes": {}},
                {"name": "отжиг", "type": "Process",
                 "attributes": {"temperature": 850, "temperature_unit": "°C",
                                "time": 1, "time_unit": "h", "cooling": "воздух"}},
                {"name": "микротвёрдость по Виккерсу", "type": "Property",
                 "attributes": {"value": 168, "unit": "HV"}},
                {"name": "СЭМ", "type": "Method", "attributes": {}},
                {"name": "α-твёрдый раствор", "type": "Phase", "attributes": {}},
                {"name": "Эксперимент 1", "type": "Experiment", "attributes": {}},
            ],
            "relations": [
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "USES",
                 "dst_name": "Cu-30Ni", "dst_type": "Material",
                 "evidence": "Образцы медно-никелевого сплава Cu-30Ni отжигали", "confidence": 0.97},
                {"src_name": "Cu-30Ni", "src_type": "Material", "rel": "HAS_COMPOSITION",
                 "dst_name": "Cu", "dst_type": "Element",
                 "evidence": "медно-никелевого сплава Cu-30Ni", "confidence": 0.95},
                {"src_name": "Cu-30Ni", "src_type": "Material", "rel": "HAS_COMPOSITION",
                 "dst_name": "Ni", "dst_type": "Element",
                 "evidence": "медно-никелевого сплава Cu-30Ni", "confidence": 0.95},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "отжиг", "dst_type": "Process",
                 "evidence": "отжигали при 850 °C в течение 1 ч", "confidence": 0.97},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "RESULTS_IN",
                 "dst_name": "микротвёрдость по Виккерсу", "dst_type": "Property",
                 "evidence": "Микротвёрдость по Виккерсу выросла до 168 HV, на 20 % выше исходной", "confidence": 0.95},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "CONTAINS_PHASE",
                 "dst_name": "α-твёрдый раствор", "dst_type": "Phase",
                 "evidence": "равноосные зёрна α-твёрдого раствора", "confidence": 0.9},
                {"src_name": "α-твёрдый раствор", "src_type": "Phase", "rel": "OBSERVED_BY",
                 "dst_name": "СЭМ", "dst_type": "Method",
                 "evidence": "изучали методом сканирующей электронной микроскопии (СЭМ)", "confidence": 0.92},
            ],
        },
    },
    {
        "text": (
            "Сталь 40Х после закалки от 860 °C в масле и низкого отпуска при 200 °C "
            "достигла твёрдости 52 HRC. Рентгенофазовый анализ (РФА) подтвердил "
            "образование мартенсита. Предел прочности при растяжении составил 1600 МПа, "
            "при этом относительное удлинение снизилось до 9 %."
        ),
        "output": {
            "entities": [
                {"name": "Сталь 40Х", "type": "Material", "attributes": {}},
                {"name": "закалка", "type": "Process",
                 "attributes": {"temperature": 860, "temperature_unit": "°C", "medium": "масло"}},
                {"name": "низкий отпуск", "type": "Process",
                 "attributes": {"temperature": 200, "temperature_unit": "°C"}},
                {"name": "твёрдость", "type": "Property",
                 "attributes": {"value": 52, "unit": "HRC"}},
                {"name": "предел прочности при растяжении", "type": "Property",
                 "attributes": {"value": 1600, "unit": "MPa"}},
                {"name": "относительное удлинение", "type": "Property",
                 "attributes": {"value": 9, "unit": "%"}},
                {"name": "мартенсит", "type": "Phase", "attributes": {}},
                {"name": "РФА", "type": "Method", "attributes": {}},
                {"name": "Эксперимент 1", "type": "Experiment", "attributes": {}},
            ],
            "relations": [
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "USES",
                 "dst_name": "Сталь 40Х", "dst_type": "Material",
                 "evidence": "Сталь 40Х после закалки", "confidence": 0.97},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "закалка", "dst_type": "Process",
                 "evidence": "закалки от 860 °C в масле", "confidence": 0.96},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "низкий отпуск", "dst_type": "Process",
                 "evidence": "низкого отпуска при 200 °C", "confidence": 0.95},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "MEASURES",
                 "dst_name": "твёрдость", "dst_type": "Property",
                 "evidence": "достигла твёрдости 52 HRC", "confidence": 0.96},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "MEASURES",
                 "dst_name": "предел прочности при растяжении", "dst_type": "Property",
                 "evidence": "Предел прочности при растяжении составил 1600 МПа", "confidence": 0.96},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "RESULTS_IN",
                 "dst_name": "относительное удлинение", "dst_type": "Property",
                 "evidence": "относительное удлинение снизилось до 9 %", "confidence": 0.9},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "CONTAINS_PHASE",
                 "dst_name": "мартенсит", "dst_type": "Phase",
                 "evidence": "подтвердил образование мартенсита", "confidence": 0.93},
                {"src_name": "мартенсит", "src_type": "Phase", "rel": "OBSERVED_BY",
                 "dst_name": "РФА", "dst_type": "Method",
                 "evidence": "Рентгенофазовый анализ (РФА) подтвердил образование мартенсита", "confidence": 0.93},
            ],
        },
    },
    # --- Металлургический домен ТЗ (не только сплавы): электроэкстракция никеля ---
    {
        "text": (
            "При электроэкстракции никеля циркуляцию католита в ваннах организуют с "
            "расходом от 0.5 до 1.2 м3/ч на ванну — это обеспечивает равномерное "
            "распределение никеля и снижает риск дендритообразования на катоде. "
            "Опыт эксплуатации ванн электроэкстракции на предприятии в России показал "
            "повышение выхода по току до 96 %."
        ),
        "output": {
            "entities": [
                {"name": "электроэкстракция никеля", "type": "Process", "domain": "гидрометаллургия",
                 "attributes": {}},
                {"name": "ванна электроэкстракции", "type": "Equipment", "attributes": {}},
                {"name": "расход католита", "type": "Condition", "attributes": {},
                 "constraints": [{"param": "расход католита", "operator": "between",
                                  "value": 0.5, "value_high": 1.2, "unit": "m3/h"}]},
                {"name": "выход по току", "type": "Property", "attributes": {"value": 96, "unit": "%"}},
                {"name": "Эксперимент 1", "type": "Experiment", "geography": "Россия",
                 "is_domestic": True, "attributes": {}},
            ],
            "relations": [
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "электроэкстракция никеля", "dst_type": "Process",
                 "evidence": "При электроэкстракции никеля", "confidence": 0.96,
                 "geography": "Россия", "is_domestic": True},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "ванна электроэкстракции", "dst_type": "Equipment",
                 "evidence": "Опыт эксплуатации ванн электроэкстракции", "confidence": 0.9},
                {"src_name": "электроэкстракция никеля", "src_type": "Process",
                 "rel": "OPERATES_AT_CONDITION", "dst_name": "расход католита", "dst_type": "Condition",
                 "evidence": "расходом от 0.5 до 1.2 м3/ч на ванну", "confidence": 0.95,
                 "verification_level": "confirmed"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "RESULTS_IN",
                 "dst_name": "выход по току", "dst_type": "Property",
                 "evidence": "повышение выхода по току до 96 %", "confidence": 0.93,
                 "geography": "Россия", "is_domestic": True, "verification_level": "confirmed"},
            ],
        },
    },
    # --- Металлургический домен ТЗ: обессоливание воды обогатительной фабрики ---
    {
        "text": (
            "Для обогатительной фабрики с исходной водой, содержащей сульфаты, хлориды, "
            "Ca, Mg, Na в концентрации 200–300 мг/л, применили метод обратного осмоса, "
            "позволивший снизить сухой остаток до значения не более 1000 мг/дм3. Метод "
            "показал устойчивую работу при производительности от 100 т/сут."
        ),
        "output": {
            "entities": [
                {"name": "вода обогатительной фабрики", "type": "Material", "attributes": {}},
                {"name": "обратный осмос", "type": "Process", "domain": "экология", "attributes": {}},
                {"name": "исходная концентрация солей", "type": "Condition", "attributes": {},
                 "constraints": [{"param": "исходная концентрация солей", "operator": "between",
                                  "value": 200, "value_high": 300, "unit": "mg/L"}]},
                {"name": "сухой остаток", "type": "Property", "attributes": {},
                 "constraints": [{"param": "сухой остаток", "operator": "<=",
                                  "value": 1000, "unit": "mg/L"}]},
                {"name": "производительность", "type": "Property", "attributes": {},
                 "constraints": [{"param": "производительность", "operator": ">=",
                                  "value": 100, "unit": "t/day"}]},
                {"name": "Эксперимент 1", "type": "Experiment", "attributes": {}},
            ],
            "relations": [
                {"src_name": "обратный осмос", "src_type": "Process", "rel": "USES_MATERIAL",
                 "dst_name": "вода обогатительной фабрики", "dst_type": "Material",
                 "evidence": "с исходной водой, содержащей сульфаты, хлориды", "confidence": 0.9},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "APPLIES",
                 "dst_name": "обратный осмос", "dst_type": "Process",
                 "evidence": "применили метод обратного осмоса", "confidence": 0.96},
                {"src_name": "обратный осмос", "src_type": "Process", "rel": "OPERATES_AT_CONDITION",
                 "dst_name": "исходная концентрация солей", "dst_type": "Condition",
                 "evidence": "в концентрации 200–300 мг/л", "confidence": 0.93},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "RESULTS_IN",
                 "dst_name": "сухой остаток", "dst_type": "Property",
                 "evidence": "снизить сухой остаток до значения не более 1000 мг/дм3",
                 "confidence": 0.95, "verification_level": "confirmed"},
                {"src_name": "Эксперимент 1", "src_type": "Experiment", "rel": "MEASURES",
                 "dst_name": "производительность", "dst_type": "Property",
                 "evidence": "производительности от 100 т/сут", "confidence": 0.9},
            ],
        },
    },
]


# --------------------------------------------------------------------------
# Системный промпт и инструкция извлечения
# --------------------------------------------------------------------------
EXTRACTION_SYSTEM = (
    "Ты — эксперт по материаловедению и металлургии. Твоя задача — извлекать "
    "структурированные знания (сущности и связи) из научных текстов для построения "
    "графа знаний. Ты отвечаешь СТРОГО валидным JSON-объектом, без пояснений, "
    "без markdown-обёрток и без текста вокруг."
)

_RULES = f"""Извлеки сущности и связи из фрагмента научно-технического текста по
горно-металлургической отрасли (гидро-/пирометаллургия, обогащение, экология,
геомеханика, переработка отходов) — не только материаловедение сплавов.

Допустимые типы узлов: {_NODE_TYPES}
Допустимые типы связей (направленные): {_REL_TYPES}

Разрешённые шаблоны связей (используй ТОЛЬКО их):
- Experiment USES Material                     — в эксперименте использовали материал/сырьё
- Process|Equipment USES_MATERIAL Material     — технологическое решение использует материал/реагент
- Experiment APPLIES Process|Equipment          — применили режим обработки или оборудование
- Process|Experiment OPERATES_AT_CONDITION Condition — условие применения (климат, диапазон концентраций, режим)
- Process|Experiment PRODUCES_OUTPUT Material|Property — что получено на выходе
- Experiment MEASURES Property                  — измерили свойство/показатель
- Experiment RESULTS_IN Property                — обработка привела к изменению свойства (эффект)
- Experiment CONTAINS_PHASE Phase                — обнаружена фаза/микроструктура
- Material HAS_COMPOSITION Element               — материал содержит элемент
- Material EXHIBITS Property                     — материал обладает свойством (вне конкретного эксперимента)
- Property|Phase OBSERVED_BY Method              — измерено/выявлено методом
- Experiment|Process|Equipment DESCRIBED_IN Publication — где описано
- Experiment|Publication VALIDATED_BY Expert     — кто подтвердил/провалидировал факт
- Experiment|Publication CONTRADICTS Experiment|Publication — противоречащие друг другу факты
- Publication AUTHORED_BY Expert / Publication CITES Publication / Publication REPORTS Experiment
- Expert EXPERT_IN Process|Material / Expert AFFILIATED_WITH Facility

Правила:
1. Используй ТОЛЬКО перечисленные типы узлов и шаблоны связей. Если факт не ложится ни в один шаблон — пропусти его, НЕ выдумывай новый тип.
2. Каждый эксперимент в фрагменте оформляй как узел Experiment с именем "Эксперимент 1", "Эксперимент 2" и т.д.
3. Для КАЖДОЙ связи добавь "evidence" — дословную фразу из текста (копируй из фрагмента, не перефразируй).
4. Для простых измеренных величин записывай числа и единицы в attributes: {{"value": 168, "unit": "HV"}} или {{"temperature": 850, "temperature_unit": "°C", "time": 1, "time_unit": "h"}}.
5. Если в тексте явно указано ОГРАНИЧЕНИЕ (не диапазон измеренного значения, а требование/условие — «не более», «не менее», «от...до») — заполни у сущности поле "constraints": список объектов {{"param": "...", "operator": "<="|">="|"="|"between", "value": число, "value_high": число_или_null, "unit": "..."}}.
6. Если в тексте явно указана география (страна/регион) или дата актуализации факта — заполни поля сущности/связи "geography", "is_domestic" (true для РФ), "actualized_at" (ISO-дата). Если этого нет в тексте — НЕ выдумывай, оставь поле пустым/не указывай.
7. Если явно можно определить домен темы (гидрометаллургия/пирометаллургия/экология/переработка отходов/обогащение/геомеханика) — заполни "domain" у Process/Experiment/Publication.
8. Названия материалов/сплавов/процессов сохраняй как в тексте (например "Cu-30Ni", "электроэкстракция никеля") — нормализацией и сопоставлением синонимов (RU/EN) займётся отдельный модуль.
9. Не извлекай факты, которых нет в тексте. Лучше меньше связей, но точных.
10. "confidence" — твоя уверенность в связи от 0 до 1. "verification_level" связи — "confirmed", если текст явно подтверждает факт (измерено/показано/получено), "preliminary" — если формулировка предположительная («предполагается», «может быть»), иначе не указывай (по умолчанию "unverified").
11. MEASURES vs RESULTS_IN — не путай и не дублируй одно свойство обеими связями. RESULTS_IN используй, когда описан ЭФФЕКТ/ИЗМЕНЕНИЕ от обработки (слова «выросла», «снизилась», «увеличилась», «повысила», «улучшила», «привело к»). MEASURES — для просто заявленного измеренного значения без формулировки изменения. Если есть эффект — ставь RESULTS_IN, а не MEASURES.
12. HAS_COMPOSITION — ТОЛЬКО когда элемент является составной частью (составом) материала. НЕ отмечай как состав элементы, которые лишь РАСПРЕДЕЛЯЮТСЯ, извлекаются или получаются на выходе (напр. «распределение Au, Ag между штейном и шлаком» — золото НЕ входит в состав штейна как HAS_COMPOSITION; это выход/распределение → PRODUCES_OUTPUT или пропусти).
13. Именование Process — называй процесс ДЕЙСТВИЕМ (полной глагольной фразой из текста: «закачка шахтных вод в глубокие горизонты», «электроэкстракция никеля»), а НЕ оборудованием/системой («система закачки» — это Equipment, не Process).

Ответ — РОВНО один JSON-объект вида {{"entities": [...], "relations": [...]}}."""


def _render_examples(examples: list[dict]) -> str:
    blocks = []
    for i, ex in enumerate(examples, start=1):
        out = json.dumps(ex["output"], ensure_ascii=False, indent=2)
        blocks.append(
            f"### Пример {i}\n"
            f'Фрагмент:\n"""\n{ex["text"]}\n"""\n'
            f"Ответ:\n{out}"
        )
    return "\n\n".join(blocks)


def build_extraction_prompt(chunk_text: str, few_shot: bool = True,
                            examples: list[dict] | None = None) -> str:
    """Собрать промпт извлечения. few_shot=True добавляет эталонные примеры.

    На MetalGPT-1 рекомендуется few_shot=True. На очень длинных чанках или при
    дефиците контекста можно отключить (few_shot=False) ради экономии токенов.

    `examples` позволяет подменить набор few-shot примеров (например, для
    leave-one-out оценки качества, чтобы не оценивать на тех же парах).
    """
    parts = [_RULES]
    if few_shot:
        ex = examples if examples is not None else FEW_SHOT_EXAMPLES
        parts.append("Примеры правильного извлечения:\n\n" + _render_examples(ex))
    parts.append(f'Теперь обработай новый фрагмент.\nФрагмент:\n"""\n{chunk_text}\n"""\nОтвет:')
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Промпт генерации ответа (QA)
# --------------------------------------------------------------------------
ANSWER_SYSTEM = (
    "Ты — научный ассистент по горно-металлургической отрасли (гидро-/"
    "пирометаллургия, обогащение, экология, геомеханика). Отвечаешь на вопрос "
    "ТОЛЬКО на основе предоставленного контекста (подграф знаний + цитаты из "
    "статей). Если данных недостаточно — честно скажи об этом. К каждому "
    "утверждению добавляй ссылку на источник в формате [doc_id]. Если у связи "
    "в контексте указан уровень верификации (verification_level) или "
    "география — упоминай их в ответе («подтверждено», «предварительные "
    "данные», «по отечественной практике» и т.п.), не игнорируй их."
)

ANSWER_INSTRUCTION = """Вопрос: {question}

Контекст из графа знаний (связи, с провенансом и верификацией):
{graph_context}

Релевантные фрагменты статей:
{passages}

Дай связный ответ на русском. Опирайся только на контекст выше. Если контекст
ограничен по числовым условиям или географии из вопроса — явно скажи, что
ответ учитывает эти ограничения. В конце перечисли использованные источники.
"""


def build_answer_prompt(question: str, graph_context: str, passages: str) -> str:
    return (
        ANSWER_INSTRUCTION
        .replace("{question}", question)
        .replace("{graph_context}", graph_context)
        .replace("{passages}", passages)
    )


# --------------------------------------------------------------------------
# Промпт «литературного обзора» — структурированный синтез, не просто Q&A
# --------------------------------------------------------------------------
REVIEW_SYSTEM = (
    "Ты — научный аналитик по горно-металлургической отрасли. Составляешь "
    "структурированный литературный обзор по теме на основе предоставленных "
    "публикаций и связей графа знаний. Обзор должен группировать источники, "
    "явно выделять зоны консенсуса и зоны разногласий, и указывать степень "
    "уверенности (сколько источников подтверждает вывод). Никогда не "
    "выдумывай источники или цифры, которых нет в контексте."
)

REVIEW_INSTRUCTION = """Тема обзора: {topic}

Публикации и связи по теме (с географией/годом/уровнем верификации, где известны):
{graph_context}

Релевантные фрагменты статей:
{passages}

Составь литературный обзор на русском со следующей структурой:
1. **Обзор источников** — сгруппируй по методу/подходу, году, географии (РФ/мир).
2. **Консенсус** — выводы, которые подтверждаются несколькими независимыми источниками.
3. **Разногласия** — где источники расходятся или противоречат друг другу (см. связи CONTRADICTS в контексте, если есть).
4. **Степень уверенности** — для каждого ключевого вывода укажи количество подтверждающих источников и их уровень верификации.
В конце перечисли все использованные источники.
"""


def build_review_prompt(topic: str, graph_context: str, passages: str) -> str:
    return (
        REVIEW_INSTRUCTION
        .replace("{topic}", topic)
        .replace("{graph_context}", graph_context)
        .replace("{passages}", passages)
    )
