"""Дешёвые эвристики для верификационных/гео/доменных полей — БЕЗ LLM.

На реальных металлургических текстах LLM часто не проставляет geography/domain
и оставляет verification_level="unverified" (в тексте нет явной страны/даты у
каждого факта). Тогда «модель верификации знаний» и гео-фильтр из ТЗ выглядят
пустыми. Эти функции добирают поля надёжными правилами по языку/ключевым словам:

  * verification_from_evidence — уровень достоверности по глаголам в цитате;
  * detect_is_domestic         — РФ vs мир по преобладанию кириллицы;
  * detect_domain              — тех. домен по ключевым словам.

Всё чистое и тестируемое без БД. Применяется в extractor (по цитате) и pipeline
(по тексту документа) как fallback, когда LLM/метаданные молчат.
"""
from __future__ import annotations

import re

# --- уровень верификации по формулировке цитаты ---
_CONFIRMED = re.compile(
    r"получен|показал|установлен|измерен|составил|достиг|обеспечива|подтвержд|"
    r"позволил|привел|доказан|зафиксирован|show|achiev|obtain|measur|demonstrat|result",
    re.IGNORECASE)
_PRELIMINARY = re.compile(
    r"предполага|вероятно|возможно|может быть|ожидается|по-видимому|потенциально|"
    r"perhaps|likely|may be|expected|potential|assum",
    re.IGNORECASE)


def verification_from_evidence(evidence: str | None) -> str:
    """Уровень достоверности по цитате: confirmed | preliminary | unverified."""
    if not evidence:
        return "unverified"
    if _PRELIMINARY.search(evidence):
        return "preliminary"
    if _CONFIRMED.search(evidence):
        return "confirmed"
    return "unverified"


# --- отечественная практика vs мировая по языку текста ---
def detect_is_domestic(text: str) -> tuple[bool | None, str | None]:
    """(is_domestic, geography) по преобладанию кириллицы.

    Корпус: русские отраслевые журналы (кириллица) vs иностранные proceedings
    (латиница). Возвращает (True,'Россия') / (False,None) / (None,None) если
    текста мало или язык неоднозначен.
    """
    cyr = len(re.findall(r"[а-яё]", text, re.IGNORECASE))
    lat = len(re.findall(r"[a-z]", text, re.IGNORECASE))
    total = cyr + lat
    if total < 200:
        return None, None
    frac = cyr / total
    if frac >= 0.6:
        return True, "Россия"
    if frac <= 0.2:
        return False, None
    return None, None


# --- технологический домен по ключевым словам ---
_DOMAIN_KEYWORDS = [
    ("гидрометаллургия", ("выщелачивани", "электроэкстракц", "электролиз", "католит",
                          "раствор", "экстракц", "сорбц", "автоклав", "leaching", "electrowinning")),
    ("пирометаллургия", ("плавк", "штейн", "шлак", "печь", "конвертир", "обжиг", "восстановлен",
                         "smelt", "furnace", "slag", "matte")),
    ("обогащение", ("флотац", "обогащени", "измельчени", "концентрат", "хвост", "флотореагент",
                    "flotation", "concentrat")),
    ("экология", ("шахтн", "сточн", "очистк", "so2", "выброс", "экологи", "обессоливани",
                  "деминерализац", "wastewater", "emission")),
    ("переработка отходов", ("отход", "техногенн", "закладк", "гипс", "утилизац", "рецикл",
                             "tailings", "waste", "recycl")),
]


def detect_domain(text: str) -> str | None:
    """Тех. домен по частоте ключевых слов; None если ничего не набралось."""
    low = text.lower()
    best, best_score = None, 0
    for domain, kws in _DOMAIN_KEYWORDS:
        score = sum(low.count(kw) for kw in kws)
        if score > best_score:
            best, best_score = domain, score
    return best if best_score >= 2 else None


# --- добор числовых атрибутов Property из цитат (слабейшая метрика — атрибуты) ---
def backfill_property_values(entities, relations) -> int:
    """Заполнить value/unit у Property-узлов из цитат связей — БЕЗ LLM.

    ТЗ прямо требует точности чисел (концентрации/температуры), а это слабейшая
    метрика извлечения: LLM часто называет свойство («содержание золота в шлаке»),
    но не выносит число в attributes, хотя оно есть в цитате («не превышает
    0.5 г/т»). Достаём первую величину из evidence связи, где это свойство —
    приёмник, и проставляем value+unit. НИКОГДА не перезаписываем то, что уже
    дал LLM (fill-if-empty). Чистая логика, тестируется без БД.

    Возвращает число заполненных узлов.
    """
    from klubok.ontology import NodeType
    from klubok.extraction.normalize import parse_quantity

    by_key = {(e.type, e.name): e for e in entities}
    filled = 0
    for r in relations:
        if r.dst_type != NodeType.PROPERTY or not r.evidence:
            continue
        ent = by_key.get((NodeType.PROPERTY, r.dst_name))
        if ent is None:
            continue
        attrs = ent.attributes or {}
        # уже есть числовое значение (value/current_density/…) — не трогаем
        if any(isinstance(v, (int, float)) for v in attrs.values()):
            continue
        q = parse_quantity(r.evidence)
        if q is None:
            continue
        ent.attributes = {**attrs, "value": q.value, "unit": q.unit}
        filled += 1
    return filled
