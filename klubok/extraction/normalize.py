"""Нормализация числовых величин и режимов из текста.

LLM плохо считает и путает единицы — поэтому численные параметры
(температура, время, давление, значения свойств) вытаскиваем regex'ом
и приводим к каноническим единицам. Чистая логика, без GPU.

    >>> q = parse_quantity("800 °C")
    >>> q.value, q.unit
    (800.0, '°C')
    >>> to_celsius(parse_quantity("1073 K")).value
    799.85

`parse_constraint` вытаскивает структурные числовые ОГРАНИЧЕНИЯ («сульфаты
≤300 мг/л», «производительность от 100 т/сут», «200–300 мг/л») — прямое
требование ТЗ («корректное распознавание числовых ограничений»). Используется
и при извлечении атрибутов сущностей, и при разборе вопроса пользователя
(klubok/retrieval/graphrag.py::extract_query_constraints).

    >>> c = parse_constraint("сульфаты ≤300 мг/л")
    >>> c.operator, c.value, c.unit
    ('<=', 300.0, 'mg/L')
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from klubok.ontology import NumericConstraint


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    raw: str


# число с десятичной точкой/запятой, опц. знак, опц. экспонента
_NUM = r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?"

# единицы, которые встречаются в материаловедении. Порядок важен (длинные сначала).
_UNIT_ALIASES = {
    "°c": "°C", "degc": "°C", "c": "°C",
    "k": "K",
    "mpa": "MPa", "мпа": "MPa",
    "gpa": "GPa", "гпа": "GPa",
    "kpa": "kPa",
    "pa": "Pa", "па": "Pa",
    "hv": "HV", "hb": "HB", "hrc": "HRC",     # твёрдость
    "h": "h", "ч": "h", "hr": "h", "hrs": "h",
    "min": "min", "мин": "min",
    "s": "s", "sec": "s", "с": "s",
    "wt%": "wt%", "wt.%": "wt%", "масс.%": "wt%", "%": "%",
    "at%": "at%", "at.%": "at%",
    "µm": "µm", "um": "µm", "mкm": "µm", "мкм": "µm",
    "nm": "nm", "нм": "nm",
    "mm": "mm", "мм": "mm",
    # металлургия/гидрометаллургия: концентрации, расходы, электрические величины
    "мг/л": "mg/L", "mg/l": "mg/L", "мг/дм3": "mg/L", "мг/дм³": "mg/L",
    "г/л": "g/L", "g/l": "g/L", "г/дм3": "g/L", "г/дм³": "g/L",
    "м3/ч": "m3/h", "м³/ч": "m3/h", "m3/h": "m3/h",
    "л/мин": "L/min", "l/min": "L/min", "л/ч": "L/h", "l/h": "L/h",
    "т/сут": "t/day", "т/сутки": "t/day", "t/day": "t/day",
    "а/м2": "A/m2", "а/м²": "A/m2", "a/m2": "A/m2",
}

# отсортированный по длине список вариантов для жадного матча
_UNIT_PATTERN = "|".join(
    re.escape(u) for u in sorted(_UNIT_ALIASES, key=len, reverse=True)
)
_QTY_RE = re.compile(rf"(?P<num>{_NUM})\s*(?P<unit>{_UNIT_PATTERN})", re.IGNORECASE)


def _to_float(s: str) -> float:
    return float(s.replace(",", "."))


def parse_quantity(text: str) -> Optional[Quantity]:
    """Первая величина в строке или None."""
    m = _QTY_RE.search(text)
    if not m:
        return None
    unit = _UNIT_ALIASES.get(m.group("unit").lower(), m.group("unit"))
    return Quantity(value=_to_float(m.group("num")), unit=unit, raw=m.group(0))


def parse_all_quantities(text: str) -> list[Quantity]:
    """Все величины в строке (для таблиц/перечислений)."""
    out: list[Quantity] = []
    for m in _QTY_RE.finditer(text):
        unit = _UNIT_ALIASES.get(m.group("unit").lower(), m.group("unit"))
        out.append(Quantity(value=_to_float(m.group("num")), unit=unit, raw=m.group(0)))
    return out


def to_celsius(q: Quantity) -> Quantity:
    """Привести температуру к °C (K -> °C)."""
    if q.unit == "K":
        return Quantity(value=round(q.value - 273.15, 2), unit="°C", raw=q.raw)
    return q


def to_hours(q: Quantity) -> Quantity:
    """Привести время к часам."""
    factor = {"h": 1.0, "min": 1 / 60, "s": 1 / 3600}.get(q.unit)
    if factor is None:
        return q
    return Quantity(value=round(q.value * factor, 4), unit="h", raw=q.raw)


# Категоризация единиц — помогает понять, к какому атрибуту относить число.
_DIMENSION = {
    "°C": "temperature", "K": "temperature",
    "h": "time", "min": "time", "s": "time",
    "MPa": "stress", "GPa": "stress", "kPa": "stress", "Pa": "stress",
    "HV": "hardness", "HB": "hardness", "HRC": "hardness",
    "wt%": "fraction", "at%": "fraction", "%": "fraction",
    "µm": "length", "nm": "length", "mm": "length",
    "mg/L": "concentration", "g/L": "concentration",
    "m3/h": "flow_rate", "L/min": "flow_rate", "L/h": "flow_rate",
    "t/day": "throughput", "A/m2": "current_density",
}


def dimension_of(q: Quantity) -> Optional[str]:
    return _DIMENSION.get(q.unit)


# --------------------------------------------------------------------------
# Числовые ограничения: операторы сравнения и диапазоны
# --------------------------------------------------------------------------
_LE_WORDS = r"(?:не\s+более|не\s+превышает|не\s+выше|до|≤|<=)"
_GE_WORDS = r"(?:не\s+менее|не\s+ниже|от|≥|>=)"
_RANGE_SEP = r"\s*(?:[-–—]|до)\s*"

_RANGE_RE = re.compile(
    rf"(?:от\s+)?(?P<lo>{_NUM})\s*{_RANGE_SEP}\s*(?P<hi>{_NUM})\s*(?P<unit>{_UNIT_PATTERN})",
    re.IGNORECASE,
)
_LE_RE = re.compile(
    rf"{_LE_WORDS}\s*(?P<num>{_NUM})\s*(?P<unit>{_UNIT_PATTERN})", re.IGNORECASE,
)
_GE_RE = re.compile(
    rf"{_GE_WORDS}\s*(?P<num>{_NUM})\s*(?P<unit>{_UNIT_PATTERN})", re.IGNORECASE,
)

# слово-кандидат на имя параметра: буквы (кириллица/латиница), дефис
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё\-]+")


def _guess_param(text: str, match_start: int, max_words: int = 3) -> str:
    """Взять несколько слов перед найденным числом как имя параметра.

    'содержание сульфатов не более 300 мг/л' + match на '300 мг/л' -> 'содержание сульфатов'.
    Эвристика, не панацея — при экстракции LLM параметр обычно переопределяется
    явным полем сущности (Property.name), это вспомогательный сигнал для вопросов
    пользователя без сущности под рукой.
    """
    before = text[:match_start]
    words = _WORD_RE.findall(before)
    stop = {"не", "более", "менее", "выше", "ниже", "превышает", "от", "до"}
    words = [w for w in words if w.lower() not in stop]
    return " ".join(words[-max_words:]).strip()


def parse_constraint(text: str) -> Optional[NumericConstraint]:
    """Первое числовое ограничение в строке (диапазон/оператор) или None.

    Порядок попыток: диапазон ('200-300 мг/л') -> '<=' -> '>=' -> просто
    величина ('=' по умолчанию, через parse_quantity).
    """
    m = _RANGE_RE.search(text)
    if m:
        unit = _UNIT_ALIASES.get(m.group("unit").lower(), m.group("unit"))
        return NumericConstraint(
            param=_guess_param(text, m.start()), operator="between",
            value=_to_float(m.group("lo")), value_high=_to_float(m.group("hi")), unit=unit,
        )

    m = _LE_RE.search(text)
    if m:
        unit = _UNIT_ALIASES.get(m.group("unit").lower(), m.group("unit"))
        return NumericConstraint(
            param=_guess_param(text, m.start()), operator="<=",
            value=_to_float(m.group("num")), unit=unit,
        )

    m = _GE_RE.search(text)
    if m:
        unit = _UNIT_ALIASES.get(m.group("unit").lower(), m.group("unit"))
        return NumericConstraint(
            param=_guess_param(text, m.start()), operator=">=",
            value=_to_float(m.group("num")), unit=unit,
        )

    q = parse_quantity(text)
    if q:
        idx = text.find(q.raw)
        return NumericConstraint(
            param=_guess_param(text, idx if idx >= 0 else 0),
            operator="=", value=q.value, unit=q.unit,
        )
    return None


def parse_all_constraints(text: str) -> list[NumericConstraint]:
    """Все ограничения в строке (вопрос может нести несколько условий сразу)."""
    out: list[NumericConstraint] = []
    for m in _RANGE_RE.finditer(text):
        unit = _UNIT_ALIASES.get(m.group("unit").lower(), m.group("unit"))
        out.append(NumericConstraint(
            param=_guess_param(text, m.start()), operator="between",
            value=_to_float(m.group("lo")), value_high=_to_float(m.group("hi")), unit=unit,
        ))
    covered = {(m.start(), m.end()) for m in _RANGE_RE.finditer(text)}

    def _overlaps(pos: int) -> bool:
        return any(s <= pos < e for s, e in covered)

    for regex, op in ((_LE_RE, "<="), (_GE_RE, ">=")):
        for m in regex.finditer(text):
            if _overlaps(m.start()):
                continue
            unit = _UNIT_ALIASES.get(m.group("unit").lower(), m.group("unit"))
            out.append(NumericConstraint(
                param=_guess_param(text, m.start()), operator=op,
                value=_to_float(m.group("num")), unit=unit,
            ))
    return out
