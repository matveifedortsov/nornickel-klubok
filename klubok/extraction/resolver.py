"""Резолвинг сущностей: канонизация имён и дедуп узлов.

Без этого граф рассыпается на дубли («CuNi», «Cu-Ni», «медно-никелевый»
станут тремя разными узлами, и обходы перестанут работать). Это главный
таймсинк трека — поэтому модуль написан и протестирован заранее.

Стратегия (дёшево -> дорого):
  1. Правила нормализации строки по типу узла.
  2. Словарь синонимов/алиасов (расширяется руками по ходу).
  3. (на этапе с железом) добор по косинусной близости эмбеддингов —
     хук `embedding_merge` оставлен, но не обязателен для MVP.
"""
from __future__ import annotations

import re
from collections import defaultdict

from klubok.ontology import Entity, NodeType
from klubok.extraction.glossary_ru_en import GLOSSARY


# Символы химических элементов — чтобы корректно разбирать формулы сплавов.
_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Pb", "Bi",
}

# Русские названия -> символ. Дополняйте по мере встреч в корпусе.
_RU_ELEMENT = {
    "медь": "Cu", "медно": "Cu", "никель": "Ni", "никелевый": "Ni",
    "железо": "Fe", "хром": "Cr", "титан": "Ti", "алюминий": "Al",
    "кобальт": "Co", "цинк": "Zn", "олово": "Sn", "свинец": "Pb",
    # благородные металлы и МПГ — ядро домена Норникеля (штейн/шлак, МПГ)
    "золото": "Au", "серебро": "Ag", "платина": "Pt", "палладий": "Pd",
    "родий": "Rh", "рутений": "Ru", "иридий": "Ir", "осмий": "Os",
    "магний": "Mg", "марганец": "Mn", "кремний": "Si", "сера": "S",
    "натрий": "Na", "кальций": "Ca", "калий": "K", "молибден": "Mo",
    "вольфрам": "W", "ванадий": "V", "мышьяк": "As", "сурьма": "Sb",
    "висмут": "Bi", "селен": "Se", "теллур": "Te", "кадмий": "Cd",
}

# Ручной словарь алиасов: любое из значений слева -> каноническое справа.
# Плюс отраслевой глоссарий RU/EN (klubok/extraction/glossary_ru_en.py) — держим
# отдельным файлом-данными, чтобы расширять терминологию без правки кода.
ALIAS_MAP: dict[str, str] = {
    "stainless steel": "Steel",
    "нержавеющая сталь": "Steel",
    "нержавейка": "Steel",
    **GLOSSARY,
}


def _split_alloy_tokens(name: str) -> list[str]:
    """Разбить 'Cu-Ni', 'CuNi', 'Cu Ni' на ['Cu','Ni'] если это похоже на сплав."""
    # явные разделители
    parts = re.split(r"[-/\s,]+", name)
    if len(parts) > 1 and all(p.capitalize() in _ELEMENTS for p in parts if p):
        return [p.capitalize() for p in parts if p]
    # слитная запись CamelCase: CuNiFe -> Cu Ni Fe
    camel = re.findall(r"[A-Z][a-z]?", name)
    if camel and "".join(camel) == name and all(c in _ELEMENTS for c in camel):
        return camel
    return []


def canonical_material(name: str) -> str:
    """Канонизировать имя материала/сплава.

    'Cu-Ni' / 'CuNi' / 'медно-никелевый' -> 'CuNi' (элементы по алфавиту).
    """
    low = name.strip().lower()
    if low in ALIAS_MAP:
        return ALIAS_MAP[low]

    # русские составные («медно-никелевый сплав»)
    ru_tokens = [_RU_ELEMENT[w] for w in re.split(r"[-\s]+", low) if w in _RU_ELEMENT]
    if ru_tokens:
        return "".join(sorted(set(ru_tokens)))

    tokens = _split_alloy_tokens(name.strip())
    if tokens:
        return "".join(sorted(set(tokens)))      # CuNi, FeNi, ...

    return name.strip()


def canonical_element(name: str) -> str:
    low = name.strip().lower()
    if low in _RU_ELEMENT:
        return _RU_ELEMENT[low]
    cap = name.strip().capitalize()
    return cap if cap in _ELEMENTS else name.strip()


def _canonical_generic(name: str) -> str:
    """Свойства/методы/процессы: схлопнуть регистр и пробелы, применить алиасы."""
    norm = re.sub(r"\s+", " ", name.strip().lower())
    return ALIAS_MAP.get(norm, norm)


def canonical_id(entity: Entity) -> str:
    """Построить стабильный canonical_id вида 'Material:CuNi'."""
    t = entity.type
    if t == NodeType.MATERIAL:
        base = canonical_material(entity.name)
    elif t == NodeType.ELEMENT:
        base = canonical_element(entity.name)
    else:
        base = _canonical_generic(entity.name)
    return f"{t.value}:{base}"


def resolve(entities: list[Entity]) -> tuple[list[Entity], dict[str, str]]:
    """Проставить canonical_id и схлопнуть дубли.

    Возвращает (уникальные сущности, отображение исходное_имя -> canonical_id).
    Атрибуты дублей объединяются.
    """
    by_canon: dict[str, Entity] = {}
    name_to_canon: dict[str, str] = {}

    for e in entities:
        cid = canonical_id(e)
        name_to_canon[f"{e.type.value}:{e.name}"] = cid
        if cid in by_canon:
            by_canon[cid].attributes.update(e.attributes)
        else:
            merged = e.model_copy(deep=True)
            merged.canonical_id = cid
            by_canon[cid] = merged

    return list(by_canon.values()), name_to_canon


def group_aliases(entities: list[Entity]) -> dict[str, list[str]]:
    """Диагностика: какие исходные имена схлопнулись в один canonical_id."""
    groups: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        groups[canonical_id(e)].append(e.name)
    return {k: sorted(set(v)) for k, v in groups.items() if len(set(v)) > 1}
