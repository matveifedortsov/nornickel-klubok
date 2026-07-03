"""Метрики качества извлечения: precision / recall / F1 по сущностям и рёбрам.

Сравнение идёт по КАНОНИЗИРОВАННЫМ ключам (через resolver.canonical_id), чтобы
'Cu-Ni' и 'CuNi' считались совпадением — нам важна правильность факта, а не
буквальное написание. evidence/confidence в матчинге не участвуют.

Чистый модуль без GPU/БД — тестируется сразу.
"""
from __future__ import annotations

from dataclasses import dataclass

from klubok.ontology import Entity, Relation, ExtractionResult, NodeType, RelType
from klubok.extraction.resolver import canonical_id
from klubok.extraction.normalize import _UNIT_ALIASES


# --------------------------------------------------------------------------
# Ключи для сравнения
# --------------------------------------------------------------------------
def entity_key(e: Entity) -> str:
    return canonical_id(e)


def relation_key(r: Relation) -> tuple[str, str, str]:
    src = canonical_id(Entity(name=r.src_name, type=r.src_type))
    dst = canonical_id(Entity(name=r.dst_name, type=r.dst_type))
    return (src, r.rel.value, dst)


# --------------------------------------------------------------------------
# Атрибуты сущностей (partial-credit: числа value/unit, температуры, времена)
# --------------------------------------------------------------------------
def _is_number(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        try:
            float(v.replace(",", "."))
            return True
        except ValueError:
            return False
    return False


def _norm_attr_value(v) -> str:
    """Канонизировать значение атрибута для сравнения.

    Числа -> округлённый float (850 == 850.0). Единицы -> канон по словарю
    алиасов ('c'/'°C' -> '°C'). Прочие строки -> lower/strip.
    """
    if _is_number(v):
        f = float(v.replace(",", ".")) if isinstance(v, str) else float(v)
        return repr(round(f, 4))
    s = str(v).strip()
    return _UNIT_ALIASES.get(s.lower(), s.lower())


def attribute_items(entities: list[Entity]) -> set[tuple[str, str, str]]:
    """Множество (canonical_id, имя_атрибута, норм_значение) по всем сущностям."""
    items: set[tuple[str, str, str]] = set()
    for e in entities:
        cid = canonical_id(e)
        for k, v in (e.attributes or {}).items():
            items.add((cid, k.strip().lower(), _norm_attr_value(v)))
    return items


# --------------------------------------------------------------------------
# Базовая P/R/F1 по множествам
# --------------------------------------------------------------------------
@dataclass
class PRF:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": round(self.precision, 3),
                "recall": round(self.recall, 3),
                "f1": round(self.f1, 3)}


def prf_from_sets(pred: set, gold: set) -> PRF:
    tp = len(pred & gold)
    return PRF(tp=tp, fp=len(pred - gold), fn=len(gold - pred))


# --------------------------------------------------------------------------
# Оценка одного извлечения и агрегирование
# --------------------------------------------------------------------------
@dataclass
class ExampleScore:
    entities: PRF
    relations: PRF
    attributes: PRF                    # partial-credit по value/unit/temperature/...
    missed_relations: list[tuple]      # FN — что не нашли
    spurious_relations: list[tuple]    # FP — что придумали лишнего
    wrong_attributes: list[tuple]      # FN по атрибутам — неверные/пропущенные числа


def score_extraction(pred: ExtractionResult, gold: ExtractionResult) -> ExampleScore:
    pe = {entity_key(e) for e in pred.entities}
    ge = {entity_key(e) for e in gold.entities}
    pr = {relation_key(r) for r in pred.relations}
    gr = {relation_key(r) for r in gold.relations}
    pa = attribute_items(pred.entities)
    ga = attribute_items(gold.entities)
    return ExampleScore(
        entities=prf_from_sets(pe, ge),
        relations=prf_from_sets(pr, gr),
        attributes=prf_from_sets(pa, ga),
        missed_relations=sorted(gr - pr),
        spurious_relations=sorted(pr - gr),
        wrong_attributes=sorted(ga - pa),
    )


def micro_average(scores: list[ExampleScore]) -> dict:
    """Микро-усреднение (суммируем tp/fp/fn по всем примерам)."""
    def agg(getter) -> PRF:
        tp = sum(getter(s).tp for s in scores)
        fp = sum(getter(s).fp for s in scores)
        fn = sum(getter(s).fn for s in scores)
        return PRF(tp, fp, fn)
    return {
        "entities": agg(lambda s: s.entities).as_dict(),
        "relations": agg(lambda s: s.relations).as_dict(),
        "attributes": agg(lambda s: s.attributes).as_dict(),
    }


# --------------------------------------------------------------------------
# Конвертация эталонного словаря (формат few-shot / gold) в ExtractionResult
# --------------------------------------------------------------------------
def gold_from_dict(output: dict, doc_id: str = "gold") -> ExtractionResult:
    entities = [
        Entity(name=e["name"], type=NodeType(e["type"]), attributes=e.get("attributes", {}) or {})
        for e in output.get("entities", [])
    ]
    relations = [
        Relation(
            src_name=r["src_name"], src_type=NodeType(r["src_type"]),
            rel=RelType(r["rel"]), dst_name=r["dst_name"], dst_type=NodeType(r["dst_type"]),
            evidence=r.get("evidence"), confidence=float(r.get("confidence", 1.0)),
        )
        for r in output.get("relations", [])
    ]
    return ExtractionResult(doc_id=doc_id, entities=entities, relations=relations)
