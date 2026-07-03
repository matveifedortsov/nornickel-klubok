"""Онтология предметной области и модели данных пайплайна.

Это «контракт» всего проекта: парсинг -> извлечение -> граф -> поиск
обмениваются именно этими структурами. Зафиксируйте схему в первый час
хакатона — менять её потом дорого.

Домен — R&D горно-металлургической отрасли (гидро-/пирометаллургия,
обогащение, экология, переработка отходов), не только материаловедение
сплавов: узлы и связи специально общие, чтобы покрыть и «CuNi после
отжига», и «циркуляцию католита при электроэкстракции никеля».

Чистый модуль без внешних зависимостей (кроме pydantic) -> тестируется
до выделения вычислительных мощностей.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------
# Типы узлов и рёбер графа знаний
# --------------------------------------------------------------------------
class NodeType(str, Enum):
    MATERIAL = "Material"        # материал / сплав / реагент / отход
    ELEMENT = "Element"          # химический элемент / компонент
    PROCESS = "Process"          # технологическое решение / режим (выщелачивание, отжиг, ...)
    EQUIPMENT = "Equipment"      # оборудование / установка (ванна электроэкстракции, ПВП, ...)
    PROPERTY = "Property"        # измеримое свойство / технико-экономический показатель
    CONDITION = "Condition"      # условие применения (климат, диапазон концентраций, география)
    EXPERIMENT = "Experiment"    # конкретный эксперимент / протокол опыта
    PUBLICATION = "Publication"  # статья / отчёт / патент / норматив — источник
    EXPERT = "Expert"            # автор / носитель компетенции
    FACILITY = "Facility"        # лаборатория / институт / команда
    METHOD = "Method"            # метод измерения (SEM, XRD, OM, ...)
    PHASE = "Phase"              # фаза / микроструктура


class RelType(str, Enum):
    HAS_COMPOSITION = "HAS_COMPOSITION"           # Material -> Element
    USES = "USES"                                 # Experiment -> Material
    USES_MATERIAL = "USES_MATERIAL"               # Process/Equipment -> Material (вне конкретного опыта)
    APPLIES = "APPLIES"                           # Experiment -> Process/Equipment
    OPERATES_AT_CONDITION = "OPERATES_AT_CONDITION"  # Process/Experiment -> Condition
    PRODUCES_OUTPUT = "PRODUCES_OUTPUT"           # Process/Experiment -> Material/Property
    MEASURES = "MEASURES"                         # Experiment -> Property
    RESULTS_IN = "RESULTS_IN"                     # Experiment -> Effect/Property change
    REPORTS = "REPORTS"                           # Publication -> Experiment
    DESCRIBED_IN = "DESCRIBED_IN"                 # Experiment/Process/Equipment -> Publication
    VALIDATED_BY = "VALIDATED_BY"                 # Experiment/Publication -> Expert
    CONTRADICTS = "CONTRADICTS"                   # Experiment<->Experiment, Publication<->Publication
    AUTHORED_BY = "AUTHORED_BY"                   # Publication -> Expert
    EXPERT_IN = "EXPERT_IN"                       # Expert -> Process/Material
    AFFILIATED_WITH = "AFFILIATED_WITH"           # Expert -> Facility
    EXHIBITS = "EXHIBITS"                         # Material -> Property
    OBSERVED_BY = "OBSERVED_BY"                   # Property/Phase -> Method
    CONTAINS_PHASE = "CONTAINS_PHASE"             # Material/Experiment -> Phase
    CITES = "CITES"                               # Publication -> Publication


# Допустимые пары (src_type, RelType, dst_type). Валидатор графа использует это,
# чтобы не пускать «мусорные» связи от LLM в базу.
ALLOWED_EDGES: set[tuple[NodeType, RelType, NodeType]] = {
    (NodeType.MATERIAL, RelType.HAS_COMPOSITION, NodeType.ELEMENT),
    (NodeType.EXPERIMENT, RelType.USES, NodeType.MATERIAL),
    (NodeType.PROCESS, RelType.USES_MATERIAL, NodeType.MATERIAL),
    (NodeType.EQUIPMENT, RelType.USES_MATERIAL, NodeType.MATERIAL),
    (NodeType.EXPERIMENT, RelType.APPLIES, NodeType.PROCESS),
    (NodeType.EXPERIMENT, RelType.APPLIES, NodeType.EQUIPMENT),
    (NodeType.PROCESS, RelType.OPERATES_AT_CONDITION, NodeType.CONDITION),
    (NodeType.EXPERIMENT, RelType.OPERATES_AT_CONDITION, NodeType.CONDITION),
    (NodeType.PROCESS, RelType.PRODUCES_OUTPUT, NodeType.MATERIAL),
    (NodeType.PROCESS, RelType.PRODUCES_OUTPUT, NodeType.PROPERTY),
    (NodeType.EXPERIMENT, RelType.PRODUCES_OUTPUT, NodeType.MATERIAL),
    (NodeType.EXPERIMENT, RelType.MEASURES, NodeType.PROPERTY),
    (NodeType.EXPERIMENT, RelType.RESULTS_IN, NodeType.PROPERTY),
    (NodeType.EXPERIMENT, RelType.CONTAINS_PHASE, NodeType.PHASE),
    (NodeType.PUBLICATION, RelType.REPORTS, NodeType.EXPERIMENT),
    (NodeType.EXPERIMENT, RelType.DESCRIBED_IN, NodeType.PUBLICATION),
    (NodeType.PROCESS, RelType.DESCRIBED_IN, NodeType.PUBLICATION),
    (NodeType.EQUIPMENT, RelType.DESCRIBED_IN, NodeType.PUBLICATION),
    (NodeType.EXPERIMENT, RelType.VALIDATED_BY, NodeType.EXPERT),
    (NodeType.PUBLICATION, RelType.VALIDATED_BY, NodeType.EXPERT),
    (NodeType.EXPERIMENT, RelType.CONTRADICTS, NodeType.EXPERIMENT),
    (NodeType.PUBLICATION, RelType.CONTRADICTS, NodeType.PUBLICATION),
    (NodeType.PUBLICATION, RelType.AUTHORED_BY, NodeType.EXPERT),
    (NodeType.EXPERT, RelType.EXPERT_IN, NodeType.PROCESS),
    (NodeType.EXPERT, RelType.EXPERT_IN, NodeType.MATERIAL),
    (NodeType.EXPERT, RelType.AFFILIATED_WITH, NodeType.FACILITY),
    (NodeType.MATERIAL, RelType.EXHIBITS, NodeType.PROPERTY),
    (NodeType.MATERIAL, RelType.CONTAINS_PHASE, NodeType.PHASE),
    (NodeType.PROPERTY, RelType.OBSERVED_BY, NodeType.METHOD),
    (NodeType.PHASE, RelType.OBSERVED_BY, NodeType.METHOD),
    (NodeType.PUBLICATION, RelType.CITES, NodeType.PUBLICATION),
}


def edge_is_valid(src: NodeType, rel: RelType, dst: NodeType) -> bool:
    return (src, rel, dst) in ALLOWED_EDGES


# --------------------------------------------------------------------------
# Числовые ограничения («сульфаты ≤300 мг/л», «производительность от 100 т/сут»)
# --------------------------------------------------------------------------
class NumericConstraint(BaseModel):
    """Структурное числовое условие — вместо того, чтобы прятать его в тексте.

    Используется и как атрибут сущности (Property/Process/Condition), и как
    результат разбора вопроса пользователя (klubok/retrieval/graphrag.py),
    чтобы фильтровать граф по диапазонам, а не только семантически.
    """
    param: str                                              # напр. "сульфаты", "производительность"
    operator: Literal["<=", ">=", "=", "between"]
    value: float
    value_high: Optional[float] = None                      # для operator="between"
    unit: str = ""


# --------------------------------------------------------------------------
# Документы и чанки (вход извлечения / источник для эмбеддингов)
# --------------------------------------------------------------------------
class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    section: Optional[str] = None
    page: Optional[int] = None
    text: str


class Document(BaseModel):
    """Результат парсинга файла — вход в извлечение, источник Publication-узла."""
    doc_id: str
    title: Optional[str] = None
    source_path: Optional[str] = None
    year: Optional[int] = None
    chunks: list[Chunk] = Field(default_factory=list)
    # метаданные публикации (не всегда извлекаемы из текста — часто дешевле взять из
    # структуры корпуса/имени файла, см. klubok/parsing/filename_meta.py)
    authors: list[str] = Field(default_factory=list)
    publication_type: Optional[str] = None      # статья / отчёт / патент / норматив / доклад
    domain: Optional[str] = None                # гидрометаллургия / пирометаллургия / экология / ...
    geography: Optional[str] = None             # страна/регион публикации
    is_domestic: Optional[bool] = None          # отечественная практика (РФ) vs мировая
    sensitivity: str = "internal"               # internal | external — для разграничения доступа


# --------------------------------------------------------------------------
# Извлечённые сущности и связи (выход LLM-экстрактора)
# --------------------------------------------------------------------------
class Entity(BaseModel):
    """Узел графа. `canonical_id` проставляет resolver после нормализации."""
    name: str
    type: NodeType
    canonical_id: Optional[str] = None
    # свободные атрибуты: value/unit для Property, temp/time для Process и т.п.
    attributes: dict[str, str | float | int] = Field(default_factory=dict)
    constraints: list[NumericConstraint] = Field(default_factory=list)
    # общие для Publication/Experiment/Expert; на прочих типах просто не используются
    geography: Optional[str] = None
    domain: Optional[str] = None
    is_domestic: Optional[bool] = None

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Entity.name пустой")
        return v

    @property
    def key(self) -> str:
        """Ключ для дедупа до резолвинга."""
        return self.canonical_id or f"{self.type.value}:{self.name.lower()}"


class Relation(BaseModel):
    """Ребро графа. Ссылается на сущности по их `name` внутри одного документа."""
    src_name: str
    src_type: NodeType
    rel: RelType
    dst_name: str
    dst_type: NodeType
    # доказательная база — на этом строятся цитаты в ответе
    evidence: Optional[str] = None      # дословная фраза из текста
    chunk_id: Optional[str] = None
    confidence: float = 1.0
    # верификация факта (ТЗ: источник, уровень достоверности, дата актуализации)
    source_type: Optional[str] = None                # publication | internal_report | experiment
    verification_level: str = "unverified"            # confirmed | preliminary | disputed | unverified
    actualized_at: Optional[str] = None                # ISO-дата
    geography: Optional[str] = None
    is_domestic: Optional[bool] = None

    def is_schema_valid(self) -> bool:
        return edge_is_valid(self.src_type, self.rel, self.dst_type)


class ExtractionResult(BaseModel):
    """Результат извлечения по одному чанку/документу."""
    doc_id: str
    chunk_id: Optional[str] = None
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    def schema_valid_relations(self) -> list[Relation]:
        """Отфильтровать связи, нарушающие онтологию (защита от галлюцинаций LLM)."""
        return [r for r in self.relations if r.is_schema_valid()]
