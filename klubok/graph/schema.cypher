// Constraints и индексы графа знаний «Научный клубок».
// Применяется один раз при инициализации БД (см. neo4j_client.apply_schema).
// Узлы идентифицируются по canonical_id (например 'Material:CuNi').

// --- Уникальность узлов по типам ---
CREATE CONSTRAINT material_id     IF NOT EXISTS FOR (n:Material)    REQUIRE n.canonical_id IS UNIQUE;
CREATE CONSTRAINT element_id      IF NOT EXISTS FOR (n:Element)     REQUIRE n.canonical_id IS UNIQUE;
CREATE CONSTRAINT process_id      IF NOT EXISTS FOR (n:Process)     REQUIRE n.canonical_id IS UNIQUE;
CREATE CONSTRAINT equipment_id    IF NOT EXISTS FOR (n:Equipment)   REQUIRE n.canonical_id IS UNIQUE;
CREATE CONSTRAINT property_id     IF NOT EXISTS FOR (n:Property)    REQUIRE n.canonical_id IS UNIQUE;
CREATE CONSTRAINT condition_id    IF NOT EXISTS FOR (n:Condition)   REQUIRE n.canonical_id IS UNIQUE;
CREATE CONSTRAINT experiment_id   IF NOT EXISTS FOR (n:Experiment)  REQUIRE n.canonical_id IS UNIQUE;
CREATE CONSTRAINT publication_id  IF NOT EXISTS FOR (n:Publication) REQUIRE n.canonical_id IS UNIQUE;
CREATE CONSTRAINT expert_id       IF NOT EXISTS FOR (n:Expert)      REQUIRE n.canonical_id IS UNIQUE;
CREATE CONSTRAINT facility_id     IF NOT EXISTS FOR (n:Facility)    REQUIRE n.canonical_id IS UNIQUE;
CREATE CONSTRAINT method_id       IF NOT EXISTS FOR (n:Method)      REQUIRE n.canonical_id IS UNIQUE;
CREATE CONSTRAINT phase_id        IF NOT EXISTS FOR (n:Phase)       REQUIRE n.canonical_id IS UNIQUE;

// --- Полнотекстовые индексы для входа в граф по названию ---
// (seed-узлы гибридного поиска: материалы/процессы/оборудование/методы + люди/лаборатории)
CREATE FULLTEXT INDEX entity_names IF NOT EXISTS
FOR (n:Material|Element|Process|Equipment|Property|Phase|Method|Expert|Facility)
ON EACH [n.name, n.canonical_id];

// --- Индексы для гео-/доменной фильтрации и числовых диапазонов (§4 гибридный поиск) ---
// Ставим на узлах, а не на рёбрах — Publication/Experiment несут geography/domain,
// что и позволяет фильтровать выдачу «отечественная практика» vs «мировая» без полного skan'а.
CREATE INDEX publication_geography IF NOT EXISTS FOR (n:Publication) ON (n.geography);
CREATE INDEX publication_domain    IF NOT EXISTS FOR (n:Publication) ON (n.domain);
CREATE INDEX publication_domestic  IF NOT EXISTS FOR (n:Publication) ON (n.is_domestic);
CREATE INDEX experiment_geography  IF NOT EXISTS FOR (n:Experiment)  ON (n.geography);
CREATE INDEX experiment_domain     IF NOT EXISTS FOR (n:Experiment)  ON (n.domain);
CREATE INDEX property_value        IF NOT EXISTS FOR (n:Property)    ON (n.value);
CREATE INDEX condition_value       IF NOT EXISTS FOR (n:Condition)   ON (n.value);

// --- Верификация/версионирование на рёбрах-фактах (см. graph/ingest.py) ---
// is_current=true — актуальная версия факта; при конфликте новых данных со старыми
// создаётся новая версия ребра, старая помечается is_current=false (не теряется).
CREATE INDEX rel_current_reports    IF NOT EXISTS FOR ()-[r:REPORTS]-()      ON (r.is_current);
CREATE INDEX rel_current_measures   IF NOT EXISTS FOR ()-[r:MEASURES]-()     ON (r.is_current);
CREATE INDEX rel_current_results_in IF NOT EXISTS FOR ()-[r:RESULTS_IN]-()   ON (r.is_current);
CREATE INDEX rel_current_exhibits   IF NOT EXISTS FOR ()-[r:EXHIBITS]-()     ON (r.is_current);
