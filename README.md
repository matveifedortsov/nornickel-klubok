# Научный клубок — R&D карта знаний горно-металлургической отрасли (Норникель AI Science Hack)

Решение трека **«Научный клубок»**: система, которая связывает публикации, эксперименты,
технологические решения, материалы, оборудование и экспертов в граф знаний и отвечает на
сложные многопараметрические вопросы вида
*«какие технические решения циркуляции католита при электроэкстракции никеля описаны в
мировой практике, и какая скорость потока считается оптимальной?»* — **с цитатами и
верификацией**, **числовыми/гео-фильтрами**, **визуализацией подграфа** и **поиском пробелов
в данных**.

## Главная идея архитектуры
Все «тяжёлые» компоненты (LLM, эмбеддер) спрятаны за интерфейсами с двумя
реализациями:

| Слой       | Без GPU (разработка)      | С железом (боевой режим)        |
|------------|---------------------------|---------------------------------|
| LLM        | `MockLLM`                 | `MetalGPTClient` (MetalGPT-1)   |
| Эмбеддинги | `MockEmbedder`            | `BGEEmbedder` (BAAI/bge-m3)     |
| Вектор     | `InMemoryVectorStore`     | `QdrantStore`                   |

Переключение — через `.env` (`LLM_BACKEND`, `EMBEDDER_BACKEND`). Поэтому **весь
пайплайн пишется и тестируется заранее**, до выделения вычислительных мощностей.

## Что можно запустить прямо сейчас (без GPU)
```bash
pip install -r requirements.txt        # для оффлайн-демо хватит core-зависимостей
python scripts/demo_offline.py         # end-to-end на Mock'ах, без БД
pytest -q                              # юнит-тесты чистой логики
```

## Оценка качества извлечения
```bash
python scripts/eval_extraction.py                 # held-out gold-набор (честная оценка)
python scripts/eval_extraction.py --dataset fewshot   # few-shot пары, leave-one-out
python scripts/eval_extraction.py --no-few-shot       # zero-shot baseline
```
Считает precision / recall / F1 по сущностям и **рёбрам** (сравнение по
канонизированным ключам, поэтому `Cu-Ni` = `CuNi`). На MockLLM метрики низкие —
это норма; реальная оценка с `LLM_BACKEND=metalgpt`. Gold-набор в
[klubok/eval/gold_set.py](klubok/eval/gold_set.py) **не пересекается** с few-shot
примерами промпта (защита от утечки проверяется тестом).

## Что добавляется, когда подняли инфраструктуру (всё ещё без GPU)
```bash
docker compose up -d                   # Neo4j + Qdrant
uvicorn klubok.api.app:app --reload    # API
streamlit run ui/streamlit_app.py      # демо-интерфейс
```
На этом этапе LLM/эмбеддер ещё Mock — но граф, ретривал, gap-анализ и UI уже живые.

## Ингест реального корпуса
```bash
python scripts/select_corpus.py --root "<Источники информации>" --out scripts/corpus_subset.txt
python scripts/ingest_corpus.py --list scripts/corpus_subset.txt
```
`select_corpus.py` отбирает компактный сабсет (статьи, обзоры, доклады, по 2
последних номера каждого журнала), исключая рыночную аналитику и архивы.
`ingest_corpus.py` — батч-ингест с ретраями и чекпоинтом: одна битая статья не
роняет весь прогон, длинный прогон можно прервать и продолжить.

## Что включается с GPU
1. Поднять MetalGPT-1 (vLLM, OpenAI-совместимый сервер) — см. docstring
   `klubok/extraction/llm_client.py`.
2. В `.env`: `LLM_BACKEND=metalgpt`, `EMBEDDER_BACKEND=bge`.
3. Прогнать `scripts/ingest_corpus.py` на реальном корпусе — качество извлечения
   и ответов резко вырастет.

## Онтология
Типы узлов: `Material, Element, Process, Equipment, Property, Condition, Experiment,
Publication, Expert, Facility, Method, Phase`. Связи (среди прочих): `USES,
USES_MATERIAL, APPLIES, OPERATES_AT_CONDITION, PRODUCES_OUTPUT, MEASURES, RESULTS_IN,
DESCRIBED_IN, VALIDATED_BY, CONTRADICTS, AUTHORED_BY, EXPERT_IN, AFFILIATED_WITH,
EXHIBITS, OBSERVED_BY, CONTAINS_PHASE, CITES, REPORTS`. Полная схема и допустимые
пары — [klubok/ontology.py](klubok/ontology.py).

Каждая связь несёт провенанс и верификацию: `evidence`, `confidence`, `source_type`,
`verification_level`, `actualized_at`, `geography`, `is_domestic`. При конфликте
числовых значений на повторном ингесте старое значение не теряется —
[klubok/graph/ingest.py](klubok/graph/ingest.py) ведёт `value_history` и
подсвечивает `has_conflicting_versions`.

## Структура
```
config.py                     конфиг из .env (+ RBAC api_keys)
klubok/
  ontology.py                 СХЕМА: типы узлов/рёбер + NumericConstraint + Pydantic-модели
  parsing/
    pdf_parser.py              PDF -> Document(chunks) + таблицы
    docx_parser.py             DOCX -> Document(chunks)
    pptx_parser.py             PPTX -> Document(chunks) (слайды + заметки)
    filename_meta.py           автор/лаборатория из имени файла — без LLM
  extraction/
    prompts.py                 промпты MetalGPT (строгий JSON, металлургический домен)
    llm_client.py               LLM-интерфейс + MockLLM + MetalGPTClient
    normalize.py                величины/единицы + parse_constraint (диапазоны/операторы)
    resolver.py                  канонизация + glossary_ru_en.py (синонимы RU/EN)
    extractor.py                 chunk -> триплеты (с фильтром по онтологии)
  graph/
    schema.cypher                constraints/индексы (в т.ч. гео/домен/числовые)
    neo4j_client.py               обёртка драйвера
    ingest.py                     запись триплетов + версионирование + ручная правка графа
    gaps.py                       ПОИСК ПРОБЕЛОВ (Cypher) — дифференциатор
  analytics/
    compare.py                    сравнение технологий «вариант А vs Б»
    recommend.py                  похожие кейсы + эксперты по теме
    dashboard.py                  метрики покрытия/активности/зон риска
  vectorstore/
    embeddings.py                 Embedder + MockEmbedder + BGEEmbedder
    store.py                      QdrantStore + InMemoryVectorStore
  retrieval/graphrag.py           ГИБРИДНЫЙ ПОИСК: вектор -> seed -> обход графа 1-4 хопа
                                   + числовые/гео-фильтры из вопроса (APOC)
  qa/answer.py                    ответ с цитатами + литературный обзор
  export/formats.py               PDF / Markdown / JSON-LD
  pipeline.py                     фасад: ingest / ask
  api/
    app.py                        FastAPI (ask/review/gaps/dashboard/compare/experts/...)
    auth.py                       RBAC по X-API-Key
    audit.py                      аудит-лог запросов
ui/streamlit_app.py               демо-интерфейс (граф, фильтры, дашборд, сравнение, роли)
scripts/
  demo_offline.py                 end-to-end демо без GPU и без БД
  select_corpus.py                отбор файлов реального корпуса
  ingest_corpus.py                батч-ингест с ретраями/чекпоинтом
  eval_extraction.py              precision/recall/F1 по gold-набору
tests/test_pure_logic.py          юнит-тесты
```

## Чек-лист «почему выигрышно»
- [x] Используется **MetalGPT-1** (модель Норникеля).
- [x] **Гибрид** граф + вектор (а не просто RAG поверх PDF) — точно по условию.
- [x] **Поиск пробелов** в данных, включая гео-пробелы (только РФ / только мир) и
      расхождения версий фактов — прямое требование трека.
- [x] Ответы с **провенансом и верификацией** (цитаты, evidence, уровень достоверности,
      дата актуализации) — против галлюцинаций.
- [x] **Числовые диапазоны и гео-фильтр** прямо в вопросе («сульфаты ≤300 мг/л», «в России»).
- [x] **RBAC + аудит** запросов, ручная корректировка графа экспертом.
- [x] **Экспорт** в PDF/Markdown/JSON-LD, дашборд для руководителя, сравнение технологий.
- [x] Фильтр связей по онтологии — барьер против мусора от LLM.
