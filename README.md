# Научный клубок — R&D карта знаний горно-металлургической отрасли (Норникель AI Science Hack)

Решение трека **«Научный клубок»**: система, которая связывает публикации, эксперименты,
технологические решения, материалы, оборудование и экспертов в граф знаний и отвечает на
сложные многопараметрические вопросы вида
*«какие технические решения циркуляции католита при электроэкстракции никеля описаны в
мировой практике, и какая скорость потока считается оптимальной?»* — **с цитатами и
верификацией**, **числовыми/гео-фильтрами**, **визуализацией подграфа** и **поиском пробелов
в данных**.

## Главная идея архитектуры — LLM/эмбеддер-агностичность
Все «тяжёлые» компоненты спрятаны за интерфейсами с несколькими взаимозаменяемыми
реализациями. Один и тот же пайплайн работает на разных бэкендах — переключение
одной строкой в `.env` (`LLM_BACKEND`, `EMBEDDER_BACKEND`):

| Слой | `mock` (разработка) | `yandex` (боевой) | `fastembed`/локальный | on-prem (прод Норникеля) |
|------|------|------|------|------|
| **LLM** | `MockLLM` | **YandexGPT** (Yandex AI Studio) | — | `MetalGPTClient` (MetalGPT-1 через vLLM) |
| **Эмбеддинги** | `MockEmbedder` | Yandex textEmbedding | **`FastEmbedEmbedder`** (ONNX, локально, без квоты) | `BGEEmbedder` (bge-m3) |
| **Вектор** | `InMemoryVectorStore` | `QdrantStore` (server/**local**) | — | `QdrantStore` |

Архитектурная выгода: сегодня — Yandex AI Studio, в проде Норникеля **тот же код**
работает с on-prem MetalGPT-1 (клиент уже написан). Эмбеддинги вынесены на
локальный ONNX (`fastembed`) — векторный слой **не зависит от облачной квоты**.

📐 Схема архитектуры: [docs/architecture.svg](docs/architecture.svg).

### Гибридный конвейер запроса
`вопрос` → распознавание фильтров (числа/гео/годы) → **векторный recall** (fastembed)
→ **seed-узлы** (fulltext Neo4j) → **обход графа 1–4 хопа** (APOC) со структурными
фильтрами → **реранкинг рёбер** по релевантности (bi-encoder, локально) →
**генерация ответа** с цитатами и верификацией. Ретривал полностью локальный
(~0.1–2.6 с, в бюджете ТЗ «3–5 с»); от облака зависит только генерация текста.

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
docker compose up -d                   # Neo4j + Qdrant + API + UI одной командой
```
На этом этапе LLM/эмбеддер ещё Mock — но граф, ретривал, gap-анализ и UI уже живые.

## Боевой запуск на Yandex AI Studio (проверено, без Docker)
Если Docker недоступен — Neo4j запускается портативно, Qdrant работает во
встроенном режиме (без сервера), LLM/эмбеддинги — через Yandex AI Studio.

1. **`.env`** (ключи только здесь, не в git):
   ```
   LLM_BACKEND=yandex
   EMBEDDER_BACKEND=yandex
   QDRANT_MODE=local            # встроенный Qdrant в data/qdrant_local
   YANDEX_API_KEY=...
   YANDEX_FOLDER_ID=...
   YANDEX_RPS=0.15              # под квоту Yandex (~10 запросов/мин)
   ```
2. **Смоук-тест** ключа/квоты/сети: `python scripts/check_yandex.py`
3. **Neo4j**: `docker compose up -d neo4j` ИЛИ портативно (JDK+Neo4j в `runtime/`).
4. **Ингест** (встроенный Qdrant → API и ingest не запускать одновременно):
   `python scripts/ingest_corpus.py --list runtime/pilot.txt`
5. **API + UI** (после ингеста):
   `uvicorn klubok.api.app:app` и `streamlit run ui/streamlit_app.py`

> **Квота Yandex.** Дефолтная квота Text Generation ≈10 запросов/мин — на батч-ингест
> её мало. В коде: `YANDEX_RPS=0.15` (без 429) + кэш извлечения по чанкам
> ([extract_cache.py](klubok/extraction/extract_cache.py), ретраи не жгут квоту) +
> кэш эмбеддингов. Для полного корпуса (176 файлов) **поднимите квоту** в Yandex
> Cloud console; для демо достаточно пилота + десятков файлов.

## Ингест реального корпуса
```bash
python scripts/select_corpus.py --root "<Источники информации>" --out scripts/corpus_subset.txt
python scripts/ingest_corpus.py --list scripts/corpus_subset.txt
```

## Пополнение корпуса из открытых источников
```bash
python scripts/fetch_articles.py                 # ~10 тем экспертных вопросов
python scripts/ingest_corpus.py --list data/openaccess/files.txt
```
`fetch_articles.py` скачивает open-access статьи (CyberLeninka CC-BY, DOAJ —
агрегатор MDPI/Springer OA/Wiley OA) по темам реальных экспертных вопросов:
обессоливание, шахтные воды, электроэкстракция, техногенный гипс, SO₂,
штейн/шлак, Pb-Zn. Вежливый rate-limit, повторный запуск докачивает только новое.
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
