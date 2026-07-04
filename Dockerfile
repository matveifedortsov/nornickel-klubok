# Единый образ для API и UI (различаются командой запуска в docker-compose).
# GPU НЕ требуется: эмбеддер — локальный ONNX (fastembed), генерация — по API (Yandex).
FROM python:3.12-slim

WORKDIR /app

# curl — для healthcheck API в compose; остальное для pymupdf/reportlab есть в slim
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Запекаем локальную ONNX-модель эмбеддера в образ (иначе ~120МБ качаются при
# первом запросе / нужен интернет в контейнере). FASTEMBED_CACHE читается конфигом.
ENV FASTEMBED_CACHE=/app/models
RUN python -c "from fastembed import TextEmbedding; \
    TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='/app/models')"

COPY . .

# API — 8000, UI (Streamlit) — 8501; конкретный сервис выбирается командой в compose
EXPOSE 8000 8501

# по умолчанию поднимаем API; UI переопределяет command в docker-compose.
# При старте API сам загрузит seed/ в пустой граф (klubok/pipeline.py::seed_if_empty).
CMD ["uvicorn", "klubok.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
