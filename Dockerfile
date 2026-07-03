# Единый образ для API и UI (различаются командой запуска в docker-compose).
FROM python:3.12-slim

WORKDIR /app

# системные зависимости для pymupdf/reportlab обычно уже в slim; ставим только pip-пакеты
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# API — 8000, UI (Streamlit) — 8501; конкретный сервис выбирается командой в compose
EXPOSE 8000 8501

# по умолчанию поднимаем API; UI переопределяет command в docker-compose
CMD ["uvicorn", "klubok.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
