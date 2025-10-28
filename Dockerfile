# Базовый образ с Python
FROM python:3.11-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Скопировать файл зависимостей и установить их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скопировать весь код приложения
COPY . .

# Открыть порт 8000 (на котором работает FastAPI)
EXPOSE 8000

# Команда для запуска приложения
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
