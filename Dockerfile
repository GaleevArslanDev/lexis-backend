# Dockerfile
FROM python:3.9-slim  # Python 3.9 стабильнее для ML

# Установка системных зависимостей МИНИМАЛЬНЫМ набором
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Создание пользователя для безопасности (опционально, но рекомендуется)
RUN useradd -m -u 1000 appuser
WORKDIR /app
RUN chown appuser:appuser /app
USER appuser

# Копируем requirements первыми для лучшего кэширования
COPY --chown=appuser:appuser requirements.txt .

# Устанавливаем Python зависимости с оптимизацией
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --user --no-cache-dir \
    --no-warn-script-location \
    -r requirements.txt

# Копируем приложение
COPY --chown=appuser:appuser . .

# Оптимизация переменных окружения Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1  # Ограничиваем потоки для OpenMP

# Запуск приложения
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]