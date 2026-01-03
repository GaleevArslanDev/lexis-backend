FROM python:3.9-slim-bullseye

# Установка системных зависимостей - ОПТИМИЗИРОВАННЫЙ НАБОР
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Создаем пользователя для безопасности
RUN useradd -m -u 1000 appuser
WORKDIR /app
RUN chown appuser:appuser /app
USER appuser

# Копируем зависимости
COPY --chown=appuser:appuser requirements.txt .

# Оптимизация установки pip
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Устанавливаем Python зависимости
RUN pip install --user --no-cache-dir --no-warn-script-location -r requirements.txt

# Добавляем bin пользователя в PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Копируем приложение
COPY --chown=appuser:appuser . .

# Оптимизация для CPU и памяти
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PADDLE_PROCESSOR=cpu \
    TF_CPP_MIN_LOG_LEVEL=3

# Запуск приложения с одним воркером для экономии памяти
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]