from sqlmodel import SQLModel, create_engine, Session, text
import os
import logging
from fastapi import HTTPException
from urllib.parse import urlparse
import time

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ["DATABASE_URL"]

# Парсим URL для диагностики
parsed = urlparse(DATABASE_URL)
logger.info(f"Connecting to Supabase: host={parsed.hostname}, database={parsed.path[1:]}")

# Оптимизированные параметры для Supabase
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_size=5,  # Уменьшаем размер пула
    max_overflow=2,  # Минимум дополнительных соединений
    pool_pre_ping=True,  # Проверять соединение перед использованием
    pool_recycle=60,  # Пересоздавать соединения каждые 60 секунд
    pool_timeout=30,  # Таймаут ожидания соединения
    connect_args={
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
        "sslmode": "require"
    }
)


def get_session():
    """Генератор сессий с повторными попытками"""
    session = None
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            session = Session(engine)
            # ИСПРАВЛЕНИЕ: используем text() для текстового SQL
            session.execute(text("SELECT 1"))
            yield session
            break
        except Exception as e:
            logger.error(f"Database session error (attempt {attempt + 1}/{max_retries}): {e}")
            if session:
                session.close()
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"Database connection failed after {max_retries} attempts: {str(e)}"
                )
        finally:
            if session:
                session.close()