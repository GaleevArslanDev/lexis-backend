from sqlmodel import SQLModel, create_engine, Session, text
import os
import logging
from fastapi import HTTPException
from urllib.parse import urlparse
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ["DATABASE_URL"]

# Парсим URL для диагностики
parsed = urlparse(DATABASE_URL)
logger.info(f"Connecting to Supabase: host={parsed.hostname}, database={parsed.path[1:]}")

# Оптимизированные параметры для Supabase
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=2,
    pool_pre_ping=True,
    pool_recycle=60,
    pool_timeout=30,
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
    """Генератор сессий"""
    with Session(engine) as session:
        try:
            # Проверяем соединение
            session.execute(text("SELECT 1"))
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            session.rollback()
            raise HTTPException(
                status_code=503,
                detail=f"Database connection failed: {str(e)}"
            )