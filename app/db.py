from sqlmodel import SQLModel, create_engine, Session
import os
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,  # Проверять соединение перед использованием
    pool_recycle=300,  # Пересоздавать соединения каждые 300 секунд
)


def get_session():
    """Генератор сессий с обработкой исключений"""
    try:
        with Session(engine) as session:
            yield session
    except Exception as e:
        logger.error(f"Database session error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")