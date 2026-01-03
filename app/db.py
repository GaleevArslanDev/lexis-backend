from sqlmodel import SQLModel, create_engine, Session
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ["DATABASE_URL"]

# УПРОЩАЕМ НАСТРОЙКИ ДЛЯ ЭКОНОМИИ ПАМЯТИ
engine = create_engine(
    DATABASE_URL,
    # Используем NullPool для минимизации использования памяти
    poolclass=None,  # SQLAlchemy по умолчанию использует QueuePool, но мы можем отключить пул
    echo=False,
    # Упрощаем параметры соединения
    connect_args={
        "connect_timeout": 5,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 3,
        "application_name": "lexis-backend"
    }
)


# Отключаем SSL проверку для экономии памяти (не для продакшена!)
# Для тестирования можно временно отключить SSL
def get_session():
    """Генератор сессий с обработкой исключений"""
    try:
        # Создаем соединение без SSL принудительно
        connection = psycopg2.connect(
            DATABASE_URL,
            connect_timeout=5,
            sslmode='disable'  # Отключаем SSL для тестирования
        )

        session = Session(bind=connection)
        try:
            yield session
        finally:
            session.close()
            connection.close()

    except Exception as e:
        logger.error(f"Database session error: {e}")
        # Попробуем снова с еще более простыми настройками
        try:
            connection = psycopg2.connect(
                DATABASE_URL,
                connect_timeout=3,
                sslmode='disable'
            )
            session = Session(bind=connection)
            yield session
            session.close()
            connection.close()
        except Exception as e2:
            logger.error(f"Retry failed: {e2}")
            raise HTTPException(status_code=500, detail="Database connection failed")