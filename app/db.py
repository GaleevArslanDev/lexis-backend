from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.pool import NullPool
import os

DATABASE_URL = os.environ["DATABASE_URL"]

# Для Render с ограниченной памятью используем более легкие настройки
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,  # Не используем пул соединений для экономии памяти
    echo=False,  # Отключаем логирование SQL
    pool_pre_ping=False,  # Отключаем проверку соединений
    connect_args={
        "sslmode": "require",
        "connect_timeout": 10,
        "application_name": "lexis-backend",
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 5,
        "keepalives_count": 5
    }
)

def get_session():
    """Генератор сессий с обработкой исключений"""
    try:
        with Session(engine) as session:
            yield session
    except Exception as e:
        print(f"Database session error: {e}")
        # Повторная попытка с новой сессией
        with Session(engine) as session:
            yield session