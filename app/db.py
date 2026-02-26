from sqlmodel import SQLModel, create_engine, Session, text
import os
import logging
from fastapi import HTTPException
from urllib.parse import urlparse
import traceback

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is not set!")
    raise ValueError("DATABASE_URL environment variable is not set")

# Маскируем пароль для логирования
masked_url = DATABASE_URL.replace(
    DATABASE_URL.split(':')[2].split('@')[0],
    '******'
) if '@' in DATABASE_URL else DATABASE_URL
logger.info(f"Connecting to database with URL: {masked_url}")

# Парсим URL для диагностики
try:
    parsed = urlparse(DATABASE_URL)
    logger.info(f"Database host: {parsed.hostname}, database: {parsed.path[1:] if parsed.path else 'unknown'}")
except Exception as e:
    logger.error(f"Failed to parse DATABASE_URL: {e}")

# Оптимизированные параметры для Supabase
try:
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
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    logger.error(traceback.format_exc())
    raise


def get_session():
    """Генератор сессий"""
    logger.debug("Attempting to create database session")

    try:
        with Session(engine) as session:
            # Проверяем соединение
            logger.debug("Testing database connection with SELECT 1")
            result = session.execute(text("SELECT 1")).first()
            logger.debug(f"Database connection test result: {result}")

            if not result:
                error_msg = "Database returned empty result for SELECT 1"
                logger.error(error_msg)
                raise HTTPException(status_code=503, detail=error_msg)

            logger.debug("Database connection successful")
            yield session

    except HTTPException:
        # Пробрасываем HTTP исключения дальше
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Database connection failed: {error_msg}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        # Проверяем конкретные типы ошибок
        if "password" in error_msg.lower() or "authentication" in error_msg.lower():
            detail = "Database authentication failed. Check username and password."
        elif "timeout" in error_msg.lower():
            detail = "Database connection timeout. Check network and firewall."
        elif "connection refused" in error_msg.lower():
            detail = "Database connection refused. Check if database is accessible."
        elif "ssl" in error_msg.lower():
            detail = "SSL connection error. Check SSL settings."
        else:
            detail = f"Database connection failed: {error_msg}"

        raise HTTPException(status_code=503, detail=detail)