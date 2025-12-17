# Инициализация пакета app
__version__ = "2.0.0"
__author__ = "Lexis Team"

# Импортируем основные модули для удобства
from .db import init_db, get_session, engine
from .models import *
from .schemas import *

# Экспортируем главное приложение
from .main import app

__all__ = [
    "app",
    "init_db",
    "get_session",
    "engine",
    "SQLModel",
    "User",
    "Class",
    "Assignment",
    "AssessmentImage",
    "RecognizedSolution",
    "AssessmentResult"
]