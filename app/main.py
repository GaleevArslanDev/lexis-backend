from fastapi import FastAPI
from .routers import auth, classes, assignments, files, ai, schools, reports, logs, users, assessit
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import gc
import os

app = FastAPI(
    title="Лексис - backend",
    description="Образовательная платформа с системой автоматической проверки работ AssessIt",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Limit permissions on production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
gc.collect()

# Подключаем все роутеры
app.include_router(auth.router)
app.include_router(classes.router)
app.include_router(assignments.router)
app.include_router(files.router)
app.include_router(ai.router)
app.include_router(schools.router)
app.include_router(reports.router)
app.include_router(logs.router)
app.include_router(users.router)
app.include_router(assessit.router)  # Новый роутер для AssessIt


@app.on_event("startup")
def on_startup():
    """Оптимизация при запуске"""
    # Устанавливаем переменные окружения для оптимизации памяти
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    print("AssessIt backend starting with memory optimization")
    print(f"Available memory optimization active")

    # Принудительный сбор мусора
    gc.collect()


@app.get("/")
async def root():
    return {
        "message": "Лексис API с системой AssessIt",
        "version": "2.0.0",
        "endpoints": {
            "auth": "/auth",
            "classes": "/classes",
            "assignments": "/assignments",
            "assessit": "/assessit",
            "ai": "/ai",
            "docs": "/docs"
        }
    }

@app.on_event("shutdown")
def shutdown_event():
    """Очистка при завершении"""
    import sys
    if 'app.processing.ocr_engine' in sys.modules:
        del sys.modules['app.processing.ocr_engine']
    gc.collect()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "lexis-backend"
    }


from datetime import datetime  # Добавляем импорт
