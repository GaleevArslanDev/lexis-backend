import asyncio
from http.client import HTTPException
from fastapi import FastAPI
from .exceptions import ErrorCode, AppException, logger
from .routers import auth, classes, assignments, files, ai, schools, reports, logs, users, assessit, assessit_ws
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import gc
import os

from .routers.assessit_ws import start_heartbeat


def add_exception_handlers(app):
    @app.exception_handler(AppException)
    async def app_exception_handler(request, exc: AppException):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": ErrorCode.INTERNAL_ERROR,
                    "message": exc.detail,
                    "details": {}
                }
            }
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": ErrorCode.INTERNAL_ERROR,
                    "message": "Internal server error",
                    "details": {"error": str(exc)}
                }
            }
        )

app = FastAPI(
    title="AssessIt - backend",
    description="Образовательная платформа с системой автоматической проверки работ AssessIt",
    version="2.0.0"
)

add_exception_handlers(app)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Limit permissions on production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
app.include_router(assessit.router)
app.include_router(assessit_ws.router)


@app.on_event("startup")
def on_startup():
    """Оптимизация при запуске"""
    # Устанавливаем переменные окружения для оптимизации памяти
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    asyncio.create_task(start_heartbeat())

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
