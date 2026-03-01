# app/main.py
import asyncio
from http.client import HTTPException
from fastapi import FastAPI
from .exceptions import ErrorCode, AppException, logger
from .routers import auth, classes, assignments, files, ai, schools, reports, logs, users, assessit, assessit_ws
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import gc
import os
from datetime import datetime

from .routers.assessit_ws import start_heartbeat, manager
from .workers.queue_worker import get_queue_worker


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

# Хранилище для фоновых задач
background_tasks = set()


@app.on_event("startup")
async def on_startup():
    """Оптимизация при запуске"""
    # Устанавливаем переменные окружения для оптимизации памяти
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Запускаем heartbeat для WebSocket
    heartbeat_task = asyncio.create_task(start_heartbeat())
    background_tasks.add(heartbeat_task)
    heartbeat_task.add_done_callback(background_tasks.discard)

    # Запускаем воркер очереди
    worker = get_queue_worker()
    worker_task = asyncio.create_task(worker.run_forever())
    background_tasks.add(worker_task)
    worker_task.add_done_callback(background_tasks.discard)

    print("AssessIt backend starting with memory optimization")
    print(f"Available memory optimization active")
    print(f"Background tasks started: {len(background_tasks)}")

    # Принудительный сбор мусора
    gc.collect()


@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при завершении"""
    logger.info("Shutting down server...")

    # Останавливаем воркер
    worker = get_queue_worker()
    worker.stop()

    # Отменяем все фоновые задачи
    if background_tasks:
        for task in background_tasks:
            task.cancel()

        # Ждем завершения задач (но не используем await в синхронной функции)
        # Вместо этого создаем новую асинхронную задачу для ожидания
        asyncio.create_task(_wait_for_tasks())

    # Очистка других ресурсов
    import sys
    if 'app.processing.ocr_engine' in sys.modules:
        del sys.modules['app.processing.ocr_engine']

    gc.collect()
    logger.info("Shutdown complete")


async def _wait_for_tasks():
    """Вспомогательная функция для ожидания завершения задач"""
    if background_tasks:
        try:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error while waiting for tasks: {e}")


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


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "lexis-backend",
        "background_tasks": len(background_tasks)
    }