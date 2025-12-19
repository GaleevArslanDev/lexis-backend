from fastapi import FastAPI
from .routers import auth, classes, assignments, files, ai, schools, reports, logs, users, assessit
from fastapi.middleware.cors import CORSMiddleware

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
    # Создаем директории для загрузок
    import os
    upload_dir = os.getenv("UPLOAD_DIR", "/app/uploads")
    processed_dir = os.getenv("PROCESSED_DIR", "/app/processed")

    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print(f"Upload directory: {upload_dir}")
    print(f"Processed directory: {processed_dir}")


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
        "service": "lexis-backend"
    }


from datetime import datetime  # Добавляем импорт
