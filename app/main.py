from fastapi import FastAPI
from .db import init_db
from .routers import auth, classes, assignments, files, ai, schools, reports, logs
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Лексис - backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # TODO: Limit permissions on production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router)
app.include_router(classes.router)
app.include_router(assignments.router)
app.include_router(files.router)
app.include_router(ai.router)
app.include_router(schools.router)
app.include_router(reports.router)
app.include_router(logs.router)


@app.on_event("startup")
def on_startup():
    init_db()
