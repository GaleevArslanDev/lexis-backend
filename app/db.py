from sqlmodel import SQLModel, create_engine, Session
import os

DB_USER = os.getenv("POSTGRES_USER", "lexis_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "lexis_pass")
DB_NAME = os.getenv("POSTGRES_DB", "lexis_db")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL, echo=False, future=True)


def init_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
