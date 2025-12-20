from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.pool import NullPool
import os

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,
    pool_pre_ping=True,
)

def get_session():
    with Session(engine) as session:
        yield session
