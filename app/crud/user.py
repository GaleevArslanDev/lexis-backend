from sqlmodel import select
from ..models import User


def create_user(session, name:str, surname:str, email: str, hashed_password: str, role: str = "teacher"):
    user = User(email=email, surname=surname, name=name, hashed_password=hashed_password, role=role)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def get_user_by_email(session, email: str):
    statement = select(User).where(User.email == email)
    result = session.exec(statement).first()
    return result


def get_user_by_id(session, user_id: int):
    return session.get(User, user_id)