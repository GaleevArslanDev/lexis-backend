from sqlmodel import select
from ..models import School
from sqlmodel import Session
from fastapi import HTTPException
from typing import Optional


def create_school(session: Session, name: str, creator_id: int, address: Optional[str] = None):
    # проверка уникальности
    existing = session.exec(select(School).where(School.name == name)).first()
    if existing:
        raise HTTPException(status_code=400, detail="School with this name already exists")

    school = School(name=name, address=address, creator_id=creator_id)
    session.add(school)
    session.commit()
    session.refresh(school)
    return school


def get_school_by_id(session: Session, school_id: int):
    return session.get(School, school_id)


def list_schools(session: Session):
    return session.exec(select(School)).all()


def update_school(session: Session, school_id: int, name: str = None, address: str = None):
    school = get_school_by_id(session, school_id)
    if not school:
        return None
    if name is not None:
        school.name = name
    if address is not None:
        school.address = address
    session.add(school)
    session.commit()
    session.refresh(school)
    return school


def delete_school(session: Session, school_id: int):
    school = get_school_by_id(session, school_id)
    if not school:
        return False
    session.delete(school)
    session.commit()
    return True
