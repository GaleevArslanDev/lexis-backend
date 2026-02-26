from typing import Optional, Dict, Any
from ..models import Assignment
from sqlmodel import Session


def create_assignment(session: Session, assignment_data: Dict[str, Any]) -> Assignment:
    """
    Создать новое задание с поддержкой всех полей AssessIt
    """
    # Создаем объект Assignment со всеми переданными полями
    assignment = Assignment(**assignment_data)

    session.add(assignment)
    session.commit()
    session.refresh(assignment)

    return assignment


def get_assignment(session: Session, assignment_id: int) -> Optional[Assignment]:
    """Получить задание по ID"""
    return session.get(Assignment, assignment_id)


def update_assignment(
        session: Session,
        assignment_id: int,
        update_data: Dict[str, Any]
) -> Optional[Assignment]:
    """Обновить задание"""
    assignment = session.get(Assignment, assignment_id)
    if not assignment:
        return None

    for key, value in update_data.items():
        if hasattr(assignment, key):
            setattr(assignment, key, value)

    session.add(assignment)
    session.commit()
    session.refresh(assignment)

    return assignment


def delete_assignment(session: Session, assignment_id: int) -> bool:
    """Удалить задание"""
    assignment = session.get(Assignment, assignment_id)
    if not assignment:
        return False

    session.delete(assignment)
    session.commit()

    return True


def list_assignments_by_class(
        session: Session,
        class_id: int,
        skip: int = 0,
        limit: int = 100
) -> list[Assignment]:
    """Получить список заданий для класса"""
    from sqlmodel import select

    statement = (
        select(Assignment)
        .where(Assignment.class_id == class_id)
        .offset(skip)
        .limit(limit)
    )

    return session.exec(statement).all()