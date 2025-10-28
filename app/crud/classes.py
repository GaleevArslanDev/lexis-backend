from sqlmodel import select
from ..models import Class, ClassStudentLink
from sqlmodel import Session
from typing import Optional


def get_class_by_id(session: Session, class_id: int):
    return session.get(Class, class_id)


def list_classes(session: Session, teacher_id: int = None):
    statement = select(Class)
    if teacher_id is not None:
        statement = statement.where(Class.teacher_id == teacher_id)
    return session.exec(statement).all()


def create_class(session: Session, name: str, teacher_id: int, school_id: Optional[int] = None):
    klass = Class(name=name, teacher_id=teacher_id, school_id=school_id)
    session.add(klass)
    session.commit()
    session.refresh(klass)
    return klass


def update_class(session: Session, class_id: int, name: Optional[str] = None, school_id: Optional[int] = None):
    klass = session.get(Class, class_id)
    if not klass:
        return None
    if name is not None:
        klass.name = name
    klass.school_id = school_id
    session.add(klass)
    session.commit()
    session.refresh(klass)
    return klass


def delete_class(session: Session, class_id: int):
    klass = get_class_by_id(session, class_id)
    if not klass:
        return False
    session.delete(klass)
    session.commit()
    return True


def add_student_to_class(session: Session, class_id: int, student_id: int):
    link = ClassStudentLink(class_id=class_id, student_id=student_id)
    session.add(link)
    session.commit()
    return link


def remove_student_from_class(session: Session, class_id: int, student_id: int):
    link = session.get(ClassStudentLink, (class_id, student_id))
    if not link:
        return False
    session.delete(link)
    session.commit()
    return True


def get_classes_by_school(session: Session, school_id: int):
    return session.exec(select(Class).where(Class.school_id == school_id)).all()
