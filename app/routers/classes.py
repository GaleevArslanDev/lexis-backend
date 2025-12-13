from fastapi import APIRouter, Depends, HTTPException
from ..db import get_session
from ..crud.classes import *
from ..crud.logs import log_action
from ..models import User, ClassStudentLink, Class, Assignment
from ..dependencies_util import require_role, get_current_user
from ..schemas import ClassCreate, ClassUpdate

router = APIRouter(prefix="/classes", tags=["classes"])


# Создать класс
@router.post("/", response_model=dict)
def create_class_endpoint(
    payload: ClassCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(require_role("teacher"))  # только учителя
):
    klass = create_class(session, name=payload.name, teacher_id=current_user.id, school_id=payload.school_id)
    log_action(session, user_id=current_user.id, action_type="create", target_type="class", target_id=klass.id,
               details=klass.name)
    return {"id": klass.id, "name": klass.name, "school_id": klass.school_id}


# Список классов учителя
@router.get("/", response_model=list[dict])
def list_classes_endpoint(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    classes = list_classes(session, teacher_id=current_user.id if current_user.role == "teacher" else None)
    return [{"id": c.id, "name": c.name, "school_id": c.school_id} for c in classes]


# Получить конкретный класс
@router.get("/{class_id}", response_model=dict)
def get_class_endpoint(class_id: int, session: Session = Depends(get_session)):
    klass = get_class_by_id(session, class_id)
    if not klass:
        raise HTTPException(status_code=404, detail="Class not found")
    return {"id": klass.id, "name": klass.name, "school_id": klass.school_id}


# Обновить класс
@router.put("/{class_id}", response_model=dict)
def update_class_endpoint(
        class_id: int,
        payload: ClassUpdate,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))  # проверяем роль
):
    klass = get_class_by_id(session, class_id)
    if not klass:
        raise HTTPException(status_code=404, detail="Class not found")

    # Проверяем, что текущий пользователь — учитель класса
    if klass.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You cannot edit this class")

    klass = update_class(session, class_id, name=payload.name, school_id=payload.school_id)
    log_action(
        session,
        user_id=current_user.id,
        action_type="update",
        target_type="class",
        target_id=class_id,
        details=f"Updated class: {payload}"
    )
    return {"id": klass.id, "name": klass.name, "school_id": klass.school_id}


# Удалить класс
@router.delete("/{class_id}", response_model=dict)
def delete_class_endpoint(
        class_id: int,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))  # проверяем роль
):
    klass = get_class_by_id(session, class_id)
    if not klass:
        raise HTTPException(status_code=404, detail="Class not found")

    if klass.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You cannot delete this class")

    success = delete_class(session, class_id)
    log_action(session, user_id=current_user.id, action_type="delete", target_type="class", target_id=class_id,
               details=klass.name)
    return {"detail": "Class deleted"}


# Добавляем ученика в класс
@router.post("/{class_id}/add-student/{student_id}", response_model=dict)
def add_student_to_class(
    class_id: int,
    student_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(require_role("teacher"))  # только учитель
):
    klass = session.get(Class, class_id)
    if not klass:
        raise HTTPException(status_code=404, detail="Class not found")

    # проверяем, что текущий пользователь — учитель класса
    if klass.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You cannot modify this class")

    # Проверяем, что студент ещё не добавлен
    existing_link = session.exec(
        select(ClassStudentLink)
        .where(ClassStudentLink.class_id == class_id)
        .where(ClassStudentLink.student_id == student_id)
    ).first()

    if existing_link:
        raise HTTPException(status_code=400, detail="Student already in class")

    link = ClassStudentLink(class_id=class_id, student_id=student_id)
    session.add(link)
    session.commit()
    log_action(session, user_id=current_user.id, action_type="add_student", target_type="class", target_id=class_id,
               details=f"Student {student_id} added")
    return {"detail": "Student added to class"}


@router.delete("/{class_id}/remove-student/{student_id}", response_model=dict)
def remove_student_from_class(
        class_id: int,
        student_id: int,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))  # только учитель
):
    klass = session.get(Class, class_id)
    if not klass:
        raise HTTPException(status_code=404, detail="Class not found")

    # Проверка владельца класса
    if klass.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You cannot modify this class")

    link = session.exec(
        select(ClassStudentLink)
        .where(ClassStudentLink.class_id == class_id)
        .where(ClassStudentLink.student_id == student_id)
    ).first()

    if not link:
        raise HTTPException(status_code=404, detail="Student not found in this class")

    session.delete(link)
    session.commit()
    log_action(session, user_id=current_user.id, action_type="remove_student", target_type="class", target_id=class_id,
               details=f"Student {student_id} removed")
    return {"detail": "Student removed from class"}


@router.get("/by-school/{school_id}", response_model=list[dict])
def list_classes_by_school(school_id: int, session: Session = Depends(get_session)):
    classes = get_classes_by_school(session, school_id)
    return [{"id": c.id, "name": c.name, "teacher_id": c.teacher_id} for c in classes]


@router.get("/{class_id}/students", response_model=list[dict])
def list_students_in_class(
    class_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(require_role("teacher"))
):
    klass = session.get(Class, class_id)
    if not klass:
        raise HTTPException(status_code=404, detail="Class not found")
    if klass.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You cannot view students of this class")

    links = session.exec(
        select(ClassStudentLink).where(ClassStudentLink.class_id == class_id)
    ).all()
    student_ids = [link.student_id for link in links]
    if not student_ids:
        return []

    students = session.exec(select(User).where(User.id.in_(student_ids))).all()
    return [{"id": s.id, "email": s.email, "role": s.role} for s in students]


@router.get("/{class_id}/assignments", response_model=list[dict])
def get_class_assignments(
    class_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    klass = session.get(Class, class_id)
    if not klass:
        raise HTTPException(status_code=404, detail="Class not found")

    # Учитель может видеть только свои классы, а ученик — только те, где он состоит
    if current_user.role == "teacher" and klass.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You cannot view assignments of this class")
    elif current_user.role == "student":
        link = session.exec(
            select(ClassStudentLink).where(
                ClassStudentLink.class_id == class_id,
                ClassStudentLink.student_id == current_user.id
            )
        ).first()
        if not link:
            raise HTTPException(status_code=403, detail="You are not in this class")

    assignments = session.exec(
        select(Assignment).where(Assignment.class_id == class_id)
    ).all()
    return [{"id": a.id, "title": a.title, "class_id": a.class_id} for a in assignments]
