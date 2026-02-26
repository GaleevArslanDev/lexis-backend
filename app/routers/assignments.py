from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select
from typing import List, Optional
from datetime import datetime

from ..db import get_session
from ..schemas import AssignmentCreate, AssignmentResponse
from ..crud.assignment import create_assignment, get_assignment, update_assignment, delete_assignment
from ..crud.logs import log_action
from ..dependencies_util import require_role, get_current_user
from ..models import Assignment, User, Class, ClassStudentLink

router = APIRouter(prefix="/assignments", tags=["assignments"])


@router.post("/", response_model=AssignmentResponse)
def create_assignment_endpoint(
        payload: AssignmentCreate,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """
    Создать новое задание
    """
    # Проверяем, принадлежит ли класс текущему учителю
    klass = session.exec(
        select(Class).where(Class.id == payload.class_id)
    ).first()

    if not klass:
        raise HTTPException(status_code=404, detail="Class not found")

    if klass.teacher_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="You do not own this class"
        )

    # Создаем задание
    assignment_data = payload.dict(exclude_unset=True)
    assignment = create_assignment(session, assignment_data)

    # Логируем действие
    log_action(
        session,
        user_id=current_user.id,
        action_type="create",
        target_type="assignment",
        target_id=assignment.id,
        details=f"Created assignment: {assignment.title}"
    )

    return assignment


@router.get("/", response_model=List[AssignmentResponse])
def list_assignments(
        class_id: Optional[int] = Query(None, description="Filter by class ID"),
        subject: Optional[str] = Query(None, description="Filter by subject"),
        difficulty: Optional[str] = Query(None, description="Filter by difficulty"),
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=100),
        session: Session = Depends(get_session),
        current_user: User = Depends(get_current_user)
):
    """
    Получить список заданий
    """
    # Базовый запрос
    if current_user.role == "teacher":
        statement = (
            select(Assignment)
            .join(Class)
            .where(Class.teacher_id == current_user.id)
        )
    else:
        # Для ученика - только задания из его классов
        statement = (
            select(Assignment)
            .join(Class)
            .join(ClassStudentLink)
            .where(ClassStudentLink.student_id == current_user.id)
        )

    # Фильтры
    if class_id:
        statement = statement.where(Assignment.class_id == class_id)

    if subject:
        statement = statement.where(Assignment.subject == subject)

    if difficulty:
        statement = statement.where(Assignment.difficulty == difficulty)

    # Пагинация
    statement = statement.offset(skip).limit(limit)

    assignments = session.exec(statement).all()
    return assignments


@router.get("/{assignment_id}", response_model=AssignmentResponse)
def get_assignment_endpoint(
        assignment_id: int,
        session: Session = Depends(get_session),
        current_user: User = Depends(get_current_user)
):
    """
    Получить задание по ID
    """
    assignment = get_assignment(session, assignment_id)

    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    # Проверка доступа
    if current_user.role == "teacher":
        if assignment.class_.teacher_id != current_user.id:
            raise HTTPException(
                status_code=403,
                detail="You don't have access to this assignment"
            )
    else:
        # Для ученика - проверяем, что он в классе
        link = session.exec(
            select(ClassStudentLink).where(
                ClassStudentLink.class_id == assignment.class_id,
                ClassStudentLink.student_id == current_user.id
            )
        ).first()

        if not link:
            raise HTTPException(
                status_code=403,
                detail="You don't have access to this assignment"
            )

    return assignment


@router.put("/{assignment_id}", response_model=AssignmentResponse)
def update_assignment_endpoint(
        assignment_id: int,
        payload: AssignmentCreate,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """
    Обновить задание
    """
    # Получаем существующее задание
    assignment = get_assignment(session, assignment_id)

    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    # Проверяем права
    if assignment.class_.teacher_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="You cannot edit this assignment"
        )

    # Обновляем
    update_data = payload.dict(exclude_unset=True)
    updated_assignment = update_assignment(session, assignment_id, update_data)

    # Логируем
    log_action(
        session,
        user_id=current_user.id,
        action_type="update",
        target_type="assignment",
        target_id=assignment_id,
        details=f"Updated assignment: {updated_assignment.title}"
    )

    return updated_assignment


@router.delete("/{assignment_id}")
def delete_assignment_endpoint(
        assignment_id: int,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """
    Удалить задание
    """
    assignment = get_assignment(session, assignment_id)

    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    # Проверяем права
    if assignment.class_.teacher_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="You cannot delete this assignment"
        )

    # Сохраняем название для лога
    title = assignment.title

    # Удаляем
    success = delete_assignment(session, assignment_id)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete assignment")

    # Логируем
    log_action(
        session,
        user_id=current_user.id,
        action_type="delete",
        target_type="assignment",
        target_id=assignment_id,
        details=f"Deleted assignment: {title}"
    )

    return {"message": "Assignment deleted successfully"}


@router.get("/class/{class_id}", response_model=List[AssignmentResponse])
def get_class_assignments(
        class_id: int,
        session: Session = Depends(get_session),
        current_user: User = Depends(get_current_user)
):
    """
    Получить все задания для конкретного класса
    """
    # Проверяем доступ к классу
    klass = session.get(Class, class_id)

    if not klass:
        raise HTTPException(status_code=404, detail="Class not found")

    if current_user.role == "teacher":
        if klass.teacher_id != current_user.id:
            raise HTTPException(
                status_code=403,
                detail="You cannot view assignments of this class"
            )
    else:
        link = session.exec(
            select(ClassStudentLink).where(
                ClassStudentLink.class_id == class_id,
                ClassStudentLink.student_id == current_user.id
            )
        ).first()

        if not link:
            raise HTTPException(
                status_code=403,
                detail="You are not in this class"
            )

    assignments = list_assignments_by_class(session, class_id)
    return assignments