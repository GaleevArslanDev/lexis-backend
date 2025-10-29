from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from ..db import get_session
from ..schemas import AssignmentCreate
from ..crud.assignment import create_assignment as crud_create_assignment
from ..crud.logs import log_action
from ..dependencies_util import require_role, get_current_user
from ..models import Assignment, User, Class, ClassStudentLink

router = APIRouter(prefix="/assignments", tags=["assignments"])


# Создание задания (только для учителей)
@router.post("/", response_model=dict)
def create_assignment(
    payload: AssignmentCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(require_role("teacher"))
):
    # Проверяем, принадлежит ли класс текущему учителю
    klass = session.exec(select(Class).where(Class.id == payload.class_id)).first()
    if not klass or klass.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="You do not own this class")

    assignment = crud_create_assignment(session, payload.dict())
    log_action(session, user_id=current_user.id, action_type="create", target_type="assignment",
               target_id=assignment.id, details=assignment.title)
    return {"id": assignment.id, "title": assignment.title}


@router.get("/", response_model=list[dict])
def list_assignments(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    if current_user.role == "teacher":
        statement = (
            select(Assignment)
            .join(Class)
            .where(Class.teacher_id == current_user.id)
        )
    else:
        # выбираем только задания из классов, где ученик состоит
        statement = (
            select(Assignment)
            .join(Class)
            .join(ClassStudentLink)
            .where(ClassStudentLink.student_id == current_user.id)
        )

    assignments = session.exec(statement).all()
    return [{"id": a.id, "title": a.title, "class_id": a.class_id} for a in assignments]


@router.delete("/{assignment_id}")
def delete_assignment(assignment_id: int, session: Session = Depends(get_session), current_user=Depends(get_current_user)):
    assignment = session.get(Assignment, assignment_id)
    if not assignment:
        raise HTTPException(404, "Assignment not found")
    session.delete(assignment)
    session.commit()

    log_action(session, user_id=current_user.id, action_type="delete", target_type="assignment", target_id=assignment_id, details=assignment.title)
    return {"message": "Assignment deleted"}
