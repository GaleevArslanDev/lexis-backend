from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from ..db import get_session
from ..crud.school import *
from ..crud.logs import log_action
from typing import Optional
from ..models import School, User
from ..dependencies import require_role
from ..schemas import SchoolUpdate, SchoolCreate

router = APIRouter(prefix="/schools", tags=["schools"])


@router.post("/", response_model=dict)
def create_school_endpoint(
    payload: SchoolCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(require_role("teacher"))
):
    school = create_school(session, name=payload.name, address=payload.address, creator_id=current_user.id)
    log_action(session, user_id=current_user.id, action_type="create", target_type="school", target_id=school.id, details=school.name)
    return {"id": school.id, "name": school.name, "address": school.address}


# Список всех школ
@router.get("/", response_model=list[dict])
def list_schools_endpoint(session: Session = Depends(get_session)):
    schools = list_schools(session)
    return [{"id": s.id, "name": s.name, "address": s.address} for s in schools]


# Получить конкретную школу
@router.get("/{school_id}", response_model=dict)
def get_school_endpoint(school_id: int, session: Session = Depends(get_session)):
    school = get_school_by_id(session, school_id)
    if not school:
        raise HTTPException(status_code=404, detail="School not found")
    return {"id": school.id, "name": school.name, "address": school.address}


# Обновить школу
@router.put("/{school_id}", response_model=dict)
def update_school_endpoint(
        school_id: int,
        payload: SchoolUpdate,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))  # проверяем роль
):
    school = get_school_by_id(session, school_id)
    if not school:
        raise HTTPException(status_code=404, detail="School not found")

    # Проверка владельца (если есть creator_id)
    if getattr(school, "creator_id", None) != current_user.id:
        raise HTTPException(status_code=403, detail="You cannot edit this school")

    school = update_school(session, school_id, name=payload.name, address=payload.address)
    log_action(
        session,
        user_id=current_user.id,
        action_type="update",
        target_type="school",
        target_id=school_id,
        details=f"Updated school: {payload}"
    )
    return {"id": school.id, "name": school.name, "address": school.address}


# Удалить школу
@router.delete("/{school_id}", response_model=dict)
def delete_school_endpoint(
        school_id: int,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))  # проверяем роль
):
    school = get_school_by_id(session, school_id)
    if not school:
        raise HTTPException(status_code=404, detail="School not found")

    # Проверка владельца
    if getattr(school, "creator_id", None) != current_user.id:
        raise HTTPException(status_code=403, detail="You cannot delete this school")

    success = delete_school(session, school_id)
    log_action(session, user_id=current_user.id, action_type="delete", target_type="school", target_id=school_id,
               details=school.name)
    return {"detail": "School deleted"}

