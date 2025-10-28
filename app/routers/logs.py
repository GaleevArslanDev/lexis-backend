from fastapi import APIRouter, Depends, Query
from sqlmodel import Session
from datetime import datetime
from ..db import get_session
from ..dependencies import get_current_user, require_role
from ..crud.logs import get_logs_for_user_with_count, get_all_logs
from fastapi import Query

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("/me", response_model=dict)
def my_logs(
        from_date: datetime | None = Query(None),
        to_date: datetime | None = Query(None),
        action_type: str | None = Query(None),
        target_type: str | None = Query(None),
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=100),
        session: Session = Depends(get_session),
        current_user=Depends(get_current_user)
):
    logs, total = get_logs_for_user_with_count(
        session=session,
        user_id=current_user.id,
        from_date=from_date,
        to_date=to_date,
        action_type=action_type,
        target_type=target_type,
        offset=offset,
        limit=limit
    )

    return {
        "total": total,
        "logs": [
            {
                "id": log.id,
                "action_type": log.action_type,
                "target_type": log.target_type,
                "target_id": log.target_id,
                "timestamp": log.timestamp,
                "details": log.details
            } for log in logs
        ]
    }


@router.get("/", response_model=dict)
def all_logs(
    user_id: int | None = Query(None),
    from_date: datetime | None = Query(None),
    to_date: datetime | None = Query(None),
    action_type: str | None = Query(None),
    target_type: str | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    session: Session = Depends(get_session),
    current_user = Depends(require_role("teacher"))
):
    logs, total = get_all_logs(
        session=session,
        user_id=user_id,
        from_date=from_date,
        to_date=to_date,
        action_type=action_type,
        target_type=target_type,
        offset=offset,
        limit=limit
    )
    return {
        "total": total,
        "logs": [
            {
                "id": log.id,
                "user_id": log.user_id,
                "action_type": log.action_type,
                "target_type": log.target_type,
                "target_id": log.target_id,
                "timestamp": log.timestamp,
                "details": log.details
            } for log in logs
        ]
    }
