from datetime import datetime
from typing import Optional

from ..models import ActionLog
from sqlmodel import Session, select, func


def log_action(session: Session, user_id: int, action_type: str, target_type: str, target_id: int, details: Optional[str] = None):
    log = ActionLog(
        user_id=user_id,
        action_type=action_type,
        target_type=target_type,
        target_id=target_id,
        details=details
    )
    session.add(log)
    session.commit()


def get_logs_for_user(
        session: Session,
        user_id: int,
        from_date: datetime = None,
        to_date: datetime = None,
        action_type: str = None,
        target_type: str = None,
        offset: int = 0,
        limit: int = 50
):
    stmt = select(ActionLog).where(ActionLog.user_id == user_id)

    if from_date:
        stmt = stmt.where(ActionLog.timestamp >= from_date)
    if to_date:
        stmt = stmt.where(ActionLog.timestamp <= to_date)
    if action_type:
        stmt = stmt.where(ActionLog.action_type == action_type)
    if target_type:
        stmt = stmt.where(ActionLog.target_type == target_type)

    stmt = stmt.order_by(ActionLog.timestamp.desc()).offset(offset).limit(limit)
    return session.exec(stmt).all()


def get_all_logs(
        session: Session,
        from_date: datetime = None,
        to_date: datetime = None,
        action_type: str = None,
        target_type: str = None,
        user_id: int = None,
        offset: int = 0,
        limit: int = 50
):
    stmt = select(ActionLog)

    if user_id:
        stmt = stmt.where(ActionLog.user_id == user_id)
    if from_date:
        stmt = stmt.where(ActionLog.timestamp >= from_date)
    if to_date:
        stmt = stmt.where(ActionLog.timestamp <= to_date)
    if action_type:
        stmt = stmt.where(ActionLog.action_type == action_type)
    if target_type:
        stmt = stmt.where(ActionLog.target_type == target_type)

    total_count = session.exec(select(func.count()).select_from(stmt.subquery())).one()
    stmt = stmt.order_by(ActionLog.timestamp.desc()).offset(offset).limit(limit)
    return session.exec(stmt).all(), total_count


def get_logs_for_user_with_count(
        session: Session,
        user_id: int,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        action_type: Optional[str] = None,
        target_type: Optional[str] = None,
        offset: int = 0,
        limit: int = 50
):
    # базовый запрос
    stmt = select(ActionLog).where(ActionLog.user_id == user_id)
    if from_date:
        stmt = stmt.where(ActionLog.timestamp >= from_date)
    if to_date:
        stmt = stmt.where(ActionLog.timestamp <= to_date)
    if action_type:
        stmt = stmt.where(ActionLog.action_type == action_type)
    if target_type:
        stmt = stmt.where(ActionLog.target_type == target_type)

    # считаем total_count через func.count()
    total_count = session.exec(select(func.count()).select_from(stmt.subquery())).one()

    # выбираем сами записи с пагинацией
    stmt = stmt.order_by(ActionLog.timestamp.desc()).offset(offset).limit(limit)
    logs = session.exec(stmt).all()

    return logs, total_count

