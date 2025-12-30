from typing import Optional

from sqlmodel import Session, select
from ..models import MistakeStat


# --- Create ---
def create_mistake_stat(session: Session, class_id: int, question_id: int, mistake_type: str, count: int = 1) -> MistakeStat:
    stat = MistakeStat(class_id=class_id, question_id=question_id, mistake_type=mistake_type, count=count)
    session.add(stat)
    session.commit()
    session.refresh(stat)
    return stat


# --- Read ---
def get_mistake_stat(session: Session, class_id: int, question_id: int, mistake_type: str) -> Optional[MistakeStat]:
    return session.exec(
        select(MistakeStat).where(
            MistakeStat.class_id == class_id,
            MistakeStat.question_id == question_id,
            MistakeStat.mistake_type == mistake_type
        )
    ).first()


# --- Update ---
def increment_mistake_count(session: Session, class_id: int, question_id: int, mistake_type: str, increment: int = 1) -> MistakeStat:
    stat = get_mistake_stat(session, class_id, question_id, mistake_type)
    if stat:
        stat.count += increment
        session.add(stat)
        session.commit()
        session.refresh(stat)
        return stat
    else:
        return create_mistake_stat(session, class_id, question_id, mistake_type, count=increment)


# --- Delete ---
def delete_mistake_stat(session: Session, stat_id: int):
    stat = session.get(MistakeStat, stat_id)
    if stat:
        session.delete(stat)
        session.commit()
        return True
    return False
