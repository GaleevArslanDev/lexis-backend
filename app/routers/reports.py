from fastapi import APIRouter, Depends
from sqlmodel import Session, select
from ..db import get_session
from ..models import MistakeStat, Class
from fastapi import HTTPException

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/mistakes/class/{class_id}", response_model=list[dict])
def get_mistakes_for_class(class_id: int, session: Session = Depends(get_session)):
    """
    Получить статистику ошибок по конкретному классу.
    """
    stats = session.exec(
        select(MistakeStat).where(MistakeStat.class_id == class_id)
    ).all()
    return [
        {"question_id": s.question_id, "mistake_type": s.mistake_type, "count": s.count}
        for s in stats
    ]


@router.get("/mistakes/school/{school_id}", response_model=list[dict])
def get_mistakes_for_school(school_id: int, session: Session = Depends(get_session)):
    """
    Получить агрегированную статистику ошибок по всем классам школы.
    """
    # Получаем все классы школы
    classes = session.exec(
        select(Class).where(Class.school_id == school_id)
    ).all()
    if not classes:
        raise HTTPException(status_code=404, detail="School not found or has no classes")

    class_ids = [c.id for c in classes]

    # Получаем ошибки по всем классам школы
    stats = session.exec(
        select(MistakeStat).where(MistakeStat.class_id.in_(class_ids))
    ).all()

    # Группируем по question_id и типу ошибки
    aggregated = {}
    for s in stats:
        key = (s.question_id, s.mistake_type)
        if key not in aggregated:
            aggregated[key] = 0
        aggregated[key] += s.count

    # Преобразуем в список словарей для ответа
    result = [
        {"question_id": qid, "mistake_type": mtype, "count": count}
        for (qid, mtype), count in aggregated.items()
    ]
    return result
