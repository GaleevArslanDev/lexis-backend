# app/crud/queue.py
from sqlmodel import Session, select, func, and_, or_
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import json
import uuid

from ..models import ProcessingQueue, AssessmentImage


def add_to_queue(
        session: Session,
        image_id: int,
        priority: int = 0,
        max_retries: int = 3
) -> ProcessingQueue:
    """Добавить задачу в очередь"""

    # Проверяем, нет ли уже такой задачи в очереди
    existing = session.exec(
        select(ProcessingQueue).where(
            ProcessingQueue.image_id == image_id,
            ProcessingQueue.status.in_(["pending", "processing"])
        )
    ).first()

    if existing:
        return existing

    queue_item = ProcessingQueue(
        image_id=image_id,
        priority=priority,
        max_retries=max_retries,
        status="pending"
    )

    session.add(queue_item)
    session.commit()
    session.refresh(queue_item)

    return queue_item


def get_next_pending(
        session: Session,
        worker_id: str,
        batch_size: int = 5,
        timeout_minutes: int = 10
) -> List[ProcessingQueue]:
    """
    Получить следующие задачи для обработки
    - Блокирует их для текущего воркера
    - Учитывает зависшие задачи (processing > timeout)
    """

    # Сначала ищем зависшие задачи
    timeout_threshold = datetime.utcnow() - timedelta(minutes=timeout_minutes)

    stuck_items = session.exec(
        select(ProcessingQueue).where(
            ProcessingQueue.status == "processing",
            ProcessingQueue.started_at < timeout_threshold,
            ProcessingQueue.retry_count < ProcessingQueue.max_retries
        )
    ).all()

    for item in stuck_items:
        # Возвращаем в очередь для повторной обработки
        item.status = "pending"
        item.worker_id = None
        item.started_at = None
        item.retry_count += 1
        session.add(item)

    session.commit()

    # Получаем новые задачи
    items = session.exec(
        select(ProcessingQueue)
        .where(
            ProcessingQueue.status == "pending",
            ProcessingQueue.retry_count < ProcessingQueue.max_retries
        )
        .order_by(ProcessingQueue.priority.desc())
        .order_by(ProcessingQueue.created_at.asc())
        .limit(batch_size)
        .with_for_update(skip_locked=True)
    ).all()

    # Блокируем за собой
    for item in items:
        item.status = "processing"
        item.started_at = datetime.utcnow()
        item.worker_id = worker_id
        session.add(item)

    session.commit()

    # Обновляем, чтобы получить актуальные данные
    for item in items:
        session.refresh(item)

    return items


def mark_completed(
        session: Session,
        queue_id: int,
        result_data: Dict[str, Any]
) -> Optional[ProcessingQueue]:
    """Отметить задачу как выполненную"""

    item = session.get(ProcessingQueue, queue_id)
    if not item:
        return None

    item.status = "completed"
    item.completed_at = datetime.utcnow()
    item.result_data = json.dumps(result_data, ensure_ascii=False)

    session.add(item)
    session.commit()
    session.refresh(item)

    # Обновляем статус изображения
    image = session.get(AssessmentImage, item.image_id)
    if image:
        image.status = "processed"
        image.processing_completed = datetime.utcnow()
        session.add(image)
        session.commit()

    return item


def mark_failed(
        session: Session,
        queue_id: int,
        error_message: str,
        should_retry: bool = True
) -> Optional[ProcessingQueue]:
    """Отметить задачу как проваленную"""

    item = session.get(ProcessingQueue, queue_id)
    if not item:
        return None

    if should_retry and item.retry_count < item.max_retries:
        # Возвращаем в очередь для повторной попытки
        item.status = "pending"
        item.worker_id = None
        item.started_at = None
        item.error_message = error_message
        item.retry_count += 1
    else:
        # Окончательная неудача
        item.status = "failed"
        item.completed_at = datetime.utcnow()
        item.error_message = error_message

        # Обновляем статус изображения
        image = session.get(AssessmentImage, item.image_id)
        if image:
            image.status = "error"
            image.error_message = error_message
            session.add(image)

    session.add(item)
    session.commit()
    session.refresh(item)

    return item


def get_queue_stats(session: Session) -> Dict[str, Any]:
    """Получить статистику очереди"""

    total = session.exec(
        select(func.count()).select_from(ProcessingQueue)
    ).one()

    pending = session.exec(
        select(func.count()).where(ProcessingQueue.status == "pending")
    ).one()

    processing = session.exec(
        select(func.count()).where(ProcessingQueue.status == "processing")
    ).one()

    completed = session.exec(
        select(func.count()).where(ProcessingQueue.status == "completed")
    ).one()

    failed = session.exec(
        select(func.count()).where(ProcessingQueue.status == "failed")
    ).one()

    # Среднее время обработки
    avg_time = session.exec(
        select(func.avg(
            func.extract('epoch', ProcessingQueue.completed_at - ProcessingQueue.started_at)
        )).where(ProcessingQueue.status == "completed")
    ).one()

    return {
        "total": total,
        "pending": pending,
        "processing": processing,
        "completed": completed,
        "failed": failed,
        "avg_processing_seconds": round(avg_time or 0, 2)
    }


def get_class_queue_items(
        session: Session,
        class_id: int,
        limit: int = 50
) -> List[Dict[str, Any]]:
    """Получить элементы очереди для класса"""

    items = session.exec(
        select(ProcessingQueue)
        .join(AssessmentImage)
        .where(AssessmentImage.class_id == class_id)
        .order_by(ProcessingQueue.created_at.desc())
        .limit(limit)
    ).all()

    result = []
    for item in items:
        image = session.get(AssessmentImage, item.image_id)
        result.append({
            "queue_id": item.id,
            "work_id": item.image_id,
            "status": item.status,
            "created_at": item.created_at,
            "started_at": item.started_at,
            "completed_at": item.completed_at,
            "retry_count": item.retry_count,
            "error": item.error_message,
            "student_name": "Unknown"  # Можно добавить JOIN
        })

    return result