from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, status
from typing import Dict, Set, Optional, List
import asyncio
import json
import logging
from datetime import datetime, timedelta
from sqlmodel import Session, select
from jose import jwt
import uuid
import time

from ..db import get_session
from ..models import User, Class, AssessmentImage, RecognizedSolution, Assignment
from ..utils.security import SECRET_KEY, ALGORITHM
from ..exceptions import AppException, ErrorCode
from ..ocr_client import get_ocr_client_v1
from ..crud.assessment import (
    update_image_status,
    create_recognized_solution_v1,
    get_recognized_solution
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


class QueueItem:
    """Элемент очереди с полным статусом"""

    def __init__(self, work_id: int, student_id: int, student_name: str,
                 image_bytes: bytes, filename: str,
                 assignment_id: int, class_id: int,
                 reference_answer: Optional[str] = None,
                 reference_formulas: Optional[list] = None):
        self.id = str(uuid.uuid4())
        self.work_id = work_id
        self.student_id = student_id
        self.student_name = student_name
        self.image_bytes = image_bytes
        self.filename = filename
        self.assignment_id = assignment_id
        self.class_id = class_id
        self.reference_answer = reference_answer
        self.reference_formulas = reference_formulas
        self.status = "queued"  # queued, processing, completed, failed
        self.position = 0
        self.queued_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None
        self.error = None
        self.result = None
        self.retry_count = 0


class ConnectionManager:
    def __init__(self):
        # Класс -> множество подключений учителей
        self.class_connections: Dict[int, Set[WebSocket]] = {}
        # Очереди по классам (используем asyncio.Queue для правильной работы)
        self.class_queues: Dict[int, asyncio.Queue] = {}
        # Словарь для хранения элементов очереди с их статусами
        self.queue_items: Dict[int, List[QueueItem]] = {}  # class_id -> list of items
        # Задачи обработки очередей
        self.processing_tasks: Dict[int, asyncio.Task] = {}
        # Блокировки для каждой очереди
        self.queue_locks: Dict[int, asyncio.Lock] = {}
        # Статистика обработки
        self.processing_stats: Dict[int, List[float]] = {}
        # Активные соединения
        self.active_connections: Set[WebSocket] = set()
        # Запущен ли heartbeat
        self.heartbeat_task = None

    async def start_heartbeat(self):
        """Запускаем фоновую задачу для heartbeat"""
        while True:
            try:
                await asyncio.sleep(30)
                disconnected = set()
                for conn in self.active_connections:
                    try:
                        await conn.send_json({
                            "type": "heartbeat",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    except:
                        disconnected.add(conn)

                # Удаляем отключившиеся соединения
                for conn in disconnected:
                    self.active_connections.discard(conn)
                    for class_id, connections in self.class_connections.items():
                        connections.discard(conn)

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def connect(self, websocket: WebSocket, class_id: int, user_id: int):
        """Подключение клиента"""
        await websocket.accept()

        self.active_connections.add(websocket)

        # Инициализируем структуры для класса если нужно
        if class_id not in self.class_connections:
            self.class_connections[class_id] = set()
            self.class_queues[class_id] = asyncio.Queue()
            self.queue_items[class_id] = []
            self.queue_locks[class_id] = asyncio.Lock()
            self.processing_stats[class_id] = []

            # Запускаем обработчик очереди
            self.processing_tasks[class_id] = asyncio.create_task(
                self.process_queue(class_id)
            )
            logger.info(f"Started queue processor for class {class_id}")

        self.class_connections[class_id].add(websocket)

        # Отправляем текущее состояние очереди
        await self.send_queue_status(class_id)

        logger.info(f"User {user_id} connected to class {class_id}")

    def disconnect(self, websocket: WebSocket, class_id: int):
        """Отключение клиента"""
        self.active_connections.discard(websocket)

        if class_id in self.class_connections:
            self.class_connections[class_id].discard(websocket)

            # Если никого не осталось, останавливаем обработку
            if not self.class_connections[class_id] and class_id in self.processing_tasks:
                self.processing_tasks[class_id].cancel()
                del self.processing_tasks[class_id]
                logger.info(f"Stopped queue processor for class {class_id} (no connections)")

    async def broadcast_to_class(self, class_id: int, message: dict):
        """Отправить сообщение всем подключенным к классу"""
        if class_id not in self.class_connections:
            return

        disconnected = set()
        for connection in self.class_connections[class_id]:
            try:
                await connection.send_json(message)
            except:
                disconnected.add(connection)

        # Удаляем отключившихся
        for conn in disconnected:
            self.class_connections[class_id].discard(conn)
            self.active_connections.discard(conn)

    async def add_to_queue(self, class_id: int, items: List[QueueItem]) -> int:
        """
        Добавить несколько работ в очередь
        Возвращает позицию первого элемента
        """
        async with self.queue_locks[class_id]:
            current_size = len(self.queue_items[class_id])

            for i, item in enumerate(items):
                item.position = current_size + i + 1
                self.queue_items[class_id].append(item)
                await self.class_queues[class_id].put(item)  # Добавляем в asyncio.Queue

                # Обновляем статус в БД
                from ..crud.assessment import update_image_status
                with next(get_session()) as session:  # Создаем временную сессию
                    update_image_status(session, item.work_id, "queued")

                logger.info(f"Added work {item.work_id} to queue for class {class_id} at position {item.position}")

            # Обновляем статус очереди
            await self.send_queue_status(class_id)

            return current_size + 1

    async def send_queue_status(self, class_id: int):
        """Отправить детальный статус очереди"""
        if class_id not in self.queue_items:
            return

        async with self.queue_locks[class_id]:
            items = self.queue_items[class_id]

            # Считаем статистику
            queued = [i for i in items if i.status == "queued"]
            processing = [i for i in items if i.status == "processing"]
            completed = [i for i in items if i.status == "completed"]
            failed = [i for i in items if i.status == "failed"]

            avg_time = self.get_avg_processing_time(class_id)
            estimated_wait = len(queued) * avg_time

            status_message = {
                "type": "queue_status",
                "data": {
                    "queue_size": len(items),
                    "queued": len(queued),
                    "processing": len(processing),
                    "completed": len(completed),
                    "failed": len(failed),
                    "estimated_wait_seconds": int(estimated_wait),
                    "estimated_wait_minutes": round(estimated_wait / 60, 1),
                    "avg_processing_time": round(avg_time, 1),
                    "items": [
                        {
                            "work_id": item.work_id,
                            "student_id": item.student_id,
                            "student_name": item.student_name,
                            "position": item.position,
                            "status": item.status,
                            "queued_at": item.queued_at.isoformat(),
                            "started_at": item.started_at.isoformat() if item.started_at else None,
                            "completed_at": item.completed_at.isoformat() if item.completed_at else None,
                            "error": item.error
                        }
                        for item in items[-20:]  # Последние 20
                    ]
                },
                "timestamp": datetime.utcnow().isoformat()
            }

            await self.broadcast_to_class(class_id, status_message)

    def get_avg_processing_time(self, class_id: int) -> float:
        """Среднее время обработки (секунды)"""
        if class_id not in self.processing_stats or not self.processing_stats[class_id]:
            return 6.0  # По умолчанию

        stats = self.processing_stats[class_id][-50:]
        return sum(stats) / len(stats) if stats else 6.0

    async def process_queue(self, class_id: int):
        """Фоновая задача для обработки очереди"""
        logger.info(f"Queue processor started for class {class_id}")
        client = get_ocr_client_v1()

        while True:
            try:
                # Ждем элемент из очереди
                item = await self.class_queues[class_id].get()

                # Обновляем статус
                async with self.queue_locks[class_id]:
                    item.status = "processing"
                    item.started_at = datetime.utcnow()

                # Обновляем в БД
                with next(get_session()) as session:
                    update_image_status(session, item.work_id, "processing")

                # Уведомляем о начале
                await self.broadcast_to_class(class_id, {
                    "type": "work_started",
                    "data": {
                        "work_id": item.work_id,
                        "student_id": item.student_id,
                        "student_name": item.student_name,
                        "position": item.position
                    },
                    "timestamp": datetime.utcnow().isoformat()
                })

                # Обрабатываем
                start_time = time.time()
                try:
                    logger.info(f"Sending to OCR API: work {item.work_id}")

                    result = client.assess_solution(
                        image_bytes=item.image_bytes,
                        filename=item.filename,
                        reference_answer=item.reference_answer,
                        reference_formulas=item.reference_formulas
                    )

                    processing_time = time.time() - start_time

                    # Сохраняем статистику
                    self.processing_stats[class_id].append(processing_time)

                    if result.get("success") and result.get("assessment"):
                        assessment = result["assessment"]

                        # Сохраняем результат в БД
                        with next(get_session()) as session:
                            # Создаем запись распознанного решения
                            solution = create_recognized_solution_v1(
                                session=session,
                                image_id=item.work_id,
                                assessment_data=assessment
                            )

                            # Обновляем статус изображения
                            update_image_status(session, item.work_id, "processed")

                            logger.info(f"Saved solution {solution.id} for work {item.work_id}")

                        # Обновляем статус в очереди
                        async with self.queue_locks[class_id]:
                            item.status = "completed"
                            item.result = assessment
                            item.completed_at = datetime.utcnow()

                        # Отправляем результат
                        await self.broadcast_to_class(class_id, {
                            "type": "work_completed",
                            "data": {
                                "work_id": item.work_id,
                                "student_id": item.student_id,
                                "student_name": item.student_name,
                                "confidence_score": assessment.get("confidence_score"),
                                "check_level": f"level_{assessment.get('confidence_level', 3)}",
                                "mark_score": assessment.get("mark_score"),
                                "teacher_comment": assessment.get("teacher_comment")
                            },
                            "timestamp": datetime.utcnow().isoformat()
                        })

                    else:
                        error_msg = result.get("error", "Unknown OCR error")

                        # Обновляем статус в БД
                        with next(get_session()) as session:
                            update_image_status(session, item.work_id, "error", error_msg)

                        async with self.queue_locks[class_id]:
                            item.status = "failed"
                            item.error = error_msg
                            item.completed_at = datetime.utcnow()

                        await self.broadcast_to_class(class_id, {
                            "type": "work_failed",
                            "data": {
                                "work_id": item.work_id,
                                "student_id": item.student_id,
                                "student_name": item.student_name,
                                "error": error_msg
                            },
                            "timestamp": datetime.utcnow().isoformat()
                        })

                except Exception as e:
                    logger.error(f"Error processing work {item.work_id}: {e}", exc_info=True)

                    # Обновляем статус в БД
                    with next(get_session()) as session:
                        update_image_status(session, item.work_id, "error", str(e))

                    async with self.queue_locks[class_id]:
                        item.status = "failed"
                        item.error = str(e)
                        item.completed_at = datetime.utcnow()

                    await self.broadcast_to_class(class_id, {
                        "type": "work_failed",
                        "data": {
                            "work_id": item.work_id,
                            "student_id": item.student_id,
                            "student_name": item.student_name,
                            "error": str(e)
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    })

                # Отправляем обновленный статус очереди
                await self.send_queue_status(class_id)

                # Помечаем задачу как выполненную
                self.class_queues[class_id].task_done()

                # Очищаем старые элементы
                await self.cleanup_completed_items(class_id)

            except asyncio.CancelledError:
                logger.info(f"Queue processing for class {class_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Error in queue processing: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def cleanup_completed_items(self, class_id: int):
        """Очистка старых завершенных элементов"""
        if class_id not in self.queue_items:
            return

        async with self.queue_locks[class_id]:
            cutoff = datetime.utcnow() - timedelta(minutes=5)
            self.queue_items[class_id] = [
                item for item in self.queue_items[class_id]
                if item.status in ["queued", "processing"] or
                   (item.completed_at and item.completed_at > cutoff)
            ]

    async def get_queue_status(self, class_id: int) -> dict:
        """Получить полный статус очереди"""
        if class_id not in self.queue_items:
            return {
                "queue_size": 0,
                "queued": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "estimated_wait_seconds": 0,
                "avg_processing_time": 6.0,
                "items": []
            }

        async with self.queue_locks[class_id]:
            items = self.queue_items[class_id]
            queued = [i for i in items if i.status == "queued"]
            processing = [i for i in items if i.status == "processing"]
            completed = [i for i in items if i.status == "completed"]
            failed = [i for i in items if i.status == "failed"]

            avg_time = self.get_avg_processing_time(class_id)
            estimated_wait = len(queued) * avg_time

            return {
                "queue_size": len(items),
                "queued": len(queued),
                "processing": len(processing),
                "completed": len(completed),
                "failed": len(failed),
                "estimated_wait_seconds": int(estimated_wait),
                "avg_processing_time": round(avg_time, 1),
                "items": [
                    {
                        "work_id": item.work_id,
                        "student_id": item.student_id,
                        "student_name": item.student_name,
                        "position": item.position,
                        "status": item.status,
                        "queued_at": item.queued_at.isoformat(),
                        "started_at": item.started_at.isoformat() if item.started_at else None,
                        "completed_at": item.completed_at.isoformat() if item.completed_at else None,
                        "error": item.error
                    }
                    for item in items
                ]
            }


# Создаем глобальный менеджер
manager = ConnectionManager()


@router.on_event("startup")
async def start_heartbeat():
    """Запускаем heartbeat при старте"""
    asyncio.create_task(manager.start_heartbeat())


async def get_current_user_ws(token: str, session: Session) -> Optional[User]:
    """Аутентификация для WebSocket"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
        return session.get(User, user_id)
    except Exception as e:
        logger.error(f"WebSocket auth error: {e}")
        return None


@router.websocket("/ws/class/{class_id}")
async def websocket_class(
        websocket: WebSocket,
        class_id: int,
        token: str = Query(...),
        session: Session = Depends(get_session)
):
    """WebSocket для real-time обновлений по классу"""

    # Аутентификация
    user = await get_current_user_ws(token, session)
    if not user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Проверяем права (только учитель класса)
    class_obj = session.get(Class, class_id)
    if not class_obj or class_obj.teacher_id != user.id:
        logger.warning(f"User {user.id} tried to connect to class {class_id} without permission")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    try:
        await manager.connect(websocket, class_id, user.id)

        while True:
            # Ждем сообщения от клиента
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "get_queue_status":
                status = await manager.get_queue_status(class_id)
                await websocket.send_json({
                    "type": "queue_status",
                    "data": status,
                    "timestamp": datetime.utcnow().isoformat()
                })

            elif message.get("type") == "get_work_result":
                work_id = message.get("work_id")
                # Ищем результат в очереди
                for item in manager.queue_items.get(class_id, []):
                    if item.work_id == work_id and item.status == "completed":
                        await websocket.send_json({
                            "type": "work_result",
                            "data": {
                                "work_id": work_id,
                                "result": item.result
                            }
                        })
                        break

            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket, class_id)
        logger.info(f"User {user.id} disconnected from class {class_id}")
    except Exception as e:
        logger.error(f"WebSocket error for class {class_id}: {e}")
        manager.disconnect(websocket, class_id)


# API функция для добавления работ в очередь
async def add_works_to_queue(
        class_id: int,
        works: List[dict],
        session: Session
) -> int:
    """
    Добавить несколько работ в очередь на обработку

    Args:
        class_id: ID класса
        works: список словарей с полями:
            - work_id: int
            - student_id: int
            - student_name: str
            - assignment_id: int
            - class_id: int
            - image_bytes: bytes
            - filename: str
            - reference_answer: Optional[str]
            - reference_formulas: Optional[list]
        session: сессия БД (не используется здесь, но оставляем для совместимости)

    Returns:
        int: позиция первого элемента в очереди
    """
    logger.info(f"Adding {len(works)} works to queue for class {class_id}")

    # Получаем задание для reference данных
    if works and works[0].get("assignment_id"):
        assignment = session.get(Assignment, works[0]["assignment_id"])
    else:
        assignment = None

    items = []
    for work in works:
        # Если reference данные не переданы, берем из задания
        ref_answer = work.get("reference_answer")
        ref_formulas = work.get("reference_formulas")

        if not ref_answer and assignment:
            ref_answer = assignment.reference_answer
        if not ref_formulas and assignment and assignment.reference_solution:
            ref_formulas = [assignment.reference_solution]

        item = QueueItem(
            work_id=work["work_id"],
            student_id=work["student_id"],
            student_name=work["student_name"],
            image_bytes=work["image_bytes"],
            filename=work["filename"],
            assignment_id=work.get("assignment_id", 0),
            class_id=class_id,
            reference_answer=ref_answer,
            reference_formulas=ref_formulas
        )
        items.append(item)

    position = await manager.add_to_queue(class_id, items)
    logger.info(f"Added {len(works)} works to queue for class {class_id}, first position: {position}")

    return position