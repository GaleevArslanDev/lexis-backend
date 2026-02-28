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
from ..models import User, Class, AssessmentImage, RecognizedSolution
from ..utils.security import SECRET_KEY, ALGORITHM
from ..exceptions import AppException, ErrorCode
from ..ocr_client import get_ocr_client_v1

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


class QueueItem:
    """Элемент очереди с полным статусом"""

    def __init__(self, work_id: int, student_id: int, student_name: str,
                 image_bytes: bytes, filename: str,
                 reference_answer: Optional[str] = None,
                 reference_formulas: Optional[list] = None):
        self.id = str(uuid.uuid4())
        self.work_id = work_id
        self.student_id = student_id
        self.student_name = student_name
        self.image_bytes = image_bytes
        self.filename = filename
        self.reference_answer = reference_answer
        self.reference_formulas = reference_formulas
        self.status = "queued"  # queued, processing, completed, failed
        self.position = 0
        self.queued_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None
        self.error = None
        self.result = None


class ConnectionManager:
    def __init__(self):
        # Класс -> множество подключений учителей
        self.class_connections: Dict[int, Set[WebSocket]] = {}
        # Очереди по классам (список QueueItem)
        self.class_queues: Dict[int, List[QueueItem]] = {}
        # Задачи обработки очередей
        self.processing_tasks: Dict[int, asyncio.Task] = {}
        # Блокировки для каждой очереди
        self.queue_locks: Dict[int, asyncio.Lock] = {}
        # Статистика обработки
        self.processing_stats: Dict[int, List[float]] = {}
        # Heartbeat задача
        self.heartbeat_task = None
        # Активные соединения для heartbeat
        self.active_connections: Set[WebSocket] = set()

    async def start_heartbeat(self):
        """Запускаем фоновую задачу для heartbeat"""
        while True:
            try:
                await asyncio.sleep(30)  # Каждые 30 секунд
                disconnected = set()
                for conn in self.active_connections:
                    try:
                        await conn.send_json({"type": "heartbeat", "timestamp": datetime.utcnow().isoformat()})
                    except:
                        disconnected.add(conn)

                # Удаляем отключившиеся соединения
                for conn in disconnected:
                    self.active_connections.discard(conn)
                    # Также удаляем из class_connections
                    for class_id, connections in self.class_connections.items():
                        if conn in connections:
                            connections.discard(conn)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def connect(self, websocket: WebSocket, class_id: int, user_id: int):
        await websocket.accept()

        # Добавляем в активные соединения
        self.active_connections.add(websocket)

        # Добавляем в класс
        if class_id not in self.class_connections:
            self.class_connections[class_id] = set()

        # Гарантируем, что структуры данных для очереди существуют
        if class_id not in self.class_queues:
            self.class_queues[class_id] = []
            self.queue_locks[class_id] = asyncio.Lock()
            self.processing_stats[class_id] = []

            # Запускаем обработчик очереди
            if class_id not in self.processing_tasks:
                self.processing_tasks[class_id] = asyncio.create_task(
                    self.process_queue(class_id)
                )
                logger.info(f"Started queue processor for class {class_id}")

        self.class_connections[class_id].add(websocket)

        # Отправляем текущее состояние очереди при подключении
        await self.send_queue_status(class_id)

        logger.info(
            f"User {user_id} connected to class {class_id}. "
            f"Total connections: {len(self.class_connections[class_id])}"
        )

    def disconnect(self, websocket: WebSocket, class_id: int):
        # Удаляем из активных
        self.active_connections.discard(websocket)

        # Удаляем из класса
        if class_id in self.class_connections:
            self.class_connections[class_id].discard(websocket)

            # Если никого не осталось, останавливаем обработку очереди
            if not self.class_connections[class_id]:
                if class_id in self.processing_tasks:
                    self.processing_tasks[class_id].cancel()
                    del self.processing_tasks[class_id]
                # Очередь сохраняем - она не должна очищаться при отключении всех

    async def broadcast_to_class(self, class_id: int, message: dict):
        """Отправить сообщение всем подключенным к классу"""
        if class_id in self.class_connections:
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
        """Добавить несколько работ в очередь"""
        # Гарантируем, что очередь инициализирована
        if class_id not in self.class_queues:
            self.class_queues[class_id] = []
            self.queue_locks[class_id] = asyncio.Lock()
            self.processing_stats[class_id] = []

        # ВАЖНО: Убеждаемся, что процессор запущен
        await self.ensure_processor_running(class_id)

        async with self.queue_locks[class_id]:
            current_size = len(self.class_queues[class_id])

            # Добавляем все элементы с правильными позициями
            for i, item in enumerate(items):
                item.position = current_size + i + 1
                self.class_queues[class_id].append(item)
                logger.info(f"📝 Added work {item.work_id} to queue for class {class_id} at position {item.position}")

            # Обновляем статус очереди
            await self.send_queue_status(class_id)

            return current_size + 1  # Позиция первого элемента

    async def send_queue_status(self, class_id: int):
        """Отправить детальный статус очереди"""
        if class_id not in self.class_queues:
            # Если очереди нет, отправляем пустой статус
            status_message = {
                "type": "queue_status",
                "data": {
                    "queue_size": 0,
                    "queued": 0,
                    "processing": 0,
                    "completed": 0,
                    "failed": 0,
                    "estimated_wait_seconds": 0,
                    "estimated_wait_minutes": 0,
                    "avg_processing_time": 6.0,
                    "items": []
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.broadcast_to_class(class_id, status_message)
            return

        async with self.queue_locks.get(class_id, asyncio.Lock()):
            queue = self.class_queues[class_id]

            # Считаем статистику
            queued = [q for q in queue if q.status == "queued"]
            processing = [q for q in queue if q.status == "processing"]
            completed = [q for q in queue if q.status == "completed"]
            failed = [q for q in queue if q.status == "failed"]

            # Оценка времени
            avg_time = self.get_avg_processing_time(class_id)
            estimated_wait = len(queued) * avg_time

            # Формируем статус
            status_message = {
                "type": "queue_status",
                "data": {
                    "queue_size": len(queue),
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
                        for item in queue[-20:]  # Последние 20
                    ]
                },
                "timestamp": datetime.utcnow().isoformat()
            }

            await self.broadcast_to_class(class_id, status_message)

    def get_avg_processing_time(self, class_id: int) -> float:
        """Среднее время обработки (секунды)"""
        if class_id not in self.processing_stats or not self.processing_stats[class_id]:
            return 6.0  # По умолчанию 6 секунд

        stats = self.processing_stats[class_id][-50:]  # Последние 50
        return sum(stats) / len(stats) if stats else 6.0

    async def process_queue(self, class_id: int):
        """Фоновая задача для обработки очереди"""
        logger.info(f"🔄 Queue processor started for class {class_id}")
        client = get_ocr_client_v1()

        while True:
            try:
                # Проверяем, существует ли еще очередь
                if class_id not in self.class_queues:
                    logger.info(f"Queue for class {class_id} no longer exists, stopping processor")
                    break

                # Логируем состояние очереди
                queue_size = len(self.class_queues.get(class_id, []))
                if queue_size > 0:
                    logger.info(f"Class {class_id} queue size: {queue_size}")

                # Ищем следующий элемент для обработки
                next_item = None
                lock = self.queue_locks.get(class_id)
                if not lock:
                    logger.error(f"Lock for class {class_id} not found")
                    await asyncio.sleep(1)
                    continue

                async with lock:
                    queue = self.class_queues.get(class_id, [])
                    for item in queue:
                        if item.status == "queued":
                            next_item = item
                            item.status = "processing"
                            item.started_at = datetime.utcnow()
                            logger.info(f"▶️ Processing work {item.work_id} for class {class_id}")
                            break

                if not next_item:
                    # Нет элементов в очереди - ждем
                    await asyncio.sleep(1)
                    continue

                # Уведомляем о начале обработки
                await self.broadcast_to_class(class_id, {
                    "type": "work_started",
                    "data": {
                        "work_id": next_item.work_id,
                        "student_id": next_item.student_id,
                        "student_name": next_item.student_name,
                        "position": next_item.position
                    },
                    "timestamp": datetime.utcnow().isoformat()
                })

                # Обрабатываем
                start_time = time.time()
                try:
                    logger.info(f"📤 Sending to OCR API: work {next_item.work_id}, file {next_item.filename}")

                    result = client.assess_solution(
                        image_bytes=next_item.image_bytes,
                        filename=next_item.filename,
                        reference_answer=next_item.reference_answer,
                        reference_formulas=next_item.reference_formulas
                    )

                    processing_time = time.time() - start_time
                    logger.info(f"✅ OCR API response received for work {next_item.work_id} in {processing_time:.2f}s")

                    # Сохраняем статистику
                    if class_id in self.processing_stats:
                        self.processing_stats[class_id].append(processing_time)

                    if result.get("success"):
                        next_item.status = "completed"
                        next_item.result = result.get("assessment", {})
                        next_item.completed_at = datetime.utcnow()

                        # Отправляем результат
                        assessment = result.get("assessment", {})
                        await self.broadcast_to_class(class_id, {
                            "type": "work_completed",
                            "data": {
                                "work_id": next_item.work_id,
                                "student_id": next_item.student_id,
                                "student_name": next_item.student_name,
                                "confidence_score": assessment.get("confidence_score"),
                                "check_level": f"level_{assessment.get('confidence_level', 3)}",
                                "mark_score": assessment.get("mark_score"),
                                "teacher_comment": assessment.get("teacher_comment")
                            },
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    else:
                        next_item.status = "failed"
                        next_item.error = result.get("error", "Unknown error")
                        next_item.completed_at = datetime.utcnow()
                        logger.error(f"❌ OCR API error for work {next_item.work_id}: {next_item.error}")

                        await self.broadcast_to_class(class_id, {
                            "type": "work_failed",
                            "data": {
                                "work_id": next_item.work_id,
                                "student_id": next_item.student_id,
                                "student_name": next_item.student_name,
                                "error": next_item.error
                            },
                            "timestamp": datetime.utcnow().isoformat()
                        })

                except Exception as e:
                    logger.error(f"🔥 Error processing work {next_item.work_id}: {e}", exc_info=True)
                    next_item.status = "failed"
                    next_item.error = str(e)
                    next_item.completed_at = datetime.utcnow()

                    await self.broadcast_to_class(class_id, {
                        "type": "work_failed",
                        "data": {
                            "work_id": next_item.work_id,
                            "student_id": next_item.student_id,
                            "student_name": next_item.student_name,
                            "error": str(e)
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    })

                # Обновляем статус очереди
                await self.send_queue_status(class_id)

                # Очищаем старые завершенные элементы
                await self.cleanup_completed_items(class_id)

            except asyncio.CancelledError:
                logger.info(f"🛑 Queue processing for class {class_id} cancelled")
                break
            except Exception as e:
                logger.error(f"💥 Error in queue processing for class {class_id}: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def ensure_processor_running(self, class_id: int):
        """Гарантирует, что процессор очереди запущен"""
        if class_id not in self.processing_tasks or self.processing_tasks[class_id].done():
            if class_id in self.processing_tasks and self.processing_tasks[class_id].done():
                # Если задача завершилась, удаляем её
                try:
                    self.processing_tasks[class_id].result()  # Может выбросить исключение
                except Exception as e:
                    logger.error(f"Processor for class {class_id} failed: {e}")
                del self.processing_tasks[class_id]

            # Запускаем новую
            self.processing_tasks[class_id] = asyncio.create_task(
                self.process_queue(class_id)
            )
            logger.info(f"🚀 (Re)started queue processor for class {class_id}")
            return True
        return False

    async def cleanup_completed_items(self, class_id: int):
        """Очистка старых завершенных элементов"""
        async with self.queue_locks[class_id]:
            # Оставляем только:
            # - элементы в очереди (queued, processing)
            # - завершенные за последние 5 минут
            cutoff = datetime.utcnow() - timedelta(minutes=5)
            self.class_queues[class_id] = [
                item for item in self.class_queues[class_id]
                if item.status in ["queued", "processing"] or
                   (item.completed_at and item.completed_at > cutoff)
            ]

    async def get_queue_status(self, class_id: int) -> dict:
        """Получить полный статус очереди"""
        if class_id not in self.class_queues:
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

        lock = self.queue_locks.get(class_id, asyncio.Lock())
        async with lock:
            queue = self.class_queues.get(class_id, [])
            queued = [q for q in queue if q.status == "queued"]
            processing = [q for q in queue if q.status == "processing"]
            completed = [q for q in queue if q.status == "completed"]
            failed = [q for q in queue if q.status == "failed"]

            avg_time = self.get_avg_processing_time(class_id)
            estimated_wait = len(queued) * avg_time

            return {
                "queue_size": len(queue),
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
                    for item in queue
                ]
            }


# Создаем глобальный менеджер
manager = ConnectionManager()


# Запускаем heartbeat при старте
@router.on_event("startup")
async def start_heartbeat():
    asyncio.create_task(manager.start_heartbeat())


async def get_current_user_ws(token: str, session: Session) -> Optional[User]:
    """Аутентификация для WebSocket"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
        return session.get(User, user_id)
    except:
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
                for item in manager.class_queues.get(class_id, []):
                    if item.work_id == work_id and item.status == "completed":
                        await websocket.send_json({
                            "type": "work_result",
                            "data": {
                                "work_id": work_id,
                                "result": item.result
                            }
                        })
                        break

    except WebSocketDisconnect:
        manager.disconnect(websocket, class_id)
        logger.info(f"User {user.id} disconnected from class {class_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
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
            - image_bytes: bytes
            - filename: str
            - reference_answer: Optional[str]
            - reference_formulas: Optional[list]
        session: сессия БД

    Returns:
        int: позиция первого элемента в очереди
    """
    logger.info(f"📦 Adding {len(works)} works to queue for class {class_id}")

    items = []
    for work in works:
        item = QueueItem(
            work_id=work["work_id"],
            student_id=work["student_id"],
            student_name=work["student_name"],
            image_bytes=work["image_bytes"],
            filename=work["filename"],
            reference_answer=work.get("reference_answer"),
            reference_formulas=work.get("reference_formulas")
        )
        items.append(item)

    position = await manager.add_to_queue(class_id, items)
    logger.info(f"✅ Added {len(works)} works to queue for class {class_id}, first position: {position}")

    # Сохраняем в БД статус queued
    for work in works:
        from ..crud.assessment import update_image_status
        update_image_status(session, work["work_id"], "queued")
        logger.info(f"💾 Updated DB: work {work['work_id']} status = queued")

    return position