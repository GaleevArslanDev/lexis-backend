from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, status
from typing import Dict, Set, Optional
import asyncio
import json
import logging
from datetime import datetime, timedelta
from sqlmodel import Session, select
from jose import jwt

from ..db import get_session
from ..models import User, Class, AssessmentImage, RecognizedSolution
from ..utils.security import SECRET_KEY, ALGORITHM
from ..exceptions import AppException, ErrorCode

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

class ConnectionManager:
    def __init__(self):
        # Класс -> множество подключений учителей
        self.class_connections: Dict[int, Set[WebSocket]] = {}
        # Словарь для хранения очередей по классам
        self.class_queues: Dict[int, asyncio.Queue] = {}
        # Задачи обработки очередей
        self.processing_tasks: Dict[int, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, class_id: int, user_id: int):
        await websocket.accept()
        if class_id not in self.class_connections:
            self.class_connections[class_id] = set()
            # Создаем очередь для этого класса
            self.class_queues[class_id] = asyncio.Queue()
            # Запускаем обработчик очереди
            self.processing_tasks[class_id] = asyncio.create_task(
                self.process_queue(class_id)
            )

        self.class_connections[class_id].add(websocket)

        # Отправляем текущее состояние очереди при подключении
        await self.send_queue_status(class_id)

        logger.info(
            f"User {user_id} connected to class {class_id}. Total connections: {len(self.class_connections[class_id])}")

    def disconnect(self, websocket: WebSocket, class_id: int):
        if class_id in self.class_connections:
            self.class_connections[class_id].discard(websocket)
            # Если никого не осталось, останавливаем обработку очереди
            if not self.class_connections[class_id]:
                if class_id in self.processing_tasks:
                    self.processing_tasks[class_id].cancel()
                    del self.processing_tasks[class_id]
                if class_id in self.class_queues:
                    del self.class_queues[class_id]

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

    async def add_to_queue(self, class_id: int, work_data: dict):
        """Добавить работу в очередь на обработку"""
        if class_id not in self.class_queues:
            self.class_queues[class_id] = asyncio.Queue()

        # Добавляем время постановки в очередь
        work_data["queued_at"] = datetime.utcnow().isoformat()
        work_data["position"] = self.class_queues[class_id].qsize() + 1

        await self.class_queues[class_id].put(work_data)

        # Обновляем статус очереди
        await self.send_queue_status(class_id)

        return work_data["position"]

    async def send_queue_status(self, class_id: int):
        """Отправить статус очереди всем подключенным"""
        if class_id in self.class_queues:
            queue_size = self.class_queues[class_id].qsize()

            # Оцениваем время ожидания (в среднем 5-7 секунд на работу)
            avg_processing_time = 6  # секунд
            estimated_wait = queue_size * avg_processing_time

            status_message = {
                "type": "queue_status",
                "data": {
                    "queue_size": queue_size,
                    "estimated_wait_seconds": estimated_wait,
                    "estimated_wait_minutes": round(estimated_wait / 60, 1),
                    "processing_speed": f"{avg_processing_time}s per work"
                },
                "timestamp": datetime.utcnow().isoformat()
            }

            await self.broadcast_to_class(class_id, status_message)

    async def process_queue(self, class_id: int):
        """Фоновая задача для обработки очереди работ"""
        while True:
            try:
                # Ждем новую работу в очереди
                work_data = await self.class_queues[class_id].get()

                # Уведомляем о начале обработки
                await self.broadcast_to_class(class_id, {
                    "type": "work_started",
                    "data": {
                        "work_id": work_data["work_id"],
                        "student_name": work_data["student_name"],
                        "position": work_data["position"]
                    },
                    "timestamp": datetime.utcnow().isoformat()
                })

                # Эмулируем обработку (в реальности здесь будет вызов ML API)
                # Вместо эмуляции нужно использовать реальную обработку
                await asyncio.sleep(2)  # Симуляция обработки

                # Получаем результат (в реальности из ML API)
                result = await self.process_work(work_data)

                # Отправляем результат
                await self.broadcast_to_class(class_id, {
                    "type": "work_completed",
                    "data": {
                        "work_id": work_data["work_id"],
                        "student_name": work_data["student_name"],
                        "result": result,
                        "check_level": result.get("check_level", "level_3"),
                        "confidence_score": result.get("confidence_score", 0)
                    },
                    "timestamp": datetime.utcnow().isoformat()
                })

                # Обновляем позиции в очереди
                await self.update_queue_positions(class_id)

            except asyncio.CancelledError:
                logger.info(f"Queue processing for class {class_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing queue for class {class_id}: {e}")
                await asyncio.sleep(1)

    async def process_work(self, work_data: dict) -> dict:
        """Реальная обработка работы через ML API"""
        # Здесь должен быть вызов реального ML API
        # Используем существующий ocr_client_v1
        from ..ocr_client import get_ocr_client_v1

        client = get_ocr_client_v1()

        try:
            result = client.assess_solution(
                image_bytes=work_data["image_bytes"],
                filename=work_data["filename"],
                reference_answer=work_data.get("reference_answer"),
                reference_formulas=work_data.get("reference_formulas")
            )

            if result.get("success"):
                assessment = result.get("assessment", {})
                return {
                    "success": True,
                    "confidence_score": assessment.get("confidence_score"),
                    "check_level": f"level_{assessment.get('confidence_level', 3)}",
                    "mark_score": assessment.get("mark_score"),
                    "teacher_comment": assessment.get("teacher_comment"),
                    "steps_analysis": assessment.get("steps_analysis", [])
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
        except Exception as e:
            logger.error(f"Error processing work: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def update_queue_positions(self, class_id: int):
        """Обновить позиции в очереди после завершения работы"""
        if class_id in self.class_queues:
            # Создаем временный список для обновления позиций
            temp_queue = []
            while not self.class_queues[class_id].empty():
                item = await self.class_queues[class_id].get()
                temp_queue.append(item)

            # Возвращаем в очередь с обновленными позициями
            for i, item in enumerate(temp_queue, 1):
                item["position"] = i
                await self.class_queues[class_id].put(item)

            # Отправляем обновленный статус
            await self.send_queue_status(class_id)


manager = ConnectionManager()


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

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

            elif message.get("type") == "get_queue_status":
                await manager.send_queue_status(class_id)

            elif message.get("type") == "cancel_work":
                # TODO: Реализовать отмену работы в очереди
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket, class_id)
        logger.info(f"User {user.id} disconnected from class {class_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, class_id)


# API для добавления работ в очередь
async def add_work_to_queue(
        class_id: int,
        work_id: int,
        student_name: str,
        image_bytes: bytes,
        filename: str,
        reference_answer: Optional[str] = None,
        reference_formulas: Optional[list] = None
) -> int:
    """Добавить работу в очередь на обработку (вызывается из основного API)"""
    position = await manager.add_to_queue(class_id, {
        "work_id": work_id,
        "student_name": student_name,
        "image_bytes": image_bytes,
        "filename": filename,
        "reference_answer": reference_answer,
        "reference_formulas": reference_formulas
    })
    return position