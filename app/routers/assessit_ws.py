# app/routers/assessit_ws.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, status
from typing import Dict, Set, Optional, List, Any
import asyncio
import json
import logging
from datetime import datetime
from sqlmodel import Session, select
from jose import jwt
import uuid

from ..db import get_session
from ..dependencies_util import require_role
from ..models import User, Class, AssessmentImage
from ..utils.security import SECRET_KEY, ALGORITHM
from ..crud.queue import get_class_queue_items

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """
    Менеджер WebSocket соединений без Redis
    Хранит все соединения в памяти
    """

    def __init__(self):
        # Класс -> множество подключений учителей
        self.class_connections: Dict[int, Set[WebSocket]] = {}
        # User ID -> множество подключений (для личных уведомлений)
        self.user_connections: Dict[int, Set[WebSocket]] = {}
        # Активные соединения (для мониторинга)
        self.active_connections: Set[WebSocket] = set()

        # Очередь сообщений для каждого класса (для отказоустойчивости)
        self.class_message_queues: Dict[int, asyncio.Queue] = {}

        # Задачи для обработки очередей
        self.queue_tasks: Dict[int, asyncio.Task] = {}

        logger.info("ConnectionManager initialized (in-memory mode)")

    async def connect(self, websocket: WebSocket, class_id: int, user_id: int):
        """Подключение клиента"""
        await websocket.accept()

        self.active_connections.add(websocket)

        # Добавляем в класс
        if class_id not in self.class_connections:
            self.class_connections[class_id] = set()
            # Создаем очередь сообщений для класса
            self.class_message_queues[class_id] = asyncio.Queue()
            # Запускаем обработчик очереди
            self.queue_tasks[class_id] = asyncio.create_task(
                self._process_class_queue(class_id)
            )

        self.class_connections[class_id].add(websocket)

        # Добавляем в пользовательские соединения
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(websocket)

        # Отправляем подтверждение подключения
        await websocket.send_json({
            "type": "connected",
            "data": {
                "class_id": class_id,
                "user_id": user_id,
                "message": "Connected to class updates",
                "timestamp": datetime.utcnow().isoformat()
            }
        })

        # Отправляем текущий статус очереди
        await self._send_queue_status(class_id, websocket)

        logger.info(f"User {user_id} connected to class {class_id}. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, class_id: int, user_id: int):
        """Отключение клиента"""
        self.active_connections.discard(websocket)

        # Удаляем из класса
        if class_id in self.class_connections:
            self.class_connections[class_id].discard(websocket)

            # Если никого не осталось в классе, останавливаем обработчик очереди
            if not self.class_connections[class_id] and class_id in self.queue_tasks:
                self.queue_tasks[class_id].cancel()
                del self.queue_tasks[class_id]
                logger.info(f"Stopped queue processor for class {class_id} (no listeners)")

        # Удаляем из пользовательских соединений
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]

        logger.info(f"User {user_id} disconnected from class {class_id}")

    async def broadcast_to_class(self, class_id: int, message: dict):
        """
        Отправить сообщение всем подключенным к классу
        Если нет активных соединений, сообщение кладется в очередь
        """
        # Добавляем timestamp если нет
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()

        # Если есть активные соединения, отправляем сразу
        if class_id in self.class_connections and self.class_connections[class_id]:
            disconnected = set()
            for connection in self.class_connections[class_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send to client: {e}")
                    disconnected.add(connection)

            # Удаляем отключившихся
            for conn in disconnected:
                self.class_connections[class_id].discard(conn)
                self.active_connections.discard(conn)

        # Если нет соединений, но есть очередь - сохраняем для последующей отправки
        elif class_id in self.class_message_queues:
            try:
                # Ограничиваем размер очереди (макс 100 сообщений)
                if self.class_message_queues[class_id].qsize() < 100:
                    await self.class_message_queues[class_id].put(message)
                    logger.debug(
                        f"Queued message for class {class_id} (queue size: {self.class_message_queues[class_id].qsize()})")
            except Exception as e:
                logger.error(f"Failed to queue message for class {class_id}: {e}")

    async def send_to_user(self, user_id: int, message: dict):
        """Отправить сообщение конкретному пользователю"""
        if user_id not in self.user_connections:
            return

        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()

        disconnected = set()
        for connection in self.user_connections[user_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to user {user_id}: {e}")
                disconnected.add(connection)

        # Удаляем отключившихся
        for conn in disconnected:
            self.active_connections.discard(conn)
            self.user_connections[user_id].discard(conn)

    async def _process_class_queue(self, class_id: int):
        """
        Фоновая задача для обработки очереди сообщений класса
        Отправляет накопленные сообщения при появлении соединений
        """
        logger.info(f"Started queue processor for class {class_id}")

        while True:
            try:
                # Ждем сообщение из очереди (с таймаутом для проверки соединений)
                try:
                    message = await asyncio.wait_for(
                        self.class_message_queues[class_id].get(),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    # Проверяем, есть ли еще слушатели
                    if class_id not in self.class_connections or not self.class_connections[class_id]:
                        # Если слушателей нет, продолжаем ждать
                        continue
                    else:
                        # Есть слушатели, но очередь пуста
                        continue

                # Если есть слушатели, отправляем
                if class_id in self.class_connections and self.class_connections[class_id]:
                    await self.broadcast_to_class(class_id, message)
                else:
                    # Если слушателей нет, возвращаем в очередь
                    await self.class_message_queues[class_id].put(message)
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info(f"Queue processor for class {class_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Error in queue processor for class {class_id}: {e}")
                await asyncio.sleep(5)

    async def _send_queue_status(self, class_id: int, websocket: WebSocket):
        """Отправить текущий статус очереди для класса"""
        try:
            from ..db import engine
            from sqlmodel import Session

            with Session(engine) as session:
                queue_items = get_class_queue_items(session, class_id, limit=20)

                # Получаем статистику
                pending = sum(1 for item in queue_items if item["status"] == "pending")
                processing = sum(1 for item in queue_items if item["status"] == "processing")

                await websocket.send_json({
                    "type": "queue_status",
                    "data": {
                        "class_id": class_id,
                        "pending": pending,
                        "processing": processing,
                        "recent_items": queue_items[:10],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                })
        except Exception as e:
            logger.error(f"Failed to send queue status: {e}")

    async def notify_work_update(self, work_id: int, class_id: int, status: str, data: dict = None):
        """Уведомить об обновлении статуса работы"""
        message = {
            "type": "work_status_update",
            "data": {
                "work_id": work_id,
                "status": status,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        if data:
            message["data"].update(data)

        await self.broadcast_to_class(class_id, message)

    async def notify_batch_upload(self, class_id: int, batch_data: dict):
        """Уведомить о пакетной загрузке"""
        message = {
            "type": "batch_upload",
            "data": {
                **batch_data,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        await self.broadcast_to_class(class_id, message)

    def get_stats(self) -> dict:
        """Получить статистику соединений"""
        return {
            "total_connections": len(self.active_connections),
            "active_classes": len(self.class_connections),
            "active_users": len(self.user_connections),
            "class_queues": {
                str(cid): q.qsize()
                for cid, q in self.class_message_queues.items()
            }
        }


# Создаем глобальный менеджер
manager = ConnectionManager()


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
    """
    WebSocket для real-time обновлений по классу
    Без Redis, использует in-memory соединения
    """
    # Аутентификация
    user = await get_current_user_ws(token, session)
    if not user:
        logger.warning(f"WebSocket authentication failed for class {class_id}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Проверяем права (только учитель класса)
    class_obj = session.get(Class, class_id)
    if not class_obj or class_obj.teacher_id != user.id:
        logger.warning(f"User {user.id} tried to connect to class {class_id} without permission")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    try:
        # Подключаем
        await manager.connect(websocket, class_id, user.id)

        # Основной цикл обработки сообщений
        while True:
            try:
                # Ждем сообщения от клиента с таймаутом
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)

                # Обрабатываем разные типы сообщений
                msg_type = message.get("type")

                if msg_type == "ping":
                    # Ответ на ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })

                elif msg_type == "request_queue_status":
                    # Клиент запросил статус очереди
                    with Session(get_session) as db_session:
                        queue_items = get_class_queue_items(db_session, class_id, limit=20)
                        await websocket.send_json({
                            "type": "queue_status",
                            "data": {
                                "class_id": class_id,
                                "items": queue_items,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        })

                elif msg_type == "subscribe_work":
                    # Подписка на обновления конкретной работы
                    work_id = message.get("work_id")
                    # Можно сохранять подписки, но пока просто подтверждаем
                    await websocket.send_json({
                        "type": "subscribed",
                        "data": {
                            "work_id": work_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    })

                else:
                    logger.debug(f"Unknown message type: {msg_type} from user {user.id}")

            except asyncio.TimeoutError:
                # Таймаут - отправляем ping для проверки соединения
                try:
                    await websocket.send_json({
                        "type": "ping",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                except:
                    # Если не можем отправить, вероятно соединение разорвано
                    break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user.id} from class {class_id}")
    except Exception as e:
        logger.error(f"WebSocket error for class {class_id}: {e}")
    finally:
        # Всегда отключаем при выходе
        manager.disconnect(websocket, class_id, user.id)


@router.get("/ws/stats")
async def websocket_stats(current_user: User = Depends(require_role("teacher"))):
    """Получить статистику WebSocket соединений (только для админов/учителей)"""
    return manager.get_stats()


# Функция для запуска heartbeat (опционально)
async def start_heartbeat():
    """Запускает периодическую отправку heartbeat для поддержания соединений"""
    while True:
        try:
            await asyncio.sleep(30)  # Каждые 30 секунд

            # Отправляем heartbeat всем активным соединениям
            for class_id, connections in manager.class_connections.items():
                for conn in connections:
                    try:
                        await conn.send_json({
                            "type": "heartbeat",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    except:
                        pass

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")