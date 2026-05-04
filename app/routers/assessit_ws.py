# app/routers/assessit_ws.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, status
from typing import Dict, Set, Optional, List, Any
import asyncio
import json
import logging
from datetime import datetime
from sqlmodel import Session
from jose import jwt

from ..db import engine, get_session
from ..dependencies_util import require_role
from ..models import User, Class
from ..utils.security import SECRET_KEY, ALGORITHM
from ..crud.queue import get_class_queue_items

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """
    Менеджер WebSocket соединений (in-memory, без Redis).
    """

    def __init__(self):
        self.class_connections: Dict[int, Set[WebSocket]] = {}
        self.user_connections: Dict[int, Set[WebSocket]] = {}
        self.active_connections: Set[WebSocket] = set()

        # Очереди накопленных сообщений (на случай отсутствия слушателей)
        self.class_message_queues: Dict[int, asyncio.Queue] = {}
        self.queue_tasks: Dict[int, asyncio.Task] = {}

        # Атрибуты для совместимости с assessit.py
        self.class_queues: Dict[int, List[Any]] = {}
        self.processing_tasks: Dict[int, asyncio.Task] = {}
        self.queue_locks: Dict[int, asyncio.Lock] = {}

        logger.info("ConnectionManager initialized (in-memory mode)")

    # ------------------------------------------------------------------
    # Подключение / отключение
    # ------------------------------------------------------------------

    async def connect(self, websocket: WebSocket, class_id: int, user_id: int):
        await websocket.accept()
        self.active_connections.add(websocket)

        if class_id not in self.class_connections:
            self.class_connections[class_id] = set()
            self.class_message_queues[class_id] = asyncio.Queue()
            self.class_queues[class_id] = []
            self.queue_locks[class_id] = asyncio.Lock()
            self.queue_tasks[class_id] = asyncio.create_task(
                self._process_class_queue(class_id)
            )

        self.class_connections[class_id].add(websocket)

        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(websocket)

        await websocket.send_json({
            "type": "connected",
            "data": {
                "class_id": class_id,
                "user_id": user_id,
                "message": "Connected to class updates",
                "timestamp": datetime.utcnow().isoformat(),
            },
        })

        await self._send_queue_status(class_id, websocket)
        logger.info(
            f"User {user_id} connected to class {class_id}. "
            f"Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket, class_id: int, user_id: int):
        self.active_connections.discard(websocket)

        if class_id in self.class_connections:
            self.class_connections[class_id].discard(websocket)
            if not self.class_connections[class_id] and class_id in self.queue_tasks:
                self.queue_tasks[class_id].cancel()
                del self.queue_tasks[class_id]
                logger.info(f"Stopped queue processor for class {class_id} (no listeners)")

        if user_id in self.user_connections:
            self.user_connections[user_id].discard(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]

        logger.info(f"User {user_id} disconnected from class {class_id}")

    # ------------------------------------------------------------------
    # Рассылка
    # ------------------------------------------------------------------

    async def broadcast_to_class(self, class_id: int, message: dict):
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()

        if class_id in self.class_connections and self.class_connections[class_id]:
            disconnected: Set[WebSocket] = set()
            for conn in self.class_connections[class_id]:
                try:
                    await conn.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send to client: {e}")
                    disconnected.add(conn)

            for conn in disconnected:
                self.class_connections[class_id].discard(conn)
                self.active_connections.discard(conn)

        elif class_id in self.class_message_queues:
            try:
                if self.class_message_queues[class_id].qsize() < 100:
                    await self.class_message_queues[class_id].put(message)
            except Exception as e:
                logger.error(f"Failed to queue message for class {class_id}: {e}")

    async def send_to_user(self, user_id: int, message: dict):
        if user_id not in self.user_connections:
            return
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()

        disconnected: Set[WebSocket] = set()
        for conn in self.user_connections[user_id]:
            try:
                await conn.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to user {user_id}: {e}")
                disconnected.add(conn)

        for conn in disconnected:
            self.active_connections.discard(conn)
            self.user_connections[user_id].discard(conn)

    # ------------------------------------------------------------------
    # Фоновая обработка очереди сообщений
    # ------------------------------------------------------------------

    async def _process_class_queue(self, class_id: int):
        logger.info(f"Started message queue processor for class {class_id}")
        while True:
            try:
                try:
                    message = await asyncio.wait_for(
                        self.class_message_queues[class_id].get(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    continue

                if class_id in self.class_connections and self.class_connections[class_id]:
                    await self.broadcast_to_class(class_id, message)
                else:
                    await self.class_message_queues[class_id].put(message)
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info(f"Message queue processor for class {class_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Error in message queue processor for class {class_id}: {e}")
                await asyncio.sleep(5)

    # ------------------------------------------------------------------
    # Отправка статуса очереди
    # ------------------------------------------------------------------

    async def _send_queue_status(self, class_id: int, websocket: WebSocket):
        """
        ИСПРАВЛЕНО: использует Session(engine), а не Session(get_session).
        get_session — это генератор-зависимость FastAPI, не фабрика сессий.
        """
        try:
            with Session(engine) as session:
                queue_items = get_class_queue_items(session, class_id, limit=20)

                pending = sum(1 for item in queue_items if item["status"] == "pending")
                processing = sum(1 for item in queue_items if item["status"] == "processing")

                await websocket.send_json({
                    "type": "queue_status",
                    "data": {
                        "class_id": class_id,
                        "pending": pending,
                        "processing": processing,
                        "recent_items": queue_items[:10],
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                })
        except Exception as e:
            logger.error(f"Failed to send queue status for class {class_id}: {e}")

    # ------------------------------------------------------------------
    # Публичные хелперы
    # ------------------------------------------------------------------

    async def notify_work_update(self, work_id: int, class_id: int, status: str, data: dict = None):
        message = {
            "type": "work_status_update",
            "data": {
                "work_id": work_id,
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }
        if data:
            message["data"].update(data)
        await self.broadcast_to_class(class_id, message)

    async def notify_batch_upload(self, class_id: int, batch_data: dict):
        message = {
            "type": "batch_upload",
            "data": {**batch_data, "timestamp": datetime.utcnow().isoformat()},
        }
        await self.broadcast_to_class(class_id, message)

    def get_stats(self) -> dict:
        return {
            "total_connections": len(self.active_connections),
            "active_classes": len(self.class_connections),
            "active_users": len(self.user_connections),
            "class_queues": {
                str(cid): len(self.class_queues.get(cid, []))
                for cid in self.class_connections
            },
        }

    async def get_queue_status(self, class_id: int) -> dict:
        """
        ИСПРАВЛЕНО: использует Session(engine).
        """
        try:
            with Session(engine) as session:
                queue_items = get_class_queue_items(session, class_id, limit=50)

                pending = sum(1 for item in queue_items if item["status"] == "pending")
                processing = sum(1 for item in queue_items if item["status"] == "processing")
                completed = sum(1 for item in queue_items if item["status"] == "completed")
                failed = sum(1 for item in queue_items if item["status"] == "failed")

                items = [
                    {
                        "work_id": item["work_id"],
                        "student_id": item.get("student_id"),
                        "student_name": item.get("student_name", "Unknown"),
                        "position": i + 1,
                        "status": item["status"],
                        "queued_at": item["created_at"].isoformat() if item.get("created_at") else None,
                        "started_at": item["started_at"].isoformat() if item.get("started_at") else None,
                        "completed_at": item["completed_at"].isoformat() if item.get("completed_at") else None,
                        "error": item.get("error"),
                    }
                    for i, item in enumerate(queue_items)
                ]

                return {
                    "queue_size": len(queue_items),
                    "pending": pending,
                    "processing": processing,
                    "completed": completed,
                    "failed": failed,
                    "items": items,
                    "estimated_wait_seconds": pending * 6,
                    "avg_processing_time": 6.0,
                }
        except Exception as e:
            logger.error(f"Error getting queue status for class {class_id}: {e}")
            return {
                "queue_size": 0, "pending": 0, "processing": 0,
                "completed": 0, "failed": 0, "items": [],
                "estimated_wait_seconds": 0, "avg_processing_time": 0,
                "error": str(e),
            }


# ---------------------------------------------------------------------------
# Глобальный менеджер
# ---------------------------------------------------------------------------
manager = ConnectionManager()


async def get_current_user_ws(token: str, session: Session) -> Optional[User]:
    """Аутентификация для WebSocket соединений."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
        return session.get(User, user_id)
    except Exception as e:
        logger.error(f"WebSocket auth error: {e}")
        return None


# ---------------------------------------------------------------------------
# WebSocket эндпоинт
# ---------------------------------------------------------------------------

@router.websocket("/ws/class/{class_id}")
async def websocket_class(
        websocket: WebSocket,
        class_id: int,
        token: str = Query(...),
):
    """
    WebSocket для real-time обновлений по классу.
    Не использует Depends(get_session) — сессия создаётся вручную через engine.
    """
    # Аутентификация
    with Session(engine) as session:
        user = await get_current_user_ws(token, session)
        if not user:
            logger.warning(f"WebSocket auth failed for class {class_id}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        class_obj = session.get(Class, class_id)
        if not class_obj or class_obj.teacher_id != user.id:
            logger.warning(
                f"User {user.id} tried to connect to class {class_id} without permission"
            )
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        user_id = user.id  # сохраняем до закрытия сессии

    try:
        await manager.connect(websocket, class_id, user_id)

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)
                msg_type = message.get("type")

                if msg_type == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                elif msg_type == "request_queue_status":
                    # ИСПРАВЛЕНО: Session(engine)
                    with Session(engine) as db_session:
                        queue_items = get_class_queue_items(db_session, class_id, limit=20)
                    await websocket.send_json({
                        "type": "queue_status",
                        "data": {
                            "class_id": class_id,
                            "items": queue_items,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    })

                elif msg_type == "subscribe_work":
                    work_id = message.get("work_id")
                    await websocket.send_json({
                        "type": "subscribed",
                        "data": {
                            "work_id": work_id,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    })

                else:
                    logger.debug(f"Unknown WS message type: {msg_type} from user {user_id}")

            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({
                        "type": "ping",
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                except Exception:
                    break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user {user_id}, class {class_id}")
    except Exception as e:
        logger.error(f"WebSocket error for class {class_id}: {e}")
    finally:
        manager.disconnect(websocket, class_id, user_id)


@router.get("/ws/stats")
async def websocket_stats(current_user: User = Depends(require_role("teacher"))):
    """Статистика WebSocket соединений."""
    return manager.get_stats()


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

async def start_heartbeat():
    """Периодически пингует все активные соединения."""
    while True:
        try:
            await asyncio.sleep(30)
            for connections in manager.class_connections.values():
                for conn in connections:
                    try:
                        await conn.send_json({
                            "type": "heartbeat",
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                    except Exception:
                        pass
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")