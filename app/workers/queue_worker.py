# app/workers/queue_worker.py
import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Optional
from sqlmodel import Session
from contextlib import contextmanager

from ..db import engine
from ..crud.queue import get_next_pending, mark_completed, mark_failed
from ..ocr_client import get_ocr_client_v1
from ..crud.assessment import create_recognized_solution_v1

logger = logging.getLogger(__name__)


class QueueWorker:
    """Фоновый воркер для обработки очереди"""

    def __init__(self, worker_id: Optional[str] = None):
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.is_running = False
        self.current_task = None

    @contextmanager
    def get_session(self):
        """Получить сессию БД"""
        with Session(engine) as session:
            yield session

    async def process_item(self, queue_item):
        """Обработать один элемент очереди"""

        logger.info(f"Worker {self.worker_id} processing queue item {queue_item.id} (image: {queue_item.image_id})")

        try:
            # Получаем данные изображения
            with self.get_session() as session:
                from ..models import AssessmentImage, Assignment

                image = session.get(AssessmentImage, queue_item.image_id)
                if not image:
                    raise ValueError(f"Image {queue_item.image_id} not found")

                # Получаем данные задания
                assignment = session.get(Assignment, image.assignment_id)

                # Загружаем изображение (нужно хранить в БД или файловой системе)
                # Здесь предполагаем, что изображения хранятся в файловой системе
                image_bytes = None
                if image.original_image_path and os.path.exists(image.original_image_path):
                    with open(image.original_image_path, 'rb') as f:
                        image_bytes = f.read()
                else:
                    # Если файла нет, можно хранить в БД как LargeBinary
                    # Для простоты пока пропускаем
                    raise ValueError(f"Image file not found: {image.original_image_path}")

            # Отправляем на OCR
            client = get_ocr_client_v1()
            result = client.assess_solution(
                image_bytes=image_bytes,
                filename=image.file_name,
                reference_answer=assignment.reference_answer if assignment else None,
                reference_formulas=[
                    assignment.reference_solution] if assignment and assignment.reference_solution else None
            )

            if result.get("success") and result.get("assessment"):
                assessment = result["assessment"]

                # Сохраняем результат
                with self.get_session() as session:
                    solution = create_recognized_solution_v1(
                        session=session,
                        image_id=queue_item.image_id,
                        assessment_data=assessment
                    )

                    # Отмечаем как выполненное
                    mark_completed(
                        session,
                        queue_item.id,
                        {
                            "solution_id": solution.id,
                            "confidence": assessment.get("confidence_score"),
                            "mark_score": assessment.get("mark_score")
                        }
                    )

                    # Отправляем уведомление через WebSocket
                    await self.send_notification(
                        class_id=image.class_id,
                        work_id=queue_item.image_id,
                        status="completed",
                        data=assessment
                    )

                logger.info(f"✅ Successfully processed queue item {queue_item.id}")

            else:
                error_msg = result.get("error", "Unknown OCR error")
                raise Exception(error_msg)

        except Exception as e:
            logger.error(f"❌ Error processing queue item {queue_item.id}: {e}")

            with self.get_session() as session:
                should_retry = queue_item.retry_count < queue_item.max_retries - 1
                mark_failed(session, queue_item.id, str(e), should_retry)

                # Отправляем уведомление об ошибке
                await self.send_notification(
                    class_id=image.class_id if 'image' in locals() else None,
                    work_id=queue_item.image_id,
                    status="failed",
                    error=str(e)
                )

    async def send_notification(self, class_id: Optional[int], work_id: int, status: str, data: dict = None,
                                error: str = None):
        """Отправить уведомление через WebSocket"""
        try:
            from ..routers.assessit_ws import manager

            if class_id:
                message = {
                    "type": "work_status_update",
                    "data": {
                        "work_id": work_id,
                        "status": status,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }

                if data:
                    message["data"].update({
                        "confidence_score": data.get("confidence_score"),
                        "check_level": f"level_{data.get('confidence_level', 3)}",
                        "mark_score": data.get("mark_score")
                    })

                if error:
                    message["data"]["error"] = error

                await manager.broadcast_to_class(class_id, message)

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    async def run_once(self, batch_size: int = 5):
        """Выполнить один цикл обработки"""

        worker_id = self.worker_id

        with self.get_session() as session:
            items = get_next_pending(session, worker_id, batch_size)

        if not items:
            return 0

        logger.info(f"Worker {worker_id} got {len(items)} items to process")

        for item in items:
            await self.process_item(item)

        return len(items)

    async def run_forever(self, sleep_seconds: int = 2):
        """Запустить бесконечный цикл обработки"""

        self.is_running = True
        logger.info(f"🚀 Worker {self.worker_id} started")

        while self.is_running:
            try:
                processed = await self.run_once()

                if processed == 0:
                    # Нет задач - спим
                    await asyncio.sleep(sleep_seconds)
                else:
                    # Были задачи - проверяем сразу еще
                    await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(5)

        logger.info(f"🛑 Worker {self.worker_id} stopped")

    def stop(self):
        """Остановить воркер"""
        self.is_running = False


# Глобальный экземпляр воркера
_queue_worker = None


def get_queue_worker() -> QueueWorker:
    """Получить или создать экземпляр воркера"""
    global _queue_worker
    if _queue_worker is None:
        _queue_worker = QueueWorker()
    return _queue_worker