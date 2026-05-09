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
from ..crud.assessment import create_recognized_solution_v1

logger = logging.getLogger(__name__)


class QueueWorker:
    """Фоновый воркер для обработки очереди через pipeline_service."""

    def __init__(self, worker_id: Optional[str] = None):
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.is_running = False

    @contextmanager
    def get_session(self):
        with Session(engine) as session:
            yield session

    async def process_item(self, queue_item):
        # Безопасное логирование - используем str() для защиты от format specifier
        logger.info(
            f"Worker {self.worker_id} processing queue item {queue_item.id} "
            f"(image_id={queue_item.image_id})"
        )

        # Загружаем данные изображения ДО try, чтобы иметь доступ в except
        image = None

        try:
            # --- Получаем данные из БД ---
            with self.get_session() as session:
                from ..models import AssessmentImage, Assignment

                image = session.get(AssessmentImage, queue_item.image_id)
                if not image:
                    raise ValueError(f"Image {queue_item.image_id} not found in DB")

                assignment = session.get(Assignment, image.assignment_id)
                reference_answer = assignment.reference_answer if assignment else None

            # --- Проверяем наличие файла ---
            if not image.original_image_path or not os.path.exists(image.original_image_path):
                # Безопасное форматирование пути
                image_path = image.original_image_path or "None"
                raise ValueError(
                    f"Image file not found on disk: {image_path}. "
                    "Возможно, /tmp был очищен после рестарта контейнера."
                )

            # --- Запускаем ML-пайплайн напрямую (без HTTP) ---
            from ..processing.pipeline_service import pipeline_service
            from ..processing.schemas import OCRRequest

            request = OCRRequest(
                image_path=image.original_image_path,
                reference_answer=reference_answer,
                reference_solution=assignment.reference_solution if assignment else None,  # NEW
            )

            logger.info(f"Queue item {queue_item.id}: calling pipeline_service...")
            response = await pipeline_service.process_assessment(request)

            if response.success and response.assessment:
                assessment = response.assessment
                # Сериализуем в dict (mode='json' преобразует datetime → str)
                assessment_dict = assessment.model_dump(mode='json')

                with self.get_session() as session:
                    solution = create_recognized_solution_v1(
                        session=session,
                        image_id=queue_item.image_id,
                        assessment_data=assessment_dict,
                    )
                    mark_completed(
                        session,
                        queue_item.id,
                        {
                            "solution_id": solution.id,
                            "confidence_score": assessment.confidence_score,
                            "mark_score": assessment.mark_score,
                            "confidence_level": assessment.confidence_level,
                        },
                    )

                await self._send_notification(
                    class_id=image.class_id,
                    work_id=queue_item.image_id,
                    status="completed",
                    data=assessment_dict,
                )
                logger.info(f"Queue item {queue_item.id} processed successfully")

            else:
                error_msg = response.error if response.error else "Pipeline returned no result"
                raise Exception(error_msg)

        except Exception as e:
            # Безопасное логирование ошибки
            error_msg = str(e)
            # Экранируем потенциальные specifier символы
            safe_error_msg = error_msg.replace('%', '%%') if '%' in error_msg else error_msg
            logger.error(f"Error processing queue item {queue_item.id}: {safe_error_msg}", exc_info=True)

            with self.get_session() as session:
                should_retry = queue_item.retry_count < queue_item.max_retries - 1
                mark_failed(session, queue_item.id, safe_error_msg, should_retry)

            await self._send_notification(
                class_id=image.class_id if image else None,
                work_id=queue_item.image_id,
                status="failed",
                error=safe_error_msg,
            )

    async def _send_notification(
        self,
        class_id: Optional[int],
        work_id: int,
        status: str,
        data: dict = None,
        error: str = None,
    ):
        if not class_id:
            return
        try:
            from ..routers.assessit_ws import manager

            message = {
                "type": "work_status_update",
                "data": {
                    "work_id": work_id,
                    "status": status,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }

            if data:
                message["data"].update({
                    "confidence_score": data.get("confidence_score"),
                    "confidence_level": data.get("confidence_level"),
                    "check_level": f"level_{data.get('confidence_level', 3)}",
                    "mark_score": data.get("mark_score"),
                    "teacher_comment": data.get("teacher_comment"),
                })

            if error:
                message["data"]["error"] = error

            await manager.broadcast_to_class(class_id, message)

        except Exception as e:
            logger.error(f"Failed to send WebSocket notification: {e}")

    async def run_once(self, batch_size: int = 5) -> int:
        with self.get_session() as session:
            items = get_next_pending(session, self.worker_id, batch_size)

        if not items:
            return 0

        logger.info(f"Worker {self.worker_id}: got {len(items)} items")
        for item in items:
            await self.process_item(item)

        return len(items)

    async def run_forever(self, sleep_seconds: int = 2):
        self.is_running = True
        logger.info(f"Worker {self.worker_id} started")

        while self.is_running:
            try:
                processed = await self.run_once()
                await asyncio.sleep(0.5 if processed > 0 else sleep_seconds)
            except Exception as e:
                error_msg = str(e).replace('%', '%%')
                logger.error(f"Error in worker loop: {error_msg}", exc_info=True)
                await asyncio.sleep(5)

        logger.info(f"Worker {self.worker_id} stopped")

    def stop(self):
        self.is_running = False


_queue_worker: Optional[QueueWorker] = None


def get_queue_worker() -> QueueWorker:
    global _queue_worker
    if _queue_worker is None:
        _queue_worker = QueueWorker()
    return _queue_worker