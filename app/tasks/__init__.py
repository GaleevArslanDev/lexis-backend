# Инициализация задач Celery
from .image_processing import (
    process_assessment_image,
    batch_process_images,
    retry_failed_images
)


def process_image_sync(image_id, session):
    """Заглушка для синхронной обработки"""
    from ..routers.assessit import process_image_sync as actual_process
    return actual_process(image_id, session)


__all__ = [
    "process_image_sync",
    "process_assessment_image",
    "batch_process_images",
    "retry_failed_images"
]
