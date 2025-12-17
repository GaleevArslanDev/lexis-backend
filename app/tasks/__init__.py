# Инициализация задач Celery
from .celery_app import celery_app
from .image_processing import (
    process_assessment_image,
    batch_process_images,
    retry_failed_images
)

__all__ = [
    "celery_app",
    "process_assessment_image",
    "batch_process_images",
    "retry_failed_images"
]