from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

# Создаем экземпляр Celery
celery_app = Celery(
    'lexis_tasks',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    include=['app.tasks.image_processing']
)

# Конфигурация Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Europe/Moscow',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 минут на задачу
    task_soft_time_limit=240,  # 4 минуты мягкий лимит
    worker_max_tasks_per_child=100,
    worker_prefetch_multiplier=1,

    # Роутинг задач
    task_routes={
        'app.tasks.image_processing.process_assessment_image': {
            'queue': 'image_processing'
        },
        'app.tasks.image_processing.batch_process_images': {
            'queue': 'batch_processing'
        }
    },

    # Расписание (если нужно периодические задачи)
    beat_schedule={
        'cleanup-old-tasks': {
            'task': 'app.tasks.cleanup.cleanup_old_tasks',
            'schedule': 3600.0,  # Каждый час
        },
    }
)

# Импортируем задачи
from . import image_processing

if __name__ == '__main__':
    celery_app.start()