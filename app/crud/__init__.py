# Инициализация CRUD модулей
from .user import get_user_by_email, create_user, get_user_by_id
from .classes import *
from .assignment import create_assignment
from .school import *
from .logs import *
from .mistakes import *
from .assessment import *

__all__ = [
    "get_user_by_email",
    "create_user",
    "get_user_by_id",
    "create_assignment",
    "create_assessment_image",
    "get_assessment_image",
    "update_image_status",
    "create_recognized_solution",
    "get_recognized_solution",
    "create_assessment_result",
    "get_assessment_result",
    "get_class_assessments",
    "get_class_assessment_summary",
    "get_teacher_dashboard_stats",
    "create_training_sample",
    "update_system_metrics"
]