import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from typing import List, Optional
import logging
import gc
from sqlalchemy import text

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Query, Form
from fastapi.responses import JSONResponse
from sqlmodel import Session, select

from .assessit_ws import manager, add_works_to_queue
from .. import app
from ..crud.queue import get_queue_stats, get_class_queue_items
from ..db import get_session
from ..dependencies.auth import get_current_user
from ..dependencies_util import require_role
from ..exceptions import AppException, ErrorCode
from ..models import User, AssessmentImage, Assignment, Class, ClassStudentLink, RecognizedSolution
from ..crud.assessment import (
    create_assessment_image,
    get_assessment_image,
    get_recognized_solution,
    create_assessment_result,
    get_class_assessments,
    get_class_assessment_summary,
    get_teacher_dashboard_stats,
    update_image_status,
    create_recognized_solution_v1, create_teacher_verification, get_common_mistakes,
    get_class_assessments_paginated,  # Новый метод
)
from ..ocr_client import get_ocr_client_v1  # Новый клиент
from ..schemas import (
    UploadWorkResponse,
    ProcessedWorkResponse,
    DetailedAnalysisResponse,
    TeacherVerificationRequest,
    TeacherVerificationResponse,
    ClassAssessmentSummary,
    StudentWorkItem,
    BatchUploadResponse,
    DashboardStats,
    CommonError, SystemHealthResponse
)
from ..workers.queue_worker import get_queue_worker

logger = logging.getLogger(__name__)

_background_workers = set()

router = APIRouter(prefix="/assessit", tags=["assessit"])

# Инициализация клиента v1
ocr_client_v1 = get_ocr_client_v1()


@app.on_event("startup")
async def startup_queue_workers():
    """Запустить фоновые воркеры при старте"""
    worker = get_queue_worker()
    # Запускаем воркер в фоне
    task = asyncio.create_task(worker.run_forever())
    _background_workers.add(task)
    task.add_done_callback(_background_workers.discard)
    logger.info("Started background queue worker")


@app.on_event("shutdown")
async def shutdown_queue_workers():
    """Остановить воркеры при завершении"""
    worker = get_queue_worker()
    worker.stop()

    # Ждем завершения задач
    if _background_workers:
        await asyncio.gather(*_background_workers, return_exceptions=True)

@router.get("/dashboard/{teacher_id}", response_model=DashboardStats)
async def get_dashboard(
        teacher_id: int,
        days: int = Query(30, ge=1, le=90),
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """Главный экран - дашборд учителя с графиками (Рис. 2, 6)"""

    # Проверяем права (учитель может смотреть только свой дашборд)
    if current_user.id != teacher_id and current_user.role != "admin":
        raise AppException(
            status_code=403,
            error_code=ErrorCode.ACCESS_DENIED,
            message="You can only view your own dashboard",
            details={"teacher_id": teacher_id, "current_user": current_user.id}
        )

    stats = get_teacher_dashboard_stats(session, teacher_id, days)
    return stats


@router.get("/class/{class_id}/summary", response_model=ClassAssessmentSummary)
async def get_class_summary(
        class_id: int,
        assignment_id: Optional[int] = None,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """Сводка по классу для экрана результатов (Рис. 5)"""

    # Проверяем права
    class_obj = session.get(Class, class_id)
    if not class_obj:
        raise AppException(
            status_code=404,
            error_code=ErrorCode.NOT_FOUND,
            message="Class not found",
            details={"class_id": class_id}
        )

    if class_obj.teacher_id != current_user.id:
        raise AppException(
            status_code=403,
            error_code=ErrorCode.ACCESS_DENIED,
            message="Not your class",
            details={"class_id": class_id}
        )

    summary = get_class_assessment_summary(session, class_id, assignment_id)
    return summary


@router.get("/class/{class_id}/works", response_model=dict)
async def get_class_works(
        class_id: int,
        assignment_id: Optional[int] = None,
        status: Optional[str] = Query(None, regex="^(pending|processing|processed|error)$"),
        check_level: Optional[str] = Query(None, regex="^(level_1|level_2|level_3)$"),
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """Список работ класса с пагинацией и фильтрацией (для экрана результатов)"""

    # Проверяем права
    class_obj = session.get(Class, class_id)
    if not class_obj or class_obj.teacher_id != current_user.id:
        raise AppException(
            status_code=403,
            error_code=ErrorCode.ACCESS_DENIED,
            message="Not your class",
            details={"class_id": class_id}
        )

    result = get_class_assessments_paginated(
        session, class_id, assignment_id, status, check_level,
        page, page_size
    )

    # Добавляем статус очереди из WebSocket
    if class_id in manager.class_queues:
        queue_size = manager.class_queues[class_id].qsize()
        result["queue_status"] = {
            "size": queue_size,
            "estimated_wait_seconds": queue_size * 6  # 6 секунд на работу
        }

    return result


@router.get("/class/{class_id}/common-mistakes", response_model=List[CommonError])
async def get_class_common_mistakes(
        class_id: int,
        assignment_id: Optional[int] = None,
        limit: int = Query(10, ge=1, le=50),
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """Типичные ошибки класса для аналитики (Рис. 6)"""

    # Проверяем права
    class_obj = session.get(Class, class_id)
    if not class_obj or class_obj.teacher_id != current_user.id:
        raise AppException(
            status_code=403,
            error_code=ErrorCode.ACCESS_DENIED,
            message="Not your class",
            details={"class_id": class_id}
        )

    mistakes = get_common_mistakes(session, class_id, assignment_id, limit)
    return mistakes


@router.post("/work/{work_id}/verify", response_model=TeacherVerificationResponse)
async def verify_work(
        work_id: int,
        verification: TeacherVerificationRequest,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """Подтверждение/корректировка работы учителем (кнопки действий на Рис. 5)"""

    # Получаем работу
    image = get_assessment_image(session, work_id)
    if not image:
        raise AppException(
            status_code=404,
            error_code=ErrorCode.NOT_FOUND,
            message="Work not found",
            details={"work_id": work_id}
        )

    # Проверяем права
    class_obj = session.get(Class, image.class_id)
    if not class_obj or class_obj.teacher_id != current_user.id:
        raise AppException(
            status_code=403,
            error_code=ErrorCode.ACCESS_DENIED,
            message="Not your class",
            details={"work_id": work_id, "class_id": image.class_id}
        )

    # Получаем решение
    solution = get_recognized_solution(session, work_id)
    if not solution:
        raise AppException(
            status_code=404,
            error_code=ErrorCode.NOT_FOUND,
            message="Solution not found for this work",
            details={"work_id": work_id}
        )

    # Создаем результат проверки
    result = create_teacher_verification(
        session=session,
        solution_id=solution.id,
        teacher_id=current_user.id,
        verdict=verification.verdict,
        score=verification.score,
        feedback=verification.feedback,
        corrected_text=verification.corrected_text,
        corrected_answer=verification.corrected_answer
    )

    # Уведомляем через WebSocket (если кто-то подключен)
    await manager.broadcast_to_class(image.class_id, {
        "type": "work_verified",
        "data": {
            "work_id": work_id,
            "verdict": verification.verdict,
            "score": verification.score
        },
        "timestamp": datetime.utcnow().isoformat()
    })

    return TeacherVerificationResponse(
        result_id=result.id,
        work_id=work_id,
        teacher_id=current_user.id,
        verdict=verification.verdict,
        score=verification.score,
        feedback=verification.feedback,
        verified_at=result.verified_at or datetime.utcnow()
    )


@router.get("/queue/status", response_model=dict)
async def get_queue_status(
        session: Session = Depends(get_session),
        current_user: User = Depends(get_current_user)
):
    """Получить глобальный статус очереди"""
    stats = get_queue_stats(session)

    # Добавляем информацию о воркере
    worker = get_queue_worker()
    stats["worker"] = {
        "id": worker.worker_id,
        "is_running": worker.is_running
    }

    return stats


@router.get("/queue/class/{class_id}", response_model=list)
async def get_class_queue(
        class_id: int,
        limit: int = 50,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """Получить очередь для конкретного класса"""

    # Проверка прав
    class_obj = session.get(Class, class_id)
    if not class_obj or class_obj.teacher_id != current_user.id:
        raise AppException(
            status_code=403,
            error_code=ErrorCode.ACCESS_DENIED,
            message="Not your class"
        )

    items = get_class_queue_items(session, class_id, limit)
    return items


@router.get("/debug/queue/{class_id}")
async def debug_queue(
        class_id: int,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """Отладочный эндпоинт для проверки состояния очереди"""

    # Проверяем права
    class_obj = session.get(Class, class_id)
    if not class_obj or class_obj.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your class")

    # Получаем статус из менеджера
    queue_exists = class_id in manager.class_queues
    processor_exists = class_id in manager.processing_tasks
    processor_alive = processor_exists and not manager.processing_tasks[class_id].done()
    lock_exists = class_id in manager.queue_locks

    queue_size = len(manager.class_queues.get(class_id, [])) if queue_exists else 0

    # Детальная информация о процессоре
    processor_info = None
    if processor_exists:
        task = manager.processing_tasks[class_id]
        processor_info = {
            "done": task.done(),
            "cancelled": task.cancelled(),
            "exception": str(task.exception()) if task.done() and task.exception() else None
        }

    return {
        "class_id": class_id,
        "queue": {
            "exists": queue_exists,
            "size": queue_size
        },
        "processor": {
            "exists": processor_exists,
            "alive": processor_alive,
            "info": processor_info
        },
        "lock_exists": lock_exists,
        "items": [
            {
                "work_id": item.work_id,
                "student_name": item.student_name,
                "status": item.status,
                "position": item.position,
                "queued_at": item.queued_at.isoformat(),
                "started_at": item.started_at.isoformat() if item.started_at else None
            }
            for item in manager.class_queues.get(class_id, [])
        ] if queue_exists else []
    }


@router.post("/batch-upload", response_model=BatchUploadResponse)
async def batch_upload_works(
        assignment_id: int = Form(...),
        class_id: int = Form(...),
        files: List[UploadFile] = File(...),
        students_json: str = Form(...),  # JSON строка с маппингом
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher")),
        background_tasks: BackgroundTasks = None
):
    """
    Пакетная загрузка работ для всего класса

    Формат students_json:
    [
        {"filename": "photo1.jpg", "student_id": 123},
        {"filename": "photo2.jpg", "student_id": 456},
        ...
    ]

    Все работы автоматически попадают в очередь на основе БД.
    Статус обработки можно отслеживать через WebSocket.
    """
    # Для сбора созданных записей и ошибок
    created_images = []
    failed_files = []
    queue_items = []
    student_names = {}

    try:
        # Парсим маппинг студентов
        try:
            student_mapping = json.loads(students_json)
            if not isinstance(student_mapping, list):
                raise HTTPException(
                    status_code=400,
                    detail="students_json must be a list"
                )
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON in students_json. Expected format: [{'filename': 'photo1.jpg', 'student_id': 123}]"
            )

        # Создаем словарь filename -> student_id
        file_to_student = {item["filename"]: item["student_id"] for item in student_mapping}

        # Проверяем, что все файлы имеют маппинг
        uploaded_filenames = [file.filename for file in files]
        for filename in uploaded_filenames:
            if filename not in file_to_student:
                raise HTTPException(
                    status_code=400,
                    detail=f"No student mapping for file: {filename}"
                )

        # Проверяем, что класс принадлежит учителю
        class_obj = session.get(Class, class_id)
        if not class_obj:
            raise HTTPException(status_code=404, detail="Class not found")

        if class_obj.teacher_id != current_user.id:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to upload to this class"
            )

        # Получаем задание
        assignment = session.get(Assignment, assignment_id)
        if not assignment:
            raise HTTPException(status_code=404, detail="Assignment not found")

        # Проверяем, что задание принадлежит этому классу
        if assignment.class_id != class_id:
            raise HTTPException(
                status_code=400,
                detail="Assignment does not belong to this class"
            )

        # Получаем всех учеников класса для проверки
        students = session.exec(
            select(User)
            .join(ClassStudentLink)
            .where(ClassStudentLink.class_id == class_id)
            .where(User.role == "student")
        ).all()

        valid_student_ids = {s.id for s in students}

        # Создаем словарь student_id -> имя для быстрого доступа
        student_names = {s.id: f"{s.name} {s.surname}" for s in students}

        # Проверяем, что все указанные student_id действительно в классе
        for filename, student_id in file_to_student.items():
            if student_id not in valid_student_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"Student {student_id} is not in class {class_id} (file: {filename})"
                )

        # Обрабатываем каждый файл
        for file in files:
            try:
                # Определяем student_id для этого файла
                student_id = file_to_student.get(file.filename)

                if not student_id:
                    raise ValueError(f"No student_id mapping for file {file.filename}")

                # Читаем содержимое файла (макс 10MB)
                file_content = await file.read()
                if len(file_content) > 10 * 1024 * 1024:  # 10MB limit
                    raise ValueError(f"File {file.filename} exceeds 10MB limit")

                # Проверяем, что файл не пустой
                if len(file_content) == 0:
                    raise ValueError(f"File {file.filename} is empty")

                # Проверяем MIME тип
                allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf']
                if file.content_type not in allowed_types:
                    raise ValueError(f"File {file.filename} has unsupported type: {file.content_type}")

                # Создаем запись в БД
                image = create_assessment_image(
                    session=session,
                    assignment_id=assignment_id,
                    student_id=student_id,
                    file_name=file.filename,
                    file_size=len(file_content),
                    mime_type=file.content_type or "application/octet-stream",
                    class_id=class_id
                )
                created_images.append(image)

                # Важно: Сохраняем файл во временное хранилище
                # Для Render можно использовать /tmp или постоянное хранилище
                file_path = save_image_file(file_content, image.id, file.filename)

                # Обновляем путь к файлу в БД
                image.original_image_path = file_path
                session.add(image)
                session.commit()

                # Добавляем в очередь через БД вместо Celery
                from ..crud.queue import add_to_queue
                queue_item = add_to_queue(
                    session=session,
                    image_id=image.id,
                    priority=0,  # Обычный приоритет
                    max_retries=3
                )
                queue_items.append(queue_item)

                # Обновляем статус
                update_image_status(session, image.id, "queued")

                logger.info(f"Added work {image.id} to queue for student {student_id}")

            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                failed_files.append({
                    "filename": file.filename,
                    "error": str(e)
                })
                # Продолжаем с другими файлами

        # Финальный коммит всех изменений
        session.commit()

        # Проверяем, есть ли успешные загрузки
        if not created_images:
            error_detail = "No valid works to process"
            if failed_files:
                error_detail += f". Failed files: {failed_files}"
            raise HTTPException(
                status_code=400,
                detail=error_detail
            )

        # Получаем статистику очереди для оценки времени ожидания
        from ..crud.queue import get_queue_stats
        queue_stats = get_queue_stats(session)

        # Оцениваем позицию в очереди (сколько задач перед нашими)
        pending_count = queue_stats.get("pending", 0)

        # Среднее время обработки (из статистики или по умолчанию 6 секунд)
        avg_processing_time = queue_stats.get("avg_processing_seconds", 6.0)

        # Оцениваем время ожидания
        # Если наши задачи в начале очереди, то ждать меньше
        # Берем минимум между всеми pending и нашими задачами
        estimated_wait = min(pending_count, len(created_images)) * avg_processing_time

        # Формируем ответ
        response = BatchUploadResponse(
            batch_id=str(uuid.uuid4()),
            total_submitted=len(created_images),
            queue_position=pending_count - len(created_images) + 1,  # Примерная позиция
            estimated_wait_seconds=int(estimated_wait),
            status="queued",
            task_ids=[str(item.id) for item in queue_items]  # Используем ID из очереди
        )

        # Отправляем уведомление через WebSocket (без Redis)
        try:
            from .assessit_ws import manager

            # Формируем список успешно добавленных работ
            successful_works = []
            for i, image in enumerate(created_images):
                successful_works.append({
                    "work_id": image.id,
                    "student_id": image.student_id,
                    "student_name": student_names.get(image.student_id, "Unknown"),
                    "queue_id": queue_items[i].id if i < len(queue_items) else None
                })

            notification = {
                "type": "new_works_added",
                "data": {
                    "count": len(successful_works),
                    "works": successful_works,
                    "queue_stats": {
                        "pending": queue_stats.get("pending", 0),
                        "processing": queue_stats.get("processing", 0),
                        "estimated_wait_seconds": estimated_wait
                    },
                    "batch_id": response.batch_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

            # Отправляем через WebSocket менеджер (прямая рассылка)
            await manager.broadcast_to_class(class_id, notification)

        except Exception as e:
            logger.error(f"Failed to send WebSocket notification: {e}")

        # Если были ошибки, добавляем их в ответ
        if failed_files:
            # Создаем словарь с ответом и добавляем failed_files
            response_dict = response.dict()
            response_dict["failed_files"] = failed_files
            response_dict["partial_success"] = True
            response_dict["message"] = f"Successfully uploaded {len(created_images)} files, {len(failed_files)} failed"
            return response_dict

        return response

    except HTTPException:
        # Пробрасываем HTTP исключения
        session.rollback()
        raise
    except Exception as e:
        # Логируем и откатываем транзакцию
        logger.error(f"Batch upload error: {str(e)}", exc_info=True)
        session.rollback()

        # Если были созданы записи в БД, помечаем их как ошибки
        if created_images:
            for image in created_images:
                try:
                    update_image_status(session, image.id, "error", str(e))
                except:
                    pass
            session.commit()

        raise AppException(
            status_code=500,
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Batch upload failed",
            details={
                "error": str(e),
                "failed_files": failed_files if failed_files else [],
                "successful_count": len(created_images) if created_images else 0
            }
        )


# Вспомогательная функция для сохранения файла
def save_image_file(file_content: bytes, image_id: int, filename: str) -> str:
    """
    Сохранить файл изображения на диск
    Для Render используем /tmp или постоянное хранилище
    """
    import os
    from datetime import datetime

    # Создаем директорию для загрузок, если её нет
    upload_dir = os.getenv("UPLOAD_DIR", "/tmp/uploads")
    os.makedirs(upload_dir, exist_ok=True)

    # Создаем поддиректорию по дате
    date_str = datetime.utcnow().strftime("%Y%m%d")
    date_dir = os.path.join(upload_dir, date_str)
    os.makedirs(date_dir, exist_ok=True)

    # Генерируем имя файла: {image_id}_{timestamp}_{original_filename}
    timestamp = datetime.utcnow().strftime("%H%M%S")
    safe_filename = "".join(c for c in filename if c.isalnum() or c in '._-')[:50]
    file_path = os.path.join(date_dir, f"{image_id}_{timestamp}_{safe_filename}")

    # Сохраняем файл
    with open(file_path, "wb") as f:
        f.write(file_content)

    logger.info(f"Saved file to {file_path}")
    return file_path


@router.get("/work-details/{work_id}", response_model=DetailedAnalysisResponse)
async def get_work_details(
        work_id: int,
        include_steps: bool = Query(True, description="Включить детальный анализ шагов"),
        session: Session = Depends(get_session),
        current_user: User = Depends(get_current_user)
):
    """Получить детальный анализ работы"""
    image = get_assessment_image(session, work_id)
    if not image:
        raise HTTPException(status_code=404, detail="Work not found")

    # Проверяем права доступа
    if current_user.role == "student" and image.student_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    if image.status != "processed":
        raise HTTPException(status_code=400, detail="Work not processed yet")

    solution = get_recognized_solution(session, work_id)
    if not solution:
        raise HTTPException(status_code=404, detail="Solution not found")

    # Получаем данные задания для эталонного ответа
    assignment = session.get(Assignment, image.assignment_id)
    reference_answer = assignment.reference_answer if assignment else None

    # Парсим формулы из JSON
    formulas = []
    if solution.extracted_formulas_json:
        try:
            formulas_data = json.loads(solution.extracted_formulas_json)
            formulas = [
                {
                    "latex": f.get("latex", ""),
                    "confidence": f.get("confidence", 0.0),
                    "position": f.get("bbox", {}),
                    "is_correct": None
                }
                for f in formulas_data[:10]
            ]
        except:
            formulas = []

    # Парсим шаги анализа
    problem_areas = []
    if include_steps and solution.steps_analysis_json:
        try:
            steps = json.loads(solution.steps_analysis_json)
            # Преобразуем шаги в проблемные области, где is_correct=False
            for step in steps:
                if not step.get("is_correct", True):
                    problem_areas.append({
                        "step_id": step.get("step_id"),
                        "found_formula": step.get("found_formula"),
                        "expected_formula": step.get("expected_formula"),
                        "description": "Неверная формула или вычисление"
                    })
        except:
            pass

    return DetailedAnalysisResponse(
        work_id=work_id,
        student_id=image.student_id,
        assignment_id=image.assignment_id,
        full_text=solution.extracted_text,
        formulas=formulas,
        extracted_answer=solution.extracted_answer,
        reference_answer=reference_answer,
        ocr_confidence=solution.ocr_confidence or 0.0,
        solution_structure_confidence=solution.solution_structure_confidence or 0.0,
        formula_confidence=solution.formula_confidence or 0.0,
        answer_match_confidence=solution.answer_match_confidence or 0.0,
        total_confidence=solution.total_confidence or 0.0,
        check_level=solution.check_level,
        suggested_grade=solution.suggested_grade,
        auto_feedback=solution.auto_feedback or solution.teacher_comment or "",
        annotated_image_url=image.processed_image_path,
        problem_areas=problem_areas,
        processing_time_ms=solution.processing_time_ms or 0,
        processed_at=solution.recognized_at
    )


@router.get("/health/ocr", response_model=dict)
async def check_ocr_health(
        session: Session = Depends(get_session),
        current_user: User = Depends(get_current_user)
):
    """Проверить состояние OCR API v1"""
    try:
        # Проверяем доступность API
        health = ocr_client_v1.health_check()
        queue_status = ocr_client_v1.get_queue_status()

        return {
            "ocr_api_available": health.get("status") == "healthy",
            "services": health.get("services", {}),
            "queue_status": queue_status,
            "ocr_api_url": os.getenv("OCR_API_URL", "not set"),
            "api_version": "v1"
        }
    except Exception as e:
        logger.error(f"Error checking OCR health: {str(e)}")
        return {
            "ocr_api_available": False,
            "error": str(e),
            "api_version": "v1"
        }


@router.get("/job/{job_id}", response_model=dict)
async def get_job_status(
        job_id: str,
        session: Session = Depends(get_session),
        current_user: User = Depends(get_current_user)
):
    """Получить статус задачи по ID (для асинхронной обработки)"""
    status = ocr_client_v1.get_job_status(job_id)
    return status


@router.get("/health", response_model=SystemHealthResponse)
async def system_health(
        session: Session = Depends(get_session),
        current_user: User = Depends(get_current_user)
):
    """
    Полный health check системы
    Доступен только авторизованным пользователям
    """
    start_time = time.time()
    version = "2.0.0"
    components = {}
    overall_status = "healthy"

    # 1. Проверка БД
    try:
        db_start = time.time()
        result = session.execute(text("SELECT 1")).first()
        db_latency = (time.time() - db_start) * 1000

        components["database"] = {
            "status": "healthy" if result else "degraded",
            "latency_ms": round(db_latency, 2),
            "message": "Connected" if result else "Query failed"
        }
        if not result:
            overall_status = "degraded"
    except Exception as e:
        components["database"] = {
            "status": "unhealthy",
            "latency_ms": None,
            "message": str(e)
        }
        overall_status = "unhealthy"

    # 2. Проверка OCR API
    try:
        ocr_start = time.time()
        from ..ocr_client import get_ocr_client_v1
        client = get_ocr_client_v1()
        health = client.health_check()
        ocr_latency = (time.time() - ocr_start) * 1000

        ocr_status = health.get("status") == "healthy"
        components["ocr_api"] = {
            "status": "healthy" if ocr_status else "degraded",
            "latency_ms": round(ocr_latency, 2),
            "services": health.get("services", {}),
            "message": "Connected" if ocr_status else "API degraded"
        }
        if not ocr_status:
            overall_status = "degraded" if overall_status == "healthy" else overall_status
    except Exception as e:
        components["ocr_api"] = {
            "status": "unhealthy",
            "latency_ms": None,
            "message": str(e)
        }
        overall_status = "unhealthy"

    # 3. Проверка WebSocket
    try:
        ws_connections = len(manager.active_connections)
        components["websocket"] = {
            "status": "healthy",
            "connections": ws_connections,
            "active_classes": len(manager.class_connections),
            "message": f"{ws_connections} active connections"
        }
    except Exception as e:
        components["websocket"] = {
            "status": "degraded",
            "connections": 0,
            "message": str(e)
        }
        overall_status = "degraded"

    # 4. Проверка очередей
    try:
        total_queue_size = sum(len(q) for q in manager.class_queues.values())
        processing_count = 0
        for q in manager.class_queues.values():
            processing_count += sum(1 for item in q if item.status == "processing")

        components["queue"] = {
            "status": "healthy",
            "size": total_queue_size,
            "processing": processing_count,
            "active_queues": len(manager.class_queues),
            "message": f"{total_queue_size} items in queues"
        }
    except Exception as e:
        components["queue"] = {
            "status": "degraded",
            "size": 0,
            "message": str(e)
        }
        overall_status = "degraded"

    total_latency = (time.time() - start_time) * 1000

    return SystemHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=version,
        components=components
    )


@router.get("/queue/{class_id}/detailed", response_model=dict)
async def get_detailed_queue_status(
        class_id: int,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """Получить детальный статус очереди для класса"""

    # Проверяем права
    class_obj = session.get(Class, class_id)
    if not class_obj or class_obj.teacher_id != current_user.id:
        raise AppException(
            status_code=403,
            error_code=ErrorCode.ACCESS_DENIED,
            message="Not your class",
            details={"class_id": class_id}
        )

    status = await manager.get_queue_status(class_id)
    return status