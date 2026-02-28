import json
import os
import time
import uuid
from datetime import datetime
from typing import List, Optional
import logging
import gc

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from sqlmodel import Session, select

from .assessit_ws import manager
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
    CommonError
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assessit", tags=["assessit"])

# Инициализация клиента v1
ocr_client_v1 = get_ocr_client_v1()


@router.post("/upload-work", response_model=UploadWorkResponse)
async def upload_student_work(
        assignment_id: int,
        student_id: Optional[int] = None,
        class_id: Optional[int] = None,
        file: UploadFile = File(...),
        wait_for_result: bool = Query(False, description="Ждать результата или добавить в очередь"),
        session: Session = Depends(get_session),
        current_user: User = Depends(get_current_user)
):
    """
    Загрузить работу ученика для проверки

    - Если wait_for_result=False (по умолчанию) - работа добавляется в очередь,
      результат будет отправлен через WebSocket
    - Если wait_for_result=True - ожидаем результат синхронно (может занять 5-7 секунд)
    - Если student_id не указан, используется ID текущего пользователя (для учеников)
    - Если указан student_id, текущий пользователь должен быть учителем этого класса
    """
    try:
        # Определяем student_id
        target_student_id = student_id
        if target_student_id is None:
            target_student_id = current_user.id
        elif current_user.role == "teacher":
            # Проверяем, что ученик в классе учителя
            if class_id:
                link = session.exec(
                    select(ClassStudentLink).where(
                        ClassStudentLink.class_id == class_id,
                        ClassStudentLink.student_id == target_student_id
                    )
                ).first()
                if not link:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Student {target_student_id} is not in class {class_id}"
                    )
        else:
            raise HTTPException(
                status_code=403,
                detail="Students can only upload their own work"
            )

        # Читаем файл с ограничением размера (10MB максимум)
        max_size = 10 * 1024 * 1024
        file_content = await file.read(max_size)

        # Получаем задание для эталонных данных
        assignment = session.get(Assignment, assignment_id)
        if not assignment:
            raise AppException(
                status_code=404,
                error_code=ErrorCode.NOT_FOUND,
                message="Assignment not found",
                details={"assignment_id": assignment_id}
            )

        # Создаем запись в БД
        image = create_assessment_image(
            session=session,
            assignment_id=assignment_id,
            student_id=target_student_id,
            file_name=file.filename,
            file_size=len(file_content),
            mime_type=file.content_type or "application/octet-stream",
            class_id=class_id or assignment.class_id
        )

        if not wait_for_result and class_id:
            student = session.get(User, target_student_id)
            student_name = f"{student.name} {student.surname}" if student else "Unknown"

            position = await add_work_to_queue(
                class_id=class_id or assignment.class_id,
                work_id=image.id,
                student_name=student_name,
                image_bytes=file_content,
                filename=file.filename,
                reference_answer=assignment.reference_answer,
                reference_formulas=[assignment.reference_solution] if assignment.reference_solution else None
            )

            return UploadWorkResponse(
                work_id=image.id,
                message=f"Work added to queue at position {position}",
                queue_position=position,
                status="queued"
            )

            # Иначе обрабатываем синхронно
            try:
                ocr_result = ocr_client_v1.assess_solution(
                    image_bytes=file_content,
                    filename=file.filename,
                    reference_answer=assignment.reference_answer,
                    reference_formulas=[assignment.reference_solution] if assignment.reference_solution else None
                )
            except Exception as e:
                # Конвертируем ошибку в AppException
                update_image_status(session, image.id, "error", str(e))
                raise handle_ocr_error(e, image.id)

            if not ocr_result.get("success", False):
                error_msg = ocr_result.get("error", "Unknown OCR error")
                update_image_status(session, image.id, "error", error_msg)

                raise AppException(
                    status_code=502,
                    error_code=ErrorCode.OCR_API_ERROR,
                    message=error_msg,
                    details={"image_id": image.id}
                )

            # Сохраняем результаты
            assessment_data = ocr_result.get("assessment", {})
            solution = create_recognized_solution_v1(session, image.id, assessment_data)
            update_image_status(session, image.id, "processed")

            return UploadWorkResponse(
                work_id=image.id,
                message="Work processed successfully",
                confidence_score=assessment_data.get("confidence_score"),
                check_level=f"level_{assessment_data.get('confidence_level', 3)}",
                solution_id=assessment_data.get("solution_id")
            )

    except AppException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise AppException(
            status_code=500,
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Upload failed",
            details={"error": str(e)}
        )


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


@router.get("/queue/{class_id}/status")
async def get_queue_status(
        class_id: int,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """Получить статус очереди для класса"""

    # Проверяем права
    class_obj = session.get(Class, class_id)
    if not class_obj or class_obj.teacher_id != current_user.id:
        raise AppException(
            status_code=403,
            error_code=ErrorCode.ACCESS_DENIED,
            message="Not your class",
            details={"class_id": class_id}
        )

    if class_id not in manager.class_queues:
        return {
            "queue_size": 0,
            "estimated_wait_seconds": 0,
            "status": "idle"
        }

    queue_size = manager.class_queues[class_id].qsize()
    return {
        "queue_size": queue_size,
        "estimated_wait_seconds": queue_size * 6,
        "estimated_wait_minutes": round(queue_size * 6 / 60, 1),
        "status": "processing" if queue_size > 0 else "idle"
    }

@router.post("/batch-upload", response_model=BatchUploadResponse)
async def batch_upload_works(
        assignment_id: int,
        class_id: int,
        files: List[UploadFile] = File(...),
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """
    Пакетная загрузка работ для всего класса
    Ожидаются файлы с именами вида: {student_id}_{filename}.ext
    """
    successful_uploads = []
    failed_uploads = []
    
    # Проверяем, что класс принадлежит учителю
    class_ = session.get(Class, class_id)
    if not class_ or class_.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your class")
    
    # Получаем задание
    assignment = session.get(Assignment, assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    
    reference_answer = assignment.reference_answer
    reference_formulas = [assignment.reference_solution] if assignment.reference_solution else None

    for file in files:
        try:
            filename = file.filename
            student_id = None

            # Извлекаем student_id из имени файла (формат: "123_solution.jpg")
            if '_' in filename:
                try:
                    student_id = int(filename.split('_')[0])
                except ValueError:
                    pass

            if not student_id:
                failed_uploads.append({
                    "filename": filename,
                    "error": "Cannot extract student ID from filename (expected format: {student_id}_*.*)"
                })
                continue

            # Проверяем, что студент в классе
            link = session.exec(
                select(ClassStudentLink).where(
                    ClassStudentLink.class_id == class_id,
                    ClassStudentLink.student_id == student_id
                )
            ).first()

            if not link:
                failed_uploads.append({
                    "filename": filename,
                    "error": f"Student {student_id} not in class"
                })
                continue

            # Читаем содержимое файла
            file_content = await file.read()
            
            # Создаем запись в БД
            image = create_assessment_image(
                session=session,
                assignment_id=assignment_id,
                student_id=student_id,
                file_name=filename,
                file_size=len(file_content),
                mime_type=file.content_type or "application/octet-stream",
                class_id=class_id
            )

            # Отправляем в OCR API
            ocr_result = ocr_client_v1.assess_solution(
                image_bytes=file_content,
                filename=filename,
                reference_answer=reference_answer,
                reference_formulas=reference_formulas
            )

            if ocr_result.get("success", False):
                assessment_data = ocr_result.get("assessment", {})
                create_recognized_solution_v1(session, image.id, assessment_data)
                update_image_status(session, image.id, "processed")
                successful_uploads.append(image.id)
            else:
                error_msg = ocr_result.get("error", "Unknown error")
                update_image_status(session, image.id, "error", error_msg)
                failed_uploads.append({
                    "filename": filename,
                    "error": error_msg,
                    "image_id": image.id
                })

        except Exception as e:
            logger.error(f"Error uploading {file.filename}: {str(e)}")
            failed_uploads.append({
                "filename": file.filename,
                "error": str(e)
            })

    return BatchUploadResponse(
        total_uploaded=len(files),
        successful_uploads=successful_uploads,
        failed_uploads=failed_uploads,
        batch_id=str(uuid.uuid4())
    )


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