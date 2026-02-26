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

from ..db import get_session
from ..dependencies.auth import get_current_user
from ..dependencies_util import require_role
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
    create_recognized_solution_v1,  # Новый метод
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
        background_tasks: BackgroundTasks = None,
        session: Session = Depends(get_session),
        current_user: User = Depends(get_current_user)
):
    """
    Загрузить работу ученика для проверки
    
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
            raise HTTPException(status_code=404, detail="Assignment not found")

        # Создаем запись в БД
        image = create_assessment_image(
            session=session,
            assignment_id=assignment_id,
            student_id=target_student_id,
            file_name=file.filename,
            file_size=len(file_content),
            mime_type=file.content_type or "application/octet-stream",
            class_id=class_id
        )

        # Получаем эталонные данные из задания
        reference_answer = assignment.reference_answer
        reference_formulas = None
        if assignment.reference_solution:
            # Можно извлечь формулы из решения, пока просто передаем как есть
            reference_formulas = [assignment.reference_solution]

        # Отправляем в OCR API v1
        logger.info(f"Sending image {image.id} to OCR API v1 for assessment")
        
        ocr_result = ocr_client_v1.assess_solution(
            image_bytes=file_content,
            filename=file.filename,
            reference_answer=reference_answer,
            reference_formulas=reference_formulas
        )

        if not ocr_result.get("success", False):
            error_msg = ocr_result.get("error", "Unknown OCR error")
            update_image_status(session, image.id, "error", error_msg)
            logger.error(f"OCR failed for image {image.id}: {error_msg}")
            
            return UploadWorkResponse(
                work_id=image.id,
                message="Work uploaded but OCR failed",
                error=error_msg
            )

        # Сохраняем результаты
        assessment_data = ocr_result.get("assessment", {})
        solution = create_recognized_solution_v1(session, image.id, assessment_data)
        
        update_image_status(session, image.id, "processed")
        
        logger.info(f"Successfully processed image {image.id} with confidence {assessment_data.get('confidence_score', 0)}")

        # Очистка памяти
        del file_content
        gc.collect()

        return UploadWorkResponse(
            work_id=image.id,
            message="Work uploaded and processed successfully",
            confidence_score=assessment_data.get("confidence_score"),
            check_level=f"level_{assessment_data.get('confidence_level', 3)}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


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