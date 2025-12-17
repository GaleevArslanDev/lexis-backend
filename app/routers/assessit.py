from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlmodel import Session, select
from typing import List, Optional
import os
import uuid
import aiofiles
from datetime import datetime
import logging

from ..db import get_session
from ..dependencies.auth import get_current_user
from ..dependencies_util import require_role
from ..models import User, AssessmentImage, Assignment, Class
from ..crud.assessment import (
    create_assessment_image,
    get_assessment_image,
    get_recognized_solution,
    create_assessment_result,
    get_class_assessments,
    get_class_assessment_summary,
    get_teacher_dashboard_stats
)
from ..tasks.image_processing import process_assessment_image, batch_process_images
from ..schemas import (
    UploadWorkRequest,
    UploadWorkResponse,
    ProcessedWorkResponse,
    DetailedAnalysisResponse,
    TeacherVerificationRequest,
    TeacherVerificationResponse,
    ClassAssessmentSummary,
    StudentWorkItem,
    BatchUploadRequest,
    BatchUploadResponse,
    DashboardStats,
    CommonError
)

router = APIRouter(prefix="/assessit", tags=["assessit"])

logger = logging.getLogger(__name__)

# Конфигурация
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploads")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024
ALLOWED_EXTENSIONS = set(os.getenv("ALLOWED_EXTENSIONS", "jpg,jpeg,png,pdf").split(','))


@router.post("/upload-work", response_model=UploadWorkResponse)
async def upload_student_work(
        assignment_id: int,
        class_id: Optional[int] = None,
        question_id: Optional[int] = None,
        file: UploadFile = File(...),
        session: Session = Depends(get_session),
        current_user: User = Depends(get_current_user)
):
    """Загрузить работу студента"""

    # Проверяем файл
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = file.filename.split('.')[-1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Проверяем размер файла
    file.file.seek(0, 2)  # Перемещаемся в конец файла
    file_size = file.file.tell()
    file.file.seek(0)  # Возвращаемся в начало

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {MAX_FILE_SIZE // (1024 * 1024)}MB"
        )

    # Проверяем существование задания
    assignment = session.get(Assignment, assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    # Если class_id не указан, берем из задания
    if not class_id:
        class_id = assignment.class_id

    # Проверяем, состоит ли студент в классе
    from ..models import ClassStudentLink
    link = session.exec(
        select(ClassStudentLink).where(
            ClassStudentLink.class_id == class_id,
            ClassStudentLink.student_id == current_user.id
        )
    ).first()

    if not link:
        raise HTTPException(status_code=403, detail="Student not in this class")

    try:
        # Сохраняем файл
        unique_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{current_user.id}_{assignment_id}_{timestamp}_{unique_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)

        # Создаем директорию если нет
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Асинхронно сохраняем файл
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # Создаем запись в БД
        image = create_assessment_image(
            session=session,
            assignment_id=assignment_id,
            student_id=current_user.id,
            file_name=file.filename,
            file_size=file_size,
            mime_type=file.content_type or "application/octet-stream",
            class_id=class_id,
            question_id=question_id
        )

        # Запускаем фоновую обработку
        process_assessment_image.delay(image.id)

        return UploadWorkResponse(
            work_id=image.id,
            message="Work uploaded successfully. Processing started."
        )

    except Exception as e:
        logger.error(f"Error uploading work: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/work-status/{work_id}", response_model=ProcessedWorkResponse)
async def get_work_status(
        work_id: int,
        session: Session = Depends(get_session),
        current_user: User = Depends(get_current_user)
):
    """Получить статус обработки работы"""

    image = get_assessment_image(session, work_id)
    if not image:
        raise HTTPException(status_code=404, detail="Work not found")

    # Проверяем права доступа
    if current_user.role == "student" and image.student_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Если работа обработана, получаем решение
    solution = None
    confidence_score = None
    check_level = None
    extracted_text = None
    extracted_answer = None
    processing_time_ms = None

    if image.status == "processed":
        solution = get_recognized_solution(session, work_id)
        if solution:
            confidence_score = solution.total_confidence
            check_level = solution.check_level
            extracted_text = solution.extracted_text[:500] + "..." if solution.extracted_text else None
            extracted_answer = solution.extracted_answer
            processing_time_ms = solution.processing_time_ms

    return ProcessedWorkResponse(
        work_id=work_id,
        status=image.status,
        confidence_score=confidence_score,
        check_level=check_level,
        extracted_text=extracted_text,
        extracted_answer=extracted_answer,
        processing_time_ms=processing_time_ms,
        error_message=image.error_message
    )


@router.get("/work-details/{work_id}", response_model=DetailedAnalysisResponse)
async def get_work_details(
        work_id: int,
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

    # Формируем формулы из JSON
    formulas = []
    if solution.extracted_formulas_json:
        import json
        try:
            formulas_data = json.loads(solution.extracted_formulas_json)
            formulas = [
                {
                    "latex": f.get("latex", ""),
                    "confidence": f.get("confidence", 0.0),
                    "position": f.get("bbox", {}),
                    "is_correct": None  # Можно добавить проверку
                }
                for f in formulas_data[:10]  # Ограничиваем количество
            ]
        except:
            formulas = []

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
        auto_feedback=solution.auto_feedback or "",
        annotated_image_url=image.processed_image_path,
        problem_areas=[],  # Можно добавить логику определения проблемных зон
        processing_time_ms=solution.processing_time_ms or 0,
        processed_at=solution.recognized_at
    )


@router.post("/verify/{work_id}", response_model=TeacherVerificationResponse)
async def verify_work(
        work_id: int,
        verification: TeacherVerificationRequest,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """Учитель проверяет и оценивает работу"""

    image = get_assessment_image(session, work_id)
    if not image:
        raise HTTPException(status_code=404, detail="Work not found")

    # Проверяем, что работа из класса учителя
    class_ = session.get(Class, image.class_id)
    if not class_ or class_.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your class")

    solution = get_recognized_solution(session, work_id)
    if not solution:
        raise HTTPException(status_code=400, detail="Solution not found")

    # Создаем запись о проверке
    result = create_assessment_result(
        session=session,
        solution_id=solution.id,
        teacher_id=current_user.id,
        teacher_verdict=verification.verdict,
        teacher_score=verification.score,
        teacher_feedback=verification.feedback,
        corrected_text=verification.corrected_text,
        corrected_answer=verification.corrected_answer,
        system_score=solution.suggested_grade,
        verified_at=datetime.utcnow()
    )

    # Если есть исправления, сохраняем как образец для обучения
    if verification.corrected_text and solution.extracted_text:
        from ..crud.assessment import create_training_sample

        create_training_sample(
            session=session,
            image_path=image.original_image_path,
            original_text=solution.extracted_text,
            corrected_text=verification.corrected_text,
            subject=image.assignment.subject if hasattr(image.assignment, 'subject') else None,
            handwriting_style="student"
        )

    return TeacherVerificationResponse(
        result_id=result.id,
        work_id=work_id,
        teacher_id=current_user.id,
        verdict=result.teacher_verdict,
        score=result.teacher_score,
        feedback=result.teacher_feedback,
        verified_at=result.verified_at or datetime.utcnow()
    )


@router.get("/class/{class_id}/works", response_model=List[StudentWorkItem])
async def get_class_works(
        class_id: int,
        assignment_id: Optional[int] = None,
        status: Optional[str] = None,
        check_level: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """Получить список работ класса"""

    # Проверяем, что класс принадлежит учителю
    class_ = session.get(Class, class_id)
    if not class_ or class_.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your class")

    works = get_class_assessments(
        session=session,
        class_id=class_id,
        assignment_id=assignment_id,
        status=status,
        check_level=check_level,
        limit=limit,
        offset=offset
    )

    result = []
    for work in works:
        # Получаем информацию о студенте
        student = session.get(User, work.student_id)
        student_name = student.name if student else "Unknown"
        student_surname = student.surname if student else "Student"

        # Получаем решение если есть
        solution = get_recognized_solution(session, work.id)

        # Получаем оценку учителя если есть
        teacher_score = None
        if solution:
            result = get_assessment_result(session, solution.id)
            if result:
                teacher_score = result.teacher_score

        result.append(StudentWorkItem(
            work_id=work.id,
            student_id=work.student_id,
            student_name=student_name,
            student_surname=student_surname,
            status=work.status,
            check_level=solution.check_level if solution else None,
            confidence_score=solution.total_confidence if solution else None,
            teacher_score=teacher_score,
            system_score=solution.suggested_grade if solution else None,
            uploaded_at=work.upload_time
        ))

    return result


@router.get("/class/{class_id}/summary", response_model=ClassAssessmentSummary)
async def get_class_summary(
        class_id: int,
        assignment_id: Optional[int] = None,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """Получить сводку по работам класса"""

    # Проверяем, что класс принадлежит учителю
    class_ = session.get(Class, class_id)
    if not class_ or class_.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your class")

    # Получаем всех студентов класса
    from ..models import ClassStudentLink
    links = session.exec(
        select(ClassStudentLink).where(ClassStudentLink.class_id == class_id)
    ).all()

    total_students = len(links)

    # Получаем сводку из CRUD
    summary = get_class_assessment_summary(session, class_id, assignment_id)

    # Получаем временные рамки
    works = get_class_assessments(session, class_id, assignment_id)
    first_upload = min([w.upload_time for w in works]) if works else None
    last_upload = max([w.upload_time for w in works]) if works else None

    # Определяем общие ошибки (упрощенно)
    common_errors = []

    return ClassAssessmentSummary(
        class_id=class_id,
        assignment_id=assignment_id,
        total_students=total_students,
        submitted_works=summary.get("total_works", 0),
        processed_works=summary.get("processed_works", 0),
        level_1_count=summary.get("level_counts", {}).get("level_1", 0),
        level_2_count=summary.get("level_counts", {}).get("level_2", 0),
        level_3_count=summary.get("level_counts", {}).get("level_3", 0),
        avg_confidence=summary.get("avg_confidence"),
        avg_processing_time_ms=None,  # Можно добавить расчет
        common_errors=common_errors,
        first_upload=first_upload,
        last_upload=last_upload
    )


@router.post("/batch-upload", response_model=BatchUploadResponse)
async def batch_upload_works(
        assignment_id: int,
        class_id: int,
        files: List[UploadFile] = File(...),
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """Пакетная загрузка работ"""

    successful_uploads = []
    failed_uploads = []

    for file in files:
        try:
            # Для каждой работы находим student_id из имени файла
            # Формат: student_id_filename.ext или student_id.ext
            filename = file.filename
            student_id = None

            # Пробуем извлечь student_id из имени файла
            if '_' in filename:
                try:
                    student_id = int(filename.split('_')[0])
                except:
                    pass

            if not student_id:
                failed_uploads.append({
                    "filename": filename,
                    "error": "Cannot extract student ID from filename"
                })
                continue

            # Проверяем, что студент в классе
            from ..models import ClassStudentLink
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

            # Сохраняем файл
            unique_id = uuid.uuid4().hex[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{student_id}_{assignment_id}_{timestamp}_{unique_id}_{filename}"
            file_path = os.path.join(UPLOAD_DIR, safe_filename)

            # Создаем директорию если нет
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Сохраняем файл
            async with aiofiles.open(file_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)

            file.file.seek(0, 2)
            file_size = file.file.tell()
            file.file.seek(0)

            # Создаем запись в БД
            image = create_assessment_image(
                session=session,
                assignment_id=assignment_id,
                student_id=student_id,
                file_name=filename,
                file_size=file_size,
                mime_type=file.content_type or "application/octet-stream",
                class_id=class_id
            )

            successful_uploads.append(image.id)

        except Exception as e:
            logger.error(f"Error uploading {file.filename}: {str(e)}")
            failed_uploads.append({
                "filename": file.filename,
                "error": str(e)
            })

    # Запускаем пакетную обработку
    if successful_uploads:
        batch_process_images.delay(successful_uploads)

    return BatchUploadResponse(
        total_uploaded=len(files),
        successful_uploads=successful_uploads,
        failed_uploads=failed_uploads,
        batch_id=str(uuid.uuid4())
    )


@router.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(
        days: int = 30,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """Получить статистику для дашборда учителя"""

    stats = get_teacher_dashboard_stats(session, current_user.id, days)

    if "error" in stats:
        raise HTTPException(status_code=404, detail=stats["error"])

    return DashboardStats(
        total_assignments=0,  # Можно добавить расчет
        total_works_processed=stats["processed_works"],
        time_saved_hours=stats["time_saved_hours"],
        avg_confidence=stats["avg_confidence"],
        accuracy_rate=stats["accuracy_rate"],
        daily_stats=stats["daily_stats"]
    )


@router.get("/common-errors/{class_id}", response_model=List[CommonError])
async def get_common_errors(
        class_id: int,
        assignment_id: Optional[int] = None,
        session: Session = Depends(get_session),
        current_user: User = Depends(require_role("teacher"))
):
    """Получить список общих ошибок в классе"""

    # Проверяем, что класс принадлежит учителю
    class_ = session.get(Class, class_id)
    if not class_ or class_.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your class")

    # Получаем все работы класса
    works = get_class_assessments(session, class_id, assignment_id, status="processed")

    errors = []

    # Анализируем автоматическую обратную связь
    error_counts = {}
    for work in works:
        solution = get_recognized_solution(session, work.id)
        if solution and solution.auto_feedback:
            # Извлекаем ключевые фразы из обратной связи
            feedback = solution.auto_feedback.lower()

            if "низкое качество распознавания" in feedback:
                error_counts["low_ocr_quality"] = error_counts.get("low_ocr_quality", 0) + 1
            if "неясная структура" in feedback:
                error_counts["unclear_structure"] = error_counts.get("unclear_structure", 0) + 1
            if "проблемы с распознаванием формул" in feedback:
                error_counts["formula_recognition"] = error_counts.get("formula_recognition", 0) + 1
            if "расхождение в ответе" in feedback:
                error_counts["answer_mismatch"] = error_counts.get("answer_mismatch", 0) + 1

    total_works = len(works)

    for error_type, count in error_counts.items():
        percentage = (count / total_works * 100) if total_works > 0 else 0

        error_names = {
            "low_ocr_quality": "Низкое качество распознавания",
            "unclear_structure": "Неясная структура решения",
            "formula_recognition": "Проблемы с формулами",
            "answer_mismatch": "Расхождение в ответе"
        }

        examples = ["Пример ошибки 1", "Пример ошибки 2"]  # Можно добавить реальные примеры

        errors.append(CommonError(
            error_type=error_names.get(error_type, error_type),
            count=count,
            percentage=round(percentage, 1),
            examples=examples
        ))

    return errors