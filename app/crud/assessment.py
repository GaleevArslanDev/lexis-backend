import json
from datetime import datetime

from sqlmodel import Session, select, func, desc
from typing import Optional, List, Dict, Any
from ..models import (
    AssessmentImage, RecognizedSolution, AssessmentResult,
    TrainingSample, SystemMetrics, User, Class, Assignment
)


def create_recognized_solution_v1(
        session: Session,
        image_id: int,
        assessment_data: Dict[str, Any]
) -> RecognizedSolution:
    """
    Создать запись о распознанном решении из данных ML-пайплайна.

    assessment_data — результат FinalAssessment.model_dump(mode='json').
    Совместим как с dict-ответом старого HTTP OCR API, так и с новым
    прямым вызовом pipeline_service.
    """
    scores = assessment_data.get("scores", {})
    steps = assessment_data.get("steps_analysis", [])

    # Уровень проверки: числовой (1/2/3) → строка
    confidence_level = assessment_data.get("confidence_level", 3)
    check_level_map = {1: "level_1", 2: "level_2", 3: "level_3"}
    check_level = check_level_map.get(confidence_level, "level_3")

    # Извлечённый текст: ищем ключ "formula" (новый пайплайн) или "found_formula" (старый API)
    extracted_texts = [
        item.get("formula") or item.get("found_formula", "")
        for item in steps
    ] if steps else []

    # Время обработки
    execution_time = assessment_data.get("execution_time", 0) or 0
    processing_time_ms = int(execution_time * 1000)

    # Дата распознавания — может прийти как str или datetime
    created_at_raw = assessment_data.get("created_at")
    if created_at_raw is None:
        recognized_at = datetime.utcnow()
    elif isinstance(created_at_raw, datetime):
        recognized_at = created_at_raw
    else:
        try:
            recognized_at = datetime.fromisoformat(str(created_at_raw))
        except (ValueError, TypeError):
            recognized_at = datetime.utcnow()

    solution = RecognizedSolution(
        image_id=image_id,
        extracted_text=json.dumps(extracted_texts, ensure_ascii=False),
        text_confidence=(scores.get("c_ocr") or 0.0) * 100,
        formulas_count=len(steps) if steps else 0,

        # Confidence scores
        ocr_confidence=scores.get("c_ocr", 0.0),
        formula_confidence=scores.get("c_llm", 0.0),
        answer_match_confidence=scores.get("m_answer", 0.0),
        total_confidence=assessment_data.get("confidence_score", 0.0),

        # Классификация
        check_level=check_level,
        suggested_grade=(assessment_data.get("mark_score") or 0.0) * 5,  # → 5-балльная
        auto_feedback=assessment_data.get("teacher_comment", ""),

        # Поля из нового пайплайна
        solution_id=assessment_data.get("solution_id"),
        mark_score=assessment_data.get("mark_score", 0.0),
        teacher_comment=assessment_data.get("teacher_comment", ""),

        # Детальные метрики
        c_ocr=scores.get("c_ocr"),
        c_llm=scores.get("c_llm"),
        m_sympy=scores.get("m_sympy"),   # None в новом пайплайне — это нормально
        m_llm=scores.get("m_llm"),
        m_answer=scores.get("m_answer"),

        # Анализ шагов
        steps_analysis_json=json.dumps(steps, ensure_ascii=False) if steps else None,

        processing_time_ms=processing_time_ms,
        recognized_at=recognized_at,
    )

    session.add(solution)
    session.commit()
    session.refresh(solution)
    return solution


# ---------------------------------------------------------------------------
# Остальные функции без изменений
# ---------------------------------------------------------------------------

def create_assessment_image(
        session: Session,
        assignment_id: int,
        student_id: int,
        file_name: str,
        file_size: int,
        mime_type: str,
        class_id: Optional[int] = None,
        question_id: Optional[int] = None
) -> AssessmentImage:
    if not class_id:
        assignment = session.get(Assignment, assignment_id)
        if assignment:
            class_id = assignment.class_id

    image = AssessmentImage(
        assignment_id=assignment_id,
        class_id=class_id,
        student_id=student_id,
        question_id=question_id,
        original_image_path="",
        file_name=file_name,
        file_size=file_size,
        mime_type=mime_type,
        status="pending",
    )
    session.add(image)
    session.commit()
    session.refresh(image)
    return image


def get_assessment_image(session: Session, image_id: int) -> Optional[AssessmentImage]:
    return session.get(AssessmentImage, image_id)


def update_image_status(
        session: Session,
        image_id: int,
        status: str,
        error_message: Optional[str] = None
) -> Optional[AssessmentImage]:
    image = session.get(AssessmentImage, image_id)
    if not image:
        return None

    image.status = status
    if status == "processing":
        image.processing_started = datetime.utcnow()
    elif status == "processed":
        image.processing_completed = datetime.utcnow()
    elif status == "error":
        image.error_message = error_message
        image.retry_count += 1

    session.add(image)
    session.commit()
    session.refresh(image)
    return image


def create_recognized_solution(
        session: Session,
        image_id: int,
        extracted_text: str,
        **kwargs
) -> RecognizedSolution:
    solution = RecognizedSolution(image_id=image_id, extracted_text=extracted_text, **kwargs)
    session.add(solution)
    session.commit()
    session.refresh(solution)
    return solution


def get_recognized_solution(session: Session, image_id: int) -> Optional[RecognizedSolution]:
    statement = (
        select(RecognizedSolution)
        .where(RecognizedSolution.image_id == image_id)
        .order_by(desc(RecognizedSolution.recognized_at))
    )
    return session.exec(statement).first()


def create_assessment_result(session: Session, solution_id: int, **kwargs) -> AssessmentResult:
    result = AssessmentResult(solution_id=solution_id, **kwargs)
    session.add(result)
    session.commit()
    session.refresh(result)
    return result


def get_assessment_result(session: Session, solution_id: int) -> Optional[AssessmentResult]:
    statement = (
        select(AssessmentResult)
        .where(AssessmentResult.solution_id == solution_id)
        .order_by(desc(AssessmentResult.created_at))
    )
    return session.exec(statement).first()


def get_class_assessments(
        session: Session,
        class_id: int,
        assignment_id: Optional[int] = None,
        status: Optional[str] = None,
        check_level: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
) -> List[AssessmentImage]:
    statement = select(AssessmentImage).where(AssessmentImage.class_id == class_id)
    if assignment_id:
        statement = statement.where(AssessmentImage.assignment_id == assignment_id)
    if status:
        statement = statement.where(AssessmentImage.status == status)
    if check_level:
        statement = statement.join(RecognizedSolution).where(RecognizedSolution.check_level == check_level)
    statement = statement.order_by(desc(AssessmentImage.upload_time)).offset(offset).limit(limit)
    return session.exec(statement).all()


def get_class_assessment_summary(
        session: Session,
        class_id: int,
        assignment_id: Optional[int] = None
) -> Dict[str, Any]:
    base_query = select(AssessmentImage).where(AssessmentImage.class_id == class_id)
    if assignment_id:
        base_query = base_query.where(AssessmentImage.assignment_id == assignment_id)

    all_works = session.exec(base_query).all()
    if not all_works:
        return {
            "total_works": 0, "processed_works": 0,
            "level_counts": {"level_1": 0, "level_2": 0, "level_3": 0},
            "avg_confidence": None,
        }

    total_works = len(all_works)
    processed_works = len([w for w in all_works if w.status == "processed"])

    level_counts = {"level_1": 0, "level_2": 0, "level_3": 0}
    confidences = []
    for work in all_works:
        if work.status == "processed":
            solution = get_recognized_solution(session, work.id)
            if solution:
                level_counts[solution.check_level] = level_counts.get(solution.check_level, 0) + 1
                if solution.total_confidence:
                    confidences.append(solution.total_confidence)

    avg_confidence = sum(confidences) / len(confidences) if confidences else None
    return {
        "total_works": total_works,
        "processed_works": processed_works,
        "level_counts": level_counts,
        "avg_confidence": avg_confidence,
    }


def create_training_sample(
        session: Session,
        image_path: str,
        original_text: str,
        corrected_text: str,
        **kwargs
) -> TrainingSample:
    sample = TrainingSample(
        image_path=image_path, original_text=original_text,
        corrected_text=corrected_text, **kwargs
    )
    session.add(sample)
    session.commit()
    session.refresh(sample)
    return sample


def update_system_metrics(session: Session, **kwargs) -> SystemMetrics:
    from sqlmodel import func as sqlfunc
    from datetime import date

    today = datetime.utcnow().date()
    statement = select(SystemMetrics).where(func.date(SystemMetrics.date) == today)
    metrics = session.exec(statement).first()
    if not metrics:
        metrics = SystemMetrics()

    for key, value in kwargs.items():
        if hasattr(metrics, key):
            current = getattr(metrics, key)
            if isinstance(current, (int, float)) and isinstance(value, (int, float)):
                if key in ["total_processed", "level_1_count", "level_2_count", "level_3_count", "error_count"]:
                    setattr(metrics, key, current + value)
                else:
                    setattr(metrics, key, value)
            else:
                setattr(metrics, key, value)

    session.add(metrics)
    session.commit()
    session.refresh(metrics)
    return metrics


def get_teacher_dashboard_stats(
        session: Session,
        teacher_id: int,
        days: int = 30
) -> Dict[str, Any]:
    from datetime import timedelta

    cutoff_date = datetime.utcnow() - timedelta(days=days)

    classes = session.exec(select(Class).where(Class.teacher_id == teacher_id)).all()
    class_ids = [c.id for c in classes]

    if not class_ids:
        return {
            "total_assignments": 0, "total_works_processed": 0,
            "time_saved_hours": 0, "avg_confidence": 0,
            "accuracy_rate": 0, "daily_stats": [],
        }

    works = session.exec(
        select(AssessmentImage)
        .where(AssessmentImage.class_id.in_(class_ids))
        .where(AssessmentImage.upload_time >= cutoff_date)
    ).all()

    assignments = session.exec(
        select(Assignment).where(Assignment.class_id.in_(class_ids))
    ).all()

    processed_works = [w for w in works if w.status == "processed"]

    confidences = []
    for work in processed_works:
        sol = get_recognized_solution(session, work.id)
        if sol and sol.total_confidence:
            confidences.append(sol.total_confidence)

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    daily: Dict[str, Dict] = {}
    for work in works:
        day = work.upload_time.date().isoformat()
        if day not in daily:
            daily[day] = {"total": 0, "processed": 0, "errors": 0}
        daily[day]["total"] += 1
        if work.status == "processed":
            daily[day]["processed"] += 1
        elif work.status == "error":
            daily[day]["errors"] += 1

    daily_stats = [{"date": day, **stats} for day, stats in sorted(daily.items())]

    time_saved_hours = len(processed_works) * 3 / 60

    level_1_count = sum(
        1 for w in processed_works
        if (sol := get_recognized_solution(session, w.id)) and sol.check_level == "level_1"
    )
    accuracy_rate = (level_1_count / len(processed_works) * 100) if processed_works else 0

    return {
        "total_assignments": len(assignments),
        "total_works_processed": len(processed_works),
        "time_saved_hours": round(time_saved_hours, 1),
        "avg_confidence": round(avg_confidence, 2),
        "accuracy_rate": round(accuracy_rate, 1),
        "daily_stats": daily_stats,
    }


def create_teacher_verification(
        session: Session,
        solution_id: int,
        teacher_id: int,
        verdict: str,
        score: Optional[float] = None,
        feedback: Optional[str] = None,
        corrected_text: Optional[str] = None,
        corrected_answer: Optional[str] = None
) -> AssessmentResult:
    solution = session.get(RecognizedSolution, solution_id)
    if not solution:
        raise ValueError(f"Solution {solution_id} not found")

    result = AssessmentResult(
        solution_id=solution_id,
        teacher_id=teacher_id,
        teacher_verdict=verdict,
        teacher_score=score,
        teacher_feedback=feedback,
        corrected_text=corrected_text,
        corrected_answer=corrected_answer,
        system_score=solution.suggested_grade,
        system_feedback=solution.auto_feedback,
        verified_at=datetime.utcnow(),
    )
    session.add(result)
    session.commit()
    session.refresh(result)

    if corrected_text and corrected_text != solution.extracted_text:
        image = session.get(AssessmentImage, solution.image_id)
        if image:
            training_sample = TrainingSample(
                image_path="",
                original_text=solution.extracted_text,
                corrected_text=corrected_text,
                subject=image.assignment.subject if image.assignment else None,
                handwriting_style="unknown",
                image_quality="unknown",
            )
            session.add(training_sample)
            session.commit()

    return result


def get_common_mistakes(
        session: Session,
        class_id: int,
        assignment_id: Optional[int] = None,
        limit: int = 10
) -> List[Dict[str, Any]]:
    query = (
        select(RecognizedSolution)
        .join(AssessmentImage)
        .where(AssessmentImage.class_id == class_id)
    )
    if assignment_id:
        query = query.where(AssessmentImage.assignment_id == assignment_id)

    solutions = session.exec(query).all()
    mistakes: Dict[str, int] = {}
    examples: Dict[str, List] = {}

    for sol in solutions:
        if sol.steps_analysis_json:
            steps = json.loads(sol.steps_analysis_json)
            for step in steps:
                if not step.get("is_correct", True):
                    mistake_type = step.get("expected_formula") or step.get("formula", "unknown")
                    found = step.get("found_formula") or step.get("formula", "")
                    mistakes[mistake_type] = mistakes.get(mistake_type, 0) + 1
                    if mistake_type not in examples:
                        examples[mistake_type] = []
                    if len(examples[mistake_type]) < 3 and found:
                        examples[mistake_type].append(found)

    sorted_mistakes = sorted(mistakes.items(), key=lambda x: x[1], reverse=True)[:limit]
    total = sum(mistakes.values()) or 1

    return [
        {
            "error_type": mistake_type,
            "count": count,
            "percentage": round((count / total) * 100, 1),
            "examples": examples.get(mistake_type, []),
        }
        for mistake_type, count in sorted_mistakes
    ]


def get_class_assessments_paginated(
        session: Session,
        class_id: int,
        assignment_id: Optional[int] = None,
        status: Optional[str] = None,
        check_level: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
) -> Dict[str, Any]:
    query = select(AssessmentImage).where(AssessmentImage.class_id == class_id)
    if assignment_id:
        query = query.where(AssessmentImage.assignment_id == assignment_id)
    if status:
        query = query.where(AssessmentImage.status == status)

    count_query = select(func.count()).select_from(query.subquery())
    total = session.exec(count_query).one()

    query = query.order_by(desc(AssessmentImage.upload_time))
    query = query.offset((page - 1) * page_size).limit(page_size)
    works = session.exec(query).all()

    result = []
    for work in works:
        student = session.get(User, work.student_id)
        solution = get_recognized_solution(session, work.id)
        result.append({
            "work_id": work.id,
            "student_id": work.student_id,
            "student_name": student.name if student else "Unknown",
            "student_surname": student.surname if student else "",
            "status": work.status,
            "check_level": solution.check_level if solution else None,
            "confidence_score": solution.total_confidence if solution else None,
            "teacher_score": None,
            "system_score": solution.suggested_grade if solution else None,
            "uploaded_at": work.upload_time,
            "processed_at": solution.recognized_at if solution else None,
        })

    return {
        "items": result,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size,
            "has_next": page * page_size < total,
            "has_prev": page > 1,
        },
    }