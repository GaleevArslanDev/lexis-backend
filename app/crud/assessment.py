import json

from sqlmodel import Session, select, func, desc
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
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
    Создать запись о распознанном решении из данных ML API v1
    
    Args:
        session: Сессия БД
        image_id: ID изображения
        assessment_data: Данные из поля assessment ответа ML API
    """
    # Извлекаем данные
    scores = assessment_data.get("scores", {})
    steps = assessment_data.get("steps_analysis", [])
    
    # Конвертируем check_level из числа в строку
    confidence_level = assessment_data.get("confidence_level", 3)
    check_level_map = {
        1: "level_1",
        2: "level_2",
        3: "level_3"
    }
    check_level = check_level_map.get(confidence_level, "level_3")
    
    solution = RecognizedSolution(
        image_id=image_id,
        extracted_text="",  # Будет заполнено позже, если нужно
        text_confidence=scores.get("c_ocr", 0.0) * 100,  # Конвертируем в проценты
        formulas_count=len(steps) if steps else 0,  # Можно извлечь из steps при необходимости
        
        # Confidence scores
        ocr_confidence=scores.get("c_ocr", 0.0),
        formula_confidence=scores.get("c_llm", 0.0),  # Используем c_llm как formula_confidence
        answer_match_confidence=scores.get("m_answer", 0.0),
        total_confidence=assessment_data.get("confidence_score", 0.0),
        
        # Классификация
        check_level=check_level,
        suggested_grade=assessment_data.get("mark_score", 0.0) * 5,  # Конвертируем в 5-балльную
        auto_feedback=assessment_data.get("teacher_comment", ""),
        
        # Новые поля
        solution_id=assessment_data.get("solution_id"),
        mark_score=assessment_data.get("mark_score", 0.0),
        teacher_comment=assessment_data.get("teacher_comment", ""),
        
        # Детальные скоринг-метрики
        c_ocr=scores.get("c_ocr"),
        c_llm=scores.get("c_llm"),
        m_sympy=scores.get("m_sympy"),
        m_llm=scores.get("m_llm"),
        m_answer=scores.get("m_answer"),
        
        # Шаги анализа
        steps_analysis_json=json.dumps(steps, ensure_ascii=False) if steps else None,
        
        processing_time_ms=assessment_data.get("execution_time", 0) * 1000,  # Конвертируем в мс
        recognized_at=datetime.fromisoformat(assessment_data.get("created_at", datetime.utcnow().isoformat()))
    )
    
    session.add(solution)
    session.commit()
    session.refresh(solution)
    return solution


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
    """Создать запись о загруженной работе (без сохранения файла)"""

    # Если есть class_id, получаем его из assignment
    if not class_id:
        assignment = session.get(Assignment, assignment_id)
        if assignment:
            class_id = assignment.class_id

    # Создаем запись без пути к файлу
    image = AssessmentImage(
        assignment_id=assignment_id,
        class_id=class_id,
        student_id=student_id,
        question_id=question_id,
        original_image_path="",  # Пусто, так как файл не сохраняем
        file_name=file_name,
        file_size=file_size,
        mime_type=mime_type,
        status="pending"
    )

    session.add(image)
    session.commit()
    session.refresh(image)
    return image


def get_assessment_image(session: Session, image_id: int) -> Optional[AssessmentImage]:
    """Получить работу по ID"""
    return session.get(AssessmentImage, image_id)


def update_image_status(
        session: Session,
        image_id: int,
        status: str,
        error_message: Optional[str] = None
) -> Optional[AssessmentImage]:
    """Обновить статус обработки работы"""
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
    """Создать запись о распознанном решении"""

    solution = RecognizedSolution(
        image_id=image_id,
        extracted_text=extracted_text,
        **kwargs
    )

    session.add(solution)
    session.commit()
    session.refresh(solution)
    return solution


def get_recognized_solution(
        session: Session,
        image_id: int
) -> Optional[RecognizedSolution]:
    """Получить распознанное решение по ID изображения"""
    statement = select(RecognizedSolution).where(
        RecognizedSolution.image_id == image_id
    ).order_by(desc(RecognizedSolution.recognized_at))

    return session.exec(statement).first()


def create_assessment_result(
        session: Session,
        solution_id: int,
        **kwargs
) -> AssessmentResult:
    """Создать результат проверки"""

    result = AssessmentResult(
        solution_id=solution_id,
        **kwargs
    )

    session.add(result)
    session.commit()
    session.refresh(result)
    return result


def get_assessment_result(
        session: Session,
        solution_id: int
) -> Optional[AssessmentResult]:
    """Получить результат проверки по ID решения"""
    statement = select(AssessmentResult).where(
        AssessmentResult.solution_id == solution_id
    ).order_by(desc(AssessmentResult.created_at))

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
    """Получить все работы класса"""

    statement = select(AssessmentImage).where(
        AssessmentImage.class_id == class_id
    )

    if assignment_id:
        statement = statement.where(AssessmentImage.assignment_id == assignment_id)

    if status:
        statement = statement.where(AssessmentImage.status == status)

    # Фильтр по check_level требует join с RecognizedSolution
    if check_level:
        statement = statement.join(RecognizedSolution).where(
            RecognizedSolution.check_level == check_level
        )

    statement = statement.order_by(desc(AssessmentImage.upload_time))
    statement = statement.offset(offset).limit(limit)

    return session.exec(statement).all()


def get_class_assessment_summary(
        session: Session,
        class_id: int,
        assignment_id: Optional[int] = None
) -> Dict[str, Any]:
    """Получить сводку по работам класса"""

    # Базовый запрос для работ
    base_query = select(AssessmentImage).where(
        AssessmentImage.class_id == class_id
    )

    if assignment_id:
        base_query = base_query.where(AssessmentImage.assignment_id == assignment_id)

    all_works = session.exec(base_query).all()

    if not all_works:
        return {
            "total_works": 0,
            "processed_works": 0,
            "level_counts": {"level_1": 0, "level_2": 0, "level_3": 0},
            "avg_confidence": None
        }

    total_works = len(all_works)
    processed_works = len([w for w in all_works if w.status == "processed"])

    # Получаем уровни проверки
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
        "avg_confidence": avg_confidence
    }


def create_training_sample(
        session: Session,
        image_path: str,
        original_text: str,
        corrected_text: str,
        **kwargs
) -> TrainingSample:
    """Создать образец для дообучения"""

    sample = TrainingSample(
        image_path=image_path,
        original_text=original_text,
        corrected_text=corrected_text,
        **kwargs
    )

    session.add(sample)
    session.commit()
    session.refresh(sample)
    return sample


def update_system_metrics(
        session: Session,
        **kwargs
) -> SystemMetrics:
    """Обновить метрики системы"""

    today = datetime.utcnow().date()

    # Ищем запись за сегодня
    statement = select(SystemMetrics).where(
        func.date(SystemMetrics.date) == today
    )

    metrics = session.exec(statement).first()

    if not metrics:
        metrics = SystemMetrics()

    # Обновляем поля
    for key, value in kwargs.items():
        if hasattr(metrics, key):
            current_value = getattr(metrics, key)
            if isinstance(current_value, (int, float)) and isinstance(value, (int, float)):
                # Для числовых полей можно суммировать
                if key in ["total_processed", "level_1_count", "level_2_count",
                           "level_3_count", "error_count"]:
                    setattr(metrics, key, current_value + value)
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
    """Получить статистику для дашборда учителя"""

    # Получаем классы учителя
    statement = select(Class).where(Class.teacher_id == teacher_id)
    classes = session.exec(statement).all()

    if not classes:
        return {"error": "No classes found"}

    class_ids = [c.id for c in classes]

    # Работы за последние N дней
    cutoff_date = datetime.utcnow() - timedelta(days=days)

    works_statement = select(AssessmentImage).where(
        AssessmentImage.class_id.in_(class_ids),
        AssessmentImage.upload_time >= cutoff_date
    )

    works = session.exec(works_statement).all()

    # Обработанные работы
    processed_works = [w for w in works if w.status == "processed"]

    # Считаем метрики
    total_works = len(works)
    processed_count = len(processed_works)

    # Confidence scores
    confidences = []
    for work in processed_works:
        solution = get_recognized_solution(session, work.id)
        if solution and solution.total_confidence:
            confidences.append(solution.total_confidence)

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Распределение по дням
    daily_stats = {}
    for work in works:
        day = work.upload_time.date().isoformat()
        if day not in daily_stats:
            daily_stats[day] = {"total": 0, "processed": 0}

        daily_stats[day]["total"] += 1
        if work.status == "processed":
            daily_stats[day]["processed"] += 1

    daily_stats_list = [
        {"date": day, "total": stats["total"], "processed": stats["processed"]}
        for day, stats in daily_stats.items()
    ]

    # Оцениваем сэкономленное время (предполагаем 3 минуты на ручную проверку)
    time_saved_minutes = processed_count * 3
    time_saved_hours = time_saved_minutes / 60

    return {
        "total_works": total_works,
        "processed_works": processed_count,
        "time_saved_hours": round(time_saved_hours, 1),
        "avg_confidence": round(avg_confidence, 2),
        "accuracy_rate": round(processed_count / total_works * 100, 1) if total_works > 0 else 0,
        "daily_stats": daily_stats_list
    }