from typing import Optional, List, Dict, Any
from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
import json


class ClassStudentLink(SQLModel, table=True):
    class_id: int = Field(foreign_key="class.id", primary_key=True)
    student_id: int = Field(foreign_key="user.id", primary_key=True)


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, nullable=False, unique=True)
    name: str = Field(nullable=False)
    surname: str = Field(nullable=False)
    hashed_password: str
    role: str = Field(default="teacher")  # 'teacher' or 'student'
    created_at: datetime = Field(default_factory=datetime.utcnow)

    classes: List["Class"] = Relationship(back_populates="teacher")

    enrolled_classes: List["Class"] = Relationship(
        back_populates="students",
        link_model=ClassStudentLink
    )

    submitted_works: List["AssessmentImage"] = Relationship(back_populates="student")
    verified_results: List["AssessmentResult"] = Relationship(back_populates="teacher")


class Class(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    teacher_id: int = Field(foreign_key="user.id")
    school_id: Optional[int] = Field(foreign_key="school.id", default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    teacher: Optional[User] = Relationship(back_populates="classes")
    students: List[User] = Relationship(
        back_populates="enrolled_classes",
        link_model=ClassStudentLink
    )
    school: Optional["School"] = Relationship(back_populates="classes")

    assignments: List["Assignment"] = Relationship(back_populates="class_")
    assessment_images: List["AssessmentImage"] = Relationship(back_populates="class_")


class School(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    address: Optional[str] = None
    creator_id: int = Field(foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    classes: List[Class] = Relationship(back_populates="school")


class Assignment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    class_id: int = Field(foreign_key="class.id")
    title: str
    description: Optional[str] = None
    type: str = Field(default="essay")  # 'mcq', 'short', 'essay', 'math_problem'
    created_at: datetime = Field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = None

    reference_solution: Optional[str] = None
    reference_answer: Optional[str] = None
    subject: Optional[str] = None  # 'math', 'physics', 'chemistry'
    difficulty: Optional[str] = None  # 'easy', 'medium', 'hard'
    max_score: Optional[float] = None

    class_: Optional[Class] = Relationship(back_populates="assignments")
    questions: List["Question"] = Relationship(back_populates="assignment")
    assessment_images: List["AssessmentImage"] = Relationship(back_populates="assignment")


class Question(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    assignment_id: int = Field(foreign_key="assignment.id")
    text: str
    options: Optional[str] = None
    correct_answer: Optional[str] = None
    rubric_criteria: Optional[str] = None

    formula_template: Optional[str] = None  # Шаблон формулы в LaTeX
    step_by_step_solution: Optional[str] = None  # JSON с пошаговым решением
    common_mistakes: Optional[str] = None  # JSON с типичными ошибками

    assignment: Optional[Assignment] = Relationship(back_populates="questions")


class StudentAnswer(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: Optional[int] = None
    question_id: Optional[int] = None
    text_answer: Optional[str] = None
    file_url: Optional[str] = None
    score: Optional[float] = None
    feedback: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Rubric(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    assignment_id: int = Field(foreign_key="assignment.id")
    criteria_json: Optional[str] = None
    max_score: Optional[float] = None


class Result(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: int
    assignment_id: int
    total_score: Optional[float] = None
    feedback_summary: Optional[str] = None


class MistakeStat(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    class_id: int = Field(foreign_key="class.id")
    question_id: int = Field(foreign_key="question.id")
    mistake_type: str  # например: "grammar", "spelling", "concept"
    count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ActionLog(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    user_id: int
    action_type: str
    target_type: str
    target_id: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: str | None = None


class RefreshToken(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    token: str
    jti: Optional[str] = Field(index=True, nullable=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime


class RevokedToken(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    jti: str = Field(index=True, unique=True)
    user_id: Optional[int] = Field(index=True, default=None, nullable=True)
    token_type: Optional[str] = None  # "access" or "refresh"
    revoked_at: datetime = Field(default_factory=datetime.utcnow)

class AssessmentImage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    assignment_id: int = Field(foreign_key="assignment.id")
    class_id: Optional[int] = Field(foreign_key="class.id", default=None)
    student_id: int = Field(foreign_key="user.id")
    question_id: Optional[int] = Field(foreign_key="question.id", default=None)

    # Пути к файлам
    original_image_path: str  # Относительный путь в контейнере
    processed_image_path: Optional[str] = None
    thumbnail_path: Optional[str] = None

    # Метаданные файла
    file_name: str
    file_size: int  # В байтах
    mime_type: str
    upload_time: datetime = Field(default_factory=datetime.utcnow)

    # Статус обработки
    status: str = Field(default="pending")  # pending, processing, processed, error, queued
    processing_started: Optional[datetime] = None
    processing_completed: Optional[datetime] = None

    # Ошибки
    error_message: Optional[str] = None
    retry_count: int = Field(default=0)

    # Связи
    assignment: Optional[Assignment] = Relationship(back_populates="assessment_images")
    student: Optional[User] = Relationship(back_populates="submitted_works")
    class_: Optional[Class] = Relationship(back_populates="assessment_images")
    recognized_solutions: List["RecognizedSolution"] = Relationship(back_populates="image")

    class Config:
        from_attributes = True


class RecognizedSolution(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    image_id: int = Field(foreign_key="assessmentimage.id")

    # Распознанный текст
    extracted_text: str
    cleaned_text: Optional[str] = None
    text_confidence: Optional[float] = None  # Средняя уверенность OCR

    # Формулы
    extracted_formulas_json: Optional[str] = None  # JSON массив формул
    formulas_count: int = Field(default=0)

    # Ответы
    extracted_answer: Optional[str] = None
    answer_confidence: Optional[float] = None

    # Шаги решения
    solution_steps_json: Optional[str] = None  # JSON с выделенными шагами

    # Confidence scores
    ocr_confidence: Optional[float] = None
    solution_structure_confidence: Optional[float] = None
    formula_confidence: Optional[float] = None
    answer_match_confidence: Optional[float] = None
    total_confidence: Optional[float] = None

    # Классификация
    check_level: str = Field(default="level_3")  # level_1, level_2, level_3
    suggested_grade: Optional[float] = None
    auto_feedback: Optional[str] = None

    # Время обработки
    processing_time_ms: Optional[int] = None
    recognized_at: datetime = Field(default_factory=datetime.utcnow)

    # Связи
    image: Optional[AssessmentImage] = Relationship(back_populates="recognized_solutions")
    assessment_results: List["AssessmentResult"] = Relationship(back_populates="recognized_solution")

    class Config:
        from_attributes = True


class AssessmentResult(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    solution_id: int = Field(foreign_key="recognizedsolution.id")
    teacher_id: Optional[int] = Field(foreign_key="user.id", default=None)

    # Вердикт учителя
    teacher_verdict: Optional[str] = None  # approved, rejected, needs_correction, returned
    teacher_score: Optional[float] = None
    teacher_feedback: Optional[str] = None

    # Исправления учителя
    corrected_text: Optional[str] = None
    corrected_formulas_json: Optional[str] = None
    corrected_answer: Optional[str] = None

    # Системная оценка (для сравнения)
    system_score: Optional[float] = None
    system_feedback: Optional[str] = None

    # Метки для обучения
    used_for_training: bool = Field(default=False)
    training_priority: int = Field(default=0)  # Приоритет для дообучения

    # Временные метки
    verified_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Связи
    recognized_solution: Optional[RecognizedSolution] = Relationship(back_populates="assessment_results")
    teacher: Optional[User] = Relationship(back_populates="verified_results")

    class Config:
        from_attributes = True


class TrainingSample(SQLModel, table=True):
    """Образцы для дообучения OCR"""
    id: Optional[int] = Field(default=None, primary_key=True)

    # Исходные данные
    image_path: str
    original_text: str  # Текст, распознанный системой
    corrected_text: str  # Исправленный учителем текст

    # Метаданные
    subject: Optional[str] = None
    handwriting_style: Optional[str] = None  # child, teen, adult
    image_quality: Optional[str] = None  # good, medium, poor

    # Использование в обучении
    used_in_training: bool = Field(default=False)
    training_iteration: Optional[int] = None
    model_version: Optional[str] = None

    # Временные метки
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    used_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SystemMetrics(SQLModel, table=True):
    """Метрики системы для мониторинга"""
    id: Optional[int] = Field(default=None, primary_key=True)

    # Метрики обработки
    date: datetime = Field(default_factory=datetime.utcnow)
    total_processed: int = Field(default=0)
    level_1_count: int = Field(default=0)
    level_2_count: int = Field(default=0)
    level_3_count: int = Field(default=0)

    # Точность
    avg_ocr_confidence: Optional[float] = None
    avg_processing_time_ms: Optional[float] = None

    # Ошибки
    error_count: int = Field(default=0)
    most_common_error: Optional[str] = None

    class Config:
        from_attributes = True