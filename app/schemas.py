from pydantic import BaseModel, EmailStr, Field, validator, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# Существующие схемы с обновленным Config

class UserCreate(BaseModel):
    email: EmailStr
    name: str
    surname: str
    password: str
    role: Optional[str] = "teacher"


class UserRead(BaseModel):
    id: int
    email: EmailStr
    name: str
    surname: str
    role: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str
    jti: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ClassCreate(BaseModel):
    name: str
    school_id: Optional[int] = None


class AssignmentCreate(BaseModel):
    class_id: int
    title: str
    description: Optional[str] = None
    type: str = "essay"

    # Новые поля для AssessIt
    reference_solution: Optional[str] = None
    reference_answer: Optional[str] = None
    subject: Optional[str] = None  # 'math', 'physics', 'chemistry'
    difficulty: Optional[str] = None  # 'easy', 'medium', 'hard'
    max_score: Optional[float] = None

    @validator('max_score')
    def validate_max_score(cls, v):
        if v is not None and v <= 0:
            raise ValueError('max_score must be positive')
        return v

    @validator('subject')
    def validate_subject(cls, v):
        if v is not None and v not in ['math', 'physics', 'chemistry']:
            raise ValueError('subject must be one of: math, physics, chemistry')
        return v

    @validator('difficulty')
    def validate_difficulty(cls, v):
        if v is not None and v not in ['easy', 'medium', 'hard']:
            raise ValueError('difficulty must be one of: easy, medium, hard')
        return v

    model_config = ConfigDict(from_attributes=True)


class AssignmentResponse(BaseModel):
    id: int
    class_id: int
    title: str
    description: Optional[str] = None
    type: str
    created_at: datetime
    due_date: Optional[datetime] = None

    # Поля AssessIt
    reference_solution: Optional[str] = None
    reference_answer: Optional[str] = None
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    max_score: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class QuestionCreate(BaseModel):
    assignment_id: int
    text: str
    options: Optional[str] = None
    correct_answer: Optional[str] = None


class ClassUpdate(BaseModel):
    name: Optional[str] = None
    school_id: Optional[int] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class SchoolCreate(BaseModel):
    name: str
    address: Optional[str] = None


class SchoolUpdate(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None


class RefreshRequest(BaseModel):
    refresh_token: str


# ========== НОВЫЕ СХЕМЫ ДЛЯ ASSESSIT ==========

class CheckLevel(str, Enum):
    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"


class AssessmentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    ERROR = "error"
    QUEUED = "queued"


class UploadWorkRequest(BaseModel):
    assignment_id: int
    class_id: Optional[int] = None
    question_id: Optional[int] = None

    @validator('assignment_id')
    def validate_assignment_id(cls, v):
        if v <= 0:
            raise ValueError('assignment_id must be positive')
        return v


class UploadWorkResponse(BaseModel):
    work_id: int
    message: str
    error: Optional[str] = None
    confidence_score: Optional[float] = None
    check_level: Optional[str] = None
    solution_id: Optional[str] = None
    queue_position: Optional[int] = None
    status: str = "processed"  # processed, queued, error

    model_config = ConfigDict(from_attributes=True)


class ProcessedWorkResponse(BaseModel):
    work_id: int
    status: str
    confidence_score: Optional[float] = None
    check_level: Optional[str] = None
    extracted_text: Optional[str] = None
    extracted_answer: Optional[str] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    solution_id: Optional[str] = None  # UUID из ML API
    mark_score: Optional[float] = None  # Итоговая оценка
    teacher_comment: Optional[str] = None  # Комментарий от LLM
    
    model_config = ConfigDict(from_attributes=True)


class StepAnalysis(BaseModel):
    step_id: int
    is_correct: bool
    formula_match: bool
    found_formula: Optional[str] = None
    expected_formula: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class FormulaInfo(BaseModel):
    latex: str
    confidence: float
    position: Dict[str, int]
    is_correct: Optional[bool] = None


class SolutionStep(BaseModel):
    step_number: int
    text: str
    formula: Optional[str] = None
    is_correct: Optional[bool] = None


class DetailedAnalysisResponse(BaseModel):
    work_id: int
    student_id: int
    assignment_id: int

    # Распознанный контент
    full_text: str
    formulas: List[FormulaInfo]
    extracted_answer: Optional[str]
    reference_answer: Optional[str]

    # Confidence scores
    ocr_confidence: float
    solution_structure_confidence: float
    formula_confidence: float
    answer_match_confidence: float
    total_confidence: float

    # Классификация
    check_level: str  # level_1, level_2, level_3
    suggested_grade: Optional[float]
    auto_feedback: str

    # Детальный анализ шагов (новое)
    steps_analysis: Optional[List[StepAnalysis]] = None
    
    # Детальные скоринг-метрики (новое)
    c_ocr: Optional[float] = None
    c_llm: Optional[float] = None
    m_sympy: Optional[float] = None
    m_llm: Optional[float] = None
    m_answer: Optional[float] = None
    mark_score: Optional[float] = None

    # Визуализация
    annotated_image_url: Optional[str] = None
    problem_areas: List[Dict[str, Any]] = []

    # Время
    processing_time_ms: int
    processed_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TeacherVerificationRequest(BaseModel):
    verdict: str = Field(..., pattern="^(approved|rejected|needs_correction|returned)$")
    score: Optional[float] = Field(None, ge=0, le=100)
    feedback: Optional[str] = None
    corrected_text: Optional[str] = None
    corrected_answer: Optional[str] = None

    @validator('score')
    def validate_score(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Score must be between 0 and 100')
        return v


class TeacherVerificationResponse(BaseModel):
    result_id: int
    work_id: int
    teacher_id: int
    verdict: str
    score: Optional[float]
    feedback: Optional[str]
    verified_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ClassAssessmentSummary(BaseModel):
    class_id: int
    assignment_id: int
    total_students: int
    submitted_works: int
    processed_works: int

    # Распределение по уровням
    level_1_count: int
    level_2_count: int
    level_3_count: int

    # Средние показатели
    avg_confidence: Optional[float]
    avg_processing_time_ms: Optional[float]
    common_errors: List[Dict[str, Any]]

    # Временные рамки
    first_upload: Optional[datetime]
    last_upload: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


class StudentWorkItem(BaseModel):
    work_id: int
    student_id: int
    student_name: str
    student_surname: str
    status: str
    check_level: Optional[str]
    confidence_score: Optional[float]
    teacher_score: Optional[float]
    system_score: Optional[float]
    uploaded_at: datetime
    processed_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)


# Схемы для дашборда
class DashboardStats(BaseModel):
    total_assignments: int
    total_works_processed: int
    time_saved_hours: float
    avg_confidence: float
    accuracy_rate: float

    # Распределение по дням
    daily_stats: List[Dict[str, Any]]

    model_config = ConfigDict(from_attributes=True)


class CommonError(BaseModel):
    error_type: str
    count: int
    percentage: float
    examples: List[str]

    model_config = ConfigDict(from_attributes=True)


class PaginationInfo(BaseModel):
    page: int
    page_size: int
    total: int
    total_pages: int
    has_next: bool
    has_prev: bool


class PaginatedResponse(BaseModel):
    items: List[Any]
    pagination: PaginationInfo


class WorkVerifiedResponse(BaseModel):
    work_id: int
    verdict: str
    score: Optional[float]
    verified_at: datetime

    model_config = ConfigDict(from_attributes=True)

class BatchUploadItem(BaseModel):
    """Один элемент пакетной загрузки"""
    student_id: int
    file: bytes  # не сериализуется, используется для передачи
    filename: str

class BatchUploadRequest(BaseModel):
    """Запрос на пакетную загрузку"""
    assignment_id: int
    class_id: int
    works: List[Dict[str, Any]]  # будет содержать student_id и file

    @validator('works')
    def validate_works(cls, v):
        if not v:
            raise ValueError('works cannot be empty')
        return v


class BatchUploadResponse(BaseModel):
    """Ответ на пакетную загрузку"""
    batch_id: str
    total_submitted: int
    queue_position: int
    estimated_wait_seconds: int
    status: str = "queued"
    task_ids: List[str] = []
    failed_files: Optional[List[Dict[str, str]]] = None
    partial_success: Optional[bool] = False
    message: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class QueueItemStatus(BaseModel):
    """Статус элемента в очереди"""
    work_id: int
    student_id: int
    student_name: str
    position: int
    status: str  # queued, processing, completed, failed
    queued_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class QueueStatusResponse(BaseModel):
    """Полный статус очереди"""
    queue_size: int
    processing: int
    completed: int
    failed: int
    items: List[QueueItemStatus]
    estimated_wait_seconds: int
    avg_processing_time: float

class SystemHealthResponse(BaseModel):
    """Health check всей системы"""
    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    version: str
    components: Dict[str, Any] = {
        "database": {"status": str, "latency_ms": float},
        "ocr_api": {"status": str, "latency_ms": float},
        "websocket": {"status": str, "connections": int},
        "queue": {"status": str, "size": int}
    }
