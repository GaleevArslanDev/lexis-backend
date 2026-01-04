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
    upload_url: Optional[str] = None
    message: str

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
    check_level: CheckLevel
    suggested_grade: Optional[float]
    auto_feedback: str

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


class BatchUploadRequest(BaseModel):
    assignment_id: int
    class_id: int
    student_works: List[Dict[str, Any]]

    @validator('student_works')
    def validate_student_works(cls, v):
        if not v:
            raise ValueError('student_works cannot be empty')
        return v


class BatchUploadResponse(BaseModel):
    total_uploaded: int
    successful_uploads: List[int]
    failed_uploads: List[Dict[str, Any]]
    batch_id: Optional[str] = None

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