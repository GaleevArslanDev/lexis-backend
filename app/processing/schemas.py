"""
Внутренние схемы ML-pipeline.
Не экспортируются наружу через API напрямую.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class OCRRequest(BaseModel):
    image_path: str
    reference_answer: Optional[str] = None


class OCRStepResult(BaseModel):
    step_id: int
    formula: str
    confidence: float
    bbox: Dict[str, int] = Field(default_factory=lambda: {"x": 0, "y": 0, "width": 0, "height": 0})


class LLMAnalysisResult(BaseModel):
    reasoning_logic: str
    step_correctness: List[Dict]
    step_explanations: List[str]
    hidden_errors: List[str]
    teacher_comment: str
    llm_score: float
    ocr_quality_score: float
    confidence: float


class ConfidenceScores(BaseModel):
    c_ocr: float
    c_llm: float
    m_llm: float
    m_answer: float
    c_total: float
    m_total: float


class FinalAssessment(BaseModel):
    solution_id: str
    confidence_level: int      # 1, 2 или 3
    confidence_score: float    # Ctotal
    mark_score: float          # Mtotal
    scores: ConfidenceScores
    teacher_comment: str
    steps_analysis: List[Dict[str, Any]]
    execution_time: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AssessmentResponse(BaseModel):
    success: bool
    assessment: Optional[FinalAssessment] = None
    error: Optional[str] = None