# Инициализация модулей обработки
from .solution_analyzer import SolutionAnalyzer
from .confidence_scorer import ConfidenceScorer
from .sympy_evaluator import SymPyEvaluator
from .ocr_service import ocr_service
from .llm_service import llm_service
from .pipeline_service import pipeline_service
from .schemas import OCRRequest, AssessmentResponse, FinalAssessment

__all__ = [
    # Старые модули (используются в ConfidenceScorer/SolutionAnalyzer)
    "SolutionAnalyzer",
    "ConfidenceScorer",
    "SymPyEvaluator",
    # Новые сервисы
    "ocr_service",
    "llm_service",
    "pipeline_service",
    # Схемы
    "OCRRequest",
    "AssessmentResponse",
    "FinalAssessment",
]