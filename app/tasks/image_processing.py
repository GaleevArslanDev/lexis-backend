# Обновляем импорты
import os
import time
import logging
from typing import Dict, Any
from sqlmodel import Session, select
from app.db import engine
import json

# Добавляем импорт OCR клиента
from app.ocr_client import get_ocr_client
from app.processing.solution_analyzer import SolutionAnalyzer
from app.processing.confidence_scorer import ConfidenceScorer
from app.processing.sympy_evaluator import SymPyEvaluator

from .celery_app import celery_app

logger = logging.getLogger(__name__)

# Инициализация обработчиков
solution_analyzer = SolutionAnalyzer()
confidence_scorer = ConfidenceScorer()
sympy_evaluator = SymPyEvaluator()

# Получаем OCR клиент
ocr_client = get_ocr_client()


@celery_app.task(bind=True, name="process_assessment_image")
def process_assessment_image(self, image_id: int, image_bytes: bytes = None) -> Dict[str, Any]:
    """Фоновая задача для обработки изображения с работой с использованием OCR API"""
    task_id = self.request.id

    try:
        start_time = time.time()

        with Session(engine) as session:
            from app.models import AssessmentImage, Assignment, Question
            from app.crud.assessment import (
                update_image_status,
                create_recognized_solution,
                update_system_metrics
            )

            # Обновляем статус на "processing"
            image = update_image_status(session, image_id, "processing")
            if not image:
                return {"success": False, "error": "Image not found"}

            logger.info(f"Starting processing for image {image_id} (task: {task_id})")

            # Если байты изображения не переданы, загружаем из файла
            if image_bytes is None and image.original_image_path:
                with open(image.original_image_path, 'rb') as f:
                    image_bytes = f.read()

            if not image_bytes:
                error_msg = "No image data available"
                update_image_status(session, image_id, "error", error_msg)
                return {"success": False, "error": error_msg}

            # 1. OCR распознавание через внешний API
            ocr_result = ocr_client.process_image_bytes(
                image_bytes,
                filename=image.file_name,
                extract_formulas=True
            )

            if not ocr_result.get("success", False):
                error_msg = f"OCR error: {ocr_result.get('error', 'Unknown')}"
                update_image_status(session, image_id, "error", error_msg)
                return {"success": False, "error": error_msg}

            # 2. Анализ решения
            structure_analysis = solution_analyzer.analyze_solution_structure(
                ocr_result.get("full_text", "")
            )

            # 3. Получаем данные задания для сравнения
            assignment = session.get(Assignment, image.assignment_id)
            reference_data = {}

            if assignment:
                reference_data["reference_solution"] = assignment.reference_solution
                reference_data["reference_answer"] = assignment.reference_answer

            # Если есть конкретный вопрос
            if image.question_id:
                question = session.get(Question, image.question_id)
                if question:
                    if not reference_data.get("reference_solution"):
                        reference_data["reference_solution"] = question.step_by_step_solution
                    if not reference_data.get("reference_answer"):
                        reference_data["reference_answer"] = question.correct_answer

            # 4. Сравнение с эталоном
            comparison_result = solution_analyzer.compare_with_reference(
                ocr_result.get("full_text", ""),
                reference_data.get("reference_solution"),
                reference_data.get("reference_answer")
            )

            # 5. Расчет confidence scores
            confidence_data = confidence_scorer.calculate_total_confidence(
                ocr_data={
                    'average_confidence': ocr_result.get('average_confidence', 50.0),
                    'character_count': ocr_result.get('character_count', 0),
                    'quality_assessment': ocr_result.get('quality_assessment', {})
                },
                solution_structure={
                    'calculation_chains': structure_analysis.get('calculation_chains', []),
                    'has_formulas': len(ocr_result.get('formulas', [])) > 0
                },
                formula_data={
                    'formulas': ocr_result.get('formulas', []),
                    'reference_formulas': None
                },
                answer_data={
                    'extracted_answer': structure_analysis.get('extracted_answer'),
                    'reference_answer': reference_data.get('reference_answer'),
                    'calculation_result': comparison_result.get('final_result') if comparison_result else None
                }
            )

            # 6. Создаем запись распознанного решения
            solution_data = {
                "image_id": image_id,
                "extracted_text": ocr_result.get("full_text", ""),
                "text_confidence": ocr_result.get("average_confidence", 0.0),
                "formulas_count": len(ocr_result.get("formulas", [])),
                "extracted_answer": structure_analysis.get('extracted_answer'),
                "answer_confidence": comparison_result.get('comparison_score', 0.0) if comparison_result else None,
                "ocr_confidence": confidence_data['component_scores'].get('ocr_confidence'),
                "solution_structure_confidence": confidence_data['component_scores'].get(
                    'solution_structure_confidence'),
                "formula_confidence": confidence_data['component_scores'].get('formula_confidence'),
                "answer_match_confidence": confidence_data['component_scores'].get('answer_match_confidence'),
                "total_confidence": confidence_data['total_confidence'],
                "check_level": confidence_data['check_level'],
                "suggested_grade": confidence_data['suggested_grade'],
                "auto_feedback": confidence_data['auto_feedback'],
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }

            # Добавляем формулы если есть
            if ocr_result.get("formulas"):
                solution_data["extracted_formulas_json"] = json.dumps(ocr_result.get("formulas", []),
                                                                      ensure_ascii=False)

            # Добавляем шаги решения если есть
            if structure_analysis.get('steps'):
                solution_data["solution_steps_json"] = json.dumps(structure_analysis.get('steps', []),
                                                                  ensure_ascii=False)

            recognized_solution = create_recognized_solution(session, **solution_data)

            # 7. Обновляем статус изображения
            update_image_status(session, image_id, "processed")

            # 8. Обновляем метрики системы
            update_system_metrics(
                session,
                total_processed=1,
                **{f"{confidence_data['check_level']}_count": 1}
            )

            processing_time = int((time.time() - start_time) * 1000)

            logger.info(f"Successfully processed image {image_id} in {processing_time}ms")

            return {
                "success": True,
                "image_id": image_id,
                "solution_id": recognized_solution.id,
                "processing_time_ms": processing_time,
                "confidence_score": confidence_data['total_confidence'],
                "check_level": confidence_data['check_level'],
                "needs_attention": confidence_data['needs_attention'],
                "attention_reasons": confidence_data['attention_reasons']
            }

    except Exception as e:
        logger.error(f"Error processing image {image_id}: {str(e)}", exc_info=True)

        # Обновляем статус в случае ошибки
        try:
            with Session(engine) as session:
                from app.crud.assessment import update_image_status, update_system_metrics
                update_image_status(session, image_id, "error", str(e))
                update_system_metrics(session, error_count=1)
        except:
            pass

        return {
            "success": False,
            "error": str(e),
            "image_id": image_id
        }