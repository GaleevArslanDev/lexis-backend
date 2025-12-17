import os
import time
import logging
from typing import Dict, Any
from sqlmodel import Session, select
from app.db import engine
from app.crud.assessment import (
    update_image_status,
    create_recognized_solution,
    update_system_metrics
)
from app.processing import (
    ImagePreprocessor,
    OCREngine,
    SolutionAnalyzer,
    ConfidenceScorer,
    SymPyEvaluator
)
from .celery_app import celery_app

logger = logging.getLogger(__name__)

# Инициализация обработчиков
image_preprocessor = ImagePreprocessor()
ocr_engine = OCREngine(language=os.getenv("OCR_LANGUAGE", "rus+eng"))
solution_analyzer = SolutionAnalyzer()
confidence_scorer = ConfidenceScorer()
sympy_evaluator = SymPyEvaluator()


@celery_app.task(bind=True, name="process_assessment_image")
def process_assessment_image(self, image_id: int) -> Dict[str, Any]:
    """Фоновая задача для обработки изображения с работой"""
    task_id = self.request.id

    try:
        start_time = time.time()

        with Session(engine) as session:
            # Обновляем статус на "processing"
            image = update_image_status(session, image_id, "processing")
            if not image:
                return {"success": False, "error": "Image not found"}

            logger.info(f"Starting processing for image {image_id} (task: {task_id})")

            # 1. Обработка изображения
            processed_dir = os.getenv("PROCESSED_DIR", "/app/processed")
            preprocessing_result = image_preprocessor.process_pipeline(
                image.original_image_path,
                processed_dir
            )

            if not preprocessing_result.get("success", False):
                error_msg = preprocessing_result.get("error", "Unknown preprocessing error")
                update_image_status(session, image_id, "error", error_msg)
                return {"success": False, "error": error_msg}

            # Обновляем пути к обработанным файлам
            image.processed_image_path = preprocessing_result.get("processed_path")
            image.thumbnail_path = preprocessing_result.get("thumbnail_path")
            session.commit()

            # 2. OCR распознавание
            ocr_result = ocr_engine.process_complete_page(
                image.original_image_path,
                extract_formulas=True,
                extract_answers=True
            )

            if "error" in ocr_result:
                error_msg = f"OCR error: {ocr_result['error']}"
                update_image_status(session, image_id, "error", error_msg)
                return {"success": False, "error": error_msg}

            # 3. Анализ решения
            structure_analysis = solution_analyzer.analyze_solution_structure(
                ocr_result.get("full_text", "")
            )

            # 4. Получаем данные задания для сравнения
            from app.models import Assignment, Question
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

            # 5. Сравнение с эталоном
            comparison_result = solution_analyzer.compare_with_reference(
                ocr_result.get("full_text", ""),
                reference_data.get("reference_solution"),
                reference_data.get("reference_answer")
            )

            # 6. Расчет confidence scores
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
                    'reference_formulas': None  # Можно добавить из эталона
                },
                answer_data={
                    'extracted_answer': structure_analysis.get('extracted_answer'),
                    'reference_answer': reference_data.get('reference_answer'),
                    'calculation_result': comparison_result.get('final_result') if comparison_result else None
                }
            )

            # 7. Создаем запись распознанного решения
            recognized_solution = create_recognized_solution(
                session,
                image_id=image_id,
                extracted_text=ocr_result.get("full_text", ""),
                text_confidence=ocr_result.get("average_confidence", 0.0),
                extracted_formulas_json=(
                    json.dumps(ocr_result.get("formulas", []), ensure_ascii=False)
                    if ocr_result.get("formulas") else None
                ),
                formulas_count=len(ocr_result.get("formulas", [])),
                extracted_answer=structure_analysis.get('extracted_answer'),
                answer_confidence=comparison_result.get('comparison_score', 0.0) if comparison_result else None,
                solution_steps_json=(
                    json.dumps(structure_analysis.get('steps', []), ensure_ascii=False)
                    if structure_analysis.get('steps') else None
                ),
                ocr_confidence=confidence_data['component_scores'].get('ocr_confidence'),
                solution_structure_confidence=confidence_data['component_scores'].get('solution_structure_confidence'),
                formula_confidence=confidence_data['component_scores'].get('formula_confidence'),
                answer_match_confidence=confidence_data['component_scores'].get('answer_match_confidence'),
                total_confidence=confidence_data['total_confidence'],
                check_level=confidence_data['check_level'],
                suggested_grade=confidence_data['suggested_grade'],
                auto_feedback=confidence_data['auto_feedback'],
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

            # 8. Обновляем статус изображения
            update_image_status(session, image_id, "processed")

            # 9. Обновляем метрики системы
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
                update_image_status(session, image_id, "error", str(e))

                # Обновляем метрики ошибок
                update_system_metrics(session, error_count=1)
        except:
            pass

        return {
            "success": False,
            "error": str(e),
            "image_id": image_id
        }


@celery_app.task(bind=True, name="batch_process_images")
def batch_process_images(self, image_ids: list) -> Dict[str, Any]:
    """Пакетная обработка нескольких изображений"""
    results = {
        "total": len(image_ids),
        "successful": 0,
        "failed": 0,
        "results": [],
        "total_time": 0
    }

    start_time = time.time()

    for image_id in image_ids:
        try:
            result = process_assessment_image.delay(image_id).get(timeout=300)
            results["results"].append(result)

            if result.get("success"):
                results["successful"] += 1
            else:
                results["failed"] += 1

        except Exception as e:
            logger.error(f"Error in batch processing for image {image_id}: {str(e)}")
            results["failed"] += 1
            results["results"].append({
                "image_id": image_id,
                "success": False,
                "error": str(e)
            })

    results["total_time"] = int((time.time() - start_time) * 1000)

    return results


import json  # Добавляем импорт для json


@celery_app.task(name="retry_failed_images")
def retry_failed_images(max_retries: int = 3) -> Dict[str, Any]:
    """Повторная обработка неудачных задач"""
    with Session(engine) as session:
        from app.models import AssessmentImage

        # Ищем изображения со статусом error и retry_count < max_retries
        statement = select(AssessmentImage).where(
            AssessmentImage.status == "error",
            AssessmentImage.retry_count < max_retries
        )

        failed_images = session.exec(statement).all()

        results = []
        for image in failed_images:
            try:
                # Запускаем повторную обработку
                result = process_assessment_image.delay(image.id).get(timeout=300)
                results.append({
                    "image_id": image.id,
                    "success": result.get("success", False)
                })
            except Exception as e:
                logger.error(f"Retry failed for image {image.id}: {str(e)}")
                results.append({
                    "image_id": image.id,
                    "success": False,
                    "error": str(e)
                })

        return {
            "total_retried": len(failed_images),
            "results": results
        }