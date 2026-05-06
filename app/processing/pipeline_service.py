"""
Pipeline Service — оркестрирует OCR → LLM → Confidence → Classification.

Ключевые исправления по сравнению с lexis-ocr версией:
1. Cocr считается по наблюдаемым сигналам, а не по self-reported VLM confidence
   (VLM всегда завышает свою уверенность, что делало Ctotal бесполезным).
2. Mtotal = 0.85*Mllm + 0.15*Manswer (Msympy убран — его нет в пайплайне).
3. Пороги классификации скорректированы.
4. Нет зависимости от внутренней queue_service — управление очередью на стороне БД.
"""
import uuid
import time
import asyncio
from typing import Optional
from loguru import logger
import numpy as np

from .ocr_service import ocr_service
from .llm_service import llm_service
from .schemas import (
    OCRRequest, OCRStepResult, LLMAnalysisResult,
    ConfidenceScores, FinalAssessment, AssessmentResponse,
)

# ---------------------------------------------------------------------------
# Пороги (можно вынести в env при необходимости)
# ---------------------------------------------------------------------------
CONFIDENCE_HIGH = 0.78    # Level 1 — автопроверка
CONFIDENCE_MEDIUM = 0.52  # Level 2 — требует внимания
MARK_ATTENTION_THRESHOLD = 0.55  # если Mtotal < этого, всегда Level 2


class PipelineService:
    def __init__(self):
        logger.info("PipelineService initialized (VLM + LLM, no internal queue)")

    # ------------------------------------------------------------------
    # Расчёт Cocr по наблюдаемым сигналам
    # ------------------------------------------------------------------

    def _calculate_ocr_confidence(
        self,
        ocr_results: list[OCRStepResult],
        llm_result: Optional[LLMAnalysisResult],
    ) -> float:
        """
        Вычисляет Cocr на основе наблюдаемых сигналов из результатов,
        а НЕ на основе self-reported confidence VLM.

        Проблема self-reported confidence: VLM всегда возвращает 0.85–0.95
        независимо от реального качества распознавания, что делает
        классификацию бесполезной — все работы попадают в Level 1.

        Наблюдаемые сигналы надёжнее:
        - Количество шагов (задача обычно имеет > 2 шагов)
        - Средняя длина формулы (слишком короткие = вероятно мусор)
        - Наличие математических символов (реальное решение содержит = + - и т.д.)
        - Оценка читаемости от LLM (дисконтированная, т.к. LLM тоже завышает)
        """
        step_count = len(ocr_results)
        if step_count == 0:
            return 0.0

        signals = []

        # Сигнал 1: количество шагов
        # 0 шагов → 0 (уже обработано выше)
        # 1 шаг → подозрительно, скорее всего плохое распознавание
        # 2-3 шага → возможно, но мало
        # 4-8 шагов → норма для задачи
        # >8 → хорошо
        if step_count == 1:
            signals.append(0.35)
        elif step_count <= 3:
            signals.append(0.65)
        elif step_count <= 8:
            signals.append(0.85)
        else:
            signals.append(0.90)

        # Сигнал 2: средняя длина формулы
        # Настоящая формула редко бывает < 5 символов
        avg_len = sum(len(r.formula) for r in ocr_results) / step_count
        if avg_len < 4:
            signals.append(0.20)
        elif avg_len < 8:
            signals.append(0.50)
        elif avg_len < 18:
            signals.append(0.78)
        else:
            signals.append(0.90)

        # Сигнал 3: доля формул с математическими символами
        # Реальное решение почти всегда содержит =, +, -, *, /, ^, \, {, }
        math_chars = set('=+-*/^{}\\()[],.')
        formulas_with_math = sum(
            1 for r in ocr_results
            if any(c in math_chars for c in r.formula)
        )
        math_ratio = formulas_with_math / step_count
        # Линейный сигнал: 0 формул с математикой → 0.1, все → 0.9
        signals.append(0.1 + math_ratio * 0.80)

        # Сигнал 4: оценка читаемости от LLM (с дисконтом 0.80)
        # LLM оценивает осмысленность формул, что объективнее VLM confidence,
        # но LLM тоже склонна к оверконфиденсу → дисконт
        if llm_result and hasattr(llm_result, 'ocr_quality_score'):
            discounted = llm_result.ocr_quality_score * 0.80
            signals.append(discounted)

        c_ocr = float(sum(signals) / len(signals))
        logger.info(
            f"Cocr signals: steps={signals[0]} len={signals[1]} "
            f"math={signals[2]} llm_disc={signals[3] if len(signals) > 3 else 'n/a'} "
            f"→ Cocr={c_ocr}"
        )
        return c_ocr

    # ------------------------------------------------------------------
    # Расчёт всех confidence scores
    # ------------------------------------------------------------------

    def calculate_confidence_scores(
        self,
        ocr_results: list,
        llm_result,
        answer_match: float,
    ) -> ConfidenceScores:
        logger.info("Calculating confidence scores...")

        c_ocr = self._calculate_ocr_confidence(ocr_results, llm_result)

        # Cllm — оценка качества OCR от LLM (тоже дисконтируем, но меньше)
        if llm_result and hasattr(llm_result, 'ocr_quality_score'):
            c_llm = float(llm_result.ocr_quality_score) * 0.90
        else:
            c_llm = 0.5

        # Mllm — оценка решения от LLM
        if llm_result and hasattr(llm_result, 'llm_score'):
            m_llm = float(llm_result.llm_score)
        else:
            m_llm = 0.5

        m_answer = float(answer_match)

        # Ctotal = Cocr * Cllm
        c_total = c_ocr * c_llm

        # Mtotal = 0.85 * Mllm + 0.15 * Manswer
        # (Msympy убран, т.к. SymPy не запускается в текущем пайплайне)
        m_total = 0.85 * m_llm + 0.15 * m_answer

        logger.info(
            f"Scores — Cocr={c_ocr} Cllm={c_llm} → Ctotal={c_total} | "
            f"Mllm={m_llm} Manswer={m_answer} → Mtotal={m_total}"
        )

        return ConfidenceScores(
            c_ocr=c_ocr, c_llm=c_llm,
            m_llm=m_llm, m_answer=m_answer,
            c_total=c_total, m_total=m_total,
        )

    # ------------------------------------------------------------------
    # Классификация
    # ------------------------------------------------------------------

    def classify_confidence_level(self, c_total: float, m_total: float) -> int:
        """
        Level 1 — Автопроверка:   Ctotal >= 0.78 И Mtotal >= 0.55
        Level 2 — Требует внимания: Ctotal >= 0.52 ИЛИ Mtotal < 0.55
        Level 3 — Ручная проверка: всё остальное
        """
        # Плохая оценка — всегда требует внимания учителя
        if m_total < MARK_ATTENTION_THRESHOLD:
            return 2

        if c_total >= CONFIDENCE_HIGH:
            return 1
        if c_total >= CONFIDENCE_MEDIUM:
            return 2
        return 3

    # ------------------------------------------------------------------
    # Проверка ответа
    # ------------------------------------------------------------------

    @staticmethod
    def _check_answer_match(
        student_answer: Optional[str],
        reference_answer: Optional[str],
    ) -> float:
        if not student_answer or not reference_answer:
            return 0.0
        try:
            sv = float(student_answer.strip().replace(',', '.'))
            rv = float(reference_answer.strip().replace(',', '.'))
            return 1.0 if abs(sv - rv) < 1e-6 else 0.0
        except ValueError:
            s1 = student_answer.strip().replace(" ", "").lower()
            s2 = reference_answer.strip().replace(" ", "").lower()
            return 1.0 if s1 == s2 else 0.0

    # ------------------------------------------------------------------
    # Ядро пайплайна
    # ------------------------------------------------------------------

    async def _process_assessment_internal(
        self,
        request: OCRRequest,
        solution_id: str,
    ) -> FinalAssessment:
        logger.info(f"[{solution_id}] Starting assessment pipeline")

        # ШАГ 1: VLM OCR
        try:
            ocr_results = await ocr_service.process_image_async(request.image_path)
        except Exception as e:
            logger.error(f"[{solution_id}] OCR failed: {e}")
            raise

        if not ocr_results:
            raise ValueError("VLM не смог распознать ни одного шага в изображении")

        steps_dicts = [
            r.model_dump() if hasattr(r, 'model_dump') else r.dict()
            for r in ocr_results
        ]
        logger.info(f"[{solution_id}] OCR extracted {len(steps_dicts)} steps")

        final_answer = ocr_service.extract_final_answer(steps_dicts)
        answer_match = self._check_answer_match(final_answer, request.reference_answer)
        logger.info(f"[{solution_id}] Final answer: '{final_answer}', match={answer_match:.2f}")

        # ШАГ 2: LLM анализ
        llm_timeout = int(os.getenv("OPENROUTER_TIMEOUT", "30"))
        try:
            llm_result = await asyncio.wait_for(
                llm_service.analyze_solution(steps_dicts, request.reference_answer or ""),
                timeout=llm_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(f"[{solution_id}] LLM timed out")
            llm_result = None
        except Exception as e:
            logger.error(f"[{solution_id}] LLM failed: {e}")
            llm_result = None

        # ШАГ 3: Confidence scores
        scores = self.calculate_confidence_scores(ocr_results, llm_result, answer_match)

        # ШАГ 4: Классификация
        confidence_level = self.classify_confidence_level(scores.c_total, scores.m_total)
        logger.info(f"[{solution_id}] Level={confidence_level}")

        # Комментарий
        teacher_comment = "Анализ выполнен"
        if llm_result and hasattr(llm_result, 'teacher_comment'):
            teacher_comment = llm_result.teacher_comment

        # Анализ шагов
        steps_analysis = []
        if llm_result and hasattr(llm_result, 'step_correctness'):
            for i, step in enumerate(steps_dicts):
                step_id = step.get("step_id", i + 1)
                correct = False
                explanation = ""

                if i < len(llm_result.step_correctness):
                    sc = llm_result.step_correctness[i]
                    correct = list(sc.values())[0] if isinstance(sc, dict) else bool(sc)

                if i < len(llm_result.step_explanations):
                    explanation = llm_result.step_explanations[i]

                steps_analysis.append({
                    "step_id": step_id,
                    "formula": step.get("formula", ""),
                    "is_correct": correct,
                    "explanation": explanation,
                })

        return FinalAssessment(
            solution_id=solution_id,
            confidence_level=confidence_level,
            confidence_score=scores.c_total,
            mark_score=scores.m_total,
            scores=scores,
            teacher_comment=teacher_comment,
            steps_analysis=steps_analysis,
            execution_time=0.0,  # заполняется в process_assessment
        )

    # ------------------------------------------------------------------
    # Публичный метод
    # ------------------------------------------------------------------

    async def process_assessment(self, request: OCRRequest) -> AssessmentResponse:
        solution_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            assessment = await self._process_assessment_internal(request, solution_id)
            assessment.execution_time = time.time() - start_time
            logger.info(
                f"[{solution_id}] Done in {assessment.execution_time:.2f if assessment.execution_time is not None else 'N/A'}s | "
                f"Level={assessment.confidence_level} "
                f"Ctotal={assessment.confidence_score:.3f if assessment.confidence_score is not None else 'N/A'} "
                f"Mtotal={assessment.mark_score:.3f if assessment.mark_score is not None else 'N/A'}"
            )
            return AssessmentResponse(success=True, assessment=assessment)

        except asyncio.TimeoutError:
            logger.error(f"[{solution_id}] Pipeline timed out")
            return AssessmentResponse(success=False, error="Processing timed out.")
        except ValueError as e:
            logger.error(f"[{solution_id}] Validation error: {e}")
            return AssessmentResponse(success=False, error=str(e))
        except Exception as e:
            logger.error(f"[{solution_id}] Pipeline failed: {e}", exc_info=True)
            return AssessmentResponse(success=False, error=f"Processing failed: {str(e)}")


import os  # noqa: E402 (нужен для env var в _process_assessment_internal)

pipeline_service = PipelineService()