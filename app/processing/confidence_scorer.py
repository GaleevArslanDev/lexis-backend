from typing import Dict, List, Optional, Any
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class CheckLevel(Enum):
    LEVEL_1 = "level_1"  # Автопроверка
    LEVEL_2 = "level_2"  # Требует внимания
    LEVEL_3 = "level_3"  # Ручная проверка


class ConfidenceScorer:
    def __init__(self):
        # Веса для различных компонентов
        self.weights = {
            'ocr_confidence': 0.30,  # Качество распознавания текста
            'solution_structure': 0.25,  # Наличие структуры решения
            'formula_detection': 0.25,  # Наличие и корректность формул
            'answer_match': 0.20  # Совпадение ответа с эталоном
        }

        # Пороги для уровней проверки
        self.thresholds = {
            'level_1': 0.85,  # >= 85% - автопроверка
            'level_2': 0.60,  # 60-84% - требует внимания
            # < 60% - ручная проверка
        }

    def calculate_ocr_confidence(
            self,
            avg_confidence: float,
            text_length: int,
            quality_assessment: Dict[str, Any]
    ) -> float:
        """Рассчитать confidence для OCR"""
        try:
            # Базовый confidence из Tesseract (нормализованный к 0-1)
            base_score = avg_confidence / 100.0

            # Учет длины текста (больше текста = более надежно)
            length_factor = min(text_length / 500.0, 1.0)  # 500 символов = максимум

            # Учет оценки качества
            quality_score = quality_assessment.get('score', 0.5)

            # Комбинируем
            ocr_score = (
                    base_score * 0.5 +
                    length_factor * 0.3 +
                    quality_score * 0.2
            )

            return min(max(ocr_score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating OCR confidence: {str(e)}")
            return 0.5

    def calculate_solution_structure_confidence(
            self,
            text: str,
            calculation_chains: List[List[str]],
            has_formulas: bool
    ) -> float:
        """Оценить структуру решения"""
        try:
            score = 0.0

            # Проверяем наличие ключевых слов
            keywords = ['решение', 'дано', 'найти', 'ответ', 'получим', 'следовательно']
            found_keywords = sum(1 for kw in keywords if kw in text.lower())
            keyword_score = min(found_keywords / 3.0, 1.0)  # 3+ ключевых слова = максимум

            score += keyword_score * 0.3

            # Проверяем наличие цепочек вычислений
            if calculation_chains:
                chain_score = min(len(calculation_chains) / 3.0, 1.0)
                # Учитываем длину самой длинной цепочки
                max_chain_length = max(len(chain) for chain in calculation_chains) if calculation_chains else 0
                length_score = min(max_chain_length / 5.0, 1.0)  # 5+ шагов = хорошо

                score += chain_score * 0.3 + length_score * 0.2
            else:
                # Нет вычислений - штраф
                score += 0.1

            # Наличие формул
            if has_formulas:
                score += 0.2

            return min(max(score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating structure confidence: {str(e)}")
            return 0.3

    def calculate_formula_confidence(
            self,
            formulas: List[Dict[str, Any]],
            reference_formulas: Optional[List[str]] = None
    ) -> float:
        """Оценить confidence для формул"""
        try:
            if not formulas:
                return 0.1  # Нет формул - низкий confidence

            # Средняя уверенность OCR для формул
            avg_formula_confidence = sum(f.get('confidence', 0.0) for f in formulas) / len(formulas)
            avg_formula_confidence_norm = avg_formula_confidence / 100.0

            # Количество формул
            formula_count = len(formulas)
            count_score = min(formula_count / 5.0, 1.0)  # 5+ формул = максимум

            # Сложность формул (по количеству символов)
            avg_symbols = sum(len(f.get('latex', '')) for f in formulas) / len(formulas)
            complexity_score = min(avg_symbols / 20.0, 1.0)  # 20+ символов = сложная формула

            # Если есть эталонные формулы, проверяем соответствие
            match_score = 0.5  # Базовый score если нет эталона

            if reference_formulas:
                # Упрощенная проверка: есть ли похожие формулы
                student_formulas = [f.get('latex', '') for f in formulas]

                # Простая проверка на вхождение подстрок
                matches = 0
                for s_formula in student_formulas:
                    for r_formula in reference_formulas:
                        # Ищем общие математические символы
                        s_symbols = set(c for c in s_formula if c in '+-*/=()[]{}^')
                        r_symbols = set(c for c in r_formula if c in '+-*/=()[]{}^')

                        if s_symbols & r_symbols:  # Есть пересечение символов
                            matches += 1
                            break

                match_score = matches / len(reference_formulas)

            # Комбинируем
            formula_score = (
                    avg_formula_confidence_norm * 0.3 +
                    count_score * 0.2 +
                    complexity_score * 0.2 +
                    match_score * 0.3
            )

            return min(max(formula_score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating formula confidence: {str(e)}")
            return 0.3

    def calculate_answer_match_confidence(
            self,
            extracted_answer: Optional[str],
            reference_answer: Optional[str],
            calculation_result: Optional[float] = None
    ) -> float:
        """Оценить совпадение ответа"""
        try:
            if not extracted_answer and not calculation_result:
                return 0.1  # Нет ответа

            # Если есть эталонный ответ
            if reference_answer:
                if not extracted_answer:
                    return 0.1

                # Пробуем числовое сравнение
                try:
                    extracted_val = float(extracted_answer.replace(',', '.'))
                    reference_val = float(reference_answer.replace(',', '.'))

                    diff = abs(extracted_val - reference_val)
                    relative_diff = diff / max(abs(reference_val), 1e-10)

                    if relative_diff < 0.01:  # 1% допуск
                        return 1.0
                    elif relative_diff < 0.05:  # 5% допуск
                        return 0.7
                    elif relative_diff < 0.10:  # 10% допуск
                        return 0.4
                    else:
                        return 0.1

                except ValueError:
                    # Не числовые ответы - проверяем строковое совпадение
                    if extracted_answer.lower() == reference_answer.lower():
                        return 1.0
                    else:
                        # Частичное совпадение
                        if extracted_answer in reference_answer or reference_answer in extracted_answer:
                            return 0.6
                        else:
                            return 0.2

            # Если нет эталона, но есть результат вычислений
            if calculation_result is not None:
                try:
                    if extracted_answer:
                        extracted_val = float(extracted_answer.replace(',', '.'))
                        diff = abs(extracted_val - calculation_result)
                        relative_diff = diff / max(abs(calculation_result), 1e-10)

                        if relative_diff < 0.01:
                            return 0.9
                        elif relative_diff < 0.05:
                            return 0.6
                        else:
                            return 0.3
                    else:
                        # Нет извлеченного ответа, но есть вычисления
                        return 0.5
                except:
                    return 0.3

            # Только извлеченный ответ без проверки
            return 0.5 if extracted_answer else 0.1

        except Exception as e:
            logger.error(f"Error calculating answer match confidence: {str(e)}")
            return 0.2

    def calculate_total_confidence(
            self,
            ocr_data: Dict[str, Any],
            solution_structure: Dict[str, Any],
            formula_data: Dict[str, Any],
            answer_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Рассчитать общий confidence score"""
        try:
            # Вычисляем отдельные confidence scores
            ocr_confidence = self.calculate_ocr_confidence(
                ocr_data.get('average_confidence', 50.0),
                ocr_data.get('character_count', 0),
                ocr_data.get('quality_assessment', {})
            )

            solution_structure_confidence = self.calculate_solution_structure_confidence(
                ocr_data.get('full_text', ''),
                solution_structure.get('calculation_chains', []),
                formula_data.get('has_formulas', False)
            )

            formula_confidence = self.calculate_formula_confidence(
                formula_data.get('formulas', []),
                formula_data.get('reference_formulas')
            )

            answer_match_confidence = self.calculate_answer_match_confidence(
                answer_data.get('extracted_answer'),
                answer_data.get('reference_answer'),
                answer_data.get('calculation_result')
            )

            # Взвешенная сумма
            total_confidence = (
                    ocr_confidence * self.weights['ocr_confidence'] +
                    solution_structure_confidence * self.weights['solution_structure'] +
                    formula_confidence * self.weights['formula_detection'] +
                    answer_match_confidence * self.weights['answer_match']
            )

            # Определяем уровень проверки
            check_level = self.determine_check_level(total_confidence)

            # Генерируем автоматическую обратную связь
            auto_feedback = self.generate_auto_feedback(
                total_confidence,
                check_level,
                {
                    'ocr_confidence': ocr_confidence,
                    'solution_structure_confidence': solution_structure_confidence,
                    'formula_confidence': formula_confidence,
                    'answer_match_confidence': answer_match_confidence
                }
            )

            # Предлагаем оценку (простая эвристика)
            suggested_grade = self.suggest_grade(total_confidence, answer_match_confidence)

            return {
                'total_confidence': round(total_confidence, 3),
                'check_level': check_level.value,
                'component_scores': {
                    'ocr_confidence': round(ocr_confidence, 3),
                    'solution_structure_confidence': round(solution_structure_confidence, 3),
                    'formula_confidence': round(formula_confidence, 3),
                    'answer_match_confidence': round(answer_match_confidence, 3)
                },
                'auto_feedback': auto_feedback,
                'suggested_grade': suggested_grade,
                'needs_attention': check_level != CheckLevel.LEVEL_1,
                'attention_reasons': self.get_attention_reasons(
                    ocr_confidence,
                    solution_structure_confidence,
                    formula_confidence,
                    answer_match_confidence
                )
            }

        except Exception as e:
            logger.error(f"Error calculating total confidence: {str(e)}")
            return {
                'total_confidence': 0.0,
                'check_level': CheckLevel.LEVEL_3.value,
                'component_scores': {},
                'auto_feedback': f'Ошибка при анализе: {str(e)}',
                'suggested_grade': None,
                'needs_attention': True,
                'attention_reasons': ['system_error']
            }

    def determine_check_level(self, total_confidence: float) -> CheckLevel:
        """Определить уровень проверки на основе confidence"""
        if total_confidence >= self.thresholds['level_1']:
            return CheckLevel.LEVEL_1
        elif total_confidence >= self.thresholds['level_2']:
            return CheckLevel.LEVEL_2
        else:
            return CheckLevel.LEVEL_3

    def generate_auto_feedback(
            self,
            total_confidence: float,
            check_level: CheckLevel,
            component_scores: Dict[str, float]
    ) -> str:
        """Сгенерировать автоматическую обратную связь"""

        feedback_parts = []

        # Общий комментарий
        if check_level == CheckLevel.LEVEL_1:
            feedback_parts.append("✅ Работа распознана хорошо, можно проверить автоматически.")
        elif check_level == CheckLevel.LEVEL_2:
            feedback_parts.append("⚠️ Требуется внимание учителя: некоторые элементы требуют проверки.")
        else:
            feedback_parts.append("❌ Требуется ручная проверка: низкое качество распознавания.")

        # Конкретные замечания
        if component_scores.get('ocr_confidence', 1.0) < 0.6:
            feedback_parts.append("- Низкое качество распознавания текста")

        if component_scores.get('solution_structure_confidence', 1.0) < 0.5:
            feedback_parts.append("- Неясная структура решения")

        if component_scores.get('formula_confidence', 1.0) < 0.5:
            feedback_parts.append("- Проблемы с распознаванием формул")

        if component_scores.get('answer_match_confidence', 1.0) < 0.5:
            feedback_parts.append("- Расхождение в ответе")

        return '\n'.join(feedback_parts)

    def suggest_grade(self, total_confidence: float, answer_confidence: float) -> Optional[float]:
        """Предложить оценку (0-100)"""
        if answer_confidence < 0.3:
            return None  # Недостаточно данных

        # Базовый балл на основе confidence
        base_score = total_confidence * 100

        # Корректировка на основе answer_confidence
        if answer_confidence > 0.8:
            # Хорошее совпадение ответа
            return min(base_score * 1.1, 100.0)
        elif answer_confidence > 0.5:
            # Удовлетворительное совпадение
            return base_score
        else:
            # Плохое совпадение ответа
            return base_score * 0.7

    def get_attention_reasons(
            self,
            ocr_confidence: float,
            structure_confidence: float,
            formula_confidence: float,
            answer_confidence: float
    ) -> List[str]:
        """Получить причины, требующие внимания"""
        reasons = []

        thresholds = {
            'ocr': 0.6,
            'structure': 0.5,
            'formula': 0.5,
            'answer': 0.5
        }

        if ocr_confidence < thresholds['ocr']:
            reasons.append('low_ocr_quality')

        if structure_confidence < thresholds['structure']:
            reasons.append('unclear_solution_structure')

        if formula_confidence < thresholds['formula']:
            reasons.append('formula_recognition_issues')

        if answer_confidence < thresholds['answer']:
            reasons.append('answer_mismatch')

        return reasons