import re
import json
from typing import Dict, List, Optional, Tuple, Any
import logging
from .sympy_evaluator import SymPyEvaluator

logger = logging.getLogger(__name__)


class SolutionAnalyzer:
    def __init__(self):
        self.sympy_evaluator = SymPyEvaluator()

        # Паттерны для поиска ключевых элементов
        self.patterns = {
            'given': r'(?:дано|given|условие)[:;\s]*(.+)',
            'find': r'(?:найти|find|требуется)[:;\s]*(.+)',
            'solution': r'(?:решение|solution)[:;\s]*',
            'answer': r'(?:ответ|answer)[:;\s=]*([^\n]+)',
            'calculation': r'([+-]?\d*\.?\d+\s*[+\-*/^]\s*[+-]?\d*\.?\d+\s*=\s*[+-]?\d*\.?\d+)',
            'formula': r'([a-zA-Zα-ωΑ-Ω]+\s*=\s*[^=\n]+)'
        }

    def analyze_solution_structure(self, text: str) -> Dict[str, Any]:
        """Проанализировать структуру решения"""
        try:
            text_lower = text.lower()

            # Проверяем наличие ключевых разделов
            has_given = bool(re.search(self.patterns['given'], text_lower, re.IGNORECASE))
            has_find = bool(re.search(self.patterns['find'], text_lower, re.IGNORECASE))
            has_solution = bool(re.search(self.patterns['solution'], text_lower, re.IGNORECASE))
            has_answer = bool(re.search(self.patterns['answer'], text_lower, re.IGNORECASE))

            # Извлекаем цепочки вычислений
            calculation_chains = self.extract_calculation_chains(text)

            # Извлекаем формулы
            formulas = self.extract_formulas(text)

            # Определяем тип решения
            solution_type = self.determine_solution_type(
                has_given, has_find, has_solution, has_answer,
                calculation_chains, formulas
            )

            # Оцениваем полноту решения
            completeness_score = self.calculate_completeness_score(
                has_given, has_find, has_solution, has_answer,
                len(calculation_chains), len(formulas)
            )

            # Извлекаем шаги решения
            steps = self.extract_solution_steps(text)

            return {
                'has_given': has_given,
                'has_find': has_find,
                'has_solution_section': has_solution,
                'has_answer': has_answer,
                'calculation_chains': calculation_chains,
                'formulas': formulas,
                'solution_type': solution_type,
                'completeness_score': completeness_score,
                'steps': steps,
                'step_count': len(steps),
                'structure_quality': self.assess_structure_quality(completeness_score)
            }

        except Exception as e:
            logger.error(f"Error analyzing solution structure: {str(e)}")
            return {
                'has_given': False,
                'has_find': False,
                'has_solution_section': False,
                'has_answer': False,
                'calculation_chains': [],
                'formulas': [],
                'solution_type': 'unknown',
                'completeness_score': 0.0,
                'steps': [],
                'step_count': 0,
                'structure_quality': 'poor'
            }

    def extract_calculation_chains(self, text: str) -> List[List[str]]:
        """Извлечь цепочки вычислений"""
        # Используем SymPy evaluator для извлечения цепочек
        return self.sympy_evaluator.extract_calculation_chains(text)

    def extract_formulas(self, text: str) -> List[Dict[str, Any]]:
        """Извлечь формулы из текста"""
        formulas = []

        # Ищем строки с формулами
        lines = text.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()

            # Проверяем, похожа ли строка на формулу
            if self.is_likely_formula(line):
                # Пробуем разобрать как формулу
                parsed = self.sympy_evaluator.parse_expression(line)

                if parsed:
                    formulas.append({
                        'text': line,
                        'line_number': i,
                        'parsed': str(parsed),
                        'symbol_count': len(line),
                        'has_equals': '=' in line,
                        'is_complex': self.is_complex_formula(line)
                    })

        return formulas

    def is_likely_formula(self, text: str) -> bool:
        """Проверить, похожа ли строка на формулу"""
        # Должна содержать математические символы
        math_symbols = set('=+-*/^()[]{}<>|√∫∑∏∂∆∇≈≠≤≥')
        has_math = any(c in math_symbols for c in text)

        # Должна содержать буквы или цифры
        has_alnum = any(c.isalnum() for c in text)

        # Не должна быть слишком длинной для простой формулы
        not_too_long = len(text) < 100

        return has_math and has_alnum and not_too_long

    def is_complex_formula(self, text: str) -> bool:
        """Проверить, сложная ли формула"""
        # Считаем количество уникальных математических операций
        operations = set('+-*/^')
        operation_count = sum(1 for c in text if c in operations)

        # Считаем скобки
        paren_count = text.count('(') + text.count(')')

        return operation_count > 2 or paren_count > 1

    def determine_solution_type(
            self,
            has_given: bool,
            has_find: bool,
            has_solution: bool,
            has_answer: bool,
            calculation_chains: List[List[str]],
            formulas: List[Dict[str, Any]]
    ) -> str:
        """Определить тип решения"""
        if has_given and has_find and has_solution:
            return "structured"  # Структурированное решение
        elif calculation_chains:
            if formulas:
                return "calculation_with_formulas"  # Вычисления с формулами
            else:
                return "pure_calculation"  # Только вычисления
        elif formulas:
            return "formula_only"  # Только формулы
        elif has_answer:
            return "answer_only"  # Только ответ
        else:
            return "unstructured"  # Неструктурированный текст

    def calculate_completeness_score(
            self,
            has_given: bool,
            has_find: bool,
            has_solution: bool,
            has_answer: bool,
            chain_count: int,
            formula_count: int
    ) -> float:
        """Рассчитать оценку полноты решения"""
        score = 0.0

        # Наличие ключевых разделов
        if has_given:
            score += 0.15
        if has_find:
            score += 0.15
        if has_solution:
            score += 0.20
        if has_answer:
            score += 0.20

        # Наличие вычислений
        if chain_count > 0:
            chain_score = min(chain_count / 3.0, 0.15)
            score += chain_score

        # Наличие формул
        if formula_count > 0:
            formula_score = min(formula_count / 3.0, 0.15)
            score += formula_score

        return min(score, 1.0)

    def extract_solution_steps(self, text: str) -> List[Dict[str, Any]]:
        """Извлечь шаги решения"""
        steps = []

        # Разделяем на строки
        lines = text.split('\n')

        step_number = 1
        current_step = []

        for i, line in enumerate(lines):
            line = line.strip()

            if not line:
                continue

            # Проверяем, начинается ли новая строка с нового шага
            is_new_step = (
                    line.lower().startswith(('шаг', 'step', '1.', '2.', '3.', '4.', '5.')) or
                    re.match(r'^\d+[\.\)]\s', line) or
                    line.startswith('-') or
                    line.startswith('•')
            )

            if is_new_step and current_step:
                # Сохраняем предыдущий шаг
                step_text = ' '.join(current_step)
                steps.append({
                    'step_number': step_number,
                    'text': step_text,
                    'line_numbers': list(range(i - len(current_step), i)),
                    'has_formula': self.is_likely_formula(step_text),
                    'has_calculation': bool(re.search(r'\d+\s*[+\-*/]\s*\d+\s*=', step_text))
                })

                step_number += 1
                current_step = [line]
            else:
                current_step.append(line)

        # Добавляем последний шаг
        if current_step:
            step_text = ' '.join(current_step)
            steps.append({
                'step_number': step_number,
                'text': step_text,
                'line_numbers': list(range(len(lines) - len(current_step), len(lines))),
                'has_formula': self.is_likely_formula(step_text),
                'has_calculation': bool(re.search(r'\d+\s*[+\-*/]\s*\d+\s*=', step_text))
            })

        # Если не нашли структурированных шагов, создаем один шаг из всего текста
        if not steps and text.strip():
            steps.append({
                'step_number': 1,
                'text': text.strip(),
                'line_numbers': list(range(len(lines))),
                'has_formula': self.is_likely_formula(text),
                'has_calculation': bool(re.search(r'\d+\s*[+\-*/]\s*\d+\s*=', text))
            })

        return steps

    def assess_structure_quality(self, completeness_score: float) -> str:
        """Оценить качество структуры"""
        if completeness_score >= 0.8:
            return "excellent"
        elif completeness_score >= 0.6:
            return "good"
        elif completeness_score >= 0.4:
            return "fair"
        elif completeness_score >= 0.2:
            return "poor"
        else:
            return "very_poor"

    def extract_final_answer(self, text: str) -> Optional[str]:
        """Извлечь финальный ответ"""
        # Пробуем несколько стратегий

        # 1. Ищем явные пометки "Ответ:"
        answer_patterns = [
            r'Ответ\s*[:=]\s*([^\n]+)',
            r'[Aa]nswer\s*[:=]\s*([^\n]+)',
            r'=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$',
            r'получим\s*([+-]?\d*\.?\d+)',
            r'равно\s*([+-]?\d*\.?\d+)',
            r'≈\s*([+-]?\d*\.?\d+)',
            r'~?\s*([+-]?\d*\.?\d+)\s*(?:$|\.|,)'
        ]

        for pattern in answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Возвращаем последний найденный (скорее всего финальный)
                return matches[-1].strip()

        # 2. Ищем числа в конце текста
        lines = text.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            # Ищем числа в последней строке
            numbers = re.findall(r'[+-]?\d+\.?\d*', last_line)
            if numbers:
                return numbers[-1]

        return None

    def compare_with_reference(
            self,
            student_solution: str,
            reference_solution: Optional[str] = None,
            reference_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """Сравнить решение студента с эталонным"""
        if not reference_solution and not reference_answer:
            return {
                'success': False,
                'error': 'No reference provided',
                'comparison_score': 0.0
            }

        try:
            # Если есть эталонное решение, используем SymPy для сравнения
            if reference_solution:
                result = self.sympy_evaluator.compare_with_reference(
                    student_solution,
                    reference_solution
                )
                return result
            elif reference_answer:
                # Только сравнение ответов
                student_answer = self.extract_final_answer(student_solution)

                if not student_answer:
                    return {
                        'success': False,
                        'error': 'Could not extract student answer',
                        'comparison_score': 0.0
                    }

                # Пробуем числовое сравнение
                try:
                    student_val = float(student_answer.replace(',', '.'))
                    reference_val = float(reference_answer.replace(',', '.'))

                    diff = abs(student_val - reference_val)
                    relative_diff = diff / max(abs(reference_val), 1e-10)

                    score = max(1.0 - relative_diff * 10, 0.0)  # 10% разница = 0 баллов

                    return {
                        'success': True,
                        'student_answer': student_answer,
                        'reference_answer': reference_answer,
                        'difference': diff,
                        'relative_difference': relative_diff,
                        'comparison_score': score,
                        'is_correct': relative_diff < 0.01  # 1% допуск
                    }

                except ValueError:
                    # Строковое сравнение
                    is_correct = student_answer.lower() == reference_answer.lower()
                    score = 1.0 if is_correct else 0.0

                    return {
                        'success': True,
                        'student_answer': student_answer,
                        'reference_answer': reference_answer,
                        'comparison_score': score,
                        'is_correct': is_correct,
                        'note': 'String comparison'
                    }

        except Exception as e:
            logger.error(f"Error comparing with reference: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'comparison_score': 0.0
            }

    def generate_detailed_report(
            self,
            text: str,
            ocr_confidence: float,
            formulas: List[Dict[str, Any]],
            calculation_chains: List[List[str]],
            reference_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Сгенерировать детальный отчет по решению"""
        try:
            # Анализируем структуру
            structure_analysis = self.analyze_solution_structure(text)

            # Извлекаем ответ
            extracted_answer = self.extract_final_answer(text)

            # Проверяем вычисления если есть цепочки
            calculation_check = None
            if calculation_chains:
                calculation_check = self.sympy_evaluator.check_calculation_chain(
                    calculation_chains[0] if calculation_chains else []
                )

            # Сравниваем с эталоном если есть
            comparison_result = None
            if reference_data:
                comparison_result = self.compare_with_reference(
                    text,
                    reference_data.get('reference_solution'),
                    reference_data.get('reference_answer')
                )

            # Формируем отчет
            report = {
                'text_analysis': {
                    'character_count': len(text),
                    'word_count': len(text.split()),
                    'line_count': len(text.split('\n')),
                    'extracted_answer': extracted_answer,
                    'has_explicit_answer': bool(extracted_answer)
                },
                'structure_analysis': structure_analysis,
                'formula_analysis': {
                    'formula_count': len(formulas),
                    'formulas': formulas[:10],  # Ограничиваем количество
                    'has_complex_formulas': any(f.get('is_complex', False) for f in formulas)
                },
                'calculation_analysis': {
                    'chain_count': len(calculation_chains),
                    'total_calculations': sum(len(chain) for chain in calculation_chains),
                    'calculation_check': calculation_check
                },
                'comparison_result': comparison_result,
                'quality_indicators': {
                    'ocr_confidence': ocr_confidence,
                    'structure_quality': structure_analysis.get('structure_quality', 'unknown'),
                    'completeness_score': structure_analysis.get('completeness_score', 0.0),
                    'solution_type': structure_analysis.get('solution_type', 'unknown')
                },
                'recommendations': self.generate_recommendations(
                    ocr_confidence,
                    structure_analysis,
                    extracted_answer,
                    comparison_result
                )
            }

            return report

        except Exception as e:
            logger.error(f"Error generating detailed report: {str(e)}")
            return {
                'error': str(e),
                'text_analysis': {'character_count': len(text)},
                'structure_analysis': {},
                'formula_analysis': {'formula_count': 0},
                'calculation_analysis': {'chain_count': 0},
                'quality_indicators': {'ocr_confidence': ocr_confidence}
            }

    def generate_recommendations(
            self,
            ocr_confidence: float,
            structure_analysis: Dict[str, Any],
            extracted_answer: Optional[str],
            comparison_result: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Сгенерировать рекомендации для проверки"""
        recommendations = []

        if ocr_confidence < 0.6:
            recommendations.append("Проверить качество распознавания текста")

        if structure_analysis.get('completeness_score', 0.0) < 0.5:
            recommendations.append("Проверить структуру решения (отсутствуют ключевые разделы)")

        if not extracted_answer:
            recommendations.append("Ответ не найден в решении")
        elif comparison_result:
            if not comparison_result.get('is_correct', False):
                recommendations.append("Проверить расчеты: расхождение в ответе")

        if structure_analysis.get('step_count', 0) < 2:
            recommendations.append("Решение может быть слишком кратким")

        return recommendations