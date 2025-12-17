import sympy as sp
import re
from typing import Dict, List, Optional, Tuple, Any
import logging
import json

logger = logging.getLogger(__name__)


class SymPyEvaluator:
    def __init__(self):
        # Инициализация символьных переменных
        self.x, self.y, self.z = sp.symbols('x y z')
        self.a, self.b, self.c = sp.symbols('a b c')

        # Словарь переменных
        self.variables = {
            'x': self.x, 'y': self.y, 'z': self.z,
            'a': self.a, 'b': self.b, 'c': self.c,
            'pi': sp.pi, 'e': sp.E
        }

    def parse_expression(self, expr_str: str) -> Optional[sp.Expr]:
        """Парсить математическое выражение"""
        try:
            # Очистка строки
            expr_str = expr_str.strip()

            # Замена русских символов и common patterns
            replacements = {
                '×': '*',
                '÷': '/',
                '^': '**',
                '√': 'sqrt',
                '∫': 'integrate',
                '∑': 'summation',
                '∏': 'product',
                '∞': 'oo',
                'π': 'pi',
                '≈': '~',
                '≠': '!=',
                '≤': '<=',
                '≥': '>=',
                'α': 'alpha',
                'β': 'beta',
                'γ': 'gamma',
                'δ': 'delta',
                'θ': 'theta',
                'Δ': 'Delta'
            }

            for rus, eng in replacements.items():
                expr_str = expr_str.replace(rus, eng)

            # Пробуем парсить
            try:
                expr = sp.sympify(expr_str, self.variables)
                return expr
            except:
                # Пробуем более простой парсинг
                expr = sp.parse_expr(expr_str, local_dict=self.variables)
                return expr

        except Exception as e:
            logger.warning(f"Failed to parse expression '{expr_str}': {str(e)}")
            return None

    def check_equivalence(self, expr1_str: str, expr2_str: str) -> Dict[str, Any]:
        """Проверить эквивалентность двух выражений"""
        try:
            expr1 = self.parse_expression(expr1_str)
            expr2 = self.parse_expression(expr2_str)

            if expr1 is None or expr2 is None:
                return {
                    "equivalent": False,
                    "reason": "Failed to parse one or both expressions",
                    "confidence": 0.0
                }

            # Пробуем упростить разницу
            difference = sp.simplify(expr1 - expr2)

            # Проверяем, равна ли разница нулю
            if difference == 0:
                return {
                    "equivalent": True,
                    "reason": "Expressions are mathematically equivalent",
                    "confidence": 1.0,
                    "simplified_forms": {
                        "expr1": str(sp.simplify(expr1)),
                        "expr2": str(sp.simplify(expr2))
                    }
                }

            # Пробуем числовую проверку с разными значениями
            numerical_match = self.numerical_equivalence_check(expr1, expr2)

            if numerical_match["equivalent"]:
                return {
                    "equivalent": True,
                    "reason": "Expressions are numerically equivalent within tolerance",
                    "confidence": numerical_match["confidence"],
                    "numerical_tests": numerical_match["tests"]
                }

            return {
                "equivalent": False,
                "reason": "Expressions are not equivalent",
                "confidence": 0.0,
                "difference": str(difference)
            }

        except Exception as e:
            logger.error(f"Error checking equivalence: {str(e)}")
            return {
                "equivalent": False,
                "reason": f"Error during equivalence check: {str(e)}",
                "confidence": 0.0
            }

    def numerical_equivalence_check(
            self,
            expr1: sp.Expr,
            expr2: sp.Expr,
            test_points: int = 10
    ) -> Dict[str, Any]:
        """Проверить числовую эквивалентность"""
        try:
            # Получаем символы в выражениях
            symbols1 = expr1.free_symbols
            symbols2 = expr2.free_symbols
            all_symbols = symbols1.union(symbols2)

            if not all_symbols:
                # Если нет переменных, сравниваем как числа
                val1 = float(expr1.evalf())
                val2 = float(expr2.evalf())
                diff = abs(val1 - val2)

                return {
                    "equivalent": diff < 1e-10,
                    "confidence": 1.0 if diff < 1e-10 else 0.0,
                    "tests": [{"values": {}, "result1": val1, "result2": val2, "diff": diff}]
                }

            # Генерируем случайные значения для тестирования
            import random
            tests = []
            matches = 0

            for _ in range(test_points):
                values = {}
                for sym in all_symbols:
                    # Генерируем значение в разумном диапазоне
                    values[str(sym)] = random.uniform(-10, 10)

                try:
                    # Подставляем значения
                    val1 = float(expr1.subs(values).evalf())
                    val2 = float(expr2.subs(values).evalf())

                    diff = abs(val1 - val2)
                    test_passed = diff < 1e-6

                    if test_passed:
                        matches += 1

                    tests.append({
                        "values": values,
                        "result1": val1,
                        "result2": val2,
                        "diff": diff,
                        "passed": test_passed
                    })
                except:
                    continue

            confidence = matches / test_points if test_points > 0 else 0.0

            return {
                "equivalent": confidence > 0.95,  # 95% совпадений
                "confidence": confidence,
                "tests": tests,
                "matches": matches,
                "total_tests": test_points
            }

        except Exception as e:
            logger.error(f"Error in numerical equivalence check: {str(e)}")
            return {
                "equivalent": False,
                "confidence": 0.0,
                "error": str(e),
                "tests": []
            }

    def evaluate_expression(self, expr_str: str, values: Dict[str, float] = None) -> Dict[str, Any]:
        """Вычислить значение выражения"""
        try:
            expr = self.parse_expression(expr_str)

            if expr is None:
                return {
                    "success": False,
                    "error": "Failed to parse expression",
                    "value": None
                }

            if values:
                # Подставляем значения
                try:
                    result = expr.subs(values)
                    numeric_result = float(result.evalf())

                    return {
                        "success": True,
                        "expression": str(expr),
                        "value": numeric_result,
                        "exact_value": str(result),
                        "substitutions": values
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Error in substitution: {str(e)}",
                        "expression": str(expr)
                    }
            else:
                # Возвращаем символьную форму
                return {
                    "success": True,
                    "expression": str(expr),
                    "value": None,
                    "symbolic_form": True
                }

        except Exception as e:
            logger.error(f"Error evaluating expression '{expr_str}': {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "value": None
            }

    def check_calculation_chain(self, chain: List[str]) -> Dict[str, Any]:
        """Проверить цепочку вычислений"""
        try:
            steps = []
            all_correct = True
            last_value = None

            for i, step in enumerate(chain):
                step = step.strip()

                # Пытаемся извлечь выражение вида "60 * 2.5 = 150"
                if '=' in step:
                    left, right = step.split('=', 1)
                    left = left.strip()
                    right = right.strip()

                    # Проверяем левую часть
                    left_result = self.evaluate_expression(left)

                    if left_result["success"] and left_result["value"] is not None:
                        left_value = left_result["value"]

                        # Пробуем вычислить правую часть
                        right_result = self.evaluate_expression(right)

                        if right_result["success"] and right_result["value"] is not None:
                            right_value = right_result["value"]

                            # Сравниваем
                            diff = abs(left_value - right_value)
                            is_correct = diff < 1e-6

                            if not is_correct:
                                all_correct = False

                            # Проверяем, продолжается ли цепочка
                            if last_value is not None:
                                # Проверяем, равно ли left_value предыдущему результату
                                continuation_diff = abs(left_value - last_value)
                                continues_chain = continuation_diff < 1e-6
                            else:
                                continues_chain = True

                            last_value = right_value

                            steps.append({
                                "step_number": i + 1,
                                "expression": step,
                                "left_value": left_value,
                                "right_value": right_value,
                                "is_correct": is_correct,
                                "continues_chain": continues_chain,
                                "difference": diff,
                                "left_parsed": left_result["expression"] if left_result["success"] else None,
                                "right_parsed": right_result["expression"] if right_result["success"] else None
                            })
                            continue

                # Если не удалось разобрать как равенство
                steps.append({
                    "step_number": i + 1,
                    "expression": step,
                    "is_correct": None,
                    "continues_chain": None,
                    "error": "Could not parse as calculation step"
                })

            return {
                "success": True,
                "all_correct": all_correct,
                "total_steps": len(steps),
                "correct_steps": sum(1 for s in steps if s.get("is_correct", False)),
                "steps": steps,
                "final_result": last_value
            }

        except Exception as e:
            logger.error(f"Error checking calculation chain: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "steps": []
            }

    def extract_calculation_chains(self, text: str) -> List[List[str]]:
        """Извлечь цепочки вычислений из текста"""
        # Ищем последовательности, похожие на вычисления
        lines = text.split('\n')

        chains = []
        current_chain = []

        for line in lines:
            line = line.strip()

            # Проверяем, похожа ли строка на вычисление
            if self.looks_like_calculation(line):
                current_chain.append(line)
            else:
                if current_chain:
                    chains.append(current_chain.copy())
                    current_chain = []

        # Добавляем последнюю цепочку если есть
        if current_chain:
            chains.append(current_chain)

        return chains

    def looks_like_calculation(self, text: str) -> bool:
        """Проверить, похож ли текст на вычисление"""
        # Содержит ли знак равенства?
        if '=' not in text:
            return False

        # Содержит ли математические операторы?
        math_ops = set('+-*/^√')
        has_math_op = any(op in text for op in math_ops)

        # Содержит ли числа?
        has_numbers = any(char.isdigit() for char in text)

        return has_math_op and has_numbers

    def compare_with_reference(
            self,
            student_solution: str,
            reference_solution: str
    ) -> Dict[str, Any]:
        """Сравнить решение студента с эталонным"""
        try:
            # Извлекаем формулы из решений
            student_formulas = self.extract_formulas(student_solution)
            reference_formulas = self.extract_formulas(reference_solution)

            # Извлекаем ответы
            student_answer = self.extract_final_answer(student_solution)
            reference_answer = self.extract_final_answer(reference_solution)

            # Проверяем эквивалентность формул
            formula_matches = []
            for s_formula in student_formulas:
                best_match = None
                best_confidence = 0.0

                for r_formula in reference_formulas:
                    result = self.check_equivalence(s_formula, r_formula)
                    if result["equivalent"] and result["confidence"] > best_confidence:
                        best_match = r_formula
                        best_confidence = result["confidence"]

                formula_matches.append({
                    "student_formula": s_formula,
                    "reference_formula": best_match,
                    "match_confidence": best_confidence,
                    "is_correct": best_confidence > 0.8
                })

            # Проверяем ответ
            answer_match = None
            if student_answer and reference_answer:
                # Пробуем числовое сравнение
                try:
                    student_val = float(student_answer)
                    reference_val = float(reference_answer)

                    diff = abs(student_val - reference_val)
                    relative_diff = diff / max(abs(reference_val), 1e-10)

                    answer_match = {
                        "student_answer": student_answer,
                        "reference_answer": reference_answer,
                        "difference": diff,
                        "relative_difference": relative_diff,
                        "is_correct": relative_diff < 0.01  # 1% допуск
                    }
                except:
                    # Символьное сравнение
                    answer_match = {
                        "student_answer": student_answer,
                        "reference_answer": reference_answer,
                        "is_correct": False,
                        "reason": "Could not compare numerically"
                    }

            # Вычисляем общую оценку
            correct_formulas = sum(1 for fm in formula_matches if fm["is_correct"])
            formula_score = correct_formulas / max(len(formula_matches), 1)

            answer_score = 1.0 if answer_match and answer_match["is_correct"] else 0.0

            # Веса: формулы 70%, ответ 30%
            total_score = formula_score * 0.7 + answer_score * 0.3

            return {
                "success": True,
                "total_score": total_score,
                "formula_score": formula_score,
                "answer_score": answer_score,
                "formula_matches": formula_matches,
                "answer_match": answer_match,
                "student_formula_count": len(student_formulas),
                "reference_formula_count": len(reference_formulas),
                "correct_formula_count": correct_formulas,
                "assessment": self.interpret_score(total_score)
            }

        except Exception as e:
            logger.error(f"Error comparing with reference: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "total_score": 0.0
            }

    def extract_formulas(self, text: str) -> List[str]:
        """Извлечь формулы из текста"""
        # Простая эвристика: ищем строки с математическими символами
        lines = text.split('\n')
        formulas = []

        math_pattern = r'[=+\-*/^()\[\]{}<>\|]'

        for line in lines:
            line = line.strip()

            # Проверяем, похоже ли на формулу
            if re.search(math_pattern, line):
                # Убираем текст до и после формулы
                # Ищем начало формулы (последнее вхождение слова или =)
                formulas.append(line)

        return formulas

    def extract_final_answer(self, text: str) -> Optional[str]:
        """Извлечь финальный ответ из текста"""
        # Ищем паттерны ответов
        answer_patterns = [
            r'Ответ\s*[:=]\s*([^\n]+)',
            r'[Aa]nswer\s*[:=]\s*([^\n]+)',
            r'=\s*([0-9.,]+(?:[eE][+-]?[0-9]+)?)\s*$',
            r'получим\s*([0-9.,]+)',
            r'равно\s*([0-9.,]+)'
        ]

        for pattern in answer_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Возвращаем последний найденный ответ (скорее всего финальный)
                return matches[-1].strip()

        return None

    def interpret_score(self, score: float) -> str:
        """Интерпретировать числовую оценку"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "fair"
        elif score >= 0.3:
            return "poor"
        else:
            return "very_poor"