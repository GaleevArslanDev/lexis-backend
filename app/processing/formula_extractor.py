import cv2
import numpy as np
import re
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FormulaExtractor:
    def __init__(self):
        # Паттерны для поиска математических символов
        self.math_symbols = set('+-*/=^()[]{}<>|√∫∑∏∂∆∇≈≠≤≥∞πθαβγδ')

    def detect_formula_regions(self, image: np.ndarray) -> List[Dict]:
        """Обнаружить области с формулами на изображении"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Бинаризация для выделения текста
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Морфологические операции для объединения символов
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(binary, kernel, iterations=1)

            # Находим контуры
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            formula_regions = []

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Фильтруем маленькие и слишком большие области
                if 20 < w < 500 and 20 < h < 200:
                    # Вырезаем область
                    region = image[y:y + h, x:x + w]

                    # Проверяем, похожа ли область на формулу
                    if self.is_likely_formula_region(region):
                        formula_regions.append({
                            'bbox': (x, y, w, h),
                            'region': region,
                            'area': w * h
                        })

            # Сортируем по горизонтали (слева направо, сверху вниз)
            formula_regions.sort(key=lambda r: (r['bbox'][1] // 50, r['bbox'][0]))

            return formula_regions

        except Exception as e:
            logger.error(f"Error detecting formula regions: {str(e)}")
            return []

    def is_likely_formula_region(self, region: np.ndarray) -> bool:
        """Проверить, похожа ли область на формулу"""
        try:
            # Конвертируем в grayscale если нужно
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region.copy()

            # Анализируем вертикальные проекции (формулы часто имеют неравномерную высоту)
            vertical_projection = np.sum(gray < 128, axis=1)

            # Вычисляем вариабельность вертикальной проекции
            variance = np.var(vertical_projection) if len(vertical_projection) > 0 else 0

            # Формулы часто имеют более высокую вариабельность из-за верхних/нижних индексов
            return variance > 100  # Эмпирический порог

        except Exception as e:
            logger.error(f"Error checking formula region: {str(e)}")
            return False

    def extract_formula_text(self, region: np.ndarray) -> str:
        """Извлечь текст формулы из области"""
        try:
            # Подготовка изображения для OCR
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region.copy()

            # Улучшение контраста для формул
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Бинаризация
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Удаление шума
            denoised = cv2.medianBlur(binary, 3)

            # Сохраняем временный файл для отладки (опционально)
            # cv2.imwrite(f"/tmp/formula_{uuid.uuid4().hex[:8]}.png", denoised)

            # Здесь будет интеграция с OCR для формул
            # Временная заглушка - возвращаем пустую строку
            return ""

        except Exception as e:
            logger.error(f"Error extracting formula text: {str(e)}")
            return ""

    def convert_to_latex(self, formula_text: str) -> str:
        """Конвертировать распознанный текст формулы в LaTeX"""
        try:
            # Простые замены
            replacements = {
                'alpha': '\\alpha',
                'beta': '\\beta',
                'gamma': '\\gamma',
                'delta': '\\delta',
                'theta': '\\theta',
                'pi': '\\pi',
                'sigma': '\\sigma',
                'inf': '\\infty',
                'sqrt': '\\sqrt',
                'sum': '\\sum',
                'int': '\\int',
                '^': '^',
                '_': '_',
                '->': '\\rightarrow',
                '<=': '\\leq',
                '>=': '\\geq',
                '!=': '\\neq',
                'approx': '\\approx'
            }

            latex = formula_text
            for pattern, replacement in replacements.items():
                latex = latex.replace(pattern, replacement)

            # Добавляем математические окружение если нужно
            if not latex.startswith('$') and not latex.endswith('$'):
                latex = f"${latex}$"

            return latex

        except Exception as e:
            logger.error(f"Error converting to LaTeX: {str(e)}")
            return formula_text

    def process_image_for_formulas(self, image_path: str) -> Dict:
        """Полная обработка изображения для извлечения формул"""
        try:
            # Загружаем изображение
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Failed to load image"}

            # Обнаруживаем области с формулами
            formula_regions = self.detect_formula_regions(image)

            formulas = []
            for i, region_data in enumerate(formula_regions):
                # Извлекаем текст формулы
                formula_text = self.extract_formula_text(region_data['region'])

                if formula_text:
                    # Конвертируем в LaTeX
                    latex = self.convert_to_latex(formula_text)

                    formulas.append({
                        'id': i + 1,
                        'latex': latex,
                        'original_text': formula_text,
                        'bbox': region_data['bbox'],
                        'confidence': 0.7  # Заглушка, нужно вычислять
                    })

            return {
                'success': True,
                'formula_count': len(formulas),
                'formulas': formulas,
                'image_shape': image.shape
            }

        except Exception as e:
            logger.error(f"Error processing image for formulas: {str(e)}")
            return {"error": str(e)}