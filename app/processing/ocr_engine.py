import pytesseract
from PIL import Image
import cv2
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OCREngine:
    def __init__(self, language: str = "rus+eng"):
        self.language = language
        self.tesseract_path = os.getenv("TESSERACT_PATH", "/usr/bin/tesseract")
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

        # Конфигурация Tesseract
        self.config = {
            'basic': r'--oem 3 --psm 6',  # Для блоков текста
            'single_line': r'--oem 3 --psm 7',  # Для одиночных строк
            'formula': r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-=*/()[]{}<>\|',  # Для формул
        }

    def process_from_bytes(self, image_bytes: bytes) -> Dict:
        """Обработать изображение из байтов"""
        try:
            # Конвертируем байты в изображение OpenCV
            import cv2
            import numpy as np

            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if image is None:
                return {"error": "Failed to decode image"}

            # Основной текст
            full_text, avg_confidence, detailed_info = self.extract_text_with_confidence(image)

            result = {
                "full_text": full_text,
                "average_confidence": avg_confidence,
                "character_count": len(full_text),
                "word_count": len(full_text.split()),
                "detailed_info": detailed_info
            }

            # Формулы
            formulas = self.extract_formulas(image)
            result["formulas"] = formulas
            result["formula_count"] = len(formulas)

            # Кандидаты на ответ
            answer_candidates = self.extract_answer_candidates(image)
            result["answer_candidates"] = answer_candidates

            if answer_candidates:
                primary_candidate = None
                for candidate in answer_candidates:
                    if "Ответ" in candidate['context']:
                        primary_candidate = candidate
                        break

                if primary_candidate is None and answer_candidates:
                    primary_candidate = answer_candidates[0]

                result["primary_answer"] = primary_candidate

            # Разбивка на строки
            lines = full_text.split('\n')
            result["lines"] = [
                {"text": line.strip(), "line_number": i}
                for i, line in enumerate(lines) if line.strip()
            ]

            # Оценка качества распознавания
            quality_score = self.assess_ocr_quality(full_text, avg_confidence)
            result["quality_assessment"] = quality_score

            return result

        except Exception as e:
            logger.error(f"Error processing image from bytes: {str(e)}")
            return {"error": str(e)}

    def image_to_pil(self, cv_image: np.ndarray) -> Image.Image:
        """Конвертировать OpenCV image в PIL Image"""
        if len(cv_image.shape) == 2:
            # Grayscale
            return Image.fromarray(cv_image)
        else:
            # BGR to RGB
            rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)

    def extract_text_with_confidence(
            self,
            image: np.ndarray,
            config: str = 'basic'
    ) -> Tuple[str, float, List[Dict]]:
        """Извлечь текст с информацией о уверенности"""
        try:
            pil_image = self.image_to_pil(image)

            # Получаем данные с уверенностью
            data = pytesseract.image_to_data(
                pil_image,
                lang=self.language,
                config=self.config[config],
                output_type=pytesseract.Output.DICT
            )

            # Собираем текст и среднюю уверенность
            text_parts = []
            total_confidence = 0
            confidence_count = 0
            detailed_info = []

            for i in range(len(data['text'])):
                word = data['text'][i].strip()
                conf = float(data['conf'][i])

                if word and conf > 0:
                    text_parts.append(word)
                    total_confidence += conf
                    confidence_count += 1

                    detailed_info.append({
                        'text': word,
                        'confidence': conf,
                        'bbox': (
                            data['left'][i],
                            data['top'][i],
                            data['width'][i],
                            data['height'][i]
                        )
                    })

            # Объединяем текст
            text = ' '.join(text_parts)

            # Средняя уверенность
            avg_confidence = total_confidence / max(confidence_count, 1)

            return text, avg_confidence, detailed_info

        except Exception as e:
            logger.error(f"Error in OCR extraction: {str(e)}")
            return "", 0.0, []

    def extract_from_region(
            self,
            image: np.ndarray,
            region: Tuple[int, int, int, int]  # x, y, w, h
    ) -> Dict:
        """Извлечь текст из определенной области"""
        x, y, w, h = region

        # Вырезаем область
        region_img = image[y:y + h, x:x + w]

        if region_img.size == 0:
            return {"text": "", "confidence": 0.0, "bbox": region}

        # Применяем OCR
        text, confidence, details = self.extract_text_with_confidence(region_img)

        return {
            "text": text,
            "confidence": confidence,
            "bbox": region,
            "details": details
        }

    def extract_formulas(self, image: np.ndarray) -> List[Dict]:
        """Выделить математические формулы"""
        try:
            # Сначала находим потенциальные формулы по паттернам
            pil_image = self.image_to_pil(image)

            # Используем специальную конфигурацию для формул
            formula_data = pytesseract.image_to_data(
                pil_image,
                lang=self.language,
                config=self.config['formula'],
                output_type=pytesseract.Output.DICT
            )

            formulas = []
            current_formula = []
            current_bbox = None

            for i in range(len(formula_data['text'])):
                symbol = formula_data['text'][i].strip()
                conf = float(formula_data['conf'][i])

                if symbol and conf > 10:  # Минимальная уверенность
                    bbox = (
                        formula_data['left'][i],
                        formula_data['top'][i],
                        formula_data['width'][i],
                        formula_data['height'][i]
                    )

                    current_formula.append(symbol)

                    if current_bbox is None:
                        current_bbox = list(bbox)
                    else:
                        # Расширяем bounding box
                        current_bbox[0] = min(current_bbox[0], bbox[0])
                        current_bbox[1] = min(current_bbox[1], bbox[1])
                        current_bbox[2] = max(current_bbox[2], bbox[0] + bbox[2])
                        current_bbox[3] = max(current_bbox[3], bbox[1] + bbox[3])

                elif current_formula:
                    # Завершаем текущую формулу
                    formula_text = ''.join(current_formula)

                    formulas.append({
                        'latex': formula_text,
                        'confidence': conf if current_formula else 0,
                        'bbox': current_bbox,
                        'symbols': len(current_formula)
                    })

                    current_formula = []
                    current_bbox = None

            # Добавляем последнюю формулу если есть
            if current_formula:
                formulas.append({
                    'latex': ''.join(current_formula),
                    'confidence': 0,
                    'bbox': current_bbox,
                    'symbols': len(current_formula)
                })

            return formulas

        except Exception as e:
            logger.error(f"Error extracting formulas: {str(e)}")
            return []

    def extract_answer_candidates(self, image: np.ndarray) -> List[Dict]:
        """Найти кандидаты на ответ (паттерны: Ответ:, Ответ= и т.д.)"""
        try:
            # Полный текст страницы
            full_text, confidence, details = self.extract_text_with_confidence(image)

            answer_patterns = [
                r'Ответ\s*[:=]\s*([^\n]+)',
                r'Ответ\s*[:=]\s*\n?\s*([^\n]+)',
                r'[Aa]nswer\s*[:=]\s*([^\n]+)',
                r'=\s*([0-9.,]+)',  # Изолированное равенство с числом
                r'получим\s*([0-9.,]+)',
                r'равно\s*([0-9.,]+)'
            ]

            import re
            candidates = []

            for pattern in answer_patterns:
                matches = re.finditer(pattern, full_text, re.IGNORECASE)
                for match in matches:
                    answer_text = match.group(1).strip()

                    # Ищем позицию в detailed_info
                    answer_start = match.start(1)
                    answer_end = match.end(1)

                    candidate = {
                        'text': answer_text,
                        'pattern': pattern,
                        'position': (answer_start, answer_end),
                        'context': full_text[max(0, match.start() - 50):match.end() + 50]
                    }

                    candidates.append(candidate)

            return candidates

        except Exception as e:
            logger.error(f"Error extracting answer candidates: {str(e)}")
            return []

    def process_complete_page(
            self,
            image_path: str,
            extract_formulas: bool = True,
            extract_answers: bool = True
    ) -> Dict:
        """Полная обработка страницы"""
        try:
            # Загружаем изображение
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return {"error": "Failed to load image"}

            # Основной текст
            full_text, avg_confidence, detailed_info = self.extract_text_with_confidence(image)

            result = {
                "full_text": full_text,
                "average_confidence": avg_confidence,
                "character_count": len(full_text),
                "word_count": len(full_text.split()),
                "detailed_info": detailed_info
            }

            # Формулы
            if extract_formulas:
                formulas = self.extract_formulas(image)
                result["formulas"] = formulas
                result["formula_count"] = len(formulas)

            # Кандидаты на ответ
            if extract_answers:
                answer_candidates = self.extract_answer_candidates(image)
                result["answer_candidates"] = answer_candidates

                if answer_candidates:
                    # Берем самый вероятный ответ (первый по порядку или с контекстом "Ответ:")
                    primary_candidate = None
                    for candidate in answer_candidates:
                        if "Ответ" in candidate['context']:
                            primary_candidate = candidate
                            break

                    if primary_candidate is None and answer_candidates:
                        primary_candidate = answer_candidates[0]

                    result["primary_answer"] = primary_candidate

            # Разбивка на строки
            lines = full_text.split('\n')
            result["lines"] = [
                {"text": line.strip(), "line_number": i}
                for i, line in enumerate(lines) if line.strip()
            ]

            # Оценка качества распознавания
            quality_score = self.assess_ocr_quality(full_text, avg_confidence)
            result["quality_assessment"] = quality_score

            return result

        except Exception as e:
            logger.error(f"Error processing page {image_path}: {str(e)}")
            return {"error": str(e)}

    def assess_ocr_quality(self, text: str, avg_confidence: float) -> Dict:
        """Оценить качество распознавания"""
        # Эвристическая оценка
        words = text.split()

        # Проверяем длину текста
        if len(text) < 10:
            text_length_score = 0.1
        elif len(text) < 50:
            text_length_score = 0.3
        elif len(text) < 200:
            text_length_score = 0.7
        else:
            text_length_score = 1.0

        # Проверяем наличие математических символов
        math_symbols = set('+-*/=<>()[]{}^√∫∑∏∂∆∇≈≠≤≥∞πθαβγδ')
        math_content = sum(1 for char in text if char in math_symbols)
        math_score = min(math_content / 10, 1.0)  # Нормализуем

        # Общая оценка
        overall_score = (avg_confidence / 100 * 0.5 +
                         text_length_score * 0.3 +
                         math_score * 0.2)

        return {
            "score": round(overall_score, 2),
            "text_length_score": round(text_length_score, 2),
            "math_content_score": round(math_score, 2),
            "avg_confidence": round(avg_confidence, 2),
            "interpretation": self.interpret_quality_score(overall_score)
        }

    def interpret_quality_score(self, score: float) -> str:
        """Интерпретировать оценку качества"""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        elif score >= 0.2:
            return "poor"
        else:
            return "very_poor"