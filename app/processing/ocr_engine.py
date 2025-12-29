import easyocr
import cv2
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class OCREngine:
    def __init__(self, language: str = "ru"):
        # Инициализируем EasyOCR
        # Для русского и английского языков
        langs = []
        if "ru" in language or "rus" in language:
            langs.append('ru')
        if "en" in language or "eng" in language:
            langs.append('en')

        if not langs:
            langs = ['ru', 'en']  # по умолчанию русский и английский

        logger.info(f"Initializing EasyOCR with languages: {langs}")
        # Используем GPU если доступно, иначе CPU
        self.reader = easyocr.Reader(langs, gpu=False)
        logger.info("EasyOCR initialized")

    def process_from_bytes(self, image_bytes: bytes) -> Dict:
        """Обработать изображение из байтов с помощью EasyOCR"""
        try:
            # Конвертируем байты в изображение OpenCV
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return {"error": "Failed to decode image"}

            # Предобработка для улучшения качества
            processed_image = self.preprocess_image(image)

            # Выполняем OCR
            result = self.reader.readtext(processed_image, paragraph=False, detail=1)

            # Формируем текст и собираем статистику
            full_text = ""
            detailed_info = []
            total_confidence = 0
            count = 0

            for detection in result:
                bbox, text, confidence = detection
                if text.strip():
                    full_text += text + " "
                    detailed_info.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
                    total_confidence += confidence
                    count += 1

            full_text = full_text.strip()
            avg_confidence = (total_confidence / count * 100) if count > 0 else 0

            # Извлекаем формулы и ответы
            formulas = self.extract_formulas_from_text(full_text)
            answer_candidates = self.extract_answer_candidates(full_text)

            response = {
                "full_text": full_text,
                "average_confidence": avg_confidence,
                "character_count": len(full_text),
                "word_count": len(full_text.split()),
                "detailed_info": detailed_info,
                "formulas": formulas,
                "formula_count": len(formulas),
                "answer_candidates": answer_candidates
            }

            if answer_candidates:
                primary_candidate = None
                for candidate in answer_candidates:
                    if "ответ" in candidate['context'].lower():
                        primary_candidate = candidate
                        break

                if primary_candidate is None and answer_candidates:
                    primary_candidate = answer_candidates[0]

                response["primary_answer"] = primary_candidate

            # Разбивка на строки
            lines = full_text.split('\n')
            response["lines"] = [
                {"text": line.strip(), "line_number": i}
                for i, line in enumerate(lines) if line.strip()
            ]

            # Оценка качества
            quality_score = self.assess_ocr_quality(full_text, avg_confidence)
            response["quality_assessment"] = quality_score

            return response

        except Exception as e:
            logger.error(f"Error processing image from bytes with EasyOCR: {str(e)}")
            return {"error": str(e)}

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Предобработка изображения для улучшения OCR"""
        try:
            # Конвертируем в оттенки серого
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Улучшение контраста
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Удаление шума
            denoised = cv2.medianBlur(enhanced, 3)

            # Бинаризация
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return binary

        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            return image

    def extract_formulas_from_text(self, text: str) -> List[Dict]:
        """Извлечь формулы из текста"""
        try:
            formulas = []

            # Ищем математические выражения
            math_patterns = [
                r'[a-zA-Zα-ωΑ-Ω]\s*=\s*[^=]+',  # x = выражение
                r'[0-9]+\s*[+\-*/]\s*[0-9]+',  # 2+2, 3*4
                r'[0-9]+\s*=\s*[0-9]+',  # 60*2.5 = 150
            ]

            for pattern in math_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    formula_text = match.group(0).strip()
                    formulas.append({
                        'latex': formula_text,
                        'confidence': 0.7,
                        'original_text': formula_text
                    })

            return formulas

        except Exception as e:
            logger.error(f"Error extracting formulas: {str(e)}")
            return []

    def extract_answer_candidates(self, text: str) -> List[Dict]:
        """Найти кандидаты на ответ"""
        try:
            answer_patterns = [
                r'Ответ\s*[:=]\s*([^\n]+)',
                r'[Aa]nswer\s*[:=]\s*([^\n]+)',
                r'=\s*([0-9.,]+)',  # = число
                r'получим\s*([0-9.,]+)',
                r'равно\s*([0-9.,]+)'
            ]

            candidates = []

            for pattern in answer_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    answer_text = match.group(1).strip()
                    candidates.append({
                        'text': answer_text,
                        'pattern': pattern,
                        'position': (match.start(1), match.end(1)),
                        'context': text[max(0, match.start() - 30):match.end() + 30]
                    })

            return candidates

        except Exception as e:
            logger.error(f"Error extracting answer candidates: {str(e)}")
            return []

    def assess_ocr_quality(self, text: str, avg_confidence: float) -> Dict:
        """Оценить качество распознавания"""
        words = text.split()

        if len(text) < 10:
            text_length_score = 0.1
        elif len(text) < 50:
            text_length_score = 0.3
        elif len(text) < 200:
            text_length_score = 0.7
        else:
            text_length_score = 1.0

        math_symbols = set('+-*/=<>()[]{}')
        math_content = sum(1 for char in text if char in math_symbols)
        math_score = min(math_content / 5.0, 1.0)

        normalized_confidence = avg_confidence / 100.0
        overall_score = (normalized_confidence * 0.5 +
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

    def process_complete_page(self, image_path: str, extract_formulas: bool = True,
                              extract_answers: bool = True) -> Dict:
        """Полная обработка страницы"""
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            return self.process_from_bytes(image_bytes)

        except Exception as e:
            logger.error(f"Error processing page {image_path}: {str(e)}")
            return {"error": str(e)}