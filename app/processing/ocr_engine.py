# app/processing/ocr_engine.py - ЛЕГКОВЕСНЫЙ ВАРИАНТ
import cv2
import numpy as np
import json
import os
from typing import Dict, List, Optional
import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

# Глобальная переменная для pipeline
_PIPELINE = None


def get_keras_ocr_pipeline():
    """Ленивая загрузка Keras-OCR pipeline"""
    global _PIPELINE

    if _PIPELINE is None:
        try:
            import keras_ocr
            logger.info("Loading Keras-OCR pipeline (this may take a moment)...")
            # Используем легковесный детектор и распознаватель
            _PIPELINE = keras_ocr.pipeline.Pipeline()
            logger.info("Keras-OCR pipeline loaded successfully")
        except ImportError as e:
            logger.error(f"Keras-OCR not installed: {e}")
            _PIPELINE = None
        except Exception as e:
            logger.error(f"Failed to load Keras-OCR: {e}")
            _PIPELINE = None

    return _PIPELINE


class OCREngine:
    def __init__(self, language: str = "ru"):
        self.language = language

    def process_from_bytes(self, image_bytes: bytes) -> Dict:
        """Обработка изображения с Keras-OCR"""
        try:
            # Конвертируем байты в изображение
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return {"error": "Failed to decode image"}

            # Предобработка для рукописного текста
            processed = self.preprocess_for_handwriting(image)

            # Получаем pipeline
            pipeline = get_keras_ocr_pipeline()
            if pipeline is None:
                return self._process_fallback(processed)

            # Выполняем OCR
            predictions = pipeline.recognize([processed])

            # Извлекаем текст и уверенность
            if predictions and len(predictions[0]) > 0:
                texts = []
                confidences = []

                for text, box in predictions[0]:
                    texts.append(text)
                    # Keras-OCR не возвращает уверенность напрямую, используем эвристику
                    confidences.append(0.8)  # Примерная уверенность

                full_text = " ".join(texts)
                avg_confidence = sum(confidences) / len(confidences) * 100 if confidences else 50
            else:
                full_text = ""
                avg_confidence = 0

            return {
                "success": True,
                "full_text": full_text,
                "average_confidence": avg_confidence,
                "character_count": len(full_text),
                "word_count": len(full_text.split()),
                "formulas": self.extract_formulas(full_text),
                "answer_candidates": self.extract_answers(full_text)
            }

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {"error": str(e), "success": False}

    def preprocess_for_handwriting(self, image: np.ndarray) -> np.ndarray:
        """Оптимизированная предобработка для рукописного текста"""
        try:
            # Конвертируем в grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Улучшение контраста (упрощенное)
            # Используем гистограммное выравнивание
            equalized = cv2.equalizeHist(gray)

            # Легкое размытие для удаления шума
            blurred = cv2.GaussianBlur(equalized, (3, 3), 0)

            # Пороговая обработка
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Инвертируем если нужно (черный текст на белом фоне)
            if np.mean(binary) > 127:
                binary = cv2.bitwise_not(binary)

            # Конвертируем обратно в RGB для Keras-OCR
            rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

            return rgb

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return image

    def extract_formulas(self, text: str) -> List[Dict]:
        """Извлечение формул из текста"""
        formulas = []

        # Простые паттерны для математических выражений
        patterns = [
            r'[a-zA-Z]\s*=\s*[^=]+',  # x = выражение
            r'\d+\s*[+\-*/]\s*\d+',  # 5 + 3, 4*2
            r'[0-9]+\s*=\s*[0-9]+',  # 150 = 150
            r'[a-zA-Z]:?\s*\d+',  # x: 91, k=12
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                formula_text = match.group(0).strip()
                formulas.append({
                    'latex': formula_text,
                    'confidence': 0.7,
                    'original_text': formula_text
                })

        return formulas

    def extract_answers(self, text: str) -> List[Dict]:
        """Извлечение ответов из текста"""
        candidates = []

        patterns = [
            r'Ответ\s*[:=]\s*([0-9.,]+)',
            r'[Aa]nswer\s*[:=]\s*([0-9.,]+)',
            r'=\s*([0-9.,]+)\s*$',
            r'получим\s*([0-9.,]+)',
            r'равно\s*([0-9.,]+)',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.lastindex:
                    answer_text = match.group(1).strip()
                    candidates.append({
                        'text': answer_text,
                        'pattern': pattern,
                        'context': match.group(0)
                    })

        return candidates

    def _process_fallback(self, image: np.ndarray) -> Dict:
        """Fallback обработка - ищем цифры и буквы"""
        try:
            # Конвертируем в grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Простая бинаризация
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Инвертируем если нужно
            if np.mean(binary) > 127:
                binary = cv2.bitwise_not(binary)

            # Ищем контуры символов
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Простая логика: считаем что каждый контур - это символ
            char_count = len(contours)

            # Очень упрощенная логика - если есть контуры, предполагаем текст
            if char_count > 10:
                text = f"[Работа содержит ~{char_count} символов]"
                confidence = 50
            else:
                text = ""
                confidence = 0

            return {
                "success": True,
                "full_text": text,
                "average_confidence": confidence,
                "character_count": len(text),
                "word_count": len(text.split()),
                "formulas": [],
                "answer_candidates": []
            }

        except Exception as e:
            logger.error(f"Fallback error: {str(e)}")
            return {"success": False, "error": str(e)}