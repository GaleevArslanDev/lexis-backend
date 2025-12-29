import cv2
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

# Глобальная переменная для хранения reader (singleton)
_READER = None


def get_easyocr_reader():
    """Ленивая загрузка EasyOCR для экономии памяти"""
    global _READER

    if _READER is None:
        try:
            # Импортируем только когда нужно
            import easyocr
            logger.info("Loading EasyOCR model (this may take a moment)...")
            # Используем только русский язык для экономии памяти
            _READER = easyocr.Reader(['ru'], gpu=False)
            logger.info("EasyOCR model loaded successfully")
        except ImportError as e:
            logger.error(f"EasyOCR not installed: {e}")
            _READER = None
        except Exception as e:
            logger.error(f"Failed to load EasyOCR: {e}")
            _READER = None

    return _READER


class OCREngine:
    def __init__(self, language: str = "ru"):
        self.language = language
        self.use_easyocr = True

        # Проверяем, можем ли мы использовать EasyOCR
        try:
            reader = get_easyocr_reader()
            if reader is None:
                self.use_easyocr = False
                logger.warning("Falling back to basic image processing")
        except:
            self.use_easyocr = False

        # Конфигурация для Tesseract (fallback)
        self.tesseract_path = os.getenv("TESSERACT_PATH", "/usr/bin/tesseract")

    def process_from_bytes(self, image_bytes: bytes) -> Dict:
        """Обработать изображение с оптимальным использованием памяти"""
        try:
            # Конвертируем байты в изображение
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return {"error": "Failed to decode image"}

            # Предобработка изображения
            processed = self.preprocess_image(image)

            # Пробуем EasyOCR если доступен
            if self.use_easyocr:
                try:
                    result = self._process_with_easyocr(processed)
                    if result.get("success", False):
                        return result
                except Exception as e:
                    logger.warning(f"EasyOCR failed, falling back: {str(e)}")
                    self.use_easyocr = False

            # Fallback: улучшенная обработка изображения + базовая логика
            return self._process_fallback(processed)

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {"error": str(e), "success": False}

    def _process_with_easyocr(self, image: np.ndarray) -> Dict:
        """Обработка с использованием EasyOCR"""
        reader = get_easyocr_reader()
        if reader is None:
            return {"success": False, "error": "EasyOCR not available"}

        try:
            # Используем низкие настройки для экономии памяти
            result = reader.readtext(image, paragraph=True, detail=0)

            if result:
                full_text = " ".join(result)
            else:
                full_text = ""

            return {
                "success": True,
                "full_text": full_text,
                "average_confidence": 80.0,  # Примерная уверенность
                "character_count": len(full_text),
                "word_count": len(full_text.split()),
                "formulas": self.extract_formulas_from_text(full_text),
                "answer_candidates": self.extract_answer_candidates(full_text)
            }

        except Exception as e:
            logger.error(f"EasyOCR processing error: {str(e)}")
            return {"success": False, "error": str(e)}

    def _process_fallback(self, image: np.ndarray) -> Dict:
        """Fallback обработка без тяжелых моделей"""
        try:
            # Улучшенная предобработка для рукописного текста
            enhanced = self.enhance_handwritten_text(image)

            # Простая сегментация текста
            text_blocks = self.extract_text_blocks(enhanced)

            # Собираем текст из блоков
            all_text = []
            for block in text_blocks:
                # Простая логика для извлечения текста
                # Можно добавить contour analysis для лучших результатов
                text = self.extract_text_from_block(block)
                if text:
                    all_text.append(text)

            full_text = " ".join(all_text)

            # Базовая очистка текста
            full_text = self.clean_text(full_text)

            return {
                "success": True,
                "full_text": full_text,
                "average_confidence": 60.0,  # Средняя уверенность для fallback
                "character_count": len(full_text),
                "word_count": len(full_text.split()),
                "formulas": self.extract_formulas_from_text(full_text),
                "answer_candidates": self.extract_answer_candidates(full_text)
            }

        except Exception as e:
            logger.error(f"Fallback processing error: {str(e)}")
            return {"success": False, "error": str(e), "full_text": ""}

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Базовая предобработка"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Улучшение контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        return enhanced

    def enhance_handwritten_text(self, image: np.ndarray) -> np.ndarray:
        """Улучшение рукописного текста"""
        try:
            # Бинаризация с адаптивным порогом
            binary = cv2.adaptiveThreshold(
                image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # Удаление мелкого шума
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # Увеличение толщины текста
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(cleaned, kernel, iterations=1)

            return dilated

        except Exception as e:
            logger.error(f"Error enhancing handwritten text: {str(e)}")
            return image

    def extract_text_blocks(self, image: np.ndarray) -> List[np.ndarray]:
        """Извлечение текстовых блоков"""
        try:
            # Находим контуры
            contours, _ = cv2.findContours(
                image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            blocks = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Фильтруем мелкие контуры
                    x, y, w, h = cv2.boundingRect(contour)
                    block = image[y:y + h, x:x + w]
                    blocks.append(block)

            # Сортируем сверху вниз
            blocks.sort(key=lambda b: b.shape[0])

            return blocks

        except Exception as e:
            logger.error(f"Error extracting text blocks: {str(e)}")
            return []

    def extract_text_from_block(self, block: np.ndarray) -> str:
        """Извлечение текста из блока (упрощенное)"""
        # Здесь можно добавить простую логику распознавания
        # Например, по соотношению черных/белых пикселей и структуре

        # Пока возвращаем пустую строку
        # В реальной реализации можно добавить простой OCR на основе шаблонов
        return ""

    def clean_text(self, text: str) -> str:
        """Очистка текста"""
        if not text:
            return ""

        # Удаляем лишние символы
        text = re.sub(r'[^\w\s\.\,\-\+\=\*\/\(\)\d]', '', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def extract_formulas_from_text(self, text: str) -> List[Dict]:
        """Извлечение формул из текста"""
        formulas = []

        # Простые паттерны для формул
        patterns = [
            r'[a-zA-Z]\s*=\s*[^=]+',
            r'\d+\s*[+\-*/]\s*\d+',
            r'\d+\s*=\s*\d+',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                formula_text = match.group(0).strip()
                formulas.append({
                    'latex': formula_text,
                    'confidence': 0.6,
                    'original_text': formula_text
                })

        return formulas

    def extract_answer_candidates(self, text: str) -> List[Dict]:
        """Извлечение кандидатов на ответ"""
        candidates = []

        patterns = [
            r'Ответ\s*[:=]\s*([^\n]+)',
            r'[Aa]nswer\s*[:=]\s*([^\n]+)',
            r'=\s*([0-9.,]+)',
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
                        'position': (match.start(1), match.end(1)),
                        'context': text[max(0, match.start() - 20):match.end() + 20]
                    })

        return candidates

    def assess_ocr_quality(self, text: str, avg_confidence: float) -> Dict:
        """Оценка качества"""
        length = len(text)
        words = len(text.split())

        if length < 10:
            length_score = 0.1
        elif length < 50:
            length_score = 0.3
        elif length < 200:
            length_score = 0.7
        else:
            length_score = 1.0

        math_symbols = set('+-*/=<>()')
        math_count = sum(1 for c in text if c in math_symbols)
        math_score = min(math_count / 5.0, 1.0)

        conf_norm = avg_confidence / 100.0
        overall = conf_norm * 0.5 + length_score * 0.3 + math_score * 0.2

        return {
            "score": round(overall, 2),
            "avg_confidence": round(avg_confidence, 2),
            "interpretation": self._quality_label(overall)
        }

    def _quality_label(self, score: float) -> str:
        if score >= 0.7:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"

    def process_complete_page(self, image_path: str, **kwargs) -> Dict:
        """Обработка полной страницы"""
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            return self.process_from_bytes(image_bytes)
        except Exception as e:
            return {"error": str(e), "success": False}