import cv2
import numpy as np
from typing import Dict, List, Optional
import logging
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)


class OCREngine:
    def __init__(self, lang: str = 'ru'):
        # ЛЕНИВАЯ ИНИЦИАЛИЗАЦИЯ - модель загрузится только при первом вызове
        self.lang = lang
        self._ocr = None
        self._model_loaded = False

    def _ensure_loaded(self):
        """Загружает модель только при необходимости"""
        if not self._model_loaded:
            logger.info("Loading PaddleOCR model (this may take a moment)...")
            # Используем легкие модели для экономии памяти
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                use_gpu=False,  # На Render нет GPU
                enable_mkldnn=True,  # Ускорение CPU
                use_tensorrt=False,
                cls_model_dir='',  # Не использовать классификатор
                show_log=False  # Отключить логи для экономии памяти
            )
            self._model_loaded = True
            logger.info("PaddleOCR model loaded")

    def process_from_bytes(self, image_bytes: bytes) -> Dict:
        """Обработка изображения из байтов"""
        try:
            # Конвертируем байты в изображение
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return {"error": "Failed to decode image"}

            # Обеспечиваем загрузку модели
            self._ensure_loaded()

            # Выполняем OCR
            result = self._ocr.ocr(image, cls=False)

            # Извлекаем текст и координаты
            full_text = ""
            confidences = []

            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], (list, tuple)) else ""
                        confidence = line[1][1] if isinstance(line[1], (list, tuple)) else 0.5
                        full_text += text + " "
                        confidences.append(confidence)

            avg_confidence = sum(confidences) / len(confidences) * 100 if confidences else 50.0

            return {
                "success": True,
                "full_text": full_text.strip(),
                "average_confidence": avg_confidence,
                "character_count": len(full_text),
                "word_count": len(full_text.split()),
                "formulas": self._extract_formulas(full_text),
                "answer_candidates": self._extract_answers(full_text),
                "raw_result": result
            }

        except Exception as e:
            logger.error(f"Error in PaddleOCR: {str(e)}")
            return {"success": False, "error": str(e)}

    def _extract_formulas(self, text: str) -> List[Dict]:
        """Извлечение формул (упрощенное)"""
        formulas = []
        # Паттерны для математических выражений
        import re
        math_patterns = [
            r'[a-zA-Zα-ωΑ-Ω]\s*[=≡≈]\s*[^=\n]+',
            r'\d+\s*[+\-*/^]\s*\d+',
            r'[Ss]\s*=\s*[^=\n]+',
            r'[Vv]\s*=\s*[^=\n]+'
        ]

        for pattern in math_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                formula = match.group(0).strip()
                formulas.append({
                    'latex': formula,
                    'confidence': 0.7,
                    'original_text': formula
                })

        return formulas

    def _extract_answers(self, text: str) -> List[Dict]:
        """Извлечение ответов"""
        import re
        candidates = []

        patterns = [
            r'Ответ\s*[:=]\s*([0-9.,]+)',
            r'[Aa]nswer\s*[:=]\s*([0-9.,]+)',
            r'=\s*([0-9.,]+)\s*$',
            r'получим\s*([0-9.,]+)',
            r'равно\s*([0-9.,]+)'
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.lastindex:
                    answer = match.group(1).strip()
                    candidates.append({
                        'text': answer,
                        'pattern': pattern,
                        'context': match.group(0)
                    })

        return candidates


# Глобальный инстанс для reuse
_OCR_ENGINE = None


def get_ocr_engine():
    """Ленивый геттер для OCR движка"""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        _OCR_ENGINE = OCREngine(lang='ru')
    return _OCR_ENGINE