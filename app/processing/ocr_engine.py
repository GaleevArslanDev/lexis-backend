import cv2
import numpy as np
from typing import Dict, List, Optional
import logging
import os

logger = logging.getLogger(__name__)


class LightweightOCREngine:
    """Облегченный OCR движок для Render"""

    def __init__(self):
        self._ocr = None
        self._initialized = False

    def _lazy_init(self):
        """Ленивая инициализация PaddleOCR только при первом использовании"""
        if not self._initialized:
            try:
                # Импортируем только при необходимости
                from paddleocr import PaddleOCR

                # Минимальная конфигурация для экономии памяти
                self._ocr = PaddleOCR(
                    use_angle_cls=False,  # Отключаем классификатор угла
                    lang='ru',  # Только русский
                    det_db_thresh=0.3,
                    det_db_box_thresh=0.5,
                    use_gpu=False,
                    show_log=False,
                    enable_mkldnn=True,  # Ускорение на CPU
                    use_tensorrt=False,
                    drop_score=0.5,
                    # Отключаем ненужные компоненты
                    cls_model_dir='',
                    rec_model_dir='',
                    det_model_dir=''
                )
                self._initialized = True
                logger.info("PaddleOCR initialized in lightweight mode")

                # Принудительный сбор мусора
                import gc
                gc.collect()

            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
                self._ocr = None

    def process_from_bytes(self, image_bytes: bytes) -> Dict:
        """Обработка изображения из байтов с минимальным использованием памяти"""
        try:
            self._lazy_init()

            if self._ocr is None:
                return {
                    "success": False,
                    "error": "OCR engine not available",
                    "full_text": "",
                    "average_confidence": 0.0
                }

            # Конвертируем байты в numpy array без промежуточных файлов
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return {
                    "success": False,
                    "error": "Failed to decode image",
                    "full_text": "",
                    "average_confidence": 0.0
                }

            # Уменьшаем размер изображения если оно слишком большое
            height, width = img.shape[:2]
            if height > 2000 or width > 2000:
                scale = 2000 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))

            # Конвертируем в grayscale для экономии памяти
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # Улучшаем контраст
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Обрабатываем только одну страницу
            result = self._ocr.ocr(enhanced, cls=False)

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

            # Очищаем память
            del img, gray, enhanced, result
            import gc
            gc.collect()

            return {
                "success": True,
                "full_text": full_text.strip(),
                "average_confidence": avg_confidence,
                "character_count": len(full_text),
                "word_count": len(full_text.split())
            }

        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "full_text": "",
                "average_confidence": 0.0
            }

    def cleanup(self):
        """Очистка памяти"""
        self._ocr = None
        self._initialized = False
        import gc
        gc.collect()


# Глобальный инстанс
_OCR_ENGINE = None


def get_ocr_engine():
    """Получить или создать OCR движок"""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        _OCR_ENGINE = LightweightOCREngine()
    return _OCR_ENGINE