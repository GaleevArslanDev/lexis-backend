import requests
import base64
from typing import Dict, Any, Optional, List
import logging
from io import BytesIO
import tempfile
import os

logger = logging.getLogger(__name__)


class OCRAPIClient:
    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def process_image_bytes(
            self,
            image_bytes: bytes,
            filename: str = "image.png",
            language: str = "rus+eng",
            extract_formulas: bool = True,
            preprocess_level: str = "auto"
    ) -> Dict[str, Any]:
        """
        Отправляет изображение в OCR API и возвращает результат
        """
        try:
            files = {"file": (filename, image_bytes, "image/png")}
            data = {
                "language": language,
                "preprocess_level": preprocess_level,
                "extract_formulas": str(extract_formulas).lower()
            }

            response = requests.post(
                f"{self.base_url}/api/ocr",
                files=files,
                data=data,
                headers=self.headers,
                timeout=60  # Увеличиваем таймаут для обработки
            )

            if response.status_code != 200:
                logger.error(f"OCR API error: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"OCR API returned {response.status_code}: {response.text}",
                    "full_text": "",
                    "average_confidence": 0.0
                }

            result = response.json()

            if result.get("status") != "success":
                return {
                    "success": False,
                    "error": f"OCR API processing failed: {result.get('detail', 'Unknown error')}",
                    "full_text": "",
                    "average_confidence": 0.0
                }

            # Преобразуем ответ OCR API в формат, совместимый с нашим приложением
            ocr_data = self._format_ocr_result(result)

            return {
                "success": True,
                **ocr_data
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling OCR API: {str(e)}")
            return {
                "success": False,
                "error": f"Network error: {str(e)}",
                "full_text": "",
                "average_confidence": 0.0
            }
        except Exception as e:
            logger.error(f"Unexpected error in OCR processing: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "full_text": "",
                "average_confidence": 0.0
            }

    def _format_ocr_result(self, api_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Форматирует результат OCR API в формат, совместимый с нашим приложением
        """
        try:
            results = api_result.get("results", {})
            ocr_data = results.get("ocr_data", {})

            # Получаем полный текст
            full_text = ocr_data.get("full_text", "")

            # Вычисляем среднюю уверенность
            text_blocks = ocr_data.get("text_blocks", [])
            if text_blocks:
                confidences = [block.get("confidence", 0.0) for block in text_blocks]
                avg_confidence = sum(confidences) / len(confidences) * 100
            else:
                avg_confidence = 0.0

            # Извлекаем формулы
            formulas = []
            formatted_results = results.get("formatted_results", [])
            for item in formatted_results:
                if item.get("type") == "formula":
                    formulas.append({
                        "latex": item.get("content", ""),
                        "confidence": item.get("metadata", {}).get("confidence", 0.8),
                        "position": item.get("position", {}),
                        "is_correct": None
                    })

            # Оценка качества
            quality_assessment = results.get("summary", {}).get("quality_assessment", {})

            return {
                "full_text": full_text,
                "average_confidence": avg_confidence,
                "character_count": len(full_text),
                "word_count": len(full_text.split()),
                "formulas": formulas,
                "quality_assessment": quality_assessment,
                "formatted_results": formatted_results,
                "text_blocks": text_blocks
            }

        except Exception as e:
            logger.error(f"Error formatting OCR result: {str(e)}")
            return {
                "full_text": "",
                "average_confidence": 0.0,
                "character_count": 0,
                "word_count": 0,
                "formulas": [],
                "quality_assessment": {},
                "formatted_results": [],
                "text_blocks": []
            }

    def batch_process(
            self,
            files: List[Dict[str, bytes]],
            language: str = "rus+eng"
    ) -> Dict[str, Any]:
        """
        Пакетная обработка нескольких файлов
        """
        try:
            files_data = []
            for i, file_data in enumerate(files):
                filename = file_data.get("filename", f"image_{i}.png")
                content = file_data.get("content")
                files_data.append(("files", (filename, content, "image/png")))

            data = {"language": language}

            response = requests.post(
                f"{self.base_url}/api/batch-ocr",
                files=files_data,
                data=data,
                headers=self.headers,
                timeout=120
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Batch OCR API error: {response.status_code}"
                }

            return response.json()

        except Exception as e:
            logger.error(f"Error in batch OCR processing: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def health_check(self) -> bool:
        """
        Проверяет доступность OCR API
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/health",
                headers=self.headers,
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def get_languages(self) -> List[str]:
        """
        Получает список поддерживаемых языков
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/languages",
                headers=self.headers,
                timeout=5
            )
            if response.status_code == 200:
                return response.json().get("available_languages", [])
            return ["rus+eng"]
        except:
            return ["rus+eng"]


# Глобальный инстанс клиента OCR
_OCR_CLIENT = None


def get_ocr_client() -> OCRAPIClient:
    """
    Получить или создать экземпляр OCR клиента
    """
    global _OCR_CLIENT

    if _OCR_CLIENT is None:
        import os
        ocr_api_url = os.getenv("OCR_API_URL", "https://galeevarslan2021-lexis-ocr.hf.space")
        ocr_api_token = os.getenv("OCR_API_TOKEN", "")

        if not ocr_api_token:
            logger.warning("OCR_API_TOKEN not set. OCR will fail.")

        _OCR_CLIENT = OCRAPIClient(ocr_api_url, ocr_api_token)

    return _OCR_CLIENT