"""
Клиент для взаимодействия с внешним OCR API (версия v1)
"""
import requests
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class OCRAPIClientV1:
    """Клиент для API v1 с эндпоинтом /ocr/assess"""

    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def assess_solution(
            self,
            image_bytes: bytes,
            filename: str = "image.png",
            reference_answer: Optional[str] = None,
            reference_formulas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Отправляет изображение на эндпоинт /api/v1/ocr/assess для полной проверки решения
        
        Args:
            image_bytes: Байты изображения
            filename: Имя файла
            reference_answer: Эталонный ответ для сравнения
            reference_formulas: Список эталонных формул через точку с запятой
        
        Returns:
            Dict с результатами проверки
        """
        try:
            files = {"file": (filename, image_bytes, self._get_mime_type(filename))}
            data = {}

            if reference_answer:
                data["reference_answer"] = reference_answer

            if reference_formulas:
                data["reference_formulas"] = ";".join(reference_formulas)

            print("Data")
            print(data)

            response = requests.post(
                f"{self.base_url}/api/v1/ocr/assess",
                files=files,
                data=data,
                headers=self.headers,
                timeout=120  # Увеличиваем таймаут для полной обработки
            )

            if response.status_code != 200:
                logger.error(f"OCR API v1 error: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"OCR API returned {response.status_code}: {response.text}",
                    "assessment": None
                }

            result = response.json()

            if not result.get("success", False):
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "assessment": None
                }

            return {
                "success": True,
                "assessment": result.get("assessment"),
                "error": None
            }

        except requests.exceptions.Timeout:
            logger.error("OCR API timeout")
            return {
                "success": False,
                "error": "OCR API timeout after 120 seconds",
                "assessment": None
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling OCR API: {str(e)}")
            return {
                "success": False,
                "error": f"Network error: {str(e)}",
                "assessment": None
            }
        except Exception as e:
            logger.error(f"Unexpected error in OCR processing: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "assessment": None
            }

    def health_check(self) -> Dict[str, Any]:
        """Проверка состояния сервиса"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/health",
                headers=self.headers,
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            return {"status": "unhealthy", "services": {}}
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}

    def get_queue_status(self) -> Dict[str, Any]:
        """Получить статус очереди"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/ocr/queue/status",
                headers=self.headers,
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            logger.error(f"Queue status check failed: {str(e)}")
            return {}

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Получить статус задачи по ID"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/ocr/queue/job/{job_id}",
                headers=self.headers,
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return {"status": "not_found"}
            return {"status": "error"}
        except Exception as e:
            logger.error(f"Job status check failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _get_mime_type(self, filename: str) -> str:
        """Определить MIME-тип по расширению файла"""
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        mime_types = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
            'tiff': 'image/tiff',
            'webp': 'image/webp'
        }
        return mime_types.get(ext, 'application/octet-stream')


# Глобальный инстанс клиента OCR v1
_OCR_CLIENT_V1 = None


def get_ocr_client_v1() -> OCRAPIClientV1:
    """
    Получить или создать экземпляр OCR клиента v1
    """
    global _OCR_CLIENT_V1

    if _OCR_CLIENT_V1 is None:
        import os
        ocr_api_url = os.getenv("OCR_API_URL", "https://galeevarslan2021-lexis-ocr.hf.space")
        ocr_api_token = os.getenv("OCR_API_TOKEN", "")

        if not ocr_api_token:
            logger.warning("OCR_API_TOKEN not set. OCR will fail.")

        _OCR_CLIENT_V1 = OCRAPIClientV1(ocr_api_url, ocr_api_token)

    return _OCR_CLIENT_V1