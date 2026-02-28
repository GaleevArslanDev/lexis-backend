from fastapi import HTTPException, status
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ErrorCode:
    # Аутентификация (1000-1999)
    INVALID_CREDENTIALS = "AUTH_1001"
    TOKEN_EXPIRED = "AUTH_1002"
    TOKEN_INVALID = "AUTH_1003"
    ACCESS_DENIED = "AUTH_1004"

    # Ресурсы (2000-2999)
    NOT_FOUND = "RES_2001"
    ALREADY_EXISTS = "RES_2002"
    VALIDATION_ERROR = "RES_2003"

    # OCR/ML ошибки (3000-3999)
    OCR_TIMEOUT = "OCR_3001"
    OCR_API_ERROR = "OCR_3002"
    OCR_QUEUE_FULL = "OCR_3003"
    OCR_NO_FORMULAS = "OCR_3004"

    # Системные ошибки (9000-9999)
    INTERNAL_ERROR = "SYS_9001"
    SERVICE_UNAVAILABLE = "SYS_9002"
    DATABASE_ERROR = "SYS_9003"


class AppException(HTTPException):
    def __init__(
            self,
            status_code: int,
            error_code: str,
            message: str,
            details: Optional[Dict[str, Any]] = None
    ):
        self.error_code = error_code
        self.details = details or {}
        content = {
            "success": False,
            "error": {
                "code": error_code,
                "message": message,
                "details": self.details
            }
        }
        super().__init__(status_code=status_code, detail=content)

        # Логируем ошибку
        logger.error(f"AppException: {error_code} - {message} | Details: {details}")


def handle_ocr_error(error: Exception, image_id: Optional[int] = None) -> AppException:
    """Конвертирует ошибки OCR API в AppException"""
    error_str = str(error).lower()

    if "timeout" in error_str:
        return AppException(
            status_code=504,
            error_code=ErrorCode.OCR_TIMEOUT,
            message="OCR API timeout. The image processing took too long.",
            details={"image_id": image_id, "timeout_seconds": 120}
        )
    elif "no formulas" in error_str:
        return AppException(
            status_code=400,
            error_code=ErrorCode.OCR_NO_FORMULAS,
            message="No formulas recognized in the image. Please ensure the image contains mathematical expressions.",
            details={"image_id": image_id}
        )
    elif "queue full" in error_str or "queue is full" in error_str:
        return AppException(
            status_code=503,
            error_code=ErrorCode.OCR_QUEUE_FULL,
            message="OCR service is busy. Please try again later.",
            details={"image_id": image_id}
        )
    else:
        return AppException(
            status_code=502,
            error_code=ErrorCode.OCR_API_ERROR,
            message=f"OCR API error: {str(error)}",
            details={"image_id": image_id}
        )