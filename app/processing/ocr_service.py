"""
OCR Service — VLM-based (Vision Language Model via OpenRouter).
Адаптирован для lexis-backend: использует os.getenv вместо settings.
"""
import asyncio
import base64
import json
import re
import os
import io
from typing import List, Optional, Tuple

import httpx
from PIL import Image
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .schemas import OCRStepResult

# ---------------------------------------------------------------------------
# Конфиг из env
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
VLM_MODEL = os.getenv("VLM_MODEL", "google/gemini-flash-1.5")
VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "2000"))
VLM_TIMEOUT = int(os.getenv("VLM_TIMEOUT", "60"))
VLM_MAX_IMAGE_SIZE = int(os.getenv("VLM_MAX_IMAGE_SIZE", "1600"))


class OCRService:
    def __init__(self):
        logger.info(f"Initializing VLM-based OCR Service — model: {VLM_MODEL}")
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # HTTP-клиент (ленивая инициализация)
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(VLM_TIMEOUT),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Подготовка изображения
    # ------------------------------------------------------------------

    def _prepare_image(self, image_path: str) -> Tuple[str, str]:
        with open(image_path, "rb") as f:
            raw = f.read()

        ext = os.path.splitext(image_path)[1].lower()
        media_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".gif": "image/gif", ".bmp": "image/bmp", ".tiff": "image/tiff", ".webp": "image/webp",
        }
        media_type = media_map.get(ext, "image/jpeg")

        img = Image.open(io.BytesIO(raw))
        if max(img.size) > VLM_MAX_IMAGE_SIZE:
            img.thumbnail((VLM_MAX_IMAGE_SIZE, VLM_MAX_IMAGE_SIZE), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            raw = buf.getvalue()
            media_type = "image/png"
            logger.info(f"Image resized to fit {VLM_MAX_IMAGE_SIZE}px")

        return base64.b64encode(raw).decode(), media_type

    # ------------------------------------------------------------------
    # Промпт
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ocr_prompt() -> str:
        return (
            "Ты — эксперт по распознаванию рукописных математических решений.\n\n"
            "Внимательно рассмотри изображение. На нём — рукописное решение задачи.\n\n"
            "Извлеки все шаги решения строго по порядку сверху вниз.\n"
            "Каждый шаг — это одна строка или одно математическое выражение.\n\n"
            "Верни ТОЛЬКО валидный JSON (без лишнего текста до или после):\n"
            "{\n"
            '    "steps": [\n'
            "        {\n"
            '            "step_id": 1,\n'
            '            "formula": "выражение в LaTeX (без обёрток $...$)",\n'
            '            "confidence": 0.9\n'
            "        }\n"
            "    ],\n"
            '    "final_answer": "итоговый ответ или null",\n'
            '    "ocr_notes": "заметки о качестве рукописи или null"\n'
            "}\n\n"
            "Правила:\n"
            "- Одна строка решения = один элемент списка steps\n"
            "- Используй LaTeX для формул\n"
            "- confidence: 0.0–1.0 — твоя уверенность в распознавании конкретного шага\n"
            "- Если текст плохо читаем — всё равно попробуй распознать и снизь confidence\n"
            "- Не пропускай шаги, даже если не уверен\n"
            "- Если решение отсутствует или изображение нечитаемо, верни пустой список steps"
        )

    # ------------------------------------------------------------------
    # VLM-вызов
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    )
    async def _call_vlm(self, image_path: str) -> Optional[str]:
        if not OPENROUTER_API_KEY:
            logger.error("OPENROUTER_API_KEY not set")
            return None

        b64, media_type = self._prepare_image(image_path)

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:10000",
            "X-Title": "AssessIt OCR",
        }

        payload = {
            "model": VLM_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                    {"type": "text", "text": self._build_ocr_prompt()},
                ],
            }],
            "max_tokens": VLM_MAX_TOKENS,
            "temperature": 0.1,
        }

        client = await self._get_client()
        try:
            response = await client.post(f"{OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=payload)
        except httpx.TimeoutException:
            logger.error("VLM call timed out")
            raise

        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            logger.info(f"VLM response received ({len(content)} chars)")
            return content
        elif response.status_code == 429:
            logger.warning("Rate limit — retrying...")
            raise httpx.TimeoutException("Rate limited")
        else:
            logger.error(f"VLM API error: {response.status_code} — {response.text[:300]}")
            return None

    # ------------------------------------------------------------------
    # Парсинг ответа
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_vlm_response(text: str) -> dict:
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
        return {"steps": [], "final_answer": None, "ocr_notes": "parse error"}

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    async def process_image_async(self, image_path: str) -> List[OCRStepResult]:
        logger.info(f"VLM OCR: {image_path}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            raw_response = await asyncio.wait_for(self._call_vlm(image_path), timeout=VLM_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error("VLM OCR timed out")
            return []

        if not raw_response:
            logger.error("Empty VLM response")
            return []

        parsed = self._parse_vlm_response(raw_response)
        steps_raw = parsed.get("steps", [])

        if not steps_raw:
            logger.warning("VLM returned no steps")

        results: List[OCRStepResult] = []
        for i, step in enumerate(steps_raw, start=1):
            formula = str(step.get("formula", "")).strip()
            if not formula:
                continue
            results.append(OCRStepResult(
                step_id=step.get("step_id", i),
                formula=formula,
                confidence=float(step.get("confidence", 0.8)),
                bbox={"x": 0, "y": 0, "width": 0, "height": 0},
            ))
            logger.info(f"Step {results[-1].step_id}: '{formula[:60]}' (conf={results[-1].confidence})")

        logger.info(f"VLM OCR complete: {len(results)} steps extracted")
        return results

    # ------------------------------------------------------------------
    # Утилита извлечения финального ответа
    # ------------------------------------------------------------------

    @staticmethod
    def extract_final_answer(steps: List[dict]) -> Optional[str]:
        if not steps:
            return None

        formula = steps[-1].get("formula", "")
        patterns = [
            r"ответ[:\s]*(.+)$",
            r"answer[:\s]*(.+)$",
            r"=\s*([-\d.,]+)$",
            r"=\s*([^\s=]+)$",
        ]
        for pat in patterns:
            m = re.search(pat, formula, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                if candidate:
                    return candidate

        if "=" in formula:
            return formula.split("=")[-1].strip()
        return None


ocr_service = OCRService()