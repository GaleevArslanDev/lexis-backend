"""
LLM Service — анализ решения через OpenRouter.
Адаптирован для lexis-backend: использует os.getenv вместо settings.
"""
import httpx
import json
import re
import asyncio
import os
from typing import List, Dict, Any, Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .schemas import LLMAnalysisResult

# ---------------------------------------------------------------------------
# Конфиг из env
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-flash-1.5")
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "1500"))
OPENROUTER_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.1"))
OPENROUTER_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "30"))


class LLMService:
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        logger.info(f"LLM Service initialized — model: {OPENROUTER_MODEL}")

    async def _get_client(self) -> httpx.AsyncClient:
        if not self._client:
            self._client = httpx.AsyncClient(
                timeout=OPENROUTER_TIMEOUT,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Промпт
    # ------------------------------------------------------------------

    def _build_analysis_prompt(self, steps: List[Dict], reference_answer: str) -> str:
        if len(steps) > 40:
            steps = steps[:40]
            logger.warning("Truncated steps to 40 for LLM analysis")

        steps_text = "\n".join([
            f"Шаг {s.get('step_id', i + 1)}: {s.get('formula', '')}"
            for i, s in enumerate(steps)
        ])

        return f"""Ты - эксперт по математике, анализирующий решение ученика.

Решение ученика:
{steps_text}

Эталонный ответ: {reference_answer}

Проанализируй решение и предоставь ТОЛЬКО JSON ответ со следующими полями:
1. "reasoning_logic": строка с восстановленной логикой решения (2-3 предложения)
2. "step_correctness": массив булевых значений для каждого шага (true/false)
3. "step_explanations": массив строк с краткими объяснениями для каждого шага
4. "hidden_errors": массив строк со скрытыми ошибками (если нет ошибок, пустой массив)
5. "teacher_comment": строка с кратким комментарием для учителя (1 предложение)
6. "llm_score": число от 0.0 до 1.0 (оценка логики и целостности решения)
7. "ocr_quality_score": число от 0.0 до 1.0 (оценка читаемости рукописи по содержанию — насколько формулы выглядят осмысленно и полно)
8. "confidence": число от 0.0 до 1.0 (уверенность в анализе)

Пример:
{{
    "reasoning_logic": "Ученик правильно решил уравнение.",
    "step_correctness": [true, true, true],
    "step_explanations": ["Верно записано уравнение", "Правильный перенос", "Корень найден верно"],
    "hidden_errors": [],
    "teacher_comment": "Решение полностью верное",
    "llm_score": 1.0,
    "ocr_quality_score": 0.9,
    "confidence": 0.95
}}

Важно для "ocr_quality_score": оценивай НЕ уверенность OCR-движка, а то, насколько
распознанные формулы выглядят осмысленно. Если формулы похожи на обрывки и символьный мусор —
ставь низкую оценку. Если формулы структурированы и читаемы — высокую.

Ответ должен быть ТОЛЬКО валидным JSON, без дополнительного текста.
"""

    # ------------------------------------------------------------------
    # Парсинг ответа
    # ------------------------------------------------------------------

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    fixed = json_str.replace("'", '"')
                    try:
                        return json.loads(fixed)
                    except Exception:
                        return self._extract_from_text(response)
            return self._extract_from_text(response)
        except Exception as e:
            logger.error(f"_parse_llm_response error: {e}")
            return self._get_fallback_analysis()

    def _extract_from_text(self, text: str) -> Dict[str, Any]:
        result = self._get_fallback_analysis()
        try:
            for line in text.lower().split('\n'):
                if 'llm_score' in line:
                    nums = re.findall(r'0\.\d+', line)
                    if nums:
                        result['llm_score'] = float(nums[0])
                if 'ocr_quality' in line:
                    nums = re.findall(r'0\.\d+', line)
                    if nums:
                        result['ocr_quality_score'] = float(nums[0])
        except Exception:
            pass
        return result

    def _get_fallback_analysis(self) -> Dict[str, Any]:
        return {
            "reasoning_logic": "Не удалось восстановить логику решения",
            "step_correctness": [],
            "step_explanations": [],
            "hidden_errors": ["LLM анализ недоступен"],
            "teacher_comment": "Требуется ручная проверка",
            "llm_score": 0.5,
            "ocr_quality_score": 0.5,
            "confidence": 0.3,
        }

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    )
    async def _call_openrouter(self, prompt: str) -> Optional[str]:
        if not OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY not set")
            return None

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:10000",
            "X-Title": "AssessIt",
        }
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": OPENROUTER_MAX_TOKENS,
            "temperature": OPENROUTER_TEMPERATURE,
        }

        try:
            client = await self._get_client()
            response = await client.post(f"{OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=payload)
        except httpx.TimeoutException:
            logger.error("OpenRouter timeout")
            raise

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        elif response.status_code == 429:
            raise httpx.TimeoutException("Rate limited")
        else:
            logger.error(f"OpenRouter error: {response.status_code} — {response.text[:300]}")
            return None

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    async def analyze_solution(self, steps: List[Dict], reference_answer: str) -> LLMAnalysisResult:
        # Нормализуем шаги
        converted = []
        for step in steps:
            if isinstance(step, dict):
                converted.append(step)
            elif hasattr(step, 'model_dump'):
                converted.append(step.model_dump())
            elif hasattr(step, 'dict'):
                converted.append(step.dict())
            else:
                converted.append({"step_id": len(converted) + 1, "formula": str(step)})
        steps = converted

        if not OPENROUTER_API_KEY:
            fallback = self._get_fallback_analysis()
            return LLMAnalysisResult(**fallback)

        try:
            prompt = self._build_analysis_prompt(steps, reference_answer)
            try:
                response_text = await asyncio.wait_for(
                    self._call_openrouter(prompt), timeout=OPENROUTER_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.error("LLM analysis timed out")
                fallback = self._get_fallback_analysis()
                fallback["hidden_errors"] = ["LLM анализ превысил время ожидания"]
                return LLMAnalysisResult(**fallback)

            if not response_text:
                return LLMAnalysisResult(**self._get_fallback_analysis())

            parsed = self._parse_llm_response(response_text)

            # Формируем step_correctness в нужном формате
            step_correctness = []
            for i, step in enumerate(steps):
                correct = False
                if i < len(parsed.get("step_correctness", [])):
                    val = parsed["step_correctness"][i]
                    correct = list(val.values())[0] if isinstance(val, dict) else bool(val)
                step_correctness.append({step.get("step_id", i + 1): correct})

            return LLMAnalysisResult(
                reasoning_logic=parsed.get("reasoning_logic", "Логика не восстановлена"),
                step_correctness=step_correctness,
                step_explanations=parsed.get("step_explanations", []),
                hidden_errors=parsed.get("hidden_errors", []),
                teacher_comment=parsed.get("teacher_comment", "Анализ выполнен автоматически"),
                llm_score=float(parsed.get("llm_score", 0.5)),
                ocr_quality_score=float(parsed.get("ocr_quality_score", 0.5)),
                confidence=float(parsed.get("confidence", 0.5)),
            )

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            fallback = self._get_fallback_analysis()
            fallback["hidden_errors"].append(f"Error: {str(e)}")
            return LLMAnalysisResult(**fallback)


llm_service = LLMService()