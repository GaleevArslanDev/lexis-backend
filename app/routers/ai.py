import os
import requests
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Body, status
from pydantic import BaseModel
from ..db import get_session
from ..dependencies import get_current_user, require_role
from sqlmodel import Session
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["ai"])

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# default working free model we tested
DEFAULT_MODEL = "minimax/minimax-m2:free"

if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY is not set. AI endpoints will fail until you set it.")


# --- Pydantic schemas for endpoints ---

class GenerateRequest(BaseModel):
    topic: str
    task_type: Optional[str] = "mixed"  # "mcq", "short", "essay", "mixed"
    num_questions: Optional[int] = 5
    model: Optional[str] = DEFAULT_MODEL


class GeneratedQuestion(BaseModel):
    text: str
    choices: Optional[List[str]] = None  # for MCQ
    correct_answer: Optional[str] = None


class GenerateResponse(BaseModel):
    questions: List[GeneratedQuestion]
    raw: str


class CheckRequest(BaseModel):
    assignment_id: Optional[int] = None
    question_id: Optional[int] = None
    student_id: Optional[int] = None
    text: str
    rubric: Optional[dict] = None  # arbitrary rubric structure
    model: Optional[str] = DEFAULT_MODEL


class CheckResponse(BaseModel):
    score: Optional[float]
    feedback: str
    raw: str


# --- Helper to call OpenRouter ---
def call_openrouter(payload: dict, timeout: int = 30) -> dict:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured on server")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        logger.exception("Network error when calling OpenRouter")
        raise HTTPException(status_code=502, detail=f"Network error calling AI provider: {e}")

    if resp.status_code == 401:
        logger.warning("OpenRouter returned 401 Unauthorized")
        raise HTTPException(status_code=401, detail="AI provider unauthorized (check OPENROUTER_API_KEY)")

    # handle rate limits or provider errors gracefully
    if resp.status_code == 429:
        logger.warning("OpenRouter rate limit / too many requests")
        raise HTTPException(status_code=429, detail="AI provider rate-limited; retry later")

    if resp.status_code >= 500:
        logger.error("AI provider internal error: %s", resp.text)
        raise HTTPException(status_code=502, detail="AI provider internal error; try again later")

    try:
        data = resp.json()
    except ValueError:
        logger.error("AI provider returned non-JSON: %s", resp.text)
        raise HTTPException(status_code=502, detail="AI provider returned invalid response")

    # If provider returns explicit error structure
    if isinstance(data, dict) and data.get("error"):
        logger.error("AI provider error: %s", data["error"])
        # propagate readable message
        msg = data["error"].get("message") if isinstance(data["error"], dict) else str(data["error"])
        raise HTTPException(status_code=502, detail=f"AI provider error: {msg}")

    return data


# --- Utility to parse questions from raw assistant text ---
def parse_questions_from_text(text: str, expected: int = 5) -> List[GeneratedQuestion]:
    """
    Простая эвристика: разбиваем текст по строчным разделителям и сохраняем чистые строки.
    Для MCQ ожидаем формат:
      1. Вопрос?
        A) ...
        B) ...
    Возвращаем минимально структурированный список.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    questions: List[GeneratedQuestion] = []
    current_q = None
    current_choices = []

    for line in lines:
        # нумерация вопросов: "1.", "1)"
        if line.split()[0].rstrip('.)').isdigit() or line.startswith(("Q:", "Q ", "Вопрос")):
            # закрываем предыдущий
            if current_q:
                questions.append(GeneratedQuestion(text=current_q, choices=current_choices or None))
            # начинаем новый
            # убираем ведущую нумерацию
            # примитивно: удаляем первые токены если содержат цифры и точку/скобку
            parts = line.split(maxsplit=1)
            if parts and parts[0].rstrip('.)').isdigit() and len(parts) > 1:
                current_q = parts[1].strip()
            else:
                current_q = line
            current_choices = []
            continue

        # варианты ответов: начинаются с "A)", "B.", "1)", "-", "–"
        if (len(line) >= 2 and (line[0].isalpha() and line[1] in ').]')) or line.startswith(('-', '–', '*')):
            # strip leading letter/marker
            # убираем "A) " или "- "
            cleaned = line
            if len(line) > 2 and line[1] in ').]':
                cleaned = line[2:].strip()
            elif line[0] in ('-', '–', '*'):
                cleaned = line[1:].strip()
            current_choices.append(cleaned)
            continue

        # если пришла строка, а current_q нет — создаём новый вопрос
        if current_q is None:
            current_q = line
        else:
            # вероятно продолжение вопроса — дописываем
            current_q = current_q + " " + line

    # добавляем последний
    if current_q:
        questions.append(GeneratedQuestion(text=current_q, choices=current_choices or None))

    # ограничиваем до expected (если больше)
    if expected and len(questions) > expected:
        questions = questions[:expected]

    return questions


# --- Endpoints ---


@router.post("/generate-questions", response_model=GenerateResponse)
def generate_questions(
    payload: GenerateRequest = Body(...),
    session: Session = Depends(get_session),
    current_user = Depends(require_role("teacher"))
):
    """
    Генерация вопросов (только для учителей).
    payload: topic, task_type, num_questions, optional model.
    """
    model = payload.model or DEFAULT_MODEL
    prompt = (
        f"Составь {payload.num_questions} вопросов по теме: \"{payload.topic}\".\n"
        f"Тип заданий: {payload.task_type}.\n"
        "Для каждого вопроса, если это возможно, дай варианты ответов (A, B, C...) — если это MCQ.\n"
        "Выведи каждый вопрос на новой строке, пронумерованный."
    )

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 700,
        "temperature": 0.2
    }

    data = call_openrouter(body)

    # OpenRouter returns structure similar to OpenAI; но всегда безопасно искать choices
    try:
        raw_text = data["choices"][0]["message"]["content"]
    except Exception:
        # fallback: try other shapes
        raw_text = str(data)

    questions = parse_questions_from_text(raw_text, expected=payload.num_questions)

    return {"questions": questions, "raw": raw_text}


@router.post("/check-answer", response_model=CheckResponse)
def check_answer(
    payload: CheckRequest = Body(...),
    session: Session = Depends(get_session),
    current_user = Depends(get_current_user)
):
    """
    Проверка ответа через модель.
    payload.text - текст ответа студента
    payload.rubric - рубрика (можно передать json)
    Возвращает score (если модель сумеет), feedback и raw текст.
    (Можно расширить, чтобы сохранять результат в БД.)
    """
    model = payload.model or DEFAULT_MODEL

    # Формируем промпт: даём инструкции модели оценить по рубрике и вернуть JSON: {"score": <0-100>, "feedback": "..."}
    rubric_text = ""
    if payload.rubric:
        rubric_text = "Рубрика для оценки (JSON):\n" + str(payload.rubric) + "\n"

    prompt = (
        "Оцени, пожалуйста, работу студента в формате JSON с полями:\n"
        '{"score": <число от 0 до 100>, "feedback": "<подробный комментарий>"}\n\n'
        f"{rubric_text}"
        "Текст работы:\n"
        f"{payload.text}\n\n"
        "Если нельзя вывести число, постарайся дать объективную шкалу и примерную оценку."
    )

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.0
    }

    data = call_openrouter(body)

    try:
        assistant_content = data["choices"][0]["message"]["content"]
    except Exception:
        assistant_content = str(data)

    # Попробуем извлечь JSON из ответа модели (если он вернул JSON)
    score = None
    feedback = assistant_content

    # Простая попытка найти JSON внутри текста
    import re, json
    json_match = re.search(r'(\{.*\})', assistant_content, re.S)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            score = parsed.get("score") or parsed.get("grade") or parsed.get("score_percent")
            if isinstance(score, (int, float)):
                # Если модель вернула 0-1, приводим к 0-100
                if 0 <= score <= 1:
                    score = float(score) * 100
                else:
                    score = float(score)
            feedback = parsed.get("feedback") or parsed.get("comment") or feedback
        except Exception:
            # не смогли распарсить — оставим весь текст в feedback
            logger.debug("Не удалось распарсить JSON из ответа модели")

    # Можно тут сохранить StudentAnswer / Result в БД — закомментировано, пример:
    # try:
    #     ans = StudentAnswer(student_id=payload.student_id, question_id=payload.question_id,
    #                         text_answer=payload.text, score=score, feedback=feedback)
    #     session.add(ans)
    #     session.commit()
    # except Exception as e:
    #     logger.exception("Не удалось сохранить ответ студента в БД: %s", e)

    return {"score": score, "feedback": feedback, "raw": assistant_content}
