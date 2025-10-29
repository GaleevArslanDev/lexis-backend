# app/routers/ai.py
import os
import requests
import re
import json
from typing import Optional, List, Tuple
from fastapi import APIRouter, Depends, HTTPException, Body, status
from pydantic import BaseModel
from ..db import get_session
from ..dependencies_util import get_current_user, require_role
from sqlmodel import Session
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["ai"])

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"

if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY is not set. AI endpoints will fail until you set it.")


class GenerateRequest(BaseModel):
    topic: str
    task_type: Optional[str] = "mixed"
    num_questions: Optional[int] = 5
    model: Optional[str] = DEFAULT_MODEL


class GeneratedQuestion(BaseModel):
    text: str
    choices: Optional[List[str]] = None
    correct_answer: Optional[str] = None  # "A", "B", ...
    correct_index: Optional[int] = None


class GenerateResponse(BaseModel):
    questions: List[GeneratedQuestion]
    raw: str


class CheckRequest(BaseModel):
    assignment_id: Optional[int] = None
    question_id: Optional[int] = None
    student_id: Optional[int] = None
    text: str
    rubric: Optional[dict] = None
    model: Optional[str] = DEFAULT_MODEL


class CheckResponse(BaseModel):
    score: Optional[float]
    feedback: str
    raw: str


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

    if isinstance(data, dict) and data.get("error"):
        logger.error("AI provider error: %s", data["error"])
        msg = data["error"].get("message") if isinstance(data["error"], dict) else str(data["error"])
        raise HTTPException(status_code=502, detail=f"AI provider error: {msg}")

    return data


# --- Parsing utilities ---


def extract_correct_answer_from_text(text: str) -> Tuple[Optional[str], str]:
    """
    Ищет в тексте метку правильного ответа: (Ответ: A), (Answer: A), Correct: A и т.п.
    Возвращает (letter_or_None, cleaned_text_without_label)
    """
    # шаблоны: (Ответ: A), (Ответ: A)*, Answer: A, Correct: A
    pattern = re.compile(r'\(?\s*(?:Ответ|Answer|Correct|Правильно)[:：]?\s*([A-Z0-9])\s*\)?\*?', flags=re.I)
    m = pattern.search(text)
    if m:
        letter = m.group(1).upper()
        cleaned = pattern.sub("", text).strip()
        return letter, cleaned
    # иногда бывает формат: *(Ответ: A) на отдельной строке
    pattern2 = re.compile(r'^\s*\*?\(?\s*(?:Ответ|Answer|Correct|Правильно)[:：]?\s*([A-Z0-9])\s*\)?\*?\s*$', flags=re.I | re.M)
    m2 = pattern2.search(text)
    if m2:
        letter = m2.group(1).upper()
        cleaned = pattern2.sub("", text).strip()
        return letter, cleaned
    return None, text


def extract_choices_from_block(block_text: str) -> Optional[List[str]]:
    """
    Извлекает варианты ответа из блока question. Возвращает список строк без меток A), B) и без вложенных '(Ответ: X)'.
    Поддерживает формы:
      - A) opt1  B) opt2  C) opt3
      - A) opt1\nB) opt2\nC) opt3
      - или варианты разделённые двойным пробелом "A) 5.35  B) 5.45 ..."
    """
    text = block_text.strip()

    # Удалим уже найденные пометки ответа внутри блока (чтобы не мешали)
    _, text = extract_correct_answer_from_text(text)

    # Находим все маркеры типа "A)" в тексте и вырезаем от одного маркера до следующего
    matches = list(re.finditer(r'([A-Z])\)\s*', text))
    if matches:
        choices = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            choice = text[start:end].strip()
            # убираем возможные остатки "(Ответ: ...)"
            choice = re.sub(r'\(?\s*(?:Ответ|Answer|Correct|Правильно)[:：]?.*?\)?', '', choice, flags=re.I).strip()
            # если в конце есть "*", удалить
            choice = choice.rstrip('*').strip()
            # убрать Markdown-звёздочки
            choice = choice.strip(" *")
            if choice:
                choices.append(choice)
        return choices if choices else None

    # fallback: ищем паттерн "A) opt B) opt" в одну строчку, разделяя по "  " (двойной пробел)
    parts = re.split(r'\s{2,}', text)
    opts = []
    for p in parts:
        m = re.match(r'^[A-Z]\)\s*(.*)', p)
        if m:
            opt = m.group(1).strip()
            opt = re.sub(r'\(?\s*(?:Ответ|Answer|Correct|Правильно)[:：]?.*?\)?', '', opt, flags=re.I).strip()
            if opt:
                opts.append(opt)
    if opts:
        return opts

    return None


def block_is_header(block_text: str) -> bool:
    # простая эвристика: если блок содержит слова "вот", "пример", "вопросов" и короткий — пометим как заголовок
    low = block_text.lower()
    if len(block_text.split()) < 10 and any(w in low for w in ("вот", "пример", "вопросов", "вопросы")):
        return True
    return False


def parse_questions_from_text(text: str, expected: int = 5) -> List[GeneratedQuestion]:
    """
    Разбивает текст ассистента на вопросы и извлекает варианты и правильный ответ.
    Работает с разными форматами вывода модели.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    # объединяем блоки: каждый вопрос обычно начинается с "1." или "1)"
    blocks = []
    current = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            # пустая строка означает разделитель
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue

        # начало нового вопрос: нумерация или "1." или "Q:" или "Вопрос"
        if re.match(r'^\s*\d+[\.\)]\s*', stripped) or re.match(r'^(вопрос|q[:\s])', stripped, flags=re.I):
            if current:
                blocks.append("\n".join(current).strip())
            current = [stripped]
        else:
            # если строка выглядит как "A) ..." или "B) ..." и текущего блока нет — начинаем новый
            if re.match(r'^[A-Z]\)\s+', stripped) and not current:
                current = [stripped]
            else:
                current.append(stripped)

    if current:
        blocks.append("\n".join(current).strip())

    questions: List[GeneratedQuestion] = []

    for block in blocks:
        if block_is_header(block):
            # пропускаем общий заголовок вроде "Вот 5 смешанных вопросов..."
            continue

        # Извлекаем правильный ответ (если есть) и очищаем блок
        correct_letter, cleaned_block = extract_correct_answer_from_text(block)

        # Разделяем заголовок вопроса и варианты (если варианты идут в отдельных строках)
        # Попробуем сначала найти первую строку, начинающуюся с "A)" — всё до неё считается текстом вопроса
        m_choice_start = re.search(r'\n?[A-Z]\)\s', cleaned_block)
        if m_choice_start:
            q_text = cleaned_block[:m_choice_start.start()].strip()
            choices_block = cleaned_block[m_choice_start.start():].strip()
        else:
            # возможно варианты в той же строке после вопроса
            # попытаемся отделить по двойному пробелу перед "A)"
            m_inline = re.search(r'\s{2,}[A-Z]\)\s', cleaned_block)
            if m_inline:
                q_text = cleaned_block[:m_inline.start()].strip()
                choices_block = cleaned_block[m_inline.start():].strip()
            else:
                # вариантов нет явных — весь блок считаем вопросом без вариантов
                q_text = cleaned_block.strip()
                choices_block = ""

        # извлечём варианты
        choices = extract_choices_from_block(choices_block) if choices_block else None

        # если правильный ответ указали, попробуем сопоставить с индексом
        correct_index = None
        if correct_letter and choices:
            # A -> 0, B -> 1, ...
            idx = ord(correct_letter.upper()) - ord('A')
            if 0 <= idx < len(choices):
                correct_index = idx
        elif not correct_letter and choices:
            # иногда модель ставит ответ как часть последнего элемента "(Ответ: A)" — уже удаляем в extract_choices
            # если нет, можно попытаться найти "(Ответ: A)" отдельно в block (we already removed)
            pass

        # если вопрос текст начинается с "1. **..." и содержит markdown, почистим markdown звездочки
        q_text = q_text.strip(" *")

        questions.append(GeneratedQuestion(
            text=q_text,
            choices=choices,
            correct_answer=(chr(ord('A') + correct_index) if correct_index is not None else (correct_letter if correct_letter else None)),
            correct_index=correct_index
        ))

        if expected and len(questions) >= expected:
            break

    # если парсер ничего не вывел — вернуть fallback: одна запись с raw текстом
    if not questions:
        return [GeneratedQuestion(text=text, choices=None, correct_answer=None)]

    return questions


# --- Endpoints ---


@router.post("/generate-questions", response_model=GenerateResponse)
def generate_questions(
    payload: GenerateRequest = Body(...),
    session: Session = Depends(get_session),
    current_user = Depends(require_role("teacher"))
):
    model = payload.model or DEFAULT_MODEL
    prompt = (
        f"Составь {payload.num_questions} вопросов по теме: \"{payload.topic}\".\n"
        f"Тип заданий: {payload.task_type}.\n"
        "Для каждого вопроса, если это возможно, дай варианты ответов (A, B, C...) — если это MCQ.\n"
        "Выведи каждый вопрос на новой строке, пронумерованный. Если можешь, укажи правильный ответ рядом в формате (Ответ: A)."
    )

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 700,
        "temperature": 0.2
    }

    data = call_openrouter(body)

    try:
        raw_text = data["choices"][0]["message"]["content"]
    except Exception:
        raw_text = str(data)

    questions = parse_questions_from_text(raw_text, expected=payload.num_questions)

    return {"questions": questions, "raw": raw_text}


@router.post("/check-answer", response_model=CheckResponse)
def check_answer(
    payload: CheckRequest = Body(...),
    session: Session = Depends(get_session),
    current_user = Depends(get_current_user)
):
    model = payload.model or DEFAULT_MODEL

    rubric_text = ""
    if payload.rubric:
        rubric_text = "Рубрика для оценки (JSON):\n" + json.dumps(payload.rubric, ensure_ascii=False) + "\n"

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

    score = None
    feedback = assistant_content

    import re as _re, json as _json
    json_match = _re.search(r'(\{.*\})', assistant_content, _re.S)
    if json_match:
        try:
            parsed = _json.loads(json_match.group(1))
            score = parsed.get("score") or parsed.get("grade") or parsed.get("score_percent")
            if isinstance(score, (int, float)):
                if 0 <= score <= 1:
                    score = float(score) * 100
                else:
                    score = float(score)
            feedback = parsed.get("feedback") or parsed.get("comment") or feedback
        except Exception:
            logger.debug("Не удалось распарсить JSON из ответа модели")

    return {"score": score, "feedback": feedback, "raw": assistant_content}
