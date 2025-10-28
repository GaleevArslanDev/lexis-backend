from fastapi import APIRouter, Body

router = APIRouter(prefix="/ai", tags=["ai"])

@router.post("/generate-questions")
def generate_questions(payload: dict = Body(...)):
    # TODO: integrate OpenAI or local LLM
    return {"questions": [{"text": "Пример вопроса 1"}, {"text": "Пример вопроса 2"}]}

@router.post("/check-answer")
def check_answer(answer: dict = Body(...)):
    # answer = {"text": "...", "rubric": {...}}
    # TODO: call LLM/AI and return score + feedback
    return {"score": 7.5, "feedback": "Пример развернутого фидбэка по эссе"}
