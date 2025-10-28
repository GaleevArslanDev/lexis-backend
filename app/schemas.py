from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: Optional[str] = "teacher"

class UserRead(BaseModel):
    id: int
    email: EmailStr
    role: str
    created_at: datetime

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class ClassCreate(BaseModel):
    name: str

class AssignmentCreate(BaseModel):
    class_id: int
    title: str
    description: Optional[str] = None
    type: str = "essay"

class QuestionCreate(BaseModel):
    assignment_id: int
    text: str
    options: Optional[str] = None
    correct_answer: Optional[str] = None
