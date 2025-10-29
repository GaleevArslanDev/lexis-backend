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


class ClassCreate(BaseModel):
    name: str
    school_id: Optional[int] = None  # теперь опционально


class ClassUpdate(BaseModel):
    name: Optional[str] = None
    school_id: Optional[int] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class SchoolCreate(BaseModel):
    name: str
    address: Optional[str] = None


class SchoolUpdate(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
