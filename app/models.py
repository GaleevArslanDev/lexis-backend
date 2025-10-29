from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime


class ClassStudentLink(SQLModel, table=True):
    class_id: int = Field(foreign_key="class.id", primary_key=True)
    student_id: int = Field(foreign_key="user.id", primary_key=True)


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, nullable=False, unique=True)
    hashed_password: str
    role: str = Field(default="teacher")  # 'teacher' or 'student'
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # связи
    classes: List["Class"] = Relationship(back_populates="teacher")

    # добавляем связь ученик → классы
    enrolled_classes: List["Class"] = Relationship(
        back_populates="students",
        link_model=ClassStudentLink
    )


class Class(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    teacher_id: int = Field(foreign_key="user.id")
    school_id: Optional[int] = Field(foreign_key="school.id", default=None)  # необязательная привязка
    created_at: datetime = Field(default_factory=datetime.utcnow)

    teacher: Optional[User] = Relationship(back_populates="classes")
    students: List[User] = Relationship(
        back_populates="enrolled_classes",
        link_model=ClassStudentLink
    )
    school: Optional["School"] = Relationship(back_populates="classes")

class School(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    address: Optional[str] = None
    creator_id: int = Field(foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    classes: List[Class] = Relationship(back_populates="school")


class Assignment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    class_id: int = Field(foreign_key="class.id")
    title: str
    description: Optional[str] = None
    type: str = Field(default="essay")  # 'mcq', 'short', 'essay'
    created_at: datetime = Field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = None


class Question(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    assignment_id: int = Field(foreign_key="assignment.id")
    text: str
    options: Optional[str] = None  # JSON string for simplicity
    correct_answer: Optional[str] = None
    rubric_criteria: Optional[str] = None  # JSON string


class StudentAnswer(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: Optional[int] = None
    question_id: Optional[int] = None
    text_answer: Optional[str] = None
    file_url: Optional[str] = None
    score: Optional[float] = None
    feedback: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Rubric(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    assignment_id: int = Field(foreign_key="assignment.id")
    criteria_json: Optional[str] = None
    max_score: Optional[float] = None


class Result(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    student_id: int
    assignment_id: int
    total_score: Optional[float] = None
    feedback_summary: Optional[str] = None


class MistakeStat(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    class_id: int = Field(foreign_key="class.id")
    question_id: int = Field(foreign_key="question.id")
    mistake_type: str  # например: "grammar", "spelling", "concept"
    count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ActionLog(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    user_id: int
    action_type: str
    target_type: str
    target_id: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: str | None = None


class RefreshToken(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    token: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime


class RevokedToken(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    jti: str = Field(index=True, unique=True)
    user_id: int = Field(index=True)
    revoked_at: datetime = Field(default_factory=datetime.utcnow)
    expired: bool = Field(default=True)
