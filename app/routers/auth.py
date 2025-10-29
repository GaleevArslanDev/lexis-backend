from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from ..schemas import UserCreate, Token, UserRead, LoginRequest
from ..db import get_session
from .. import auth
from ..crud.user import *

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=UserRead)
def signup(data: UserCreate, session: Session = Depends(get_session)):
    existing = get_user_by_email(session, data.email)
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    hashed = auth.hash_password(data.password)
    user = create_user(session, email=data.email, hashed_password=hashed, role=data.role)
    return user


@router.post("/login", response_model=Token)
def login(data: LoginRequest, session: Session = Depends(get_session)):
    user = get_user_by_email(session, data.email)
    if not user or not auth.verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    # Роль берём из базы, не из запроса
    token = auth.create_access_token({"sub": str(user.id), "role": user.role})
    return {"access_token": token, "token_type": "bearer"}
