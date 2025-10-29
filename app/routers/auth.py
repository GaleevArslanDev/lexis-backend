from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session
from sqlalchemy.exc import IntegrityError
from ..schemas import UserCreate, Token, UserRead, LoginRequest, RefreshRequest
from ..db import get_session
from .. import auth
from app.models import RevokedToken
from app.utils.security import security_logger, decode_token, create_access_token, is_token_revoked
from ..dependencies.auth import get_current_user
from ..crud.user import *
from ..auth import create_access_token, verify_password
import uuid
from datetime import timedelta

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
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    email = form_data.username
    password = form_data.password

    # Используем один контекст для сессии
    with next(get_session()) as session:
        # Находим пользователя
        statement = select(User).where(User.email == email)
        user = session.exec(statement).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        # Проверяем пароль
        if not verify_password(password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        # Генерируем уникальный jti
        jti = str(uuid.uuid4())

        # Создаем access токен
        access_token = create_access_token(
            data={"sub": str(user.id), "role": user.role, "jti": jti},
            expires_delta=timedelta(minutes=60)
        )

        # Сохраняем jti в таблице RevokedToken (пока не отозван)
        revoked_token = RevokedToken(user_id=user.id, jti=jti, expired=False)
        session.add(revoked_token)

        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            raise HTTPException(status_code=500, detail="Token registration error")

        # Возвращаем результат
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "jti": jti
        }


@router.post("/logout")
def logout(refresh_token: str, current_user = Depends(get_current_user), session: Session = Depends(get_session)):
    payload = decode_token(refresh_token)
    if not payload or payload.get("sub") != str(current_user.id):
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    jti = payload.get("jti")
    if not jti:
        raise HTTPException(status_code=400, detail="Missing token ID (jti)")

    # Проверяем, не был ли токен уже отозван
    statement = select(RevokedToken).where(RevokedToken.jti == jti)
    existing = session.exec(statement).first()
    if existing:
        raise HTTPException(status_code=401, detail="Token already revoked")

    revoked = RevokedToken(jti=jti, user_id=current_user.id)
    session.add(revoked)
    session.commit()

    security_logger.info(f"User {current_user.email} logged out, token revoked jti={jti}")
    return {"message": "Logged out successfully"}


@router.post("/refresh", response_model=dict)
def refresh_token_endpoint(payload: RefreshRequest, session: Session = Depends(get_session)):
    decoded = decode_token(payload.refresh_token)
    if not decoded or decoded.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    jti = decoded.get("jti")
    if is_token_revoked(session, jti):
        raise HTTPException(status_code=401, detail="Token has been revoked")

    user_id = decoded.get("sub")
    access_token, access_jti = create_access_token({"sub": user_id})
    return {"access_token": access_token, "token_type": "bearer"}
