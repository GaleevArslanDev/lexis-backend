from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select
from datetime import timedelta, datetime
from ..db import get_session
from ..schemas import UserCreate, UserRead, Token, RefreshRequest
from ..crud.user import get_user_by_email, create_user, get_user_by_id
from ..models import RefreshToken, RevokedToken, User
from ..utils.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    is_token_revoked,
    revoke_token,
    log_login_attempt,
    log_refresh_attempt
)
from ..dependencies.auth import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=UserRead)
def signup(data: UserCreate, session: Session = Depends(get_session)):
    existing = get_user_by_email(session, data.email)
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    hashed = hash_password(data.password)
    user = create_user(session, email=data.email, name=data.name, surname=data.surname, hashed_password=hashed, role=data.role)
    return user


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_session)):
    email = form_data.username
    password = form_data.password

    user = get_user_by_email(session, email)
    if not user or not verify_password(password, user.hashed_password):
        log_login_attempt(email, False)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    # Генерируем access и refresh токены
    access_token, access_jti = create_access_token({"sub": str(user.id), "role": user.role})
    refresh_token, refresh_jti = create_refresh_token({"sub": str(user.id), "role": user.role})

    # Сохраняем refresh token в БД (таблица RefreshToken)
    expires_at = datetime.utcnow() + timedelta(days=int(refresh_token_exp_days := int(__import__("os").getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))))
    db_refresh = RefreshToken(user_id=user.id, token=refresh_token, jti=refresh_jti, expires_at=expires_at)
    session.add(db_refresh)
    session.commit()

    log_login_attempt(email, True)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "jti": access_jti,
        "refresh_token": refresh_token
    }


@router.post("/refresh", response_model=dict)
def refresh_token_endpoint(payload: RefreshRequest = Body(...), session: Session = Depends(get_session)):
    decoded = decode_token(payload.refresh_token)
    if not decoded or decoded.get("type") != "refresh":
        log_refresh_attempt(None, False)
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    jti = decoded.get("jti")
    user_id = decoded.get("sub")

    # Проверяем, не отозван ли jti
    if is_token_revoked(session, jti):
        log_refresh_attempt(user_id, False)
        raise HTTPException(status_code=401, detail="Token has been revoked")

    # Проверяем, есть ли такой refresh в БД и не просрочен ли
    stmt = select(RefreshToken).where(RefreshToken.jti == jti, RefreshToken.user_id == int(user_id))
    db_token = session.exec(stmt).first()
    if not db_token:
        log_refresh_attempt(user_id, False)
        raise HTTPException(status_code=401, detail="Refresh token not found")

    if db_token.expires_at < datetime.utcnow():
        # помечаем как отозванный и удаляем запись
        revoke_token(session, jti=jti, user_id=int(user_id), token_type="refresh")
        log_refresh_attempt(user_id, False)
        raise HTTPException(status_code=401, detail="Refresh token expired")

    # успешное использование refresh -> выдаём новый access
    access_token, access_jti = create_access_token({"sub": str(user_id)})
    log_refresh_attempt(user_id, True)
    return {"access_token": access_token, "token_type": "bearer", "jti": access_jti}


@router.post("/logout", response_model=dict)
def logout(payload: RefreshRequest = Body(...), current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    """
    Logout: ожидаем refresh token от клиента. Отзываем refresh-token (и, при желании, access-token указанный в payload).
    """
    decoded = decode_token(payload.refresh_token)
    if not decoded or decoded.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if decoded.get("sub") != str(current_user.id):
        raise HTTPException(status_code=401, detail="Refresh token does not belong to current user")

    jti = decoded.get("jti")
    # Отмечаем revocation
    revoke_token(session, jti=jti, user_id=current_user.id, token_type="refresh")

    # Также удалим refresh запись из таблицы RefreshToken (чтобы не занимал место)
    stmt = select(RefreshToken).where(RefreshToken.jti == jti, RefreshToken.user_id == current_user.id)
    db_token = session.exec(stmt).first()
    if db_token:
        session.delete(db_token)
        session.commit()

    return {"message": "Logged out successfully"}
