from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from app.db import get_session
from app.models import User, RefreshToken
from app.utils.security import (
    verify_password, hash_password,
    create_access_token, create_refresh_token,
    log_login_attempt, log_refresh_attempt
)
from jose import jwt
import os

router = APIRouter(prefix="/auth", tags=["auth"])

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

@router.post("/login")
def login(email: str, password: str, db: Session = Depends(get_session)):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        log_login_attempt(email, False)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token({"sub": str(user.id)})
    refresh_token = create_refresh_token({"sub": str(user.id)})

    # Сохраняем refresh в БД
    db_token = RefreshToken(
        user_id=user.id,
        token=refresh_token,
        expires_at=datetime.utcnow() + timedelta(days=7)
    )
    db.add(db_token)
    db.commit()

    log_login_attempt(email, True)
    return {"access_token": access_token, "refresh_token": refresh_token}

@router.post("/refresh")
def refresh_token(token: str, db: Session = Depends(get_session)):
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    user_id = int(payload.get("sub"))

    db_token = db.query(RefreshToken).filter(RefreshToken.token == token).first()
    if not db_token or db_token.expires_at < datetime.utcnow():
        log_refresh_attempt(user_id, False)
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    new_access = create_access_token({"sub": str(user_id)})
    log_refresh_attempt(user_id, True)
    return {"access_token": new_access}
