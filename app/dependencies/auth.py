from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session
from app.db import get_session
from app.models import User
from app.utils.security import decode_token, is_token_revoked, security_logger
from typing import Optional

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_session)) -> User:
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # token must be access
    if payload.get("type") != "access":
        security_logger.warning("Token with wrong type used for access endpoint")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")

    jti = payload.get("jti")
    if is_token_revoked(db, jti):
        security_logger.warning(f"Attempt with revoked token jti={jti}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked")

    user_id = payload.get("sub")
    if user_id is None:
        security_logger.warning("Access token without subject (sub) field")
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = db.get(User, int(user_id))
    if not user:
        security_logger.warning(f"Access attempt with deleted user_id={user_id}")
        raise HTTPException(status_code=401, detail="User no longer exists")

    return user
