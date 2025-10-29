import os
import uuid
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlmodel import Session, select
import logging
from typing import Optional
from ..models import RevokedToken

# ========== CONFIG ==========
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

# ========== LOGGER ==========
security_logger = logging.getLogger("security")
handler = logging.FileHandler("security.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
security_logger.addHandler(handler)
security_logger.setLevel(logging.INFO)

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

# ========== PASSWORDS ==========
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# ========== JWT ==========
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    jti = str(uuid.uuid4())
    to_encode.update({"jti": jti, "type": "access"})
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token

def create_refresh_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    jti = str(uuid.uuid4())
    to_encode.update({"jti": jti, "type": "refresh"})
    if expires_delta is None:
        expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token, jti

def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def revoke_token(session: Session, jti: str, token_type: str):
    revoked = RevokedToken(jti=jti, token_type=token_type)
    session.add(revoked)
    session.commit()

def is_token_revoked(session: Session, jti: str):
    stmt = select(RevokedToken).where(RevokedToken.jti == jti)
    revoked = session.exec(stmt).first()
    return revoked is not None

# ========== LOGGING HELPERS ==========
def log_login_attempt(username: str, success: bool):
    if success:
        security_logger.info(f"Successful login for user: {username}")
    else:
        security_logger.warning(f"Failed login attempt for user: {username}")

def log_refresh_attempt(user_id: int, success: bool):
    if success:
        security_logger.info(f"Refresh token used successfully for user_id={user_id}")
    else:
        security_logger.warning(f"Invalid refresh token attempt for user_id={user_id}")
