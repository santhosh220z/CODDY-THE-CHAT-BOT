from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.auth.security import (
    PasswordPolicyError,
    create_access_token,
    decode_access_token,
    hash_password,
    validate_password_strength,
    verify_password,
)
from backend.database.connection import get_db
from backend.models.models import User
from backend.utils.logging import logger
from backend.utils.rate_limit import client_key, login_limiter, register_limiter
from backend.utils.validators import Email

router = APIRouter(prefix="/api/auth", tags=["auth"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

CREDENTIALS_INVALID = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials.",
    headers={"WWW-Authenticate": "Bearer"},
)


class UserRegister(BaseModel):
    """Registration input.

    `role` is intentionally absent. The previous version accepted it from the
    request body and wrote it straight onto the User row, so anyone could POST
    {"role": "admin"} and become an administrator. Privilege now comes only
    from the first-account bootstrap inside `register()`.
    """

    # Validates syntax without a DNS lookup - see utils/validators.py.
    email: Email
    password: str = Field(min_length=1, max_length=256)


class UserLogin(BaseModel):
    email: Email
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    email: str


class UserProfile(BaseModel):
    id: int
    email: str
    role: str
    created_at: str


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    if not token:
        raise CREDENTIALS_INVALID

    claims = decode_access_token(token)
    if not claims:
        raise CREDENTIALS_INVALID

    email = claims.get("sub")
    if not email:
        raise CREDENTIALS_INVALID

    # The token carries a role, but the database is the source of truth: a
    # role changed after issue takes effect immediately.
    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user is None or not user.is_active:
        raise CREDENTIALS_INVALID

    return user


def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator privileges required.",
        )
    return current_user


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(
    payload: UserRegister,
    request: Request,
    db: Session = Depends(get_db),
) -> dict:
    allowed, retry_after = register_limiter.check(client_key(request), "register")
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many registrations. Try again in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    email = payload.email.lower().strip()
    if db.execute(select(User).where(func.lower(User.email) == email)).scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists.",
        )

    try:
        validate_password_strength(payload.password)
    except PasswordPolicyError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # Bootstrap: the very first account gets admin so there is a way in.
    # Every later registration is a plain user. Promotion happens out of band.
    is_first_user = (db.scalar(select(func.count()).select_from(User)) or 0) == 0
    role = "admin" if is_first_user else "user"

    user = User(email=email, hashed_password=hash_password(payload.password), role=role)
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info(f"Registered {user.email} as {user.role}")
    return {"email": user.email, "role": user.role}


@router.post("/login", response_model=Token)
def login(payload: UserLogin, request: Request, db: Session = Depends(get_db)) -> Token:
    email = payload.email.lower().strip()
    allowed, retry_after = login_limiter.check(client_key(request), email)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed sign-in attempts. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )

    user = db.execute(select(User).where(func.lower(User.email) == email)).scalar_one_or_none()
    # Hash regardless of whether the user exists so the response time does not
    # reveal which emails are registered.
    stored_hash = user.hashed_password if user else "$2b$12$" + "." * 53

    if not verify_password(payload.password, stored_hash) or user is None:
        # The failed attempt stays recorded in the limiter.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="This account is disabled."
        )

    login_limiter.reset(client_key(request), email)
    token = create_access_token(subject=user.email, role=user.role)
    logger.info(f"Login OK for {user.email}")
    return Token(access_token=token, token_type="bearer", role=user.role, email=user.email)


@router.get("/profile", response_model=UserProfile)
def profile(current_user: User = Depends(get_current_user)) -> UserProfile:
    return UserProfile(
        id=current_user.id,
        email=current_user.email,
        role=current_user.role,
        created_at=current_user.created_at.isoformat(),
    )