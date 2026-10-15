from __future__ import annotations

import secrets
import threading
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
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

# Guards the read-then-insert sequence in register(). A per-process lock is
# sufficient here: this app is served by a single worker.
_registration_lock = threading.Lock()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

# The single account the UI runs as. The web client has no sign-in step, so the
# server provisions this row on the first request that needs it. The address is
# never deliverable to: it names the product's own local account, and the
# account has no usable password besides.
LOCAL_EMAIL = "local@coddy.home"

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


@router.get("/local", response_model=Token)
def local_session(db: Session = Depends(get_db)) -> Token:
    """Issue a session for the single local account.

    The web client has no sign-in step, so the app signs itself in on boot.
    This provisions the account on first use rather than assuming it exists,
    which keeps a fresh database working with no setup step.

    The stored password is a random secret that is generated here and thrown
    away, so the account can own documents and pass tenant scoping but cannot
    be signed into through `/api/auth/login`. Anyone who can reach this server
    can obtain a session, which is the intended trade for a local tool: bind
    it to localhost and it is reachable only from this machine.
    """
    with _registration_lock:
        user = db.execute(select(User).where(User.email == LOCAL_EMAIL)).scalar_one_or_none()

        if user is None:
            user = User(
                email=LOCAL_EMAIL,
                hashed_password=hash_password(secrets.token_urlsafe(32)),
                role="admin",
                is_active=True,
            )
            db.add(user)
            try:
                db.commit()
            except IntegrityError:
                # Another request provisioned it between the check and insert.
                db.rollback()
                user = db.execute(
                    select(User).where(User.email == LOCAL_EMAIL)
                ).scalar_one_or_none()
                if user is None:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Could not provision the local account.",
                    )
        db.refresh(user)

    logger.info("Local session issued")
    return Token(
        access_token=create_access_token(subject=user.email, role=user.role),
        token_type="bearer",
        role=user.role,
        email=user.email,
    )


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

    try:
        validate_password_strength(payload.password)
    except PasswordPolicyError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # Both the duplicate check and the first-user bootstrap read state that
    # another request can change between the SELECT and the INSERT. Serialise
    # them, otherwise two simultaneous registrations both observe an empty
    # users table and both become administrators.
    with _registration_lock:
        email = payload.email
        if db.execute(select(User).where(User.email == email)).scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with that email already exists.",
            )

        # Bootstrap: the very first account gets admin so there is a way in.
        # Everyone after that is a plain user. Promotion happens out of band.
        is_first_user = (db.scalar(select(func.count()).select_from(User)) or 0) == 0
        role = "admin" if is_first_user else "user"

        user = User(email=email, hashed_password=hash_password(payload.password), role=role)
        db.add(user)
        try:
            db.commit()
        except IntegrityError:
            # Another writer won the race between the check and the insert.
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with that email already exists.",
            )
        db.refresh(user)

    logger.info(f"Registered {user.email} as {user.role}")
    return {"email": user.email, "role": user.role}


@router.post("/login", response_model=Token)
def login(payload: UserLogin, request: Request, db: Session = Depends(get_db)) -> Token:
    allowed, retry_after = login_limiter.check(client_key(request), payload.email)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed sign-in attempts. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )

    # Addresses are normalised by the Email validator, so this is an exact
    # match rather than a case-folded comparison.
    user = db.execute(select(User).where(User.email == payload.email)).scalar_one_or_none()
    # Verify against a dummy hash when the user is unknown so response time
    # does not reveal which emails are registered.
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

    login_limiter.reset(client_key(request), payload.email)
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