from __future__ import annotations

import datetime as dt
import secrets
from typing import Any, Dict, Optional

import bcrypt
from jose import JWTError, jwt

from backend.utils.config import settings
from backend.utils.logging import logger


class PasswordPolicyError(ValueError):
    """Password cannot be accepted. Message is safe to show the user."""


def validate_password_strength(password: str) -> None:
    """Reject passwords bcrypt cannot represent faithfully.

    bcrypt hashes at most 72 bytes and silently discards the rest, so two
    different long passphrases would authenticate each other. Reject instead.
    """
    if not isinstance(password, str) or not password:
        raise PasswordPolicyError("Password is required.")
    if len(password) < settings.MIN_PASSWORD_LENGTH:
        raise PasswordPolicyError(
            f"Password must be at least {settings.MIN_PASSWORD_LENGTH} characters."
        )
    if len(password.encode("utf-8")) > settings.MAX_PASSWORD_BYTES:
        raise PasswordPolicyError(
            f"Password must be at most {settings.MAX_PASSWORD_BYTES} bytes."
        )
    if password.strip() == "":
        raise PasswordPolicyError("Password cannot be only whitespace.")


def hash_password(password: str) -> str:
    validate_password_strength(password)
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time check. Never raises on malformed input - a corrupt hash
    means the login fails, it does not mean the request errors out."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError) as exc:
        logger.warning("Password verification rejected a malformed hash.")
        return False


def create_access_token(subject: str, role: str, expires_delta: Optional[dt.timedelta] = None) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    expire = now + (
        expires_delta
        if expires_delta is not None
        else dt.timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    claims: Dict[str, Any] = {
        "sub": subject,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        # Distinguishes this token from any other JWT in the same process.
        "jti": secrets.token_urlsafe(8),
    }
    return jwt.encode(claims, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Return the claims, or None if the token is invalid/expired/forged."""
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError as exc:
        logger.debug(f"JWT rejected: {exc.__class__.__name__}")
        return None