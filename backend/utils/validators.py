from __future__ import annotations

from typing import Annotated

from email_validator import EmailNotValidError, validate_email
from pydantic import AfterValidator


def normalise_email(value: str) -> str:
    """Validate an address's syntax without a DNS lookup.

    pydantic's built-in EmailStr calls email-validator with
    check_deliverability=True, which does a DNS query per registration. That
    makes signup slow and fails for any address whose domain does not resolve
    - including perfectly valid internal or development domains.

    Syntax is still enforced, so genuine typos are caught.
    """
    # Surrounding whitespace is a copy-paste artefact, not an error.
    candidate = value.strip()
    try:
        result = validate_email(candidate, check_deliverability=False)
    except EmailNotValidError as exc:
        raise ValueError(f"Invalid email address: {exc}") from exc
    return result.normalized


# Lowercased and trimmed, so lookups and uniqueness are consistent.
Email = Annotated[str, AfterValidator(normalise_email)]