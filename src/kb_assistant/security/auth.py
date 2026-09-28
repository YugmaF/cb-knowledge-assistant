"""Authentication: Option A from the brief, hardcoded users and roles.

Passwords are stored as salted PBKDF2 hashes (never plaintext) and a successful login returns a
short-lived HS256 JWT. Swapping in Keycloak means replacing `authenticate` and `decode_token`
with OIDC token verification; the rest of the app only ever sees a `Principal`.

Demo accounts (documented in README):  vera / viewer-pass,  anil / analyst-pass,  amal / admin-pass
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt

from kb_assistant.config import Settings
from kb_assistant.errors import AuthError
from kb_assistant.security.rbac import Principal, Role


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    name: str
    role: Role
    department: str
    password_hash: str


USERS: dict[str, UserRecord] = {
    "vera": UserRecord(
        "vera", "Vera Perera", Role.VIEWER, "retail-banking",
        "pbkdf2_sha256$200000$48ea43c8b61194481a06d8e079a52d3a$"
        "5db1a1bfe6b7d0cfdeb373b9bbb735fdbb062c62b66c27356435ef65925711ef",
    ),
    "anil": UserRecord(
        "anil", "Anil Jayasuriya", Role.ANALYST, "payments",
        "pbkdf2_sha256$200000$f406f37179a9c1cb936fa4fd5b8f96f7$"
        "dc447b027326fbe5cfce68e739eccc22e1deeec4eafee9ff958ccf1a03481cc9",
    ),
    "amal": UserRecord(
        "amal", "Amal Silva", Role.ADMIN, "platform",
        "pbkdf2_sha256$200000$318bcb303eaebf59f8517fbc2476d07e$"
        "43c38dc63b465e4ad848e846fef1349fa5474c2e9e98a31ca3b625561ca761f9",
    ),
}

_JWT_ALG = "HS256"


def _verify_password(password: str, encoded: str) -> bool:
    _, iterations, salt, expected = encoded.split("$")
    actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations))
    return hmac.compare_digest(actual.hex(), expected)


def authenticate(username: str, password: str) -> Principal:
    user = USERS.get(username.strip().lower())
    # Same error for unknown user and wrong password: do not reveal which usernames exist.
    if user is None or not _verify_password(password, user.password_hash):
        raise AuthError("invalid credentials", public_message="Invalid username or password.")
    return Principal.for_role(user.user_id, user.name, user.role, user.department)


def issue_token(principal: Principal, settings: Settings) -> str:
    now = datetime.now(UTC)
    claims = {
        "sub": principal.user_id,
        "name": principal.name,
        "role": principal.role.value,
        "dept": principal.department,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_ttl_minutes),
    }
    return jwt.encode(claims, settings.jwt_secret, algorithm=_JWT_ALG)


def decode_token(token: str, settings: Settings) -> Principal:
    try:
        claims = jwt.decode(token, settings.jwt_secret, algorithms=[_JWT_ALG])
    except jwt.PyJWTError as exc:
        raise AuthError(str(exc), public_message="Your session is invalid or has expired.") from exc
    # The role comes from our user table, not from the token alone: a revoked or changed role
    # takes effect immediately instead of when the token expires.
    user = USERS.get(claims.get("sub", ""))
    if user is None:
        raise AuthError("unknown subject", public_message="Your session is invalid or has expired.")
    return Principal.for_role(user.user_id, user.name, user.role, user.department)
