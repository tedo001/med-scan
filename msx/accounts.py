"""Sign-in accounts: clinicians and administrators.

Passwords are stored as PBKDF2-HMAC-SHA256 (200 000 iterations, per-account
salt), never in clear. Five failed attempts lock an account for five minutes.
Two demo accounts are created on first run and shown on the sign-in screen so
judges can get in:

=========  ==========  ===============  ====================================
username   password    role             workspace
=========  ==========  ===============  ====================================
doctor     medscan     clinician        Home, Analyse, Review, Dashboard,
                                        Evaluation
admin      techgium    admin            Dashboard, Bias Monitor, Audit Trail,
                                        Engines, Accounts, Settings
=========  ==========  ===============  ====================================
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from . import paths

__all__ = ["Account", "Accounts", "AuthError", "DEMO", "CLINICIAN", "ADMIN"]

CLINICIAN, ADMIN = "clinician", "admin"
ITERATIONS = 200_000
LOCK_AFTER, LOCK_SECONDS = 5, 300
DEMO = (("doctor", "medscan", "Dr. A. Kumar", CLINICIAN, "Radiologist"),
        ("admin", "techgium", "System Administrator", ADMIN, "Administrator"))


class AuthError(Exception):
    pass


@dataclass
class Account:
    username: str
    display: str
    role: str
    title: str
    salt: str
    hash: str
    failed: int = 0
    locked_until: float = 0.0
    last_login: str = ""

    @property
    def initials(self) -> str:
        parts = [p for p in self.display.replace(".", " ").split() if p[:1].isalpha()]
        return "".join(p[0] for p in parts[-2:]).upper() or self.username[:2].upper()


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ITERATIONS).hex()


class Accounts:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(paths.data_directory(), "accounts.json")
        self._accounts: Dict[str, Account] = {}
        self._load()
        if not self._accounts:
            for username, password, display, role, title in DEMO:
                self.add(username, password, display, role, title, save=False)
            self._save()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as handle:
                self._accounts = {a["username"]: Account(**a) for a in json.load(handle)}
        except (OSError, ValueError, TypeError):
            self._accounts = {}

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump([asdict(a) for a in self._accounts.values()], handle, indent=1)

    def all(self) -> List[Account]:
        return sorted(self._accounts.values(), key=lambda a: (a.role, a.username))

    def get(self, username: str) -> Optional[Account]:
        return self._accounts.get(username.strip().lower())

    def add(self, username: str, password: str, display: str, role: str = CLINICIAN,
            title: str = "Clinician", save: bool = True) -> Account:
        username = username.strip().lower()
        if not username or not password:
            raise ValueError("username and password are required")
        if username in self._accounts:
            raise ValueError(f"{username} already exists")
        if len(password) < 6:
            raise ValueError("password must be at least 6 characters")
        salt = secrets.token_hex(16)
        account = Account(username, display or username, role, title, salt, _hash(password, salt))
        self._accounts[username] = account
        if save:
            self._save()
        return account

    def remove(self, username: str) -> None:
        if username in self._accounts:
            del self._accounts[username]
            self._save()

    def set_password(self, username: str, password: str) -> None:
        account = self._accounts[username]
        if len(password) < 6:
            raise ValueError("password must be at least 6 characters")
        account.salt = secrets.token_hex(16)
        account.hash = _hash(password, account.salt)
        self._save()

    def authenticate(self, username: str, password: str, role: Optional[str] = None) -> Account:
        account = self.get(username)
        if account is None:
            raise AuthError("Unknown username or wrong password.")
        now = time.time()
        if account.locked_until > now:
            raise AuthError(f"Account locked for {int(account.locked_until - now) // 60 + 1} more "
                            "minute(s) after repeated failures.")
        if not hmac.compare_digest(_hash(password, account.salt), account.hash):
            account.failed += 1
            if account.failed >= LOCK_AFTER:
                account.locked_until, account.failed = now + LOCK_SECONDS, 0
            self._save()
            raise AuthError("Unknown username or wrong password.")
        if role and account.role != role:
            raise AuthError(f"This is a {account.role} account - use the "
                            f"{'Admin' if account.role == ADMIN else 'Clinician'} Login tab.")
        account.failed, account.locked_until = 0, 0.0
        account.last_login = time.strftime("%Y-%m-%d %H:%M")
        self._save()
        return account
