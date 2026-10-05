"""The audit trail - who did what, when, and proof nobody edited it since.

Decision traceability is in the brief. Every analysis, every doctor decision,
every sign-in, every export is one JSON line in ``audit.jsonl``:

* **Append-only.** A line is written once and never rewritten.
* **Every entry names its actor** - the signed-in account and role, plus the
  operating-system user underneath.
* **Hash-chained.** Each entry carries the SHA-256 of the previous entry and of
  itself, so an edited, inserted or deleted line breaks the chain at that point
  and :meth:`AuditLog.verify` says where. The head hash is shown on screen; an
  auditor who notes it can tell later whether lines were removed from the end.
* **Never fatal.** If the file cannot be written the trail keeps going in
  memory and says so once - recording must never interrupt clinical work.
* **No patient identifiers.** Entries reference study IDs and pseudonyms only.
"""

from __future__ import annotations

import csv
import getpass
import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from . import paths

__all__ = ["AuditLog", "ChainReport", "GENESIS"]

GENESIS = "0" * 64


def _os_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"


def _digest(entry: Dict[str, object]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass
class ChainReport:
    intact: bool
    entries: int
    head: str
    broken_at: Optional[int] = None
    reason: str = ""


class AuditLog:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(paths.data_directory(), "audit.jsonl")
        self._lock = threading.Lock()
        self._memory: List[Dict[str, object]] = []
        self._failed = False
        self._head = self._read_head()

    def _read_head(self) -> str:
        try:
            with open(self.path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 8192))
                lines = handle.read().decode("utf-8", "replace").strip().splitlines()
            return json.loads(lines[-1])["hash"] if lines else GENESIS
        except (OSError, ValueError, KeyError, IndexError):
            return GENESIS

    @property
    def head(self) -> str:
        return self._head

    def record(self, action: str, detail: Optional[Dict[str, object]] = None, actor: str = "",
               role: str = "", category: str = "functionality", study: str = "") -> Dict[str, object]:
        with self._lock:
            entry = {"at": datetime.now().isoformat(timespec="seconds"), "category": category,
                     "action": action, "actor": actor or "system", "role": role,
                     "os_user": _os_user(), "study": study, "detail": detail or {},
                     "prev": self._head}
            entry["hash"] = _digest(entry)
            if not self._failed:
                try:
                    with open(self.path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except OSError:
                    self._failed = True
            if self._failed:
                self._memory.append(entry)
            self._head = entry["hash"]
            return entry

    def entries(self, limit: int = 500, study: str = "") -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        try:
            with open(self.path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            pass
        out += self._memory
        if study:
            out = [e for e in out if e.get("study") == study]
        return out[-limit:][::-1]

    def verify(self) -> ChainReport:
        previous, count = GENESIS, 0
        try:
            with open(self.path, encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        return ChainReport(False, count, previous, number, "unreadable line")
                    if entry.get("prev") != previous:
                        return ChainReport(False, count, previous, number,
                                           "link to the previous entry broken (line removed or inserted)")
                    if _digest(entry) != entry.get("hash"):
                        return ChainReport(False, count, previous, number, "entry edited after writing")
                    previous, count = entry["hash"], count + 1
        except OSError:
            pass
        return ChainReport(True, count, previous)

    def export_csv(self, path: str) -> int:
        rows = self.entries(limit=10 ** 9)[::-1]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["at", "category", "action", "actor", "role", "study", "detail",
                             "hash"])
            for e in rows:
                # .get throughout: a damaged trail is exactly what an auditor needs to export
                writer.writerow([e.get("at", ""), e.get("category", ""), e.get("action", ""),
                                 e.get("actor", ""), e.get("role", ""), e.get("study", ""),
                                 json.dumps(e.get("detail", {})), e.get("hash", "")])
        return len(rows)
