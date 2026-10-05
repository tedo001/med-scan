"""Shared helpers for the GUI tests (imported by conftest and the test modules)."""

from __future__ import annotations

import csv
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLES = os.path.join(ROOT, "samples")


def pump(app, ms: int = 30) -> None:
    """Let Qt process events for roughly ``ms`` milliseconds."""
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


def wait_until(app, condition, timeout: float = 120.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return False


def labels():
    with open(os.path.join(SAMPLES, "labels.csv"), newline="", encoding="utf-8") as handle:
        return {r["file"]: r for r in csv.DictReader(handle)}
