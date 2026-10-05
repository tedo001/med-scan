"""Where MEDSCAN keeps its data.

Everything a clinician produces - the study database, de-identified images, the
audit trail, settings - lives in one folder that follows the operator's profile:
``%APPDATA%\\MEDSCAN`` on Windows, ``~/.medscan`` elsewhere. ``MEDSCAN_HOME``
overrides it (tests and the smoke check point it at a temporary folder).
"""

from __future__ import annotations

import os

APP_NAME = "MEDSCAN"
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)
KNOWLEDGE_DIR = os.path.join(PACKAGE_DIR, "knowledge")
SAMPLES_DIR = os.path.join(PROJECT_DIR, "samples")


def data_directory() -> str:
    """The operator's MEDSCAN folder, created on first use."""
    folder = os.environ.get("MEDSCAN_HOME")
    if not folder:
        base = os.environ.get("APPDATA") if os.name == "nt" else None
        folder = os.path.join(base, APP_NAME) if base else os.path.join(
            os.path.expanduser("~"), ".medscan")
    os.makedirs(folder, exist_ok=True)
    return folder


def images_directory() -> str:
    """De-identified copies of every analysed scan."""
    folder = os.path.join(data_directory(), "images")
    os.makedirs(folder, exist_ok=True)
    return folder
