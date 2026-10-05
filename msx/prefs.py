"""Settings, kept as JSON beside the data. Unknown keys are ignored, missing ones defaulted."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from . import paths

DEFAULTS: Dict[str, Any] = {
    "engine": "hybrid",               # hybrid (DenseNet + measurements) or builtin
    "tta": 4,                         # test-time augmentations per routed module
    "default_context": "Routine OPD",
    "site": "Chennai GH",
    "positive_at": 0.5,
    "temperatures": {},               # group -> temperature (from Evaluation > Calibrate)
    "route_overrides": {},            # context -> {"route": x, "exit": y}
    "subgroup_thresholds": {},        # "sex:F" -> threshold, from Evaluation > Reduce bias
    "blinded_first_read": True,       # ask the doctor's impression before showing the AI
    "llm_enabled": False,
    "llm_url": "http://localhost:11434",
    "llm_model": "gemma2:latest",
    "show_demo_credentials": True,
}


def _path() -> str:
    return os.path.join(paths.data_directory(), "settings.json")


def load() -> Dict[str, Any]:
    data = dict(DEFAULTS)
    try:
        with open(_path(), encoding="utf-8") as handle:
            stored = json.load(handle)
        data.update({k: v for k, v in stored.items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return data


def save(settings: Dict[str, Any]) -> None:
    with open(_path(), "w", encoding="utf-8") as handle:
        json.dump({k: settings.get(k, v) for k, v in DEFAULTS.items()}, handle, indent=1)
