"""Doctor-chosen support modes - how MEDSCAN helps *this* doctor, on *this* case.

The problem statement asks the system to "adjust how it helps different doctors
... without making assumptions about a doctor's ability". So the mode is
**chosen by the doctor**, never inferred from seniority, title or past
decisions. It is saved on the doctor's account as a default and can be switched
on any case, at any time, in one click. The clinical context (Emergency,
Routine OPD, ...) still sets the router's thresholds; the mode sets how the
result is *presented*.

==================  =========================================================
mode                what changes
==================  =========================================================
Guided              detailed explanation, a "how to read this" checklist per
                    finding, why it matters, recommendations, evidence
Concise             a short summary line per finding with its measurements;
                    evidence and checklists folded away
Second opinion      the AI stays hidden until the doctor commits a read; then
                    only the findings where AI and doctor *disagree* are
                    expanded - the AI as a check, not a guide
Evidence-first      guideline passages and citations lead, findings follow
==================  =========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

__all__ = ["SupportMode", "MODES", "DEFAULT_MODE", "get"]


@dataclass(frozen=True)
class SupportMode:
    key: str
    name: str
    depth: str                 # explanation depth: brief / standard / detailed
    checklist: bool            # "how to read this" steps per finding
    significance: bool         # "why it matters" block expanded
    evidence_first: bool       # guideline evidence before findings
    blind_first: bool          # AI hidden until the doctor records a read
    disagreements_only: bool   # after the read, expand only where AI and doctor differ
    summary: str


MODES: Dict[str, SupportMode] = {m.key: m for m in (
    SupportMode("guided", "Guided", "detailed", True, True, False, False, False,
                "Full explanation, reading checklist, why it matters and next steps."),
    SupportMode("concise", "Concise", "brief", False, False, False, False, False,
                "One line per finding with its measurement and confidence."),
    SupportMode("second_opinion", "Second opinion", "standard", False, True, False, True, True,
                "You read first; the AI then shows only where it disagrees with you."),
    SupportMode("evidence_first", "Evidence-first", "standard", False, True, True, False, False,
                "Guideline passages and citations first, findings after."),
)}
DEFAULT_MODE = "concise"


def get(key: str) -> SupportMode:
    return MODES.get(key or DEFAULT_MODE, MODES[DEFAULT_MODE])
