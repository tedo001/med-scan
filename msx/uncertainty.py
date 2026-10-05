"""Stage 6 - how far to trust each finding, and when to say "I don't know".

A probability says how likely a finding is. A *confidence* says how much the
engine trusts that probability. They differ: 0.55 from a stable measurement on
a crisp film is a confident "borderline"; 0.85 that drops to 0.4 when the image
is shifted by eight pixels is not confident at all.

Confidence is built from four things, each reported to the doctor:

=================  ===========================================================
decisiveness       distance of the probability from 0.5 - ``|2p - 1|``
stability          1 - (std across test-time augmentations / 0.2); a finding
                   that flips under small perturbations is unstable
agreement          when both engines spoke: 1 - 0.5 x their disagreement
quality            1 - 0.08 per quality warning (floored at 0.6)
=================  ===========================================================

``confidence = (0.5 x decisiveness + 0.5 x stability) x agreement x quality``

**Abstention.** If confidence is below :data:`ABSTAIN_BELOW` while the
probability sits in the grey zone (0.3-0.7), the finding is marked
``uncertain``: the engine declines to call it and asks for a human read. This
is the "provide confidence and limitations for every finding" and "request
human review instead of an unreliable result" of the brief, applied per
finding rather than per scan.

**Calibration.** :func:`calibrate` applies per-group temperature scaling
(``p' = sigmoid(logit(p) / T)``); :func:`fit_temperature` finds T on labelled
data by minimising log loss - run it from the Evaluation page.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

from .findings import NEGATIVE, POSITIVE, POSSIBLE, UNCERTAIN, Finding

__all__ = ["score", "calibrate", "fit_temperature", "level", "ABSTAIN_BELOW"]

ABSTAIN_BELOW = 0.40
POSITIVE_AT = 0.50
POSSIBLE_AT = 0.35


def level(confidence: float) -> str:
    return "High" if confidence >= 0.75 else "Moderate" if confidence >= 0.5 else "Low"


def calibrate(probability: float, temperature: float = 1.0) -> float:
    if abs(temperature - 1.0) < 1e-6:
        return probability
    p = min(max(probability, 1e-6), 1 - 1e-6)
    return 1 / (1 + math.exp(-math.log(p / (1 - p)) / temperature))


def fit_temperature(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    """The temperature in [0.25, 4] minimising log loss on labelled examples."""
    best, best_loss = 1.0, float("inf")
    for step in range(0, 161):
        t = 0.25 * (16 ** (step / 160))          # log-spaced 0.25 .. 4
        loss = 0.0
        for p, y in zip(probabilities, labels):
            q = min(max(calibrate(p, t), 1e-6), 1 - 1e-6)
            loss -= y * math.log(q) + (1 - y) * math.log(1 - q)
        if loss < best_loss:
            best, best_loss = t, loss
    return round(best, 3)


def score(finding: Finding, quality_warnings: int = 0,
          temperatures: Dict[str, float] | None = None,
          positive_at: float = POSITIVE_AT) -> Finding:
    """Fill ``confidence``, ``confidence_level`` and ``status`` on ``finding``."""
    temperature = (temperatures or {}).get(finding.group, 1.0)
    p = calibrate(finding.probability, temperature)
    finding.probability = round(p, 4)
    std = float(finding.uncertainty.get("std", 0.0))
    decisiveness = abs(2 * p - 1)
    stability = max(0.0, 1 - std / 0.2)
    agreement = 1.0
    sources = finding.sources
    if "deep" in sources and "measurement" in sources and abs(sources["deep"] - 0.5) > 0.1:
        agreement = 1 - 0.5 * abs(sources["deep"] - sources["measurement"])
    single_source = set(sources) - {"learned", "dataset model"} == {"deep"}
    quality = max(0.6, 1 - 0.08 * quality_warnings)
    confidence = (0.5 * decisiveness + 0.5 * stability) * agreement * quality
    if single_source:
        confidence *= 0.85
    finding.confidence = round(confidence, 3)
    finding.confidence_level = level(confidence)
    finding.uncertainty.update({"decisiveness": round(decisiveness, 3),
                                "stability": round(stability, 3),
                                "agreement": round(agreement, 3), "quality": round(quality, 3)})
    if single_source and p < positive_at:
        finding.status = NEGATIVE          # one uncorroborated source never raises a doubt alone
    elif 0.3 <= p <= 0.7 and confidence < ABSTAIN_BELOW:
        finding.status = UNCERTAIN
        finding.limitations.insert(0, "Engine abstains: evidence unstable or contradictory - "
                                      "human read required")
    elif p >= positive_at:
        finding.status = POSITIVE
    elif p >= POSSIBLE_AT:
        finding.status = POSSIBLE
    else:
        finding.status = NEGATIVE
    return finding


def summarise(findings: Iterable[Finding]) -> Dict[str, int]:
    counts = {POSITIVE: 0, POSSIBLE: 0, UNCERTAIN: 0, NEGATIVE: 0}
    for finding in findings:
        counts[finding.status] = counts.get(finding.status, 0) + 1
    return counts
