"""Stage 2 - is this scan fit to analyse?

The challenge asks for poor-quality or incomplete scans to be *sent to a human*
rather than analysed into an unreliable answer. This is that gate. Seven checks,
each measured, each with a pass / warn / fail and a sentence saying why:

========================  ==================================================
resolution                shortest side of the original image
sharpness                 strongest edges relative to contrast - blur
                          (motion, defocus, heavy compression) flattens them
contrast                  5th-95th percentile spread inside the body
exposure                  share of the body clipped to white or black
lung fields               were both lungs found at all?
field coverage            do the lungs touch the image edge (apex or
                          costophrenic angles cut off)?
symmetry / rotation       lung area ratio - rotation or a missing lung field
========================  ==================================================

The score is 100 minus a penalty per warn and fail. **Any fail rejects the
scan** - the pipeline stops, the study goes to the review queue marked
"Human review - scan quality" and no findings are produced. Warnings let the
analysis continue but lower every finding's confidence (see
:mod:`msx.uncertainty`) and are listed as limitations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np

from .anatomy import Anatomy

__all__ = ["Check", "QualityReport", "assess", "OK", "WARN", "FAIL"]

OK, WARN, FAIL = "ok", "warn", "fail"
PENALTY = {OK: 0, WARN: 12, FAIL: 40}


@dataclass
class Check:
    name: str
    status: str
    value: float
    detail: str


@dataclass
class QualityReport:
    score: int
    checks: List[Check] = field(default_factory=list)

    @property
    def grade(self) -> str:
        """'reject' if anything failed, 'usable' with warnings, else 'good'."""
        if any(c.status == FAIL for c in self.checks):
            return "reject"
        return "usable" if any(c.status == WARN for c in self.checks) else "good"

    @property
    def accepted(self) -> bool:
        return self.grade != "reject"

    @property
    def problems(self) -> List[Check]:
        return [c for c in self.checks if c.status != OK]

    def to_dict(self) -> Dict[str, object]:
        return {"score": self.score, "grade": self.grade,
                "checks": [asdict(c) for c in self.checks]}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "QualityReport":
        return cls(score=int(data.get("score", 0)),
                   checks=[Check(**c) for c in data.get("checks", [])])


def _grade(value, fail_below, warn_below):
    return FAIL if value < fail_below else WARN if value < warn_below else OK


def assess(work: np.ndarray, anatomy: Anatomy, original_shape) -> QualityReport:
    """Run every check on the 512 working image and its segmentation."""
    from scipy import ndimage

    checks: List[Check] = []
    body = anatomy.body if anatomy.body.any() else work > 0.02

    side = min(original_shape) if original_shape and min(original_shape) else min(work.shape)
    checks.append(Check("Resolution", _grade(side, 256, 512), float(side),
                        f"shortest side {side} px" + (" - too small for fine detail" if side < 512 else "")))

    values = work[body] if body.any() else work.ravel()
    p5, p95 = np.percentile(values, (5, 95))
    spread = float(p95 - p5)
    # edge strength: the strongest gradients (99th percentile) relative to the image's
    # own contrast, after a 1-px smooth so sensor and quantisation noise do not count
    smooth = ndimage.gaussian_filter(work, 1.0)
    gradient = np.hypot(ndimage.sobel(smooth, 0), ndimage.sobel(smooth, 1))
    sharp = float(np.percentile(gradient[body], 99)) / max(spread, 1e-3) if body.any() else 0.0
    checks.append(Check("Sharpness", _grade(sharp, 0.33, 0.45), round(sharp, 2),
                        "edges crisp" if sharp >= 0.45 else
                        "blurred - motion or defocus; fine lines (vessels, pleura) unreliable"))

    checks.append(Check("Contrast", _grade(spread, 0.25, 0.38), round(spread, 3),
                        "lungs and soft tissue well separated" if spread >= 0.38 else
                        "flat image - lungs barely separated from soft tissue"))

    clipped = float(np.mean((values > 0.985) | (values < 0.015)))
    status = FAIL if clipped > 0.30 else WARN if clipped > 0.12 else OK
    checks.append(Check("Exposure", status, round(clipped, 3),
                        f"{clipped:.0%} of the body clipped" + (
                            " - over/under-exposed, detail lost" if status != OK else "")))

    if anatomy.found:
        checks.append(Check("Lung fields", OK, 2.0, "both lung fields identified"))
    else:
        found = len(anatomy.lungs)
        checks.append(Check("Lung fields", FAIL, float(found),
                            "; ".join(anatomy.notes) or "lung fields not identified"
                            " - incomplete, non-frontal or not a chest radiograph"))

    if anatomy.found:
        content = np.argwhere(work > 0.01)
        (y0, x0), (y1, x1) = content.min(axis=0), content.max(axis=0)
        margin = 4
        cut = []
        for lung in anatomy.lungs.values():
            if lung.top <= y0 + margin:
                cut.append(f"{lung.side} apex")
            if lung.bottom >= y1 - margin:
                cut.append(f"{lung.side} base")
            x_min, _, x_max, _ = lung.bbox()
            if x_min <= x0 + margin or x_max >= x1 - margin:
                cut.append(f"{lung.side} lateral wall")
        checks.append(Check("Field coverage", FAIL if len(cut) >= 2 else WARN if cut else OK,
                            float(len(cut)), "whole thorax in view" if not cut else
                            "cut off: " + ", ".join(cut)))

        right, left = anatomy.lungs["right"].area, anatomy.lungs["left"].area
        ratio = min(right, left) / max(right, left)
        checks.append(Check("Symmetry", WARN if ratio < 0.6 else OK, round(ratio, 2),
                            f"lung area ratio {ratio:.2f}" + (
                                " - rotated, or one lung largely opaque (check findings)"
                                if ratio < 0.6 else "")))

    score = max(0, 100 - sum(PENALTY[c.status] for c in checks))
    return QualityReport(score=score, checks=checks)
