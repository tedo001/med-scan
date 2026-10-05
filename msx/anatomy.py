"""Lung fields and the measurements built on them.

Every later stage needs to know *where the lungs are*: the quality gate asks
whether both are in the picture, the cardiac module measures the heart between
them, the pleural module looks at their bases, the parenchyma module divides
each into upper, middle and lower zones. This module finds them once per scan.

**How.** Classical, deterministic segmentation on the 512 x 512 working image:

1. Smooth (Gaussian, sigma 2) and threshold with Otsu's method. Lungs are air,
   so they fall on the dark side together with the air outside the body.
2. Drop dark regions that touch the image border - that is the outside air.
3. The largest remaining dark region on each half is a lung. Image-left is the
   *patient's right* (radiographic convention); names here are anatomical.
4. The aerated mask is what is dark. The **envelope** is the aerated mask with
   holes filled and gaps up to ~25 px closed, i.e. where lung *should* be. Envelope
   minus aerated is lung that has gone dense: consolidation, mass, fluid.

It is fast (~20 ms), needs no model and no network, and is honest about failure:
if two lungs cannot be found the quality gate stops the analysis rather than
guessing. When the optional deep stack is installed a learned segmenter can
replace step 1-3 without changing anything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["Lung", "Anatomy", "segment", "ZONES"]

ZONES = ("upper", "middle", "lower")


def _otsu(values: np.ndarray) -> float:
    histogram, edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    histogram = histogram.astype(np.float64)
    centres = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(histogram)
    mean = np.cumsum(histogram * centres)
    total, grand = weight[-1], mean[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (grand * weight - mean * total) ** 2 / (weight * (total - weight))
    between[~np.isfinite(between)] = 0
    return float(centres[int(np.argmax(between))])


def _disc(radius: int) -> np.ndarray:
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


_DISC = _disc(12)


def _convex_fill(mask: np.ndarray) -> np.ndarray:
    import cv2

    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        return mask.copy()
    hull = cv2.convexHull(points)
    filled = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(filled, hull, 1)
    return filled.astype(bool)


@dataclass
class Lung:
    """One lung field and its geometry, in working-image pixels."""

    side: str                          # "right" (image-left) or "left" (image-right)
    aerated: np.ndarray                # bool mask of air-filled lung
    envelope: np.ndarray               # bool mask of where lung should be
    filled: Optional[np.ndarray] = None  # aerated with holes filled: true margins
    top: int = 0
    bottom: int = 0
    lateral: int = 0                   # outermost x
    medial: int = 0                    # innermost x
    area: int = 0

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)

    @property
    def width(self) -> int:
        return max(1, abs(self.lateral - self.medial))

    def zone_masks(self) -> Dict[str, np.ndarray]:
        """Upper, middle, lower thirds of the envelope, apex to base."""
        rows = np.arange(self.envelope.shape[0])[:, None]
        cuts = [self.top + self.height * k / 3 for k in range(4)]
        return {name: self.envelope & (rows >= cuts[i]) & (rows < cuts[i + 1])
                for i, name in enumerate(ZONES)}

    def bbox(self) -> Tuple[int, int, int, int]:
        x0, x1 = sorted((self.lateral, self.medial))
        return x0, self.top, x1, self.bottom


@dataclass
class Anatomy:
    """Both lungs, the body, and the derived measurements."""

    image: np.ndarray                  # the smoothed working image
    body: np.ndarray                   # bool mask of the patient
    lungs: Dict[str, Lung] = field(default_factory=dict)
    threshold: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return set(self.lungs) == {"right", "left"}

    @property
    def lung_mask(self) -> np.ndarray:
        mask = np.zeros(self.image.shape, dtype=bool)
        for lung in self.lungs.values():
            mask |= lung.envelope
        return mask

    @property
    def midline(self) -> float:
        if not self.found:
            return self.image.shape[1] / 2
        return (self.lungs["right"].medial + self.lungs["left"].medial) / 2

    # -- measurements -------------------------------------------------------
    def thoracic_width(self) -> Tuple[int, int, int]:
        """(width, x_right_outer, x_left_outer) across the outer lung margins."""
        right, left = self.lungs["right"], self.lungs["left"]
        return left.lateral - right.lateral, right.lateral, left.lateral

    def cardiac_width(self) -> Tuple[int, int, int, int]:
        """(width, x_right_border, x_left_border, row) - the widest heart shadow.

        Measured as the largest horizontal gap between the two lungs' medial
        margins over the lower half of the lung fields (where the heart sits),
        excluding the bottom 12 % where the diaphragm domes curve the margins.
        """
        right, left = self.lungs["right"], self.lungs["left"]
        top = int(max(right.top, left.top) + 0.48 * min(right.height, left.height))
        bottom = int(min(right.bottom, left.bottom) - 0.12 * min(right.height, left.height))
        best = (0, right.medial, left.medial, top)
        midline = self.midline
        for row in range(top, max(top + 1, bottom)):
            r = np.flatnonzero(right.filled[row])
            l = np.flatnonzero(left.filled[row])
            if not len(r) or not len(l):
                continue
            right_border, left_border = int(r.max()), int(l.min())
            if right_border >= midline or left_border <= midline:
                continue
            width = left_border - right_border
            if width > best[0]:
                best = (width, right_border, left_border, row)
        return best

    def cardiothoracic_ratio(self) -> Optional[float]:
        if not self.found:
            return None
        thoracic = self.thoracic_width()[0]
        cardiac = self.cardiac_width()[0]
        return cardiac / thoracic if thoracic > 0 and cardiac > 0 else None

    def costophrenic_blunting(self, side: str) -> float:
        """How far the lateral base sits *above* the medial base, as a fraction of height.

        A healthy costophrenic recess is the lowest point of the lung (lateral
        base at or below medial base -> value <= 0). Fluid fills the recess
        first and climbs the lateral wall (the meniscus), so the lateral base
        rises: values above ~0.05 mean a blunted angle.
        """
        lung = self.lungs[side]
        columns = np.flatnonzero(lung.aerated.any(axis=0))
        if len(columns) < 6:
            return 0.0
        third = max(2, len(columns) // 3)
        lateral_cols = columns[:third] if side == "right" else columns[-third:]
        medial_cols = columns[-third:] if side == "right" else columns[:third]

        def lowest(cols):
            rows = [np.flatnonzero(lung.aerated[:, c]).max() for c in cols
                    if lung.aerated[:, c].any()]
            return float(np.percentile(rows, 75)) if rows else float(lung.bottom)

        return (lowest(medial_cols) - lowest(lateral_cols)) / lung.height

    def height_asymmetry(self, side: str) -> float:
        """How much shorter ``side`` is than the other lung (0 = equal, 0.25 = a quarter)."""
        own = self.lungs[side].height
        other = self.lungs["left" if side == "right" else "right"].height
        return max(0.0, 1 - own / other) if other else 0.0

    def dense_fraction(self, mask: np.ndarray, side: str) -> float:
        """Share of ``mask`` (within the envelope) that is not aerated."""
        lung = self.lungs[side]
        region = mask & lung.envelope
        total = int(region.sum())
        return 1 - float((region & lung.aerated).sum()) / total if total else 0.0


def _candidates(smooth, content, threshold):
    """Dark regions inside the body at ``threshold``: (area, centroid x, mask) each."""
    from scipy import ndimage

    height, width = smooth.shape
    dark = ndimage.binary_opening((smooth < threshold) & content, iterations=2)
    labels, count = ndimage.label(dark)
    if count == 0:
        return []
    perimeter = np.zeros_like(dark)
    perimeter[[0, -1], :] = perimeter[:, [0, -1]] = True
    perimeter |= ndimage.binary_dilation(~content, iterations=3)  # letterbox padding
    contact = ndimage.sum(perimeter, labels, index=np.arange(1, count + 1))
    sizes = ndimage.sum(np.ones_like(dark), labels, index=np.arange(1, count + 1))
    found = []
    for index in range(1, count + 1):
        # outside air touches the edge broadly; a lung in a tight crop only at its apex
        if sizes[index - 1] < 0.01 * height * width or contact[index - 1] > 0.02 * (height + width):
            continue
        mask = labels == index
        found.append((int(sizes[index - 1]), float(np.mean(np.nonzero(mask)[1])), mask))
    centre = width / 2
    split = []
    for area, cx, mask in found:  # one region straddling the midline: split it
        columns = np.flatnonzero(mask.any(axis=0))
        if columns.min() < centre - 0.05 * width and columns.max() > centre + 0.05 * width:
            for half in (np.arange(width) < centre, np.arange(width) >= centre):
                part = mask & half[None, :]
                if part.sum() > 0.01 * height * width:
                    split.append((int(part.sum()), float(np.mean(np.nonzero(part)[1])), part))
        else:
            split.append((area, cx, mask))
    return split


def _plausible(options, height, width):
    """Both halves hold a region big and tall enough to be a lung."""
    centre = width / 2
    for test in (lambda x: x < centre, lambda x: x >= centre):
        side = [c for c in options if test(c[1])]
        if not side:
            return False
        area, _, mask = max(side, key=lambda c: c[0])
        rows = np.flatnonzero(mask.any(axis=1))
        if area < 0.025 * height * width or rows.max() - rows.min() < 0.25 * height:
            return False
    return True


def segment(work: np.ndarray) -> Anatomy:
    """Find the lungs in a 512 x 512 working image (bright = dense).

    Otsu first; if the lungs bleed into the soft tissue at that level (common on
    bright, tightly cropped films) the threshold steps down until both lung
    fields stand apart, or gives up and says so.
    """
    from scipy import ndimage

    smooth = ndimage.gaussian_filter(work.astype(np.float32), 2.0)
    content = smooth > 0.01
    otsu = _otsu(smooth[content]) if content.any() else 0.5
    height, width = work.shape
    threshold, options = otsu, []
    for factor in (1.0, 0.92, 0.85, 0.78, 0.7, 0.62):
        options = _candidates(smooth, content, otsu * factor)
        if _plausible(options, height, width):
            threshold = otsu * factor
            break
    body = ndimage.binary_fill_holes(smooth > otsu * 0.6)
    anatomy = Anatomy(image=smooth, body=body, threshold=threshold)
    if not options:
        anatomy.notes.append("no lung-like dark regions - not a chest radiograph or fully opaque")
        return anatomy

    centre = width / 2
    for side, test in (("right", lambda x: x < centre), ("left", lambda x: x >= centre)):
        side_options = [c for c in options if test(c[1])]
        if not side_options:
            anatomy.notes.append(f"{side} lung field not identified")
            continue
        _, _, aerated = max(side_options, key=lambda c: c[0])
        aerated = ndimage.binary_closing(aerated, iterations=3)
        filled = ndimage.binary_fill_holes(aerated)
        # where lung should be: close gaps up to ~25 px (a dense patch at the edge,
        # a blunted angle) but keep the heart's concavity, which is far wider
        envelope = ndimage.binary_fill_holes(ndimage.binary_closing(
            np.pad(filled, 14), structure=_DISC, iterations=1)[14:-14, 14:-14]) & body
        rows, cols = np.nonzero(envelope)
        lung = Lung(side=side, aerated=aerated & envelope, envelope=envelope, filled=filled,
                    top=int(rows.min()), bottom=int(rows.max()), area=int(aerated.sum()))
        lung.lateral = int(cols.min()) if side == "right" else int(cols.max())
        lung.medial = int(cols.max()) if side == "right" else int(cols.min())
        anatomy.lungs[side] = lung
    return anatomy
