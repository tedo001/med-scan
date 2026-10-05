"""Region measurements the specialist modules are built on.

Each function looks at one kind of abnormality in one place and returns
numbers a radiologist would recognise, not an opaque score:

* :func:`zone_opacity`   - per lung zone, how much of the expected lung is
  dense, and how much denser than the same zone on the other side
* :func:`pneumothorax_rim` - per lung, how much darker its lateral band is
  than the other side's, with lung-marking density as support - the
  signature of air outside the lung
* :func:`find_blobs`     - round dense spots inside the lungs at nodule
  (<= 30 mm) and mass (> 30 mm) scales, by scale-normalised Laplacian of
  Gaussian with a local-contrast test

They run only when the router sends a scan to the module that needs them;
that is where adaptive routing saves time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .anatomy import ZONES, Anatomy

__all__ = ["ZoneOpacity", "Blob", "zone_opacity", "opacity_map", "pneumothorax_rim",
           "find_blobs"]


@dataclass
class ZoneOpacity:
    side: str
    zone: str
    dense_fraction: float      # share of the zone's envelope that is not aerated
    excess_density: float      # mean density above the contralateral zone, in lung SDs
    bbox: Tuple[int, int, int, int]


def opacity_map(anatomy: Anatomy) -> np.ndarray:
    """Smoothed density above each lung's own baseline, 0 outside the lungs.

    Thin structures (ribs, vessels) are averaged away by the 6-px smoothing;
    what remains bright is broad, confluent density - consolidation, fluid,
    mass - which is what the parenchyma module looks for.
    """
    from scipy import ndimage

    broad = ndimage.gaussian_filter(anatomy.image, 6.0)
    result = np.zeros_like(broad)
    for lung in anatomy.lungs.values():
        aerated = broad[lung.aerated]
        if not aerated.size:
            continue
        baseline = float(np.median(aerated))
        spread = float(np.median(np.abs(aerated - baseline))) * 1.4826 + 1e-3
        result[lung.envelope] = np.clip((broad[lung.envelope] - baseline) / spread, 0, None)
    return result


def zone_opacity(anatomy: Anatomy) -> List[ZoneOpacity]:
    """Six zones (upper / middle / lower, each side), each compared with its mirror."""
    zones: Dict[Tuple[str, str], np.ndarray] = {}
    for side, lung in anatomy.lungs.items():
        for name, mask in lung.zone_masks().items():
            zones[(side, name)] = mask
    out = []
    image = anatomy.image
    for (side, name), mask in zones.items():
        if not mask.any():
            continue
        other = zones.get(("left" if side == "right" else "right", name))
        own = float(image[mask].mean())
        if other is not None and other.any():
            pooled = float(np.sqrt((image[mask].var() + image[other].var()) / 2)) + 1e-3
            excess = (own - float(image[other].mean())) / pooled
        else:
            excess = 0.0
        rows, cols = np.nonzero(mask)
        out.append(ZoneOpacity(side, name, round(anatomy.dense_fraction(mask, side), 3),
                               round(excess, 2), (int(cols.min()), int(rows.min()),
                                                  int(cols.max()), int(rows.max()))))
    return out


def _markings(image: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Local density of lung markings inside ``inside``: fine detail across x.

    Fine detail is the image minus its 3-px blur; taking its derivative along x
    means near-horizontal posterior ribs, which stay visible over a
    pneumothorax, count far less than vessels, which do not. The local average
    is normalised by ``inside`` so the lung's own outline never counts as a
    marking.
    """
    from scipy import ndimage

    detail = image - ndimage.gaussian_filter(image, 3.0)
    across = np.abs(ndimage.sobel(ndimage.gaussian_filter(detail, 0.8), axis=1))
    weight = ndimage.uniform_filter(inside.astype(np.float32), 15)
    return ndimage.uniform_filter(across * inside, 15) / np.maximum(weight, 1e-3)


def pneumothorax_rim(anatomy: Anatomy) -> Dict[str, Dict[str, object]]:
    """Per lung: how much more lucent its lateral / apical band is than the other side's.

    Air outside the lung lets more X-rays through than aerated lung does, so a
    pneumothorax makes the periphery of that hemithorax *darker* than the same
    band on the healthy side; lung markings stop at the pleural line. The band
    is the outer 30 % of each lung's width on its lateral half, above the
    costophrenic region. Reported per side:

    * ``lucency``        - (other band density - own) / other; > ~0.12 is suspicious
    * ``marking_ratio``  - own band's markings / other band's (supporting only)
    * ``mask``           - the part of the band darker than the other side's mean
    """
    from scipy import ndimage

    bands, marks = {}, {}
    for side, lung in anatomy.lungs.items():
        inside = ndimage.binary_erosion(lung.aerated, iterations=5)
        depth = ndimage.distance_transform_edt(lung.envelope)
        rows = np.arange(inside.shape[0])[:, None]
        cols = np.arange(inside.shape[1])[None, :]
        centre = (lung.lateral + lung.medial) / 2
        lateral_half = cols < centre if side == "right" else cols > centre
        bands[side] = inside & (depth < 0.3 * lung.width) & lateral_half & (
            rows < lung.top + 0.8 * lung.height)
        marks[side] = _markings(anatomy.image, inside) if inside.any() else np.zeros_like(depth)
    out: Dict[str, Dict[str, object]] = {}
    smooth = anatomy.image
    for side in anatomy.lungs:
        other = "left" if side == "right" else "right"
        own_band, other_band = bands.get(side), bands.get(other)
        if own_band is None or other_band is None or not own_band.any() or not other_band.any():
            out[side] = {"lucency": 0.0, "marking_ratio": 1.0,
                         "mask": np.zeros(smooth.shape, bool), "bbox": anatomy.lungs[side].bbox()}
            continue
        own_density = float(smooth[own_band].mean())
        other_density = float(smooth[other_band].mean())
        lucency = (other_density - own_density) / max(other_density, 1e-3)
        ratio = float(marks[side][own_band].mean()) / max(float(marks[other][other_band].mean()), 1e-6)
        dark = ndimage.binary_opening(own_band & (smooth < other_density - 0.5 * (
            other_density - own_density)), iterations=2) if lucency > 0 else np.zeros_like(own_band)
        if dark.any():
            r, c = np.nonzero(dark)
            box = (int(c.min()), int(r.min()), int(c.max()), int(r.max()))
        else:
            box = anatomy.lungs[side].bbox()
        out[side] = {"lucency": round(lucency, 3), "marking_ratio": round(ratio, 2),
                     "mask": dark, "bbox": box}
    return out


@dataclass
class Blob:
    side: str
    x: int
    y: int
    radius: float              # working-image pixels
    contrast: float            # mean inside minus surrounding ring
    zone: str
    diameter_mm: Optional[float] = None

    @property
    def kind(self) -> str:
        if self.diameter_mm is not None:
            return "Mass" if self.diameter_mm > 30 else "Nodule"
        return "Mass" if self.radius > 16 else "Nodule"


def find_blobs(anatomy: Anatomy, mm_per_pixel: Optional[float] = None,
               min_contrast: float = 0.06) -> List[Blob]:
    """Round dense spots inside the lung envelopes, strongest first (at most 6)."""
    from scipy import ndimage

    lungs_full = anatomy.lung_mask
    if not lungs_full.any():
        return []
    rows, cols = np.nonzero(lungs_full)   # work on the lungs' bounding box only
    oy, ox = max(0, rows.min() - 40), max(0, cols.min() - 40)
    image = anatomy.image[oy:rows.max() + 41, ox:cols.max() + 41]
    lungs = lungs_full[oy:rows.max() + 41, ox:cols.max() + 41]
    sigmas = (4.0, 5.5, 7.5, 10.0, 13.0, 17.0, 22.0)
    stack = np.stack([-(s ** 2) * ndimage.gaussian_laplace(image, s) for s in sigmas])
    peak = ndimage.maximum_filter(stack, size=(3, 9, 9))
    candidates = np.argwhere((stack == peak) & (stack > 0.02))
    distance = ndimage.distance_transform_edt(lungs)
    yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    blobs: List[Blob] = []
    for scale, y, x in sorted(candidates, key=lambda c: -stack[tuple(c)]):
        radius = sigmas[scale] * np.sqrt(2)
        if not lungs[y, x] or distance[y, x] < radius * 0.9:
            continue                       # on the lung edge: heart border, diaphragm, rib end
        if any((b.x - ox - x) ** 2 + (b.y - oy - y) ** 2 < (b.radius + radius) ** 2 for b in blobs):
            continue
        d2 = (yy - y) ** 2 + (xx - x) ** 2
        inside = d2 <= (radius * 0.7) ** 2
        ring = (d2 > (radius * 1.3) ** 2) & (d2 <= (radius * 2.0) ** 2) & lungs
        if inside.sum() < 5 or ring.sum() < 10:
            continue
        contrast = float(image[inside].mean() - image[ring].mean())
        roundness = float(image[inside].std())
        if contrast < min_contrast or roundness > contrast:
            continue
        gx, gy = int(x) + ox, int(y) + oy
        side = "right" if gx < anatomy.midline else "left"
        lung = anatomy.lungs.get(side)
        zone = "middle"
        if lung is not None:
            zone = ZONES[max(0, min(2, int(3 * (gy - lung.top) / lung.height)))]
        diameter = 2 * radius * mm_per_pixel if mm_per_pixel else None
        blobs.append(Blob(side, gx, gy, round(float(radius), 1), round(contrast, 3),
                          zone, round(diameter, 1) if diameter else None))
        if len(blobs) >= 6:
            break
    return blobs
