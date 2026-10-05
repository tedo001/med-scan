"""Stage 5 - region-specific specialist modules.

The router decides which of these a scan needs; each then looks closely at its
own region and returns :class:`~msx.findings.Finding` objects.

=================  ====================  ================================================
module             region                how it decides
=================  ====================  ================================================
CardiacModule      heart between lungs   cardiothoracic ratio (CTR) from the medial lung
                                         margins vs the outer thoracic width; > 0.50 on a
                                         PA film is cardiomegaly (> 0.56 on AP)
PleuralModule      lung bases, lateral   effusion: lung shorter than the other side and /
                   and apical rim        or lateral costophrenic recess raised (blunted);
                                         pneumothorax: lateral band more lucent than the
                                         other side's, markings absent
ParenchymaModule   six lung zones        zone density above its mirror zone plus the share
                                         of expected lung that is no longer aerated
NoduleModule       inside lung fields    round dense spots by multi-scale LoG with a
                                         local-contrast test; > 30 mm is a mass
=================  ====================  ================================================

**Two sources, one answer.** Every module measures. When the deep engine is
installed its probability for the same label is fused with the measurement in
log-odds space (:func:`fuse`): independent evidence adds, a model sitting on
the fence (0.5) adds nothing, and disagreement pulls the answer back towards
0.5 - where the uncertainty stage will flag it. The measurement says *where*
and *how much*; the model adds pattern knowledge from ~800 000 training films.

**Test-time augmentation.** :meth:`Module.run` re-measures on slightly shifted,
scaled and re-exposed copies of the scan. A finding that survives those small
perturbations is stable; one that flips is not, and its confidence drops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import measures
from .anatomy import Anatomy
from .findings import Finding
from .screening import CURVES, logistic

__all__ = ["Candidate", "ModuleContext", "Module", "CardiacModule", "PleuralModule",
           "ParenchymaModule", "NoduleModule", "MODULES", "fuse"]

DEEP_WEIGHT = 0.8
MEASURE_WEIGHT = 0.8


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def fuse(measured: Optional[float], deep: Optional[float]) -> float:
    """Combine two probabilities as independent evidence in log-odds space."""
    if deep is None and measured is None:
        return 0.0
    if deep is None:
        return float(measured)
    if measured is None:
        return float(deep)
    z = DEEP_WEIGHT * max(-5, min(5, _logit(deep))) + MEASURE_WEIGHT * max(
        -5, min(5, _logit(measured)))
    return 1 / (1 + math.exp(-z))


@dataclass
class Candidate:
    """One thing a module measured - positive or not - at one place."""

    label: str
    side: str
    probability: float
    measurements: Dict[str, object] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    zones: List[str] = field(default_factory=list)
    bbox: Optional[Tuple[int, int, int, int]] = None
    heat: Optional[np.ndarray] = None

    @property
    def key(self) -> Tuple[str, str]:
        return self.label, self.side


@dataclass
class ModuleContext:
    """Everything a module may look at for one scan."""

    work: np.ndarray
    anatomy: Anatomy
    view: str = "PA"
    mm_per_pixel: Optional[float] = None
    deep: object = None                                  # DeepScreen or None
    deep_labels: Dict[str, float] = field(default_factory=dict)
    deep_tta: Dict[str, List[float]] = field(default_factory=dict)
    augmented: List[Tuple[np.ndarray, Anatomy]] = field(default_factory=list)
    quality_warnings: int = 0


class Module:
    name = "module"
    group = ""
    #: our label -> the deep model's labels that speak for it
    deep_map: Dict[str, Tuple[str, ...]] = {}
    limitations: Dict[str, List[str]] = {}

    def measure(self, work: np.ndarray, anatomy: Anatomy, ctx: ModuleContext) -> List[Candidate]:
        raise NotImplementedError

    def _deep(self, label: str, ctx: ModuleContext) -> Optional[float]:
        names = self.deep_map.get(label, ())
        values = [ctx.deep_labels[n] for n in names if n in ctx.deep_labels]
        return max(values) if values else None

    def _deep_tta(self, label: str, ctx: ModuleContext) -> List[Optional[float]]:
        names = [n for n in self.deep_map.get(label, ()) if n in ctx.deep_tta]
        if not names:
            return [None] * len(ctx.augmented)
        return [max(ctx.deep_tta[n][i] for n in names) for i in range(len(ctx.augmented))]

    def run(self, ctx: ModuleContext) -> List[Finding]:
        """Measure, re-measure under augmentation, fuse with the deep engine."""
        main = self.measure(ctx.work, ctx.anatomy, ctx)
        repeats: List[Dict[Tuple[str, str], float]] = []
        for work, anatomy in ctx.augmented:
            if not anatomy.found:
                repeats.append({})
                continue
            repeats.append({c.key: c.probability for c in self.measure(work, anatomy, ctx)})
        findings = []
        for candidate in main:
            deep = self._deep(candidate.label, ctx)
            fused = fuse(candidate.probability, deep)
            samples = [fused]
            for index, repeat in enumerate(repeats):
                measured = repeat.get(candidate.key, candidate.probability)
                samples.append(fuse(measured, self._deep_tta(candidate.label, ctx)[index]))
            sources = {"measurement": round(candidate.probability, 4)}
            if deep is not None:
                sources["deep"] = round(deep, 4)
            finding = Finding(
                label=candidate.label, group=self.group, module=self.name,
                probability=round(float(np.mean(samples)), 4), side=candidate.side,
                zones=candidate.zones, bbox=candidate.bbox, measurements=candidate.measurements,
                evidence=list(candidate.evidence), sources=sources,
                limitations=list(self.limitations.get(candidate.label, [])),
                uncertainty={"mean": round(float(np.mean(samples)), 4),
                             "std": round(float(np.std(samples)), 4), "n": len(samples),
                             "single": round(fused, 4)})
            if deep is not None:
                finding.evidence.append(f"DenseNet-121 probability {deep:.2f} for "
                                        f"{'/'.join(self.deep_map.get(candidate.label, ()))}")
            finding._heat = candidate.heat  # type: ignore[attr-defined]
            findings.append(finding)
        return self._deep_only(findings, ctx)

    def _deep_only(self, findings: List[Finding], ctx: ModuleContext) -> List[Finding]:
        """Labels the deep model reports that this module cannot measure."""
        return findings

    def heatmap(self, finding: Finding, ctx: ModuleContext) -> Optional[np.ndarray]:
        """Grad-CAM inside the lungs if the deep engine is on, else the measured region."""
        heat = getattr(finding, "_heat", None)
        names = self.deep_map.get(finding.label, ())
        if ctx.deep is not None and names:
            cam = ctx.deep.gradcam(ctx.work, max(names, key=lambda n: ctx.deep_labels.get(n, 0)))
            if cam is not None:
                from scipy import ndimage

                # keep the model's attention that falls on or near the lungs
                region = ndimage.binary_dilation(ctx.anatomy.lung_mask, iterations=16) \
                    if ctx.anatomy.found else ctx.anatomy.body
                cam = cam * region.astype(np.float32)
                heat = cam if heat is None else np.maximum(cam * 0.6, heat.astype(np.float32))
        return heat


def _box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if not mask.any():
        return None
    rows, cols = np.nonzero(mask)
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


# ----------------------------------------------------------------- cardiac -----
class CardiacModule(Module):
    name = "Cardiac"
    group = "cardiac"
    deep_map = {"Cardiomegaly": ("Cardiomegaly", "Enlarged Cardiomediastinum")}
    limitations = {"Cardiomegaly": [
        "CTR is measured from lung margins: pericardial effusion or fat pad looks the same",
        "AP and supine films magnify the heart; poor inspiration narrows the thorax",
        "CTR is a screening sign, not a diagnosis - echocardiography confirms",
    ]}

    def measure(self, work, anatomy, ctx):
        if not anatomy.found:
            return []
        thoracic, x_right, x_left = anatomy.thoracic_width()
        cardiac, b_right, b_left, row = anatomy.cardiac_width()
        ratio = cardiac / thoracic if thoracic else 0.0
        curve = CURVES["ctr_ap" if ctx.view == "AP" else "ctr"]
        probability = logistic(ratio, *curve)
        heart = np.zeros(work.shape, dtype=np.float32)
        top = int(row - 0.25 * thoracic)
        heart[max(0, top):min(work.shape[0], row + int(0.18 * thoracic)), b_right:b_left] = 1
        heart *= ~anatomy.lung_mask
        evidence = [f"Cardiothoracic ratio {ratio:.2f} (heart {cardiac} px / thorax {thoracic} px)",
                    f"{'above' if ratio > curve[0] - 0.02 else 'within'} the "
                    f"{0.50 if ctx.view != 'AP' else 0.56:.2f} {ctx.view} threshold"]
        if ctx.mm_per_pixel:
            evidence.append(f"heart width ~{cardiac * ctx.mm_per_pixel:.0f} mm")
        return [Candidate("Cardiomegaly", "", probability,
                          measurements={"ctr": round(ratio, 3), "cardiac_px": cardiac,
                                        "thoracic_px": thoracic, "row": row,
                                        "lines": [[b_right, row, b_left, row],
                                                  [x_right, row + 12, x_left, row + 12]]},
                          evidence=evidence, bbox=(b_right, max(0, top), b_left,
                                                   min(work.shape[0] - 1, row + int(0.18 * thoracic))),
                          heat=heart)]


# ----------------------------------------------------------------- pleural -----
class PleuralModule(Module):
    name = "Pleural"
    group = "pleural"
    deep_map = {"Effusion": ("Effusion",), "Pneumothorax": ("Pneumothorax",)}
    limitations = {
        "Effusion": ["Small effusions (< ~200 ml) may not blunt the angle on a frontal film",
                     "A raised hemidiaphragm or lower-lobe collapse can mimic it - lateral "
                     "film or ultrasound confirms",
                     "Bilateral symmetric effusions reduce the left/right asymmetry signal"],
        "Pneumothorax": ["Small apical pneumothoraces are easily missed on a supine film",
                         "Skin folds and bullae can mimic a pleural line",
                         "Compares sides: bilateral pneumothorax lowers sensitivity"],
    }

    def measure(self, work, anatomy, ctx):
        if not anatomy.found:
            return []
        out = []
        rim = measures.pneumothorax_rim(anatomy)
        for side, lung in anatomy.lungs.items():
            asym = anatomy.height_asymmetry(side)
            blunt = anatomy.costophrenic_blunting(side)
            score = max(asym, blunt)
            p = logistic(score, *CURVES["effusion"])
            base = np.zeros(work.shape, dtype=np.float32)
            y0 = int(lung.top + 0.62 * lung.height)
            x0, _, x1, _ = lung.bbox()
            y1 = min(work.shape[0] - 1, int(lung.bottom + 0.22 * lung.height))
            base[y0:y1, x0:x1] = anatomy.body[y0:y1, x0:x1] & ~lung.aerated[y0:y1, x0:x1]
            evidence = []
            if asym > 0.03:
                evidence.append(f"{side} lung field {asym:.0%} shorter than the other side")
            if blunt > 0.03:
                evidence.append(f"{side} lateral costophrenic recess raised by {blunt:.0%} of "
                                "lung height (blunted angle / meniscus)")
            if not evidence:
                evidence.append(f"{side} costophrenic angle sharp, lung heights symmetric")
            out.append(Candidate("Effusion", side, p, {"asymmetry": round(asym, 3),
                                                      "blunting": round(blunt, 3)},
                                 evidence, ["lower"], (x0, y0, x1, y1), base))

            r = rim.get(side, {})
            lucency = float(r.get("lucency", 0.0))
            p = logistic(lucency, *CURVES["pneumothorax"])
            evidence = [f"{side} lateral band {lucency:+.0%} lucency vs the other side"]
            if r.get("marking_ratio", 1.0) < 0.85:
                evidence.append(f"lung markings reduced to {r['marking_ratio']:.0%} of the "
                                "other side's")
            mask = r.get("mask")
            out.append(Candidate("Pneumothorax", side, p,
                                 {"lucency": round(lucency, 3),
                                  "marking_ratio": r.get("marking_ratio", 1.0)},
                                 evidence, ["upper", "middle"], r.get("bbox"),
                                 mask.astype(np.float32) if mask is not None else None))
        return out


# -------------------------------------------------------------- parenchyma -----
class ParenchymaModule(Module):
    name = "Parenchyma"
    group = "parenchymal"
    deep_map = {"Consolidation": ("Consolidation", "Pneumonia", "Lung Opacity", "Infiltration")}
    deep_only = ("Atelectasis", "Edema")
    limitations = {
        "Consolidation": ["Infection, oedema, haemorrhage and tumour all look dense - clinical "
                          "context decides",
                          "Overlying soft tissue (breast, large habitus) raises lower-zone density",
                          "Bilateral symmetric disease reduces the left/right comparison"],
        "Atelectasis": ["Reported by the deep model only - no independent measurement"],
        "Edema": ["Reported by the deep model only - no independent measurement",
                  "Correlate with heart size, effusions and clinical fluid status"],
    }

    def measure(self, work, anatomy, ctx):
        if not anatomy.found:
            return []
        zones = measures.zone_opacity(anatomy)
        out = []
        for side in ("right", "left"):
            own = [z for z in zones if z.side == side]
            if not own:
                continue
            scored = [(z.excess_density + 2 * z.dense_fraction, z) for z in own]
            index, worst = max(scored, key=lambda s: s[0])
            p = logistic(index, *CURVES["opacity"])
            affected = [z for s, z in scored if s >= CURVES["opacity"][0] * 0.85] or [worst]
            mask = np.zeros(work.shape, dtype=bool)
            for z in affected:
                x0, y0, x1, y1 = z.bbox
                mask[y0:y1 + 1, x0:x1 + 1] = True
            heat = measures.opacity_map(anatomy) * mask
            heat = heat / heat.max() if heat.max() > 0 else heat
            evidence = [f"{side} {worst.zone} zone {worst.excess_density:+.2f} SD denser than "
                        f"the {('left' if side == 'right' else 'right')} {worst.zone} zone"]
            if worst.dense_fraction > 0.03:
                evidence.append(f"{worst.dense_fraction:.0%} of the expected {side} "
                                f"{worst.zone}-zone lung is no longer aerated")
            out.append(Candidate("Consolidation", side, p,
                                 {"opacity_index": round(index, 3),
                                  "zone": worst.zone,
                                  "excess_density": worst.excess_density,
                                  "dense_fraction": worst.dense_fraction},
                                 evidence, [z.zone for z in affected], _box(mask), heat))
        return out

    def _deep_only(self, findings, ctx):
        for label in self.deep_only:
            p = ctx.deep_labels.get(label)
            if p is None:
                continue
            tta = ctx.deep_tta.get(label, [])
            samples = [p] + list(tta)
            finding = Finding(label=label, group=self.group, module=self.name,
                              probability=round(float(np.mean(samples)), 4),
                              sources={"deep": round(p, 4)},
                              evidence=[f"DenseNet-121 probability {p:.2f} for {label}"],
                              limitations=list(self.limitations[label]),
                              uncertainty={"mean": round(float(np.mean(samples)), 4),
                                           "std": round(float(np.std(samples)), 4),
                                           "n": len(samples), "single": round(p, 4)})
            finding._heat = None  # type: ignore[attr-defined]
            self.deep_map = dict(self.deep_map, **{label: (label,)})
            findings.append(finding)
        return findings


# ------------------------------------------------------------------ nodule -----
class NoduleModule(Module):
    name = "Nodule"
    group = "focal"
    deep_map = {"Nodule": ("Nodule", "Lung Lesion"), "Mass": ("Mass", "Lung Lesion")}
    limitations = {
        "Nodule": ["Nipple shadows, rib-end calcification and vessels seen end-on mimic nodules",
                   "Nodules < 8 mm and those behind the heart or diaphragm are often occult "
                   "on plain film",
                   "Size without pixel spacing is approximate - CT characterises"],
        "Mass": ["A mass needs CT characterisation; a plain film cannot tell benign from "
                 "malignant"],
    }

    def measure(self, work, anatomy, ctx):
        if not anatomy.found:
            return []
        blobs = measures.find_blobs(anatomy, ctx.mm_per_pixel, min_contrast=0.08)
        blobs = [b for b in blobs if b.radius >= 6.5 and not self._hilar(b, anatomy)]
        # a spot inside an already-dense zone, or one of several in a zone, is patchy
        # airspace disease (the parenchyma module's), not a nodule
        dense = {(z.side, z.zone) for z in measures.zone_opacity(anatomy)
                 if z.excess_density + 2 * z.dense_fraction >= 0.4}
        crowded = dense | {(b.side, b.zone) for b in blobs
                           if sum(1 for o in blobs if (o.side, o.zone) == (b.side, b.zone)) >= 2}
        out = []
        for label in ("Nodule", "Mass"):
            kind = [b for b in blobs if b.kind == label and (
                label == "Mass" or (b.side, b.zone) not in crowded)]
            if not kind:
                out.append(Candidate(label, "", logistic(0.0, *CURVES["focal"]),
                                     {"count": 0}, [f"no discrete {label.lower()} found "
                                                    "inside the lung fields"]))
                continue
            best = max(kind, key=lambda b: b.contrast)
            p = logistic(best.contrast, *CURVES["focal"])
            heat = np.zeros(work.shape, dtype=np.float32)
            yy, xx = np.ogrid[:work.shape[0], :work.shape[1]]
            for b in kind:
                heat = np.maximum(heat, np.exp(-((yy - b.y) ** 2 + (xx - b.x) ** 2)
                                               / (2 * (b.radius * 0.9) ** 2)))
            size = (f"~{best.diameter_mm:.0f} mm" if best.diameter_mm
                    else f"{2 * best.radius:.0f} px across")
            evidence = [f"round density in the {best.side} {best.zone} zone, {size}",
                        f"local contrast {best.contrast:.2f} above the surrounding lung"]
            if len(kind) > 1:
                evidence.append(f"{len(kind)} candidate {label.lower()}s in total")
            r = best.radius * 1.6
            out.append(Candidate(label, best.side, p,
                                 {"count": len(kind), "radius_px": best.radius,
                                  "diameter_mm": best.diameter_mm, "contrast": best.contrast,
                                  "spots": [[b.x, b.y, b.radius] for b in kind]},
                                 evidence, [best.zone],
                                 (int(best.x - r), int(best.y - r), int(best.x + r),
                                  int(best.y + r)), heat))
        return out

    @staticmethod
    def _hilar(blob, anatomy) -> bool:
        """A faint spot in the hilum, where vessels seen end-on make round shadows."""
        lung = anatomy.lungs.get(blob.side)
        if lung is None or blob.contrast >= 0.3:
            return False
        from_medial = abs(blob.x - lung.medial) / lung.width
        depth = (blob.y - lung.top) / lung.height
        return from_medial < 0.22 and 0.3 < depth < 0.65


MODULES: Dict[str, Callable[[], Module]] = {
    "cardiac": CardiacModule, "pleural": PleuralModule,
    "parenchymal": ParenchymaModule, "focal": NoduleModule,
}
