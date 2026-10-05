"""The analyser engine: eight stages, one call.

    engine = AnalysisEngine()
    analysis = engine.analyse_file("chest.png", context="Emergency")

``analyse`` runs, in order, and times each stage:

1. preprocess   - letterbox to 512 x 512, segment the lungs (:mod:`msx.anatomy`)
2. quality      - seven checks; any fail stops here with "human review" (:mod:`msx.quality`)
3. screen       - built-in measurements, plus the DenseNet when the hybrid engine is on;
                  their group probabilities are fused (:mod:`msx.screening`)
4. route        - which specialist modules this scan needs (:mod:`msx.router`)
5. specialists  - only the routed modules run, each with test-time augmentation
                  (:mod:`msx.specialists`)
6. confidence   - calibration, confidence, abstention (:mod:`msx.uncertainty`)
7. explain      - adaptive-depth, evidence-grounded explanation (:mod:`msx.explain`)
8. record       - the caller stores the analysis and the doctor's decisions
                  (:mod:`msx.datastore`, :mod:`msx.audit`)

The returned :class:`Analysis` carries every intermediate result, so the
interface can show the doctor *how* the answer was reached, and the evaluation
can count what each scan cost (modules activated, model calls, milliseconds).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from . import anatomy as anatomy_mod
from . import bias, explain, imaging, learner as learner_mod, quality as quality_mod, recommend, \
    router, screening, specialists, uncertainty
from .findings import NEGATIVE, POSITIVE, POSSIBLE, UNCERTAIN, Finding
from .imaging import Scan

__all__ = ["Stage", "Analysis", "AnalysisEngine", "STAGES", "ENGINES"]

STAGES = ("Preprocess", "Quality", "Screen", "Route", "Specialists", "Confidence", "Explain",
          "Record")
ENGINES = {"hybrid": "Hybrid - DenseNet-121 + measurements", "builtin": "Built-in measurements"}


@dataclass
class Stage:
    name: str
    ms: float
    status: str          # ok, skipped, stopped
    note: str = ""


@dataclass
class Analysis:
    id: str
    created: str
    scan: Dict[str, object]
    context: str
    mode: str
    engine: str
    quality: Dict[str, object]
    screen: Dict[str, object] = field(default_factory=dict)
    routing: Dict[str, object] = field(default_factory=dict)
    findings: List[Finding] = field(default_factory=list)
    explanation: Dict[str, object] = field(default_factory=dict)
    recommendations: List[Dict[str, object]] = field(default_factory=list)
    positive_at: float = 0.5
    stages: List[Stage] = field(default_factory=list)
    status: str = "findings"
    model_calls: int = 0
    total_ms: float = 0.0
    # in-memory only: images for the viewer
    work: Optional[np.ndarray] = None
    heat: Optional[np.ndarray] = None
    lung_mask: Optional[np.ndarray] = None

    @property
    def shown(self) -> List[Finding]:
        return [f for f in self.findings if f.shown]

    @property
    def top(self) -> Optional[Finding]:
        shown = explain._ranked(self.shown)
        return shown[0] if shown else None

    @property
    def urgent(self) -> bool:
        return any(f.status == POSITIVE and (explain.URGENT.get(f.label, 0) >= 3 or (
            explain.URGENT.get(f.label, 0) >= 2 and f.probability >= 0.8)) for f in self.findings)

    @property
    def modules_activated(self) -> int:
        return sum(1 for r in self.routing.get("routes", []) if r.get("run"))

    def to_dict(self) -> Dict[str, object]:
        return {"id": self.id, "created": self.created, "scan": self.scan, "context": self.context,
                "mode": self.mode, "engine": self.engine, "quality": self.quality,
                "screen": self.screen, "routing": self.routing,
                "findings": [f.to_dict() for f in self.findings],
                "explanation": self.explanation, "recommendations": self.recommendations,
                "positive_at": self.positive_at,
                "stages": [s.__dict__ for s in self.stages], "status": self.status,
                "model_calls": self.model_calls, "total_ms": round(self.total_ms, 1)}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Analysis":
        return cls(id=data["id"], created=data["created"], scan=data.get("scan", {}),
                   context=data.get("context", ""), mode=data.get("mode", "adaptive"),
                   engine=data.get("engine", ""), quality=data.get("quality", {}),
                   screen=data.get("screen", {}), routing=data.get("routing", {}),
                   findings=[Finding.from_dict(f) for f in data.get("findings", [])],
                   explanation=data.get("explanation", {}),
                   recommendations=data.get("recommendations", []),
                   positive_at=float(data.get("positive_at", 0.5)),
                   stages=[Stage(**s) for s in data.get("stages", [])],
                   status=data.get("status", ""), model_calls=int(data.get("model_calls", 0)),
                   total_ms=float(data.get("total_ms", 0)))


def _matrices(size: int, k: int):
    """Small, plausible perturbations: (affine shift/scale matrix, exposure gamma)."""
    import cv2

    recipes = [(8, 0, 1.0, 1.0), (-8, 6, 1.0, 0.85), (0, -6, 0.96, 1.15), (4, 4, 1.04, 1.0),
               (-4, -4, 1.0, 0.92), (0, 8, 0.98, 1.08)][:k]
    out = []
    for dx, dy, scale, gamma in recipes:
        matrix = cv2.getRotationMatrix2D((size / 2, size / 2), 0, scale)
        matrix[:, 2] += (dx, dy)
        out.append((matrix, gamma))
    return out


def _warp(image: np.ndarray, matrix, nearest: bool = False) -> np.ndarray:
    import cv2

    size = image.shape[0]
    return cv2.warpAffine(image.astype(np.float32), matrix, (size, size),
                          flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR, borderValue=0.0)


def _augment(work: np.ndarray, k: int) -> List[np.ndarray]:
    return [np.clip(_warp(work, m), 0, 1) ** g for m, g in _matrices(work.shape[0], k)]


class AnalysisEngine:
    """Configured once (engine, thresholds, TTA count), then analyses many scans."""

    def __init__(self, engine: str = "hybrid", tta: int = 4,
                 temperatures: Optional[Dict[str, float]] = None,
                 route_overrides: Optional[Dict[str, Dict[str, float]]] = None,
                 narrator: Optional[explain.LLMNarrator] = None,
                 positive_at: float = uncertainty.POSITIVE_AT,
                 subgroup_thresholds: Optional[Dict[str, float]] = None,
                 learner: Optional[learner_mod.Learner] = None):
        self.engine_name = engine
        self.tta = tta
        self.temperatures = temperatures or {}
        self.route_overrides = route_overrides or {}
        self.narrator = narrator
        self.positive_at = positive_at
        self.subgroup_thresholds = subgroup_thresholds or {}
        self.learner = learner
        self.measure_screen = screening.MeasurementScreen()

    @property
    def deep(self) -> Optional[screening.DeepScreen]:
        return screening.get_deep() if self.engine_name == "hybrid" else None

    @property
    def engine_label(self) -> str:
        if self.engine_name == "hybrid" and self.deep is not None:
            return screening.DeepScreen.short + " + measurements"
        return "built-in measurements"

    def analyse_file(self, path: str, context: str = "Routine OPD", mode: str = "adaptive",
                     depth: Optional[str] = None, **facts) -> Analysis:
        started = time.perf_counter()
        scan = imaging.load_scan(path, **facts)
        load_ms = (time.perf_counter() - started) * 1000
        analysis = self.analyse(scan, context, mode, depth)
        analysis.stages[0].ms += load_ms
        analysis.total_ms += load_ms
        return analysis

    def analyse(self, scan: Scan, context: str = "Routine OPD", mode: str = "adaptive",
                depth: Optional[str] = None) -> Analysis:
        clock = time.perf_counter
        t0 = clock()
        stages: List[Stage] = []

        # 1. preprocess
        t = clock()
        work = imaging.standardise(scan.pixels)
        anatomy = anatomy_mod.segment(work)
        learned_masks = None
        if self.engine_name == "hybrid" and screening.deep_available():
            segmenter = screening.get_segmenter()
            if segmenter is not None:
                learned_masks = segmenter.masks(work)
                candidate = anatomy_mod.from_masks(work, learned_masks["right"],
                                                   learned_masks["left"], learned_masks["heart"])
                if anatomy_mod.plausible(candidate):
                    anatomy = candidate
                else:
                    learned_masks = None
        stages.append(Stage("Preprocess", (clock() - t) * 1000, "ok",
                            f"{scan.original_shape[1]}x{scan.original_shape[0]} -> 512x512; "
                            f"lungs {'found' if anatomy.found else 'not found'} ({anatomy.source})"
                            + (f"; {len(scan.removed_tags)} identifiers removed"
                               if scan.removed_tags else "")))

        # 2. quality
        t = clock()
        quality = quality_mod.assess(work, anatomy, scan.original_shape)
        warnings = sum(1 for c in quality.checks if c.status == quality_mod.WARN)
        stages.append(Stage("Quality", (clock() - t) * 1000,
                            "ok" if quality.accepted else "stopped",
                            f"score {quality.score}, {quality.grade}"))
        analysis = Analysis(id=uuid.uuid4().hex[:10].upper(),
                            created=datetime.now().isoformat(timespec="seconds"),
                            scan=scan.facts(), context=context, mode=mode,
                            engine=self.engine_label, quality=quality.to_dict(), work=work,
                            lung_mask=anatomy.lung_mask if anatomy.found else None)
        positive_at = bias.threshold_for(scan.facts(), self.subgroup_thresholds, self.positive_at)
        analysis.positive_at = positive_at
        base_depth = depth or str(router.CONTEXTS.get(context, router.CONTEXTS["Routine OPD"])["depth"])

        if not quality.accepted:
            for name in STAGES[2:7]:
                stages.append(Stage(name, 0.0, "skipped", "scan held for human review"))
            analysis.status = "quality-hold"
            analysis.explanation = explain.build([], analysis.quality, {}, context, base_depth)
            analysis.recommendations = [r.to_dict() for r in recommend.recommend(
                [], context, analysis.quality)]
            analysis.stages = stages
            analysis.total_ms = (clock() - t0) * 1000
            return analysis

        # 3. screen
        t = clock()
        calls = 0
        result = self.measure_screen.screen(work, anatomy, scan.view)
        groups = dict(result.groups)
        deep_labels: Dict[str, float] = {}
        deep = self.deep
        if deep is not None:
            deep_result = deep.screen(work, anatomy, scan.view)
            calls += 1
            deep_labels = deep_result.labels
            groups = {g: round(specialists.fuse(result.groups[g], deep_result.groups[g]), 4)
                      for g in screening.GROUPS}
        analysis.screen = {"groups": groups, "measured": result.groups, "deep": deep_labels,
                           "features": result.features}
        stages.append(Stage("Screen", (clock() - t) * 1000, "ok",
                            ", ".join(f"{g} {p:.2f}" for g, p in groups.items())))

        # 4. route
        t = clock()
        decision = router.route(screening.ScreenResult(result.engine, groups), context, mode,
                                warnings, self.route_overrides.get(context))
        analysis.routing = decision.to_dict()
        stages.append(Stage("Route", (clock() - t) * 1000, "ok",
                            "early exit - no specialist needed" if decision.early_exit else
                            f"{decision.activated} of 4 modules: " + ", ".join(
                                r.module for r in decision.routes if r.run)))

        # 5. specialists (+ test-time augmentation, only if something is routed)
        t = clock()
        findings: List[Finding] = []
        heat = np.zeros(work.shape, dtype=np.float32)
        if decision.groups:
            recipes = _matrices(work.shape[0], self.tta)
            augmented_works = [np.clip(_warp(work, m), 0, 1) ** g for m, g in recipes]
            if learned_masks is not None:      # move the learned masks with the image
                augmented = [(w, anatomy_mod.from_masks(
                    w, _warp(learned_masks["right"], m, True) > 0.5,
                    _warp(learned_masks["left"], m, True) > 0.5,
                    _warp(learned_masks["heart"], m, True) > 0.5))
                    for w, (m, _) in zip(augmented_works, recipes)]
            else:
                augmented = [(w, anatomy_mod.segment(w)) for w in augmented_works]
            deep_tta: Dict[str, List[float]] = {}
            if deep is not None and augmented_works:
                rows = deep.predict_batch(augmented_works)
                calls += 1
                deep_tta = {label: [row[label] for row in rows] for label in rows[0]}
            ctx = specialists.ModuleContext(work=work, anatomy=anatomy, view=scan.view,
                                            mm_per_pixel=scan.mm_per_work_pixel, deep=deep,
                                            deep_labels=deep_labels, deep_tta=deep_tta,
                                            augmented=augmented, quality_warnings=warnings)
            for group in decision.groups:
                module = specialists.MODULES[group]()
                produced = module.run(ctx)
                calls += 1 + len(augmented)
                for finding in produced:
                    self._learned(finding, quality.score, scan.view)
                    uncertainty.score(finding, warnings, self.temperatures, positive_at)
                    if finding.shown:
                        map_ = module.heatmap(finding, ctx)
                        if deep is not None and module.deep_map.get(finding.label):
                            calls += 1
                        if map_ is not None and map_.max() > 0:
                            heat = np.maximum(heat, map_ / map_.max() * max(0.4, finding.probability))
                findings.extend(produced)
            stages.append(Stage("Specialists", (clock() - t) * 1000, "ok",
                                f"{len(decision.groups)} module(s) x {1 + len(augmented)} "
                                f"views (TTA)"))
        else:
            stages.append(Stage("Specialists", 0.0, "skipped", "early exit"))

        # 6. confidence (scored per finding above; here: keep one finding per label/side)
        t = clock()
        findings = self._tidy(findings)
        for index, finding in enumerate(findings):
            finding.key = f"F{index + 1}"
        counts = uncertainty.summarise(findings)
        stages.append(Stage("Confidence", (clock() - t) * 1000, "ok",
                            f"{counts[POSITIVE]} positive, {counts[POSSIBLE]} possible, "
                            f"{counts[UNCERTAIN]} abstained"))

        # 7. explain
        t = clock()
        explanation = explain.build(findings, analysis.quality, analysis.routing, context,
                                    base_depth)
        if self.narrator is not None and self.narrator.enabled:
            explanation = self.narrator.narrate(explanation, findings)
        stages.append(Stage("Explain", (clock() - t) * 1000, "ok",
                            f"{explanation['depth']}" + (" (escalated)" if explanation["escalated"]
                                                         else "") + f", {len(explanation['citations'])} citations"))
        stages.append(Stage("Record", 0.0, "ok", "awaiting doctor review"))

        analysis.findings = findings
        analysis.explanation = explanation
        analysis.recommendations = [r.to_dict() for r in recommend.recommend(
            findings, context, analysis.quality)]
        analysis.heat = heat if heat.max() > 0 else None
        analysis.model_calls = calls
        analysis.stages = stages
        analysis.status = ("findings" if counts[POSITIVE] or counts[POSSIBLE] else
                           "uncertain" if counts[UNCERTAIN] else "no-finding")
        analysis.total_ms = (clock() - t0) * 1000
        return analysis

    def _learned(self, finding: Finding, quality_score: float, view: str) -> None:
        """Blend in the feedback-trained model, weighted by how much feedback it has seen."""
        if self.learner is None or not self.learner.ready or finding.label not in learner_mod.LABELS:
            return
        p = self.learner.predict(finding.to_dict(), quality_score, view)
        if p is None:
            return
        w = self.learner.weight
        finding.sources["learned"] = round(p, 4)
        finding.probability = round((1 - w) * finding.probability + w * p, 4)
        finding.evidence.append(f"Feedback-trained model {p:.2f} (weight {w:.2f}, "
                                f"{self.learner.info.get('n', 0)} labelled findings)")

    @staticmethod
    def _tidy(findings: List[Finding]) -> List[Finding]:
        """Per label: keep every shown side; collapse negatives to the single strongest."""
        by_label: Dict[str, List[Finding]] = {}
        for f in findings:
            by_label.setdefault(f.label, []).append(f)
        out: List[Finding] = []
        for label, group in by_label.items():
            shown = [f for f in group if f.shown]
            if shown:
                out.extend(shown)
            else:
                strongest = max(group, key=lambda f: f.probability)
                strongest.side = ""
                out.append(strongest)
        order = {POSITIVE: 0, UNCERTAIN: 1, POSSIBLE: 2, NEGATIVE: 3}
        out.sort(key=lambda f: (order[f.status], -f.probability))
        return out
