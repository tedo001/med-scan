"""Stage 4 - the Adaptive Analysis Router.

The centre of the design. Instead of pushing every scan through every model,
the router reads the screen's four group probabilities and decides:

* **Early exit** - every group below the exit threshold on a good-quality
  scan: report "no significant abnormality", run no specialist at all.
* **Route** - each group at or above the route threshold sends the scan to
  that group's specialist module.
* **Borderline** - a group just under the route threshold is also routed when
  the clinical context asks for sensitivity (Emergency) or the scan has quality
  warnings (the screen itself is less reliable).

The thresholds move with the **clinical context**, because the cost of a miss
is not the same everywhere:

==================  ======  ======  =========================================
context             route   exit    rationale
==================  ======  ======  =========================================
Emergency / ED      0.20    0.08    a miss is costly - favour sensitivity
Routine OPD         0.30    0.12    balanced
Screening camp      0.40    0.18    high volume, low prevalence - favour speed
Teaching / audit    0.00    -       run every module, show everything
==================  ======  ======  =========================================

Every decision is written down with its reason, so the doctor (and the audit
trail) can see *why* a module ran or did not. ``mode="fixed"`` bypasses the
router and runs all four modules - the baseline the evaluation compares to.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .screening import GROUPS, ScreenResult

__all__ = ["CONTEXTS", "Route", "RoutingDecision", "route"]

#: context -> (route threshold, early-exit threshold, borderline band, explanation depth)
CONTEXTS: Dict[str, Dict[str, object]] = {
    "Emergency": {"route": 0.20, "exit": 0.08, "band": 0.08, "depth": "brief",
                  "note": "sensitivity first; brief, action-first explanations"},
    "Routine OPD": {"route": 0.30, "exit": 0.12, "band": 0.05, "depth": "standard",
                    "note": "balanced; findings with evidence and limitations"},
    "Screening camp": {"route": 0.40, "exit": 0.18, "band": 0.0, "depth": "brief",
                       "note": "throughput first; deep analysis only when the screen is clear"},
    "Teaching": {"route": 0.0, "exit": -1.0, "band": 0.0, "depth": "detailed",
                 "note": "every module runs; full reasoning and literature shown"},
}
MODULE_NAMES = {"cardiac": "Cardiac", "pleural": "Pleural", "parenchymal": "Parenchyma",
                "focal": "Nodule"}


@dataclass
class Route:
    group: str
    module: str
    run: bool
    probability: float
    reason: str


@dataclass
class RoutingDecision:
    mode: str
    context: str
    early_exit: bool
    routes: List[Route] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)

    @property
    def groups(self) -> List[str]:
        return [r.group for r in self.routes if r.run]

    @property
    def activated(self) -> int:
        return len(self.groups)

    def to_dict(self) -> Dict[str, object]:
        return {"mode": self.mode, "context": self.context, "early_exit": self.early_exit,
                "routes": [asdict(r) for r in self.routes], "thresholds": self.thresholds}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "RoutingDecision":
        return cls(mode=data.get("mode", "adaptive"), context=data.get("context", ""),
                   early_exit=bool(data.get("early_exit")),
                   routes=[Route(**r) for r in data.get("routes", [])],
                   thresholds=dict(data.get("thresholds", {})))


def route(screen: ScreenResult, context: str = "Routine OPD", mode: str = "adaptive",
          quality_warnings: int = 0, overrides: Optional[Dict[str, float]] = None) -> RoutingDecision:
    settings = dict(CONTEXTS.get(context, CONTEXTS["Routine OPD"]))
    settings.update(overrides or {})
    route_at, exit_at, band = float(settings["route"]), float(settings["exit"]), float(settings["band"])
    if quality_warnings:
        band = max(band, 0.05)
    decision = RoutingDecision(mode=mode, context=context, early_exit=False,
                               thresholds={"route": route_at, "exit": exit_at, "band": band})
    groups = screen.groups

    if mode == "fixed" or context == "Teaching":
        why = "fixed pipeline - every module runs" if mode == "fixed" else \
            "teaching context - every module runs"
        decision.routes = [Route(g, MODULE_NAMES[g], True, groups.get(g, 0.0), why) for g in GROUPS]
        return decision

    if all(groups.get(g, 0.0) < exit_at for g in GROUPS) and not quality_warnings:
        decision.early_exit = True
        decision.routes = [Route(g, MODULE_NAMES[g], False, groups.get(g, 0.0),
                                 f"screen {groups.get(g, 0.0):.2f} < exit {exit_at:.2f}")
                           for g in GROUPS]
        return decision

    for g in GROUPS:
        p = groups.get(g, 0.0)
        if p >= route_at:
            decision.routes.append(Route(g, MODULE_NAMES[g], True, p,
                                         f"screen {p:.2f} >= route {route_at:.2f}"))
        elif band and p >= route_at - band:
            decision.routes.append(Route(g, MODULE_NAMES[g], True, p,
                                         f"borderline {p:.2f} within {band:.2f} of route "
                                         f"{route_at:.2f}" + (" (quality warnings)"
                                                              if quality_warnings else "")))
        else:
            decision.routes.append(Route(g, MODULE_NAMES[g], False, p,
                                         f"screen {p:.2f} < route {route_at:.2f} - skipped"))
    return decision
