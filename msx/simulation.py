"""Doctor + AI simulation study - does the team beat either alone?

The problem statement allows "representative users or simulations". Real
reader studies come later; this module simulates them, transparently, on the
AI's *actual* outputs for a labelled image set.

**Simulated readers.** Each reader has a skill (per-label sensitivity /
specificity when reading alone, from typical published ranges for chest
radiograph reading) and a trust style (how readily they change their mind
when the AI disagrees):

===============  =====================================================
skill            solo sensitivity / specificity
===============  =====================================================
Trainee          0.70 / 0.88
Registrar        0.80 / 0.92
Consultant       0.90 / 0.96
===============  =====================================================

===============  =====================================================
trust            P(switch to the AI's call when they disagree)
===============  =====================================================
Sceptical        0.15
Calibrated       0.90 if AI confidence >= 0.75, else 0.10
                 (uses MEDSCAN's confidence as intended)
Over-trusting    0.85                          (automation bias)
===============  =====================================================

**AI stress test.** On synthetic phantoms the AI is near-perfect, which would
make the study trivial. ``ai_error`` flips that share of the AI's calls to
emulate real-world accuracy, and gives each flipped call a confidence drawn
from 0.30-0.85 - so roughly a fifth of the AI's errors still *look* confident,
as with any imperfectly calibrated model. Results are reported for the clean and the
stressed AI side by side.

**Measures** (mean over Monte-Carlo runs): sensitivity, specificity and
accuracy for Doctor alone, AI alone, Doctor + AI; **automation bias** - share
of wrong AI calls the doctor adopted after initially being right; **rescue
rate** - share of the doctor's own errors fixed by a correct AI call.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

__all__ = ["SKILLS", "TRUST", "simulate", "cases_from_benchmark"]

SKILLS = {"Trainee": (0.70, 0.88), "Registrar": (0.80, 0.92), "Consultant": (0.90, 0.96)}
TRUST = ("Sceptical", "Calibrated", "Over-trusting")


def cases_from_benchmark(results, labels: Sequence[str]) -> List[Dict[str, object]]:
    """[(row, analysis)] -> cases: truth, AI call and confidence per label."""
    from .findings import POSITIVE

    cases = []
    for row, analysis in results:
        if analysis.status == "quality-hold":
            continue
        ai, conf = {}, {}
        for label in labels:
            hits = [f for f in analysis.findings if f.label == label]
            best = max(hits, key=lambda f: f.probability) if hits else None
            ai[label] = int(bool(best and best.status == POSITIVE))
            conf[label] = float(best.confidence) if best else 0.9
        cases.append({"truth": {l: int(l in row["truth"]) for l in labels}, "ai": ai, "conf": conf})
    return cases


def _switch_probability(trust: str, confidence: float) -> float:
    if trust == "Sceptical":
        return 0.15
    if trust == "Over-trusting":
        return 0.85
    return 0.90 if confidence >= 0.75 else 0.10


def _rates(pred, truth):
    pred, truth = np.asarray(pred), np.asarray(truth)
    tp = ((pred == 1) & (truth == 1)).sum()
    tn = ((pred == 0) & (truth == 0)).sum()
    sens = tp / max(1, (truth == 1).sum())
    spec = tn / max(1, (truth == 0).sum())
    return float(sens), float(spec), float((pred == truth).mean())


def simulate(cases: List[Dict[str, object]], labels: Sequence[str], ai_error: float = 0.0,
             runs: int = 300, seed: int = 7) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    out: Dict[str, object] = {"cases": len(cases), "labels": list(labels), "ai_error": ai_error,
                              "runs": runs, "readers": []}
    if not cases:
        return out
    for skill, (sens, spec) in SKILLS.items():
        for trust in TRUST:
            acc = {k: [] for k in ("doctor", "ai", "team")}
            bias_hits = bias_n = rescue_hits = rescue_n = 0
            for _ in range(runs):
                truths, solo, ai_calls, team = [], [], [], []
                for case in cases:
                    for label in labels:
                        t = case["truth"][label]
                        a, c = case["ai"][label], case["conf"][label]
                        if ai_error and rng.random() < ai_error:
                            a, c = 1 - a, rng.uniform(0.30, 0.85)
                        d = int(rng.random() < (sens if t else 1 - spec))
                        final = d
                        if a != d and rng.random() < _switch_probability(trust, c):
                            final = a
                        if d == t and a != t:
                            bias_n += 1
                            bias_hits += int(final != t)
                        if d != t and a == t:
                            rescue_n += 1
                            rescue_hits += int(final == t)
                        truths.append(t)
                        solo.append(d)
                        ai_calls.append(a)
                        team.append(final)
                acc["doctor"].append(_rates(solo, truths))
                acc["ai"].append(_rates(ai_calls, truths))
                acc["team"].append(_rates(team, truths))
            row = {"skill": skill, "trust": trust}
            for key, values in acc.items():
                arr = np.array(values)
                row[key] = {"sensitivity": round(float(arr[:, 0].mean()), 3),
                            "specificity": round(float(arr[:, 1].mean()), 3),
                            "accuracy": round(float(arr[:, 2].mean()), 3),
                            "accuracy_sd": round(float(arr[:, 2].std()), 3)}
            row["automation_bias"] = round(bias_hits / bias_n, 3) if bias_n else None
            row["rescue_rate"] = round(rescue_hits / rescue_n, 3) if rescue_n else None
            row["team_beats_both"] = row["team"]["accuracy"] > max(row["doctor"]["accuracy"],
                                                                   row["ai"]["accuracy"])
            out["readers"].append(row)
    return out
