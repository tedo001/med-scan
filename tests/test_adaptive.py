"""Support modes, recommendations, report drafting, learning, simulation, bias reduction."""

from __future__ import annotations

import os

import numpy as np
import pytest

from msx import bias, explain, report, simulation, support
from msx.datastore import DataStore
from msx.findings import POSITIVE, UNCERTAIN, Finding
from msx.learner import MIN_SAMPLES, Learner
from msx.pipeline import AnalysisEngine
from msx.recommend import recommend

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")


@pytest.fixture(scope="module")
def effusion_cardio():
    return AnalysisEngine(engine="builtin", tta=1).analyse_file(
        os.path.join(SAMPLES, "cxr_12_effusion.png"), context="Emergency")


def test_modes_are_chosen_not_inferred():
    assert set(support.MODES) == {"guided", "concise", "second_opinion", "evidence_first"}
    assert support.get("nonsense").key == support.DEFAULT_MODE
    assert support.MODES["second_opinion"].blind_first


def test_guided_adds_checklist_and_significance(effusion_cardio):
    a = effusion_cardio
    guided = explain.build(a.findings, a.quality, a.routing, a.context, "detailed", mode="guided")
    first = guided["sections"][0]
    assert first.get("checklist") and first.get("why")
    evidence = explain.build(a.findings, a.quality, a.routing, a.context, "standard",
                             mode="evidence_first")
    assert evidence["sections"][0]["title"] == "Guideline evidence"


def test_recommendations_combine_patterns(effusion_cardio):
    recs = recommend(effusion_cardio.findings, "Emergency", effusion_cardio.quality)
    actions = " ".join(r.action for r in recs)
    assert "Heart-failure work-up" in actions and recs[0].tier == "Now"
    unsure = Finding("Nodule", "focal", "Nodule", 0.5, status=UNCERTAIN, confidence=0.3)
    assert recommend([unsure])[0].action.startswith("Confirm before acting")


def test_nodule_size_bands():
    f = Finding("Nodule", "focal", "Nodule", 0.9, status=POSITIVE, confidence=0.9,
                measurements={"diameter_mm": 7.0})
    assert "6-12 months" in recommend([f])[0].action


def test_report_is_grounded(effusion_cardio):
    drafted = report.draft(effusion_cardio)
    text = drafted["text"]
    assert "FINDINGS:" in text and "IMPRESSION:" in text and "RECOMMENDATIONS:" in text
    assert "Effusion" in text and "Cardiomegaly" in text and report.DISCLAIMER in text
    assert drafted["engine"] == "template"


def test_learner_trains_from_ground_truth(tmp_path):
    store = DataStore(str(tmp_path / "db.sqlite"))
    engine = AnalysisEngine(engine="builtin", tta=1)
    import csv
    with open(os.path.join(SAMPLES, "labels.csv")) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows[:21]:
        a = engine.analyse_file(os.path.join(SAMPLES, row["file"]))
        store.save_analysis(a, "", row["labels"])
    learner = Learner(str(tmp_path / "learner.json"))
    info = learner.train(store)
    assert info["n"] >= MIN_SAMPLES and info["status"] == "trained" and learner.ready
    assert info["cv_accuracy"] > 0.8
    engine.learner = learner
    a = engine.analyse_file(os.path.join(SAMPLES, "cxr_10_effusion.png"))
    hit = next(f for f in a.findings if f.label == "Effusion" and f.status == POSITIVE)
    assert "learned" in hit.sources


def test_simulation_measures_automation_bias():
    rng = np.random.default_rng(1)
    cases = [{"truth": {"Effusion": int(t)}, "ai": {"Effusion": int(t)}, "conf": {"Effusion": 0.9}}
             for t in rng.integers(0, 2, 60)]
    result = simulation.simulate(cases, ["Effusion"], ai_error=0.2, runs=40)
    by = {(r["skill"], r["trust"]): r for r in result["readers"]}
    assert by[("Trainee", "Over-trusting")]["automation_bias"] > by[("Trainee", "Sceptical")]["automation_bias"]
    assert by[("Trainee", "Calibrated")]["team"]["accuracy"] > by[("Trainee", "Calibrated")]["doctor"]["accuracy"]


def test_bias_mitigation_closes_a_gap():
    records = []
    for i in range(40):
        sex = "F" if i % 2 else "M"
        truth = ["Effusion"] if i % 4 < 2 else []
        score = (0.8 if sex == "M" else 0.42) if truth else 0.1   # women under-called
        records.append({"file": str(i), "sex": sex, "age_band": "40-64", "truth": truth,
                        "held": False, "scores": {"Effusion": score}})
    m = bias.mitigate(records, ["Effusion"])
    assert m["thresholds"].get("sex:F", 0.5) < 0.5
    assert m["max_gap_after"] < m["max_gap_before"]
    assert bias.threshold_for({"sex": "F"}, m["thresholds"], 0.5) == m["thresholds"]["sex:F"]
