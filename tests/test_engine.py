"""Engine tests on the synthetic phantoms (built-in engine: deterministic, no network)."""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from msx import anatomy, explain, imaging, quality, router, screening, uncertainty
from msx.audit import AuditLog
from msx.findings import NEGATIVE, POSITIVE, UNCERTAIN, Finding
from msx.knowledge import default_kb
from msx.pipeline import AnalysisEngine

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")


def truth():
    with open(os.path.join(SAMPLES, "labels.csv"), newline="") as handle:
        return {r["file"]: r for r in csv.DictReader(handle)}


@pytest.fixture(scope="module")
def engine():
    return AnalysisEngine(engine="builtin", tta=2)


def _work(name):
    return imaging.standardise(imaging.load_scan(os.path.join(SAMPLES, name)).pixels)


def test_dicom_is_deidentified():
    scan = imaging.load_scan(os.path.join(SAMPLES, "cxr_25_effusion_dicom.dcm"))
    assert "PatientName" in scan.removed_tags and "PatientID" in scan.removed_tags
    assert scan.patient_ref.startswith("PT-") and "TG-0001" not in scan.patient_ref
    assert scan.sex == "F" and scan.age_band == "40-64" and scan.pixel_spacing_mm == 0.5
    assert "PHANTOM" not in str(scan.facts())


def test_pseudonym_is_stable_and_not_reversible():
    assert imaging.pseudonym("12345") == imaging.pseudonym("12345")
    assert imaging.pseudonym("12345") != imaging.pseudonym("12346")
    assert "12345" not in imaging.pseudonym("12345")


def test_lungs_found_and_ctr_separates_cardiomegaly():
    normal = anatomy.segment(_work("cxr_01_normal.png"))
    big = anatomy.segment(_work("cxr_07_cardiomegaly.png"))
    assert normal.found and big.found
    assert normal.cardiothoracic_ratio() < 0.47 < 0.55 < big.cardiothoracic_ratio()


@pytest.mark.parametrize("name", ["cxr_22_quality.png", "cxr_23_quality.png", "cxr_24_quality.png"])
def test_quality_gate_rejects_bad_scans(name):
    work = _work(name)
    report = quality.assess(work, anatomy.segment(work), (768, 768))
    assert report.grade == "reject"


def test_quality_gate_passes_good_scan():
    work = _work("cxr_02_normal.png")
    assert quality.assess(work, anatomy.segment(work), (768, 768)).accepted


def test_normal_scan_exits_early(engine):
    analysis = engine.analyse_file(os.path.join(SAMPLES, "cxr_03_normal.png"))
    assert analysis.routing["early_exit"] and analysis.modules_activated == 0
    assert analysis.status == "no-finding"


def test_quality_hold_stops_pipeline(engine):
    analysis = engine.analyse_file(os.path.join(SAMPLES, "cxr_22_quality.png"))
    assert analysis.status == "quality-hold" and not analysis.findings
    assert "human review" in analysis.explanation["headline"].lower()


@pytest.mark.parametrize("name,label,side", [
    ("cxr_08_cardiomegaly.png", "Cardiomegaly", ""),
    ("cxr_10_effusion.png", "Effusion", "right"),
    ("cxr_11_effusion.png", "Effusion", "left"),
    ("cxr_13_consolidation.png", "Consolidation", "right"),
    ("cxr_16_pneumothorax.png", "Pneumothorax", "right"),
    ("cxr_17_pneumothorax.png", "Pneumothorax", "left"),
    ("cxr_20_nodule.png", "Nodule", "left"),
    ("cxr_21_mass.png", "Mass", "right"),
])
def test_findings_detected_and_localised(engine, name, label, side):
    analysis = engine.analyse_file(os.path.join(SAMPLES, name))
    hits = [f for f in analysis.findings if f.label == label and f.status == POSITIVE]
    assert hits, [(f.title, f.probability, f.status) for f in analysis.findings]
    assert hits[0].side == side
    assert hits[0].evidence and hits[0].limitations and 0 <= hits[0].confidence <= 1
    assert hits[0].bbox is not None or label == "Cardiomegaly"


def test_router_only_runs_needed_modules(engine):
    analysis = engine.analyse_file(os.path.join(SAMPLES, "cxr_09_cardiomegaly.png"))
    ran = [r["module"] for r in analysis.routing["routes"] if r["run"]]
    assert ran == ["Cardiac"]


def test_fixed_mode_runs_everything(engine):
    analysis = engine.analyse_file(os.path.join(SAMPLES, "cxr_04_normal.png"), mode="fixed")
    assert analysis.modules_activated == 4 and not analysis.routing["early_exit"]


def test_context_changes_thresholds():
    screen = screening.ScreenResult("t", {"cardiac": 0.25, "pleural": 0.0, "parenchymal": 0.0,
                                          "focal": 0.0})
    assert router.route(screen, "Emergency").groups == ["cardiac"]
    assert router.route(screen, "Screening camp").groups == []
    assert router.route(screen, "Teaching").activated == 4


def test_abstention_on_unstable_finding():
    finding = Finding("Effusion", "pleural", "Pleural", 0.52, sources={"measurement": 0.52},
                      uncertainty={"std": 0.25, "n": 5})
    uncertainty.score(finding)
    assert finding.status == UNCERTAIN and finding.confidence < uncertainty.ABSTAIN_BELOW


def test_single_deep_source_below_threshold_is_negative():
    finding = Finding("Edema", "parenchymal", "Parenchyma", 0.4, sources={"deep": 0.4},
                      uncertainty={"std": 0.01, "n": 5})
    assert uncertainty.score(finding).status == NEGATIVE


def test_temperature_fit_recovers_overconfidence():
    rng = np.random.default_rng(0)
    honest = rng.uniform(0.15, 0.85, 2000)                  # true event probabilities
    labels = (rng.uniform(size=honest.size) < honest).astype(int)
    overconfident = [uncertainty.calibrate(p, 1 / 3) for p in honest]   # a model 3x too sharp
    t = uncertainty.fit_temperature(overconfident, labels.tolist())
    assert 2.0 < t < 4.0                                     # recovers T close to 3


def test_explanation_escalates_when_unsure():
    unsure = Finding("Nodule", "focal", "Nodule", 0.5, status=UNCERTAIN)
    assert explain.escalate("brief", [unsure]) == "standard"
    assert explain.escalate("detailed", [unsure]) == "detailed"


def test_rag_retrieves_cited_passages():
    hits = default_kb().search("small nodule follow up CT", "Nodule", 2)
    assert hits and hits[0].finding == "Nodule" and "Fleischner" in " ".join(h.source for h in hits)


def test_answer_intents(engine):
    analysis = engine.analyse_file(os.path.join(SAMPLES, "cxr_10_effusion.png"))
    finding = next(f for f in analysis.findings if f.label == "Effusion" and f.status == POSITIVE)
    assert explain.answer("How sure are you?", finding)["intent"] == "confidence"
    assert explain.answer("What could mimic this?", finding)["intent"] == "mimic"
    assert "ultrasound" in explain.answer("What should I do next?", finding)["answer"].lower()


def test_llm_guard_rejects_invented_facts():
    guard = explain.LLMNarrator.check
    grounded = "Effusion - right: 97% (High). Confirm with ultrasound."
    assert guard("Right effusion, 97%, high confidence.", grounded, ["Effusion"]) is None
    assert "number" in guard("Right effusion of 450 ml.", grounded, ["Effusion"])
    assert "Pneumothorax" in guard("Effusion and a small pneumothorax.", grounded, ["Effusion"])


def test_audit_chain_detects_tampering(tmp_path):
    log = AuditLog(str(tmp_path / "audit.jsonl"))
    for i in range(5):
        log.record("event", {"i": i}, "doctor", "clinician")
    assert log.verify().intact and log.verify().entries == 5
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    lines[2] = lines[2].replace('"i": 2', '"i": 9')
    (tmp_path / "audit.jsonl").write_text("\n".join(lines) + "\n")
    report = log.verify()
    assert not report.intact and report.broken_at == 3
