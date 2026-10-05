"""Functional tests: end-to-end workflows, privacy, robustness, persistence, determinism.

Each test runs in its own temporary MEDSCAN_HOME; nothing touches real data.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import subprocess
import sys
import threading

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLES = os.path.join(ROOT, "samples")
sys.path.insert(0, ROOT)

from msx import accounts as accounts_mod  # noqa: E402
from msx import evaluation, imaging, prefs, report  # noqa: E402
from msx.audit import AuditLog  # noqa: E402
from msx.datastore import SIGNED, DataStore  # noqa: E402
from msx.findings import NEGATIVE, POSITIVE  # noqa: E402
from msx.learner import Learner  # noqa: E402
from msx.pipeline import AnalysisEngine  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("MEDSCAN_HOME", str(home))
    return home


@pytest.fixture(scope="module")
def engine():
    return AnalysisEngine(engine="builtin", tta=1)


def sample(name):
    return os.path.join(SAMPLES, name)


# ------------------------------------------------------------ workflows -----
def test_full_clinical_workflow(engine, isolated_home):
    """Analyse -> store -> blinded read -> decide -> sign -> audit -> learn -> reader study."""
    store, audit = DataStore(), AuditLog()
    analysis = engine.analyse_file(sample("cxr_12_effusion.png"), context="Emergency")
    store.save_analysis(analysis, "Site A", "Effusion;Cardiomegaly", "doctor")
    audit.record("scan analysed", {"status": analysis.status}, "doctor", "clinician", study=analysis.id)
    loaded, row = store.load(analysis.id)
    assert row["review_state"] == "open" and loaded.work.shape == (512, 512)
    store.record_first_read(analysis.id, ["Effusion"], 30000)
    for f in loaded.findings:
        if f.status == POSITIVE:
            store.record_decision(analysis.id, f.key, f.label, "accept", "doctor", "clinician")
            audit.record("finding accepted", {"finding": f.title}, "doctor", "clinician", study=analysis.id)
    final = store.final_labels(analysis.id, loaded)
    assert set(final) == {"Effusion", "Cardiomegaly"}
    store.sign_off(analysis.id, "doctor", final, 20000, report.draft(loaded)["text"])
    audit.record("report signed", {"final": final}, "doctor", "clinician", study=analysis.id)
    assert store.study(analysis.id)["review_state"] == SIGNED
    assert "IMPRESSION" in store.study(analysis.id)["report"]
    chain = audit.verify()
    assert chain.intact and chain.entries == 4
    study = evaluation.reader_study(store)
    assert study["studies"] == 1 and study["doctor_ai"]["sensitivity"] == 1.0
    assert study["doctor"]["sensitivity"] == 0.5


def test_quality_hold_workflow(engine):
    analysis = engine.analyse_file(sample("cxr_23_quality.png"))
    assert analysis.status == "quality-hold" and analysis.findings == []
    assert analysis.routing == {} and analysis.recommendations[0]["action"].startswith("Re-acquire")
    assert "Non-diagnostic" in report.draft(analysis)["text"]
    assert [s.status for s in analysis.stages][:2] == ["ok", "stopped"]


def test_every_phantom_class_detected_on_correct_side(engine):
    with open(os.path.join(SAMPLES, "labels.csv"), newline="") as handle:
        rows = list(csv.DictReader(handle))
    misses = []
    for row in rows:
        analysis = engine.analyse_file(sample(row["file"]))
        truth = {l for l in row["labels"].split(";") if l != "No Finding"}
        called = {f.label for f in analysis.findings if f.status == POSITIVE}
        if row["quality"] != "ok":
            if analysis.status != "quality-hold":
                misses.append((row["file"], "not held"))
            continue
        if not truth <= called:
            misses.append((row["file"], truth - called))
    assert not misses, misses


def test_context_changes_routing_not_findings(engine):
    emergency = engine.analyse_file(sample("cxr_10_effusion.png"), context="Emergency")
    camp = engine.analyse_file(sample("cxr_10_effusion.png"), context="Screening camp")
    teaching = engine.analyse_file(sample("cxr_10_effusion.png"), context="Teaching")
    assert emergency.routing["thresholds"]["route"] < camp.routing["thresholds"]["route"]
    assert teaching.modules_activated == 4
    pos = lambda a: {f.title for f in a.findings if f.status == POSITIVE}  # noqa: E731
    assert pos(emergency) == pos(camp) == {"Effusion - right"}
    assert emergency.explanation["depth"] in ("brief", "standard")
    assert teaching.explanation["depth"] == "detailed"


def test_adaptive_matches_fixed_accuracy(engine, tmp_path):
    folder = tmp_path / "set"
    folder.mkdir()
    names = ["cxr_01_normal.png", "cxr_03_normal.png", "cxr_08_cardiomegaly.png",
             "cxr_11_effusion.png", "cxr_17_pneumothorax.png", "cxr_19_nodule.png"]
    with open(os.path.join(SAMPLES, "labels.csv"), newline="") as handle:
        rows = {r["file"]: r for r in csv.DictReader(handle)}
    with open(folder / "labels.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[names[0]]))
        writer.writeheader()
        for n in names:
            os.symlink(sample(n), folder / n)
            writer.writerow(rows[n])
    result = evaluation.benchmark(str(folder), "builtin", simulate_runs=5)
    f, a = result["modes"]["fixed"], result["modes"]["adaptive"]
    assert a["any"] == f["any"]
    assert a["mean_modules"] < f["mean_modules"] and a["mean_calls"] < f["mean_calls"]


def test_evaluation_cli_writes_report(tmp_path):
    out = tmp_path / "r.json"
    folder = tmp_path / "set"
    folder.mkdir()
    for n in ("cxr_02_normal.png", "cxr_09_cardiomegaly.png"):
        os.symlink(sample(n), folder / n)
    (folder / "labels.csv").write_text("file,labels\ncxr_02_normal.png,No Finding\n"
                                       "cxr_09_cardiomegaly.png,Cardiomegaly\n")
    env = dict(os.environ, PYTHONPATH=ROOT)
    proc = subprocess.run([sys.executable, "-m", "msx.evaluation", str(folder), "--json", str(out)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert "adaptive vs fixed" in proc.stdout
    data = json.loads(out.read_text())
    assert data["images"] == 2 and "simulation" in data


# -------------------------------------------------------------- privacy -----
def test_no_patient_identifiers_reach_disk(engine, isolated_home):
    """The DICOM carries a fake name, ID, birth date, institution and physician."""
    store, audit = DataStore(), AuditLog()
    analysis = engine.analyse_file(sample("cxr_25_effusion_dicom.dcm"))
    store.save_analysis(analysis, "Site", "Effusion", "doctor")
    audit.record("scan analysed", analysis.scan, "doctor", "clinician", study=analysis.id)
    assert "PatientName" in analysis.scan["removed_tags"]
    forbidden = [b"PHANTOM", b"TG-0001", b"19700101", b"Synthetic Hospital", b"DOE^JANE"]
    for path in glob.glob(str(isolated_home / "**" / "*"), recursive=True):
        if os.path.isfile(path):
            blob = open(path, "rb").read()
            for word in forbidden:
                assert word not in blob, f"{word!r} found in {path}"


def test_pseudonym_links_same_patient_only():
    a = imaging.pseudonym("MRN-1")
    assert a == imaging.pseudonym("MRN-1") and a != imaging.pseudonym("MRN-2")


# ----------------------------------------------------------- robustness -----
def test_bad_inputs_are_rejected_cleanly(engine, tmp_path):
    with pytest.raises(ValueError):
        engine.analyse_file(str(tmp_path / "scan.gif"))
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"\x89PNG not really")
    with pytest.raises(Exception):
        engine.analyse_file(str(broken))


@pytest.mark.parametrize("kind", ["black", "white", "noise", "tiny"])
def test_non_chest_images_are_held_not_guessed(engine, tmp_path, kind):
    rng = np.random.default_rng(0)
    array = {"black": np.zeros((600, 600)), "white": np.full((600, 600), 255.0),
             "noise": rng.uniform(0, 255, (600, 600)), "tiny": rng.uniform(0, 255, (64, 64))}[kind]
    path = tmp_path / f"{kind}.png"
    Image.fromarray(array.astype(np.uint8)).save(path)
    analysis = engine.analyse_file(str(path))
    assert analysis.status == "quality-hold"
    assert not [f for f in analysis.findings if f.status == POSITIVE]


def test_formats_rgb_jpeg_16bit_and_large(engine, tmp_path):
    src = np.asarray(Image.open(sample("cxr_07_cardiomegaly.png")).convert("L"))
    Image.fromarray(src).convert("RGB").save(tmp_path / "rgb.jpg", quality=92)
    Image.fromarray((src.astype(np.uint16) * 257)).save(tmp_path / "deep.png")
    big = np.asarray(Image.fromarray(src).resize((2304, 2304)))
    Image.fromarray(big).save(tmp_path / "big.png")
    for name in ("rgb.jpg", "deep.png", "big.png"):
        analysis = engine.analyse_file(str(tmp_path / name))
        assert any(f.label == "Cardiomegaly" and f.status == POSITIVE for f in analysis.findings), name


def test_analysis_is_deterministic(engine):
    a = engine.analyse_file(sample("cxr_14_consolidation.png"))
    b = engine.analyse_file(sample("cxr_14_consolidation.png"))
    assert [(f.label, f.side, f.probability, f.status) for f in a.findings] == \
           [(f.label, f.side, f.probability, f.status) for f in b.findings]


# ---------------------------------------------------------- persistence -----
def test_store_is_thread_safe(engine, isolated_home):
    store = DataStore()
    analysis = engine.analyse_file(sample("cxr_02_normal.png"))
    store.save_analysis(analysis)
    errors = []

    def worker(n):
        try:
            for i in range(25):
                store.record_decision(analysis.id, f"K{n}-{i}", "Effusion", "accept", "u", "clinician")
                store.counts()
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(store.decisions(analysis.id)) == 150


def test_settings_round_trip_ignores_unknown_keys(isolated_home):
    s = prefs.load()
    s["tta"] = 3
    s["not_a_setting"] = 1
    prefs.save(s)
    reloaded = prefs.load()
    assert reloaded["tta"] == 3 and "not_a_setting" not in reloaded
    (isolated_home / "settings.json").write_text("{broken json")
    assert prefs.load()["tta"] == prefs.DEFAULTS["tta"]          # corrupt file -> defaults


def test_accounts_persist_and_never_store_passwords(isolated_home):
    accounts = accounts_mod.Accounts()
    accounts.add("tester", "secret123", "Dr. T")
    accounts.set_support_mode("tester", "guided")
    again = accounts_mod.Accounts()
    assert again.get("tester").support_mode == "guided"
    assert again.authenticate("tester", "secret123").username == "tester"
    assert b"secret123" not in (isolated_home / "accounts.json").read_bytes()


def test_audit_detects_deleted_middle_line(isolated_home):
    log = AuditLog()
    for i in range(4):
        log.record("e", {"i": i})
    lines = open(log.path).read().splitlines()
    del lines[1]
    open(log.path, "w").write("\n".join(lines) + "\n")
    report_ = log.verify()
    assert not report_.intact and report_.broken_at == 2


def test_learner_improves_with_feedback(engine, isolated_home):
    store = DataStore()
    with open(os.path.join(SAMPLES, "labels.csv"), newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        a = engine.analyse_file(sample(row["file"]))
        store.save_analysis(a, "", row["labels"])
    info = Learner().train(store)
    assert info["status"] == "trained" and info["cv_accuracy"] >= 0.9


@pytest.mark.skipif(not __import__("msx.screening", fromlist=["x"]).deep_available(),
                    reason="deep stack not installed")
def test_hybrid_engine_on_real_style_image():
    hybrid = AnalysisEngine(engine="hybrid", tta=1)
    analysis = hybrid.analyse_file(sample("cxr_07_cardiomegaly.png"))
    assert analysis.engine.startswith("densenet121")
    hit = [f for f in analysis.findings if f.label == "Cardiomegaly"]
    assert hit and "deep" in hit[0].sources


@pytest.mark.skipif(not __import__("msx.screening", fromlist=["x"]).deep_available(),
                    reason="deep stack not installed")
def test_hybrid_engine_keeps_phantom_results_and_rescues_failed_segmentation(tmp_path):
    """Hybrid (the default engine) must give the same calls as built-in where classical
    segmentation works, and fall back to the learned segmenter only where it fails."""
    hybrid = AnalysisEngine(engine="hybrid", tta=1)
    normal = hybrid.analyse_file(sample("cxr_01_normal.png"))
    assert "classical" in normal.stages[0].note
    assert not [f for f in normal.findings if f.status == POSITIVE]
    for name, label in (("cxr_10_effusion.png", "Effusion"), ("cxr_16_pneumothorax.png", "Pneumothorax")):
        hits = [f for f in hybrid.analyse_file(sample(name)).findings if f.status == POSITIVE]
        assert {f.label for f in hits} == {label}, name


def test_audit_export_survives_damaged_trail(isolated_home, tmp_path):
    log = AuditLog()
    log.record("a")
    log.record("b")
    lines = open(log.path).read().splitlines()
    lines[0] = lines[0].replace('"category"', '"categorY"')
    open(log.path, "w").write("\n".join(lines) + "\n")
    assert log.export_csv(str(tmp_path / "x.csv")) == 2
