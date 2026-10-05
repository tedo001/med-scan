"""UI: Analyse page - queue, clinical context, background worker, live pipeline, result."""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from gui_helpers import SAMPLES, pump, wait_until


def page(services):
    from ui.pages.analyse import AnalysePage

    p = AnalysePage(services)
    p.show()
    return p


def test_demo_samples_fill_queue_with_facts(qapp, services):
    p = page(services)
    p._demo()
    assert len(p.jobs) == 25 and p.table.rowCount() == 25
    assert "Analyse 25 scan(s)" == p.run_button.text()
    effusion = next(j for j in p.jobs if j["path"].endswith("cxr_10_effusion.png"))
    assert effusion["truth"] == "Effusion" and effusion["facts"]["sex"] in ("M", "F")
    p._demo()                                   # adding again must not duplicate
    assert len(p.jobs) == 25
    p._clear()
    assert p.jobs == [] and p.table.rowCount() == 0


def test_context_buttons_update_note(qapp, services):
    p = page(services)
    QTest.mouseClick(p.context.buttons["Emergency"], Qt.MouseButton.LeftButton)
    pump(qapp)
    assert p.context.value == "Emergency" and "0.20" in p.context_note.text()
    QTest.mouseClick(p.context.buttons["Teaching"], Qt.MouseButton.LeftButton)
    assert "detailed" in p.context_note.text()


def test_run_analyses_in_background_and_saves(qapp, services):
    p = page(services)
    p.add_files([os.path.join(SAMPLES, "cxr_07_cardiomegaly.png"),
                 os.path.join(SAMPLES, "cxr_22_quality.png")])
    opened = []
    services.open_study.connect(opened.append)
    QTest.mouseClick(p.run_button, Qt.MouseButton.LeftButton)
    assert not p.run_button.isEnabled()            # disabled while the worker runs
    assert wait_until(qapp, lambda: p.thread is None, 180)
    assert [j["status"] for j in p.jobs] == ["done", "held"]
    assert "Cardiomegaly" in p.jobs[0]["result"] and p.jobs[1]["result"] == "quality hold"
    assert services.store.counts()["total"] == 2
    assert all(chip.property("state") in ("ok", "skipped", "stopped") for chip in p.chips)
    assert p.open_button.isEnabled()
    QTest.mouseClick(p.open_button, Qt.MouseButton.LeftButton)
    assert opened == [p.last_id]
    actions = [e["action"] for e in services.audit.entries(10)]
    assert "analysis batch started" in actions and actions.count("scan analysed") == 2


def test_fixed_pipeline_option_runs_all_modules(qapp, services):
    p = page(services)
    p.add_files([os.path.join(SAMPLES, "cxr_02_normal.png")])
    p.mode.setCurrentIndex(1)
    p._run()
    assert wait_until(qapp, lambda: p.thread is None, 120)
    loaded, _ = services.store.load(p.last_id)
    assert loaded.mode == "fixed" and loaded.modules_activated == 4


def test_override_facts_are_applied(qapp, services):
    p = page(services)
    p.add_files([os.path.join(SAMPLES, "cxr_02_normal.png")])
    p.sex.setCurrentText("F")
    p.age.setValue(70)
    p.view.setCurrentText("AP")
    p._run()
    assert wait_until(qapp, lambda: p.thread is None, 120)
    loaded, _ = services.store.load(p.last_id)
    assert loaded.scan["sex"] == "F" and loaded.scan["age_band"] == "65+" and loaded.scan["view"] == "AP"


def test_unreadable_file_is_reported_not_fatal(qapp, services, tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    p = page(services)
    p.add_files([str(bad), os.path.join(SAMPLES, "cxr_02_normal.png")])
    p._run()
    assert wait_until(qapp, lambda: p.thread is None, 120)
    assert [j["status"] for j in p.jobs] == ["failed", "done"]
    assert any(e["action"] == "analysis failed" for e in services.audit.entries(10))


def test_quality_hold_leaves_no_stage_running(qapp, services):
    p = page(services)
    p.add_files([os.path.join(SAMPLES, "cxr_23_quality.png")])
    p._run()
    assert wait_until(qapp, lambda: p.thread is None, 120)
    assert [c.property("state") for c in p.chips][-1] == "skipped"
    assert "running" not in " ".join(c.time.text() for c in p.chips)
