"""UI: Model Training and Benchmark pages (admin workspace)."""

from __future__ import annotations

import csv
import json
import os

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from gui_helpers import pump, wait_until


@pytest.fixture
def training(qapp, services):
    from ui.pages.training_page import TrainingPage

    services.user = services.accounts.get("admin")
    page = TrainingPage(services)
    page.resize(1440, 900)
    page.show()
    page.refresh()
    return page


def test_train_from_folder_versions_and_rollback(qapp, training, services):
    from gui_helpers import SAMPLES

    training.use_db.setChecked(False)
    training.use_folder.setChecked(True)
    training.folder.setText(SAMPLES)                 # 25 phantoms -> enough labelled findings
    training.note.setText("first")
    QTest.mouseClick(training.train_button, Qt.MouseButton.LeftButton)
    assert wait_until(qapp, lambda: training.job and not training.job.running, 300)
    first = services.learner.info
    assert first["status"] == "trained" and first["note"] == "first"
    assert training.runs.rowCount() == 1 and training.runs.item(0, 8).text() == "✓"
    assert training.weights.items, "feature weights chart filled"
    training.l2.setValue(0.5)
    training.note.setText("second")
    training._train()
    assert wait_until(qapp, lambda: not training.job.running, 300)
    assert training.runs.rowCount() == 2 and services.learner.info["note"] == "second"
    row = next(r for r in range(2) if training.runs.item(r, 0).text() == first["version"])
    training.runs.selectRow(row)
    training._activate()
    assert services.learner.info["version"] == first["version"]
    assert any(e["action"] == "feedback model activated" for e in services.audit.entries(5))
    training._deactivate()
    assert not services.learner.ready and training.active_stats.values[0].text() == "—"
    assert training.weights.items == []


def test_training_with_no_source_is_refused(qapp, training):
    training.use_db.setChecked(False)
    training.use_folder.setChecked(False)
    training._train()
    assert "at least one source" in training.status.text() and training.job is None


def test_too_little_data_reports_need(qapp, training, services):
    training.use_db.setChecked(True)
    training.use_folder.setChecked(False)
    training._train()
    assert wait_until(qapp, lambda: not training.job.running, 60)
    assert "need" in training.status.text() and not services.learner.ready


def test_fit_and_apply_calibration_and_fairness(qapp, training, services, subset_folder):
    from msx import prefs

    training.folder.setText(str(subset_folder))
    training._fit()
    assert wait_until(qapp, lambda: not training.job.running, 300)
    assert training.fit and training.apply_thresholds.isEnabled()
    assert training.calib_grid.count() >= 4
    training._apply_thresholds()
    assert prefs.load()["subgroup_thresholds"] == training.fit["mitigation"]["thresholds"]
    if training.apply_temps.isEnabled():
        training._apply_temperatures()
        assert prefs.load()["temperatures"] == training.fit["temperatures"]


def test_missing_labels_fails_cleanly(qapp, training, tmp_path):
    training.use_db.setChecked(False)
    training.use_folder.setChecked(True)
    training.folder.setText(str(tmp_path))
    training._train()
    assert wait_until(qapp, lambda: not training.job.running, 60)
    assert "labels.csv" in training.status.text()
    assert training.train_button.isEnabled()


# ---------------------------------------------------------------- benchmark ----
@pytest.fixture
def bench(qapp, services):
    from ui.pages.benchmark_page import BenchmarkPage

    services.user = services.accounts.get("admin")
    page = BenchmarkPage(services)
    page.resize(1440, 900)
    page.show()
    page.refresh()
    return page


def run_bench(qapp, bench, folder, engine="builtin", fixed=True):
    bench.folder.setText(str(folder))
    bench.engine.setCurrentText(engine)
    bench.fixed.setChecked(fixed)
    bench.sim_runs.setValue(10)
    QTest.mouseClick(bench.run_button, Qt.MouseButton.LeftButton)
    assert wait_until(qapp, lambda: bench.job and not bench.job.running, 400)


def test_run_saves_and_shows_everything(qapp, bench, services, subset_folder):
    assert bench.list.count() == 0
    run_bench(qapp, bench, subset_folder)
    assert bench.list.count() == 1 and bench.report
    assert bench.stats.values[0].text() == "6"
    assert bench.metrics.rowCount() > 8
    assert bench.table.rowCount() == 6
    assert bench.auc_chart.categories and bench.cost_chart.series
    assert any(e["action"] == "benchmark finished" for e in services.audit.entries(5))
    bench.errors_only.setChecked(True)
    assert bench.table.rowCount() <= 6


def test_compare_two_runs_shows_deltas(qapp, bench, subset_folder):
    run_bench(qapp, bench, subset_folder)
    pump(qapp, 1100)                         # distinct run timestamps
    run_bench(qapp, bench, subset_folder, fixed=False)
    assert bench.list.count() == 2
    bench.compare.setCurrentIndex(2)          # the older run
    deltas = [bench.metrics.item(r, 3).text() for r in range(bench.metrics.rowCount())]
    assert any(d not in ("—",) for d in deltas)
    assert bench.metrics.item(0, 1).text() == "—"     # newest run had no fixed pipeline


def test_validation_messages(qapp, bench, tmp_path):
    bench.fixed.setChecked(False)
    bench.adaptive.setChecked(False)
    bench._run()
    assert "at least one pipeline" in bench.status.text()
    bench.adaptive.setChecked(True)
    bench.folder.setText(str(tmp_path))
    bench._run()
    assert "No labels.csv" in bench.status.text()
    bench._use_real()
    assert "Fetch real samples" in bench.status.text()


def test_exports_and_delete(qapp, bench, subset_folder, tmp_path, monkeypatch):
    run_bench(qapp, bench, subset_folder)
    out_csv, out_json = tmp_path / "img.csv", tmp_path / "run.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out_csv) if "CSV" in a[3] else str(out_json), "")))
    bench._export_csv()
    bench._export_json()
    assert len(list(csv.DictReader(open(out_csv)))) == 6
    assert json.load(open(out_json))["images"] == 6
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    bench._delete()
    assert bench.list.count() == 0 and bench.report is None


def test_evaluation_page_sees_benchmark_runs(qapp, bench, services, subset_folder):
    from ui.pages.evaluation_page import EvaluationPage

    run_bench(qapp, bench, subset_folder)
    evaluation = EvaluationPage(services)
    assert evaluation.report and evaluation.report["images"] == 6


def test_engines_page_links_and_columns(qapp, services):
    from ui.window import MainWindow

    services.user = services.accounts.get("admin")
    window = MainWindow(services)
    window.show()
    window.show_page("Engines")
    engines = window.pages["Engines"]
    engines._go("Benchmark")
    assert window.stack.currentWidget() is window.pages["Benchmark"]
    engines._go("Model Training")
    assert window.stack.currentWidget() is window.pages["Model Training"]
    window.close()
