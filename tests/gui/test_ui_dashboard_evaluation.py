"""UI: Dashboard (filters, trend, radar, export) and Evaluation (benchmark, simulation, bias, ML)."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QFileDialog, QLabel

from gui_helpers import pump, wait_until


@pytest.fixture
def dashboard(qapp, services, seeded):
    from ui.pages.dashboard import DashboardPage

    # spread the studies over time so period filters have something to cut
    for i, study in enumerate(seeded.values()):
        when = datetime.now() - timedelta(days=[1, 5, 20, 40, 100, 200, 300][i])
        services.store.set_created(study, when.isoformat(timespec="seconds"))
    page = DashboardPage(services)
    page.resize(1440, 900)
    page.show()
    page.refresh()
    pump(qapp)
    return page


def test_period_buttons_filter_studies(qapp, dashboard):
    assert len(dashboard.rows) == 4                      # last 90 days: 1, 5, 20, 40 days ago
    QTest.mouseClick(dashboard.period.buttons["Last 30 days"], Qt.MouseButton.LeftButton)
    pump(qapp)
    assert len(dashboard.rows) == 3
    QTest.mouseClick(dashboard.period.buttons["12 months"], Qt.MouseButton.LeftButton)
    pump(qapp)
    assert len(dashboard.rows) == 7
    assert dashboard.stats.values[0].text() == "7"
    assert "12 months" in dashboard.caption.text()


def test_site_and_finding_filters_and_clear(qapp, dashboard):
    QTest.mouseClick(dashboard.period.buttons["12 months"], Qt.MouseButton.LeftButton)
    dashboard.finding.setCurrentText("Effusion")
    pump(qapp)
    assert 1 <= len(dashboard.rows) < 7
    site = dashboard.site.itemText(1)
    dashboard.finding.setCurrentIndex(0)
    dashboard.site.setCurrentText(site)
    pump(qapp)
    assert all(r["site"] == site for r in dashboard.rows)
    dashboard._clear()
    assert dashboard.site.currentIndex() == 0 and dashboard.finding.currentIndex() == 0
    assert dashboard.period.value == "Last 90 days"


def test_trend_ranges_and_modes(qapp, dashboard):
    for name, buckets in (("Today", 24), ("Week", 7), ("Month", 30), ("Year", 12)):
        QTest.mouseClick(dashboard.trend_range.buttons[name], Qt.MouseButton.LeftButton)
        pump(qapp)
        assert len(dashboard.trend.categories) == buckets, name
    QTest.mouseClick(dashboard.trend_range.buttons["All"], Qt.MouseButton.LeftButton)
    assert len(dashboard.trend.categories) >= 10
    QTest.mouseClick(dashboard.trend_mode.buttons["▮ Bar"], Qt.MouseButton.LeftButton)
    assert dashboard.trend.mode == "bar"
    assert dashboard.trend.grab().width() > 0
    QTest.mouseClick(dashboard.trend_mode.buttons["↗ Line"], Qt.MouseButton.LeftButton)
    assert dashboard.trend.mode == "line"
    total = sum(dashboard.trend.series[2][1])
    assert total >= 1


def test_radar_profiles(qapp, dashboard):
    for kind, n in (("Findings", 8), ("Lung zones", 6), ("Modules", 4)):
        QTest.mouseClick(dashboard.profile_kind.buttons[kind], Qt.MouseButton.LeftButton)
        pump(qapp)
        assert len(dashboard.radar.categories) == n
        assert dashboard.radar.grab().width() > 0


def test_export_csv(dashboard, tmp_path, monkeypatch):
    target = tmp_path / "out.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), "")))
    dashboard._export()
    rows = list(csv.DictReader(open(target, encoding="utf-8")))
    assert len(rows) == len(dashboard.rows) and "top_label" in rows[0]


def test_recent_table_double_click_opens_study(qapp, dashboard, services):
    opened = []
    services.open_study.connect(opened.append)
    dashboard.table.cellDoubleClicked.emit(0, 0)
    assert opened == [dashboard.table.item(0, 0).text()]


# --------------------------------------------------------------- evaluation ----
@pytest.fixture
def evaluation(qapp, services):
    from ui.pages.evaluation_page import EvaluationPage

    page = EvaluationPage(services)
    page.resize(1440, 900)
    page.show()
    return page


def test_benchmark_runs_in_background_and_renders_all_sections(qapp, evaluation, services, subset_folder):
    evaluation.folder.setText(str(subset_folder))
    evaluation.engine.setCurrentText("builtin")
    QTest.mouseClick(evaluation.run_button, Qt.MouseButton.LeftButton)
    assert wait_until(qapp, lambda: evaluation.thread is None, 300)
    report = evaluation.report
    assert report and set(report["modes"]) == {"fixed", "adaptive"}
    assert report["simulation"]["clean"]["readers"]
    assert evaluation.bench_grid.count() > 0 and evaluation.sim_grid.count() > 0
    assert evaluation.bias_grid.count() > 0
    assert evaluation.calibrate.isEnabled()
    assert any(e["action"] == "benchmark finished" for e in services.audit.entries(10))


def test_calibrate_and_apply_bias_persist_settings(qapp, evaluation, services, subset_folder):
    from msx import evaluation as ev
    from msx import prefs

    report = ev.benchmark(str(subset_folder), "builtin", simulate_runs=10)
    evaluation.report = report
    evaluation._show_bench(report)
    evaluation._calibrate()
    assert isinstance(prefs.load()["temperatures"], dict)
    evaluation._apply_bias()
    assert prefs.load()["subgroup_thresholds"] == evaluation.mitigation["thresholds"]


def test_retrain_button_shows_learner_state(qapp, evaluation, services, seeded):
    evaluation._retrain()
    labels = " ".join(l.text() for l in evaluation.learn_card.findChildren(QLabel))
    assert "Labelled findings" in labels


def test_reader_study_message_when_empty(qapp, evaluation):
    evaluation.refresh()
    labels = " ".join(l.text() for l in evaluation.reader.findChildren(QLabel))
    assert "No eligible studies" in labels
