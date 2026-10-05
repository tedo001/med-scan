"""UI: admin pages (audit, bias, engines, accounts, settings), Home, shared widgets, visual sweep."""

from __future__ import annotations

import os

import numpy as np
import pytest
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QFileDialog, QInputDialog, QLabel, QLineEdit, QMessageBox

from gui_helpers import pump

SHOTS = os.environ.get("MEDSCAN_UI_SHOTS")       # set to a folder to keep the visual-sweep images


def labels_of(widget):
    return " ".join(l.text() for l in widget.findChildren(QLabel))


def test_audit_page_verifies_detects_tampering_and_exports(qapp, services, seeded, tmp_path, monkeypatch):
    from ui.pages.audit_page import AuditPage

    for i in range(3):
        services.log("test event", {"i": i})
    page = AuditPage(services)
    page.show()
    page.refresh()
    assert page.table.rowCount() == 3
    assert page.table.item(0, 2).text() == "test event"            # newest first
    assert "CHAIN INTACT" in labels_of(page.chain)
    page.search.setText("zzz-no-such-entry")
    assert page.table.rowCount() == 0
    page.search.clear()
    # tamper with one line of the (temporary) trail - the page must say so
    path = services.audit.path
    lines = open(path, encoding="utf-8").read().splitlines()
    lines[0] = lines[0].replace('"category"', '"categorY"', 1)
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    page._verify(log=False)
    assert "CHAIN BROKEN" in labels_of(page.chain)
    target = tmp_path / "audit.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), "")))
    page._export()
    assert target.exists() and target.stat().st_size > 0


def test_bias_page_renders_tables_and_records_review(qapp, services, seeded):
    from ui.pages.bias_page import BiasPage

    page = BiasPage(services)
    page.show()
    page.refresh()
    assert page.grid.count() == 4                                   # sex, age, site, view
    page._record()
    assert services.audit.entries(1)[0]["action"] == "bias review"


def test_engines_page_lists_all_stages(qapp, services):
    from ui.pages.engines_page import EnginesPage

    page = EnginesPage(services)
    page.show()
    page.refresh()
    text = labels_of(page)
    from PyQt6.QtWidgets import QTableWidget

    table = page.status.findChildren(QTableWidget)[0]
    stages = {table.item(r, 0).text() for r in range(table.rowCount())}
    for stage in ("1 Preprocess", "2 Quality", "4 Route", "5b Learning", "7b Recommend", "7c Report"):
        assert stage in stages
    assert "Emergency" in text


def test_accounts_create_reset_remove(qapp, services, monkeypatch):
    from ui.pages.accounts_page import AccountsPage

    services.user = services.accounts.get("admin")
    page = AccountsPage(services)
    page.show()
    page.refresh()
    before = page.table.rowCount()
    page.username.setText("rad2")
    page.display.setText("Dr. Test")
    page.password.setText("short")
    page._create()
    assert "at least 6" in page.note.text()
    page.password.setText("longenough")
    page._create()
    assert page.table.rowCount() == before + 1
    row = next(r for r in range(page.table.rowCount()) if page.table.item(r, 0).text() == "rad2")
    page.table.selectRow(row)
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("newpassword", True)))
    page._reset()
    assert services.accounts.authenticate("rad2", "newpassword").username == "rad2"
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    page.table.selectRow(row)
    page._remove()
    assert services.accounts.get("rad2") is None


def test_admin_cannot_remove_self(qapp, services, monkeypatch):
    from ui.pages.accounts_page import AccountsPage

    services.user = services.accounts.get("admin")
    page = AccountsPage(services)
    page.refresh()
    row = next(r for r in range(page.table.rowCount()) if page.table.item(r, 0).text() == "admin")
    page.table.selectRow(row)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    page._remove()
    assert services.accounts.get("admin") is not None


def test_settings_save_persists_and_rebuilds_engine(qapp, services):
    from msx import prefs
    from ui.pages.settings_page import SettingsPage

    page = SettingsPage(services)
    page.show()
    page.refresh()
    engine_before = services.engine
    page.tta.setValue(2)
    page.site.setText("Test Site")
    page.blinded.setChecked(False)
    page._save()
    stored = prefs.load()
    assert stored["tta"] == 2 and stored["site"] == "Test Site" and stored["blinded_first_read"] is False
    assert services.engine is not engine_before


def test_settings_llm_test_reports_unreachable(qapp, services):
    from ui.pages.settings_page import SettingsPage

    page = SettingsPage(services)
    page.refresh()
    page.llm_url.setText("http://127.0.0.1:9")            # nothing listens here
    page._test()
    assert "not reachable" in page.llm_state.text()


def test_settings_wipe_keeps_audit(qapp, services, seeded, monkeypatch):
    from ui.pages.settings_page import SettingsPage

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    page = SettingsPage(services)
    page._wipe()
    assert services.store.counts()["total"] == 0
    assert services.audit.verify().intact
    assert services.audit.entries(1)[0]["action"] == "all studies deleted"


def test_home_page_stats_and_recent(qapp, services, seeded):
    from ui.pages.home import HomePage

    page = HomePage(services)
    page.show()
    page.refresh()
    assert page.stats.values[0].text() == str(len(seeded))
    assert "Dr." in page.greeting.text()


# ------------------------------------------------------------------ widgets -----
def test_charts_paint_empty_and_with_data(qapp):
    from ui.components import (ColumnChart, Donut, HBarChart, ProbBar, RadarChart, TrendChart)

    charts = [HBarChart(), Donut(), ColumnChart(), TrendChart(), RadarChart(), ProbBar(0.7)]
    for chart in charts:
        chart.resize(400, 260)
        assert not chart.grab().isNull()                    # empty state paints
    charts[0].set([("a", 3), ("b", 1)])
    charts[1].set([("x", 2, "#123456"), ("y", 0, "#654321")], "2")
    charts[2].set(["c1", "c2"], [("s", [1, 2], "#123456")])
    charts[3].set(["a", "b", "c"], [("s", [1, 3, 2], "#123456", "solid", "left"),
                                    ("t", [5, 6, 7], "#999999", "dot", "right")], average=2)
    charts[4].set(["p", "q", "r"], [("s", [1, 2, 3], "#123456", "solid")])
    for chart in charts:
        assert not chart.grab().isNull()


def test_viewer_handles_missing_heat_and_mask(qapp):
    from ui.components import XrayViewer

    viewer = XrayViewer()
    viewer.resize(500, 500)
    assert not viewer.grab().isNull()                      # "No scan selected"
    viewer.set_scan(np.random.rand(512, 512).astype(np.float32))
    viewer.set_overlays([{"key": "F1", "bbox": (10, 10, 60, 60), "text": "x"}], [(0, 0, 50, 0, "#38BDF8")])
    viewer.blind = True
    assert not viewer.grab().isNull()


def test_visual_sweep_every_page_renders_non_blank(qapp, services, seeded):
    """Render every page of both workspaces at 1440x900; each must have real content."""
    from ui.window import MainWindow

    for username in ("doctor", "admin"):
        services.user = services.accounts.get(username)
        window = MainWindow(services)
        window.resize(QSize(1440, 900))
        window.show()
        for name, page in window.pages.items():
            window.show_page(name)
            pump(qapp, 60)
            image = window.grab().toImage()
            # the window must fit a 1440 x 900 laptop screen on every page
            assert image.width() == 1440 and image.height() == 900, \
                f"{username}/{name} forces the window to {image.width()}x{image.height()}"
            colours = {image.pixel(x, y) for x in range(0, 1440, 37) for y in range(120, 900, 41)}
            assert len(colours) > 8, f"{username}/{name} looks blank"
            if SHOTS:
                os.makedirs(SHOTS, exist_ok=True)
                image.save(os.path.join(SHOTS, f"{username}-{name.replace(' ', '_')}.png"))
        window.close()
