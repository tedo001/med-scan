"""Fixtures for the UI tests: an offscreen QApplication and an isolated MEDSCAN.

Every test gets its own temporary MEDSCAN_HOME (database, audit trail, settings,
accounts, learner), so nothing a test does can reach the operator's real data
or another test. The engine is forced to the built-in analyser with one
test-time augmentation so the suite is fast and needs no network.
"""

from __future__ import annotations

import csv
import os
import shutil

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from gui_helpers import SAMPLES, labels, pump, wait_until  # noqa: F401


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication

    import medscan

    app = QApplication.instance() or medscan.create_application(["medscan-ui-tests"])
    yield app


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "medscan-home"
    home.mkdir()
    monkeypatch.setenv("MEDSCAN_HOME", str(home))
    return home


@pytest.fixture
def services(qapp, isolated_home):
    from ui.services import Services

    s = Services()
    s.settings.update(engine="builtin", tta=1)
    s._engine = None
    s.user = s.accounts.get("doctor")
    return s


SEED = ("cxr_01_normal.png", "cxr_07_cardiomegaly.png", "cxr_10_effusion.png",
        "cxr_12_effusion.png", "cxr_16_pneumothorax.png", "cxr_20_nodule.png",
        "cxr_22_quality.png")


@pytest.fixture
def seeded(services):
    """A handful of analysed studies covering normal, findings and a quality hold."""
    from msx import imaging

    truth = labels()
    ids = {}
    for name in SEED:
        row = truth[name]
        analysis = services.engine.analyse_file(
            os.path.join(SAMPLES, name), context="Routine OPD", sex=row["sex"],
            age_band=imaging.age_band(float(row["age"])))
        services.store.save_analysis(analysis, row["site"], row["labels"], "doctor")
        ids[name] = analysis.id
    return ids


@pytest.fixture
def subset_folder(tmp_path):
    """A small labelled folder (copies of phantoms) for fast benchmark runs."""
    folder = tmp_path / "subset"
    folder.mkdir()
    truth = labels()
    names = ("cxr_01_normal.png", "cxr_02_normal.png", "cxr_07_cardiomegaly.png",
             "cxr_10_effusion.png", "cxr_16_pneumothorax.png", "cxr_23_quality.png")
    with open(folder / "labels.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(truth.values()))))
        writer.writeheader()
        for name in names:
            shutil.copy(os.path.join(SAMPLES, name), folder / name)
            writer.writerow(truth[name])
    return folder
