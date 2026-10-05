"""UI: Model Training → "Train on an image dataset" (class folders or a .zip).

The DenseNet is replaced by a small fake feature extractor so the tests are fast
and need no PyTorch or network; the real model is covered in tests/functional.
"""

from __future__ import annotations

import os
import sys

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QFileDialog

from gui_helpers import wait_until

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_fixtures import FakeDeep, make_dataset, make_zip  # noqa: E402


@pytest.fixture
def training(qapp, services, monkeypatch):
    from msx import screening
    from ui.pages.training_page import TrainingPage

    monkeypatch.setattr(screening, "deep_available", lambda: True)
    monkeypatch.setattr(screening, "get_deep", lambda: FakeDeep())
    services.user = services.accounts.get("admin")
    page = TrainingPage(services)
    page.resize(1440, 900)
    page.show()
    page.refresh()
    return page


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(str(tmp_path / "kaggle"))


def _inspect(qapp, page, source):
    page.ds_source.setText(source)
    QTest.mouseClick(page.ds_inspect, Qt.MouseButton.LeftButton)
    assert wait_until(qapp, lambda: page.dataset is not None and not page.job.running, 60)


def test_inspect_folder_lists_classes_splits_and_mapping(qapp, training, dataset):
    _inspect(qapp, training, dataset)
    assert training.classes.rowCount() == 3
    mapping = training._mapping()
    assert mapping == {"Bacterial Pneumonia": "Consolidation", "Normal": "No Finding",
                       "Tuberculosis": "Skip"}
    assert set(training.split_boxes) == {"train", "test"}
    assert training.split_boxes["train"].isChecked() and not training.split_boxes["test"].isChecked()
    assert training.test_button.isEnabled()
    assert "48 images" in training.ds_summary.text()


def test_train_button_trains_versions_and_rolls_back(qapp, training, services, dataset):
    _inspect(qapp, training, dataset)
    training.ds_note.setText("first")
    QTest.mouseClick(training.head_button, Qt.MouseButton.LeftButton)
    assert wait_until(qapp, lambda: not training.job.running, 120)
    head = services.head
    assert head.ready and head.labels == ["Consolidation"]
    first = head.info["version"]
    assert head.info["n"] == 24 and "Tuberculosis" not in str(head.info["classes"])
    assert training.head_runs.rowCount() == 1 and training.head_runs.item(0, 5).text() == "✓"
    assert training.head_stats.values[2].text() not in ("—", "")
    assert services._engine is None, "the analyser is rebuilt with the new dataset model"
    assert services.engine.head is head
    assert any(e["action"] == "dataset model trained" for e in services.audit.entries(5))
    # the manifest the head was trained from is a normal labels.csv folder
    assert os.path.isfile(os.path.join(os.environ["MEDSCAN_HOME"], "datasets",
                                       "Lung_Disease_Dataset_train", "labels.csv"))

    training.ds_note.setText("second")
    training._train_head()
    assert wait_until(qapp, lambda: not training.job.running, 120)
    assert training.head_runs.rowCount() == 2
    row = next(r for r in range(2) if training.head_runs.item(r, 0).text() == first)
    training.head_runs.selectRow(row)
    training._activate_head()
    assert services.head.info["version"] == first
    training._deactivate_head()
    assert not services.head.ready and training.head_stats.values[0].text() == "—"


def test_zip_source_and_benchmark_set(qapp, training, services, dataset, tmp_path):
    from ui.pages.benchmark_page import BenchmarkPage

    bench = BenchmarkPage(services)
    archive = make_zip(str(tmp_path), dataset)
    _inspect(qapp, training, archive)
    assert training.dataset.kind == "zip" and training.classes.rowCount() == 3
    training._build_test()
    assert wait_until(qapp, lambda: not training.job.running, 60)
    folder = bench.folder.text()
    assert folder.endswith("lungs-disease-dataset-4-types_test")
    assert bench.engine.currentText() == "hybrid"
    with open(os.path.join(folder, "labels.csv"), encoding="utf-8") as handle:
        rows = handle.read().splitlines()
    assert len(rows) == 1 + 8                      # 4 Normal + 4 Pneumonia; TB skipped
    assert all(os.path.isfile(os.path.join(folder, r.split(",")[0])) for r in rows[1:])


def test_refusals_without_dataset_deep_or_mapping(qapp, training, dataset, monkeypatch):
    from msx import screening

    training._train_head()
    assert "Inspect" in training.ds_summary.text() and training.job is None
    _inspect(qapp, training, dataset)
    for box in training.map_boxes.values():
        box.setCurrentText("Skip")
    training._train_head()
    assert "two classes" in training.ds_summary.text()
    monkeypatch.setattr(screening, "deep_available", lambda: False)
    training._train_head()
    assert "PyTorch" in training.ds_summary.text()


def test_bad_source_fails_cleanly(qapp, training, tmp_path):
    training.ds_source.setText(str(tmp_path / "nothing-here"))
    training._inspect()
    assert wait_until(qapp, lambda: not training.job.running, 30)
    assert "not a folder" in training.status.text() and training.dataset is None


def test_folder_and_zip_pickers(qapp, training, dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: dataset))
    training._choose_ds_folder()
    assert wait_until(qapp, lambda: training.dataset is not None and not training.job.running, 60)
    archive = make_zip(str(tmp_path), dataset)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (archive, "")))
    training._choose_ds_zip()
    assert wait_until(qapp, lambda: training.dataset.kind == "zip" and not training.job.running, 60)


def test_engines_page_shows_dataset_model(qapp, services):
    from ui.pages.engines_page import EnginesPage

    page = EnginesPage(services)
    page.refresh()
    from PyQt6.QtWidgets import QTableWidget
    table = page.findChildren(QTableWidget)[0]
    rows = [table.item(r, 0).text() for r in range(table.rowCount())]
    assert "5c Dataset model" in rows
    r = rows.index("5c Dataset model")
    assert table.item(r, 2).text() == "not trained"
