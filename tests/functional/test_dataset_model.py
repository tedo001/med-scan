"""Functional: importing an image dataset and the dataset model (transfer-learning head)."""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_fixtures import SAMPLES, FakeDeep, make_dataset, make_zip  # noqa: E402

from msx import datasets, deephead, screening  # noqa: E402
from msx.pipeline import AnalysisEngine  # noqa: E402

DEEP = pytest.mark.skipif(not screening.deep_available(), reason="deep stack not installed")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDSCAN_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(str(tmp_path / "kaggle"))


def test_default_mapping_of_kaggle_class_names():
    assert datasets.default_label("Bacterial Pneumonia") == "Consolidation"
    assert datasets.default_label("Viral Pneumonia") == "Consolidation"
    assert datasets.default_label("Corona Virus Disease") == "Consolidation"
    assert datasets.default_label("COVID-19") == "Consolidation"
    assert datasets.default_label("Normal") == "No Finding"
    assert datasets.default_label("Tuberculosis") == datasets.SKIP
    assert datasets.default_label("TB") == datasets.SKIP
    assert datasets.default_label("Pleural_Effusion") == "Effusion"
    assert datasets.default_label("Tablets") == datasets.SKIP      # "tb" only as a whole word


@pytest.mark.parametrize("as_zip", [False, True])
def test_inspect_folder_and_zip_agree(dataset, tmp_path, as_zip):
    source = make_zip(str(tmp_path), dataset) if as_zip else dataset
    info = datasets.inspect(source)
    assert info.kind == ("zip" if as_zip else "class-folders")
    assert info.total == 48 and info.splits == ["test", "train"]
    assert info.classes["Normal"] == {"train": 12, "test": 4}


def test_manifest_is_balanced_skips_and_is_readable(dataset, home):
    from msx import evaluation

    info = datasets.inspect(dataset)
    folder = datasets.build_manifest(info, info.mapping(), ["train"], 5, "kaggle train")
    assert folder == os.path.join(str(home), "datasets", "kaggle_train")
    with open(os.path.join(folder, "labels.csv"), encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert sorted(r["labels"] for r in rows) == ["Consolidation"] * 5 + ["No Finding"] * 5
    assert {r["source_class"] for r in rows} == {"Normal", "Bacterial Pneumonia"}
    assert all("_train_" in r["file"] for r in rows), "only the chosen split"
    loaded = evaluation.load_labels(folder)
    assert len(loaded) == 10 and all(os.path.isfile(r["path"]) for r in loaded)
    again = datasets.build_manifest(info, info.mapping(), ["train"], 5, "kaggle train")
    with open(os.path.join(again, "labels.csv"), encoding="utf-8") as handle:
        assert [r["file"] for r in csv.DictReader(handle)] == [r["file"] for r in rows], "seeded"


def test_zip_extracts_only_sampled_images(dataset, tmp_path):
    info = datasets.inspect(make_zip(str(tmp_path), dataset))
    folder = datasets.build_manifest(info, info.mapping(), ["test"], 3, "z")
    assert len(os.listdir(os.path.join(folder, "images"))) == 6


def test_errors_are_clear(tmp_path, dataset):
    with pytest.raises(FileNotFoundError):
        datasets.inspect(str(tmp_path / "missing"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        datasets.inspect(str(tmp_path / "empty"))
    info = datasets.inspect(dataset)
    with pytest.raises(ValueError):
        datasets.build_manifest(info, {c: datasets.SKIP for c in info.classes}, ["train"], 5, "x")


def _trained_head(dataset, per_class=12):
    info = datasets.inspect(dataset)
    folder = datasets.build_manifest(info, info.mapping(), ["train"], per_class, "train")
    X, y, _ = deephead.embed_folder(folder, FakeDeep())
    head = deephead.DatasetHead()
    return head, head.train(X, y, note="t", source="kaggle")


def test_head_trains_predicts_persists_and_rolls_back(dataset, home):
    head, info = _trained_head(dataset)
    assert info["status"] == "trained" and head.ready and head.labels == ["Consolidation"]
    assert info["per_label"]["Consolidation"]["cv_auc"] >= 0.9
    test = datasets.build_manifest(datasets.inspect(dataset), datasets.inspect(dataset).mapping(),
                                   ["test"], 4, "test")
    X, y, _ = deephead.embed_folder(test, FakeDeep())
    probs = [head.predict(x) for x in X]
    assert all(ok for _, ok, _ in probs), "held-out images of the same kind are in distribution"
    assert deephead._auc([p["Consolidation"] for p, _, _ in probs],
                         [l == "Consolidation" for l in y]) >= 0.9
    noise = np.random.default_rng(1).uniform(0, 1, 1024)
    assert head.predict(noise)[1] is False, "very different images are out of distribution"
    reloaded = deephead.DatasetHead()
    assert reloaded.ready and reloaded.info["version"] == info["version"]
    assert len(head.history()) == 1
    head.reset()
    assert not deephead.DatasetHead().ready
    head.activate(info["version"])
    assert deephead.DatasetHead().ready


def test_head_needs_enough_data(home):
    head = deephead.DatasetHead()
    info = head.train(np.zeros((10, 4)), ["No Finding"] * 5 + ["Consolidation"] * 5)
    assert "need" in info["status"] and not head.ready
    info = head.train(np.random.default_rng(0).normal(size=(30, 4)), ["No Finding"] * 30)
    assert "need" in info["status"] and not head.ready


def test_head_only_corroborates():
    assert deephead.fuse_head(0.10, 0.99) == 0.10, "never creates a finding on its own"
    assert deephead.fuse_head(0.40, 0.90) > 0.40, "raises a finding the measurements suggest"
    assert deephead.fuse_head(0.80, 0.05) < 0.80, "can always argue against"
    assert abs(deephead.fuse_head(0.6, 0.5) - 0.6) < 1e-9, "0.5 is no evidence"


def test_no_dataset_model_means_original_behaviour(home):
    """Untrained head: the analysis is exactly what it was before this feature."""
    from ui.services import Services  # noqa: F401 - the app wires the head in this way

    base = AnalysisEngine(engine="builtin", tta=1)
    wired = AnalysisEngine(engine="builtin", tta=1, head=deephead.DatasetHead())
    for name in ("cxr_01_normal.png", "cxr_13_consolidation.png", "cxr_17_pneumothorax.png"):
        a = base.analyse_file(os.path.join(SAMPLES, name))
        b = wired.analyse_file(os.path.join(SAMPLES, name))
        assert [(f.label, f.probability) for f in a.findings] == \
               [(f.label, f.probability) for f in b.findings]
        assert b.screen["head_note"] == "" and "dataset model" not in str(b.findings)


@DEEP
def test_dataset_model_end_to_end_with_densenet(dataset, home):
    deep = screening.get_deep()
    if deep is None:
        pytest.skip("DenseNet weights unavailable")
    info = datasets.inspect(dataset)
    folder = datasets.build_manifest(info, info.mapping(), ["train"], 10, "train")
    X, y, _ = deephead.embed_folder(folder, deep)
    assert X.shape == (20, 1024)
    head = deephead.DatasetHead()
    assert head.train(X, y)["status"] == "trained"
    engine = AnalysisEngine(engine="hybrid", tta=1, head=head)
    analysis = engine.analyse_file(os.path.join(SAMPLES, "cxr_13_consolidation.png"))
    assert "dataset model" in analysis.screen["head_note"]
    hit = [f for f in analysis.findings if "dataset model" in f.sources]
    assert hit and hit[0].label == "Consolidation"
    assert any("Dataset model" in e for e in hit[0].evidence)
