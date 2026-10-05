"""Datastore, accounts, evaluation and the desktop app (offscreen)."""

from __future__ import annotations

import os

import pytest

from msx import accounts as accounts_mod, evaluation
from msx.datastore import SIGNED, DataStore
from msx.findings import POSITIVE
from msx.pipeline import AnalysisEngine

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")


def test_accounts_hash_and_lockout(tmp_path):
    store = accounts_mod.Accounts(str(tmp_path / "accounts.json"))
    assert store.authenticate("doctor", "medscan").role == accounts_mod.CLINICIAN
    raw = (tmp_path / "accounts.json").read_text()
    assert "medscan" not in raw and "techgium" not in raw
    with pytest.raises(accounts_mod.AuthError):
        store.authenticate("doctor", "medscan", role=accounts_mod.ADMIN)
    for _ in range(accounts_mod.LOCK_AFTER):
        with pytest.raises(accounts_mod.AuthError):
            store.authenticate("doctor", "wrong")
    with pytest.raises(accounts_mod.AuthError, match="locked"):
        store.authenticate("doctor", "medscan")


def test_store_round_trip_and_final_read(tmp_path):
    store = DataStore(str(tmp_path / "db.sqlite"))
    analysis = AnalysisEngine(engine="builtin", tta=1).analyse_file(
        os.path.join(SAMPLES, "cxr_12_effusion.png"))
    store.save_analysis(analysis, "Site A", "Effusion;Cardiomegaly", "doctor")
    loaded, row = store.load(analysis.id)
    assert loaded.work is not None and loaded.work.shape == (512, 512)
    assert [f.label for f in loaded.findings] == [f.label for f in analysis.findings]
    positives = [f for f in loaded.findings if f.status == POSITIVE]
    store.record_first_read(analysis.id, ["Effusion"], 30000)
    store.record_decision(analysis.id, positives[0].key, positives[0].label, "accept", "doctor", "clinician")
    store.record_decision(analysis.id, positives[1].key, positives[1].label, "correct", "doctor",
                          "clinician", "Mass")
    store.record_decision(analysis.id, "ADD-Nodule", "Nodule", "add", "doctor", "clinician")
    final = store.final_labels(analysis.id, loaded)
    assert positives[0].label in final and "Mass" in final and "Nodule" in final
    store.sign_off(analysis.id, "doctor", final, 20000)
    assert store.study(analysis.id)["review_state"] == SIGNED
    study = evaluation.reader_study(store)
    assert study["studies"] == 1 and study["doctor"]["sensitivity"] == 0.5


def test_auc_and_rates():
    assert evaluation.auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert evaluation.auc([0.1, 0.9], [1, 0]) == 0.0
    r = evaluation.rates([1, 0, 1, 0], [1, 1, 0, 0])
    assert r["sensitivity"] == 0.5 and r["specificity"] == 0.5


def test_app_builds_every_page(tmp_path):
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication

    import medscan
    from ui.services import Services
    from ui.window import MainWindow

    app = QApplication.instance() or medscan.create_application(["test"])
    services = Services(DataStore(str(tmp_path / "db.sqlite")),
                        accounts=accounts_mod.Accounts(str(tmp_path / "acc.json")))
    services.settings["engine"] = "builtin"
    services.user = services.accounts.get("doctor")
    analysis = services.engine.analyse_file(os.path.join(SAMPLES, "cxr_16_pneumothorax.png"))
    services.store.save_analysis(analysis, "Site", "Pneumothorax", "doctor")
    for username in ("doctor", "admin"):
        services.user = services.accounts.get(username)
        window = MainWindow(services)
        for name in window.pages:
            window.show_page(name)
            app.processEvents()
        if username == "doctor":
            services.settings["blinded_first_read"] = False
            window.pages["Review"].select(analysis.id)
            assert window.pages["Review"].cards, "finding cards rendered"
        window.close()
