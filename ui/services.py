"""What every page shares: settings, the database, the audit trail, the engine, the user."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from msx import accounts as accounts_mod, explain, prefs, screening
from msx.audit import AuditLog
from msx.datastore import DataStore
from msx.knowledge import default_kb
from msx.learner import Learner
from msx.pipeline import AnalysisEngine


class Services(QObject):
    studies_changed = pyqtSignal()
    settings_changed = pyqtSignal()
    open_study = pyqtSignal(str)

    def __init__(self, store: Optional[DataStore] = None, audit: Optional[AuditLog] = None,
                 accounts: Optional[accounts_mod.Accounts] = None):
        super().__init__()
        self.settings = prefs.load()
        self.store = store or DataStore()
        self.audit = audit or AuditLog()
        self.accounts = accounts or accounts_mod.Accounts()
        self.kb = default_kb()
        self.user: Optional[accounts_mod.Account] = None
        self._engine: Optional[AnalysisEngine] = None
        self.learner = Learner()

    # -- the engine is rebuilt when settings change --------------------------
    @property
    def narrator(self) -> explain.LLMNarrator:
        s = self.settings
        return explain.LLMNarrator(s["llm_url"], s["llm_model"], bool(s["llm_enabled"]))

    @property
    def engine(self) -> AnalysisEngine:
        if self._engine is None:
            s = self.settings
            self._engine = AnalysisEngine(engine=s["engine"], tta=int(s["tta"]),
                                          temperatures=s["temperatures"],
                                          route_overrides=s["route_overrides"],
                                          narrator=self.narrator,
                                          positive_at=float(s["positive_at"]),
                                          subgroup_thresholds=s["subgroup_thresholds"],
                                          learner=self.learner)
        return self._engine

    def save_settings(self) -> None:
        prefs.save(self.settings)
        self._engine = None
        self.settings_changed.emit()

    def retrain(self) -> dict:
        """Retrain the feedback model from every decision and ground truth so far."""
        info = self.learner.train(self.store)
        self.log("feedback model retrained", {k: info.get(k) for k in
                                              ("n", "status", "cv_accuracy", "cv_brier")},
                 category="system")
        return info

    def engine_status(self) -> str:
        if self.settings["engine"] == "hybrid" and screening.deep_available():
            return "densenet121 + measure"
        return "built-in analyser"

    # -- audit with the signed-in actor --------------------------------------
    def log(self, action: str, detail=None, study: str = "", category: str = "functionality"):
        user = self.user
        return self.audit.record(action, detail or {}, user.username if user else "system",
                                 user.role if user else "", category, study)

    @property
    def is_admin(self) -> bool:
        return bool(self.user and self.user.role == accounts_mod.ADMIN)
