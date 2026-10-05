"""Settings - engine, routing, explanation, privacy and data."""

from __future__ import annotations

import os

from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QLineEdit,
                             QMessageBox, QSpinBox)

from msx import paths, screening
from msx.router import CONTEXTS

from ..components import Card, Page, button, hbox, label


class SettingsPage(Page):
    def __init__(self, services):
        super().__init__("Settings", paths.data_directory())
        self.services = services
        s = services.settings
        self.add_action(button("Save", "Primary", self._save))

        engine = Card("Analyser")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        self.engine = QComboBox()
        self.engine.addItem("Hybrid - DenseNet-121 + measurements", "hybrid")
        self.engine.addItem("Built-in measurements only", "builtin")
        self.tta = QSpinBox()
        self.tta.setRange(0, 6)
        self.positive = QDoubleSpinBox()
        self.positive.setRange(0.3, 0.9)
        self.positive.setSingleStep(0.05)
        self.context = QComboBox()
        self.context.addItems(list(CONTEXTS))
        self.site = QLineEdit()
        rows = (("Engine", self.engine, "hybrid needs torch + torchxrayvision" +
                 ("" if screening.deep_available() else " - NOT installed, built-in used")),
                ("Test-time augmentations", self.tta, "per routed module; 0 disables stability scoring"),
                ("Positive at", self.positive, "probability at which a finding is called positive"),
                ("Default context", self.context, "pre-selected on Analyse"),
                ("Site", self.site, "recorded with each study for bias monitoring"))
        for r, (name, widget, note) in enumerate(rows):
            grid.addWidget(label(name, "Small"), r, 0)
            grid.addWidget(widget, r, 1)
            grid.addWidget(label(note, "Faint"), r, 2)
        engine.body.addLayout(grid)
        self.body.addWidget(engine)

        review = Card("Review")
        self.blinded = QCheckBox("Ask for a blinded first read before showing AI findings "
                                 "(enables the Doctor vs AI vs Doctor + AI study)")
        self.credentials = QCheckBox("Show demo credentials on the sign-in screen")
        review.body.addWidget(self.blinded)
        review.body.addWidget(self.credentials)
        self.body.addWidget(review)

        llm = Card("LLM narrator (optional)", "rewrites grounded text; output is fact-checked")
        self.llm = QCheckBox("Use a local LLM through Ollama")
        self.llm_url = QLineEdit()
        self.llm_model = QLineEdit()
        self.llm_state = label("", "MonoSmall")
        llm.body.addWidget(self.llm)
        llm.body.addLayout(hbox(label("URL", "Small"), self.llm_url, label("Model", "Small"),
                                self.llm_model, button("Test", "", self._test)))
        llm.body.addWidget(self.llm_state)
        self.body.addWidget(llm)

        data = Card("Data")
        data.body.addWidget(label(f"Database: {services.store.path}", "Mono"))
        data.body.addWidget(label(f"Audit trail: {services.audit.path}", "Mono"))
        data.body.addWidget(label(f"De-identified images: {paths.images_directory()}", "Mono"))
        data.body.addLayout(hbox(None, button("Delete all studies…", "Danger", self._wipe)))
        self.body.addWidget(data)
        self.body.addStretch(1)

    def refresh(self):
        s = self.services.settings
        self.engine.setCurrentIndex(0 if s["engine"] == "hybrid" else 1)
        self.tta.setValue(int(s["tta"]))
        self.positive.setValue(float(s["positive_at"]))
        self.context.setCurrentText(s["default_context"])
        self.site.setText(s["site"])
        self.blinded.setChecked(bool(s["blinded_first_read"]))
        self.credentials.setChecked(bool(s["show_demo_credentials"]))
        self.llm.setChecked(bool(s["llm_enabled"]))
        self.llm_url.setText(s["llm_url"])
        self.llm_model.setText(s["llm_model"])

    def _save(self):
        s = self.services.settings
        s.update(engine=self.engine.currentData(), tta=self.tta.value(),
                 positive_at=round(self.positive.value(), 2), default_context=self.context.currentText(),
                 site=self.site.text().strip(), blinded_first_read=self.blinded.isChecked(),
                 show_demo_credentials=self.credentials.isChecked(), llm_enabled=self.llm.isChecked(),
                 llm_url=self.llm_url.text().strip(), llm_model=self.llm_model.text().strip())
        self.services.save_settings()
        self.services.log("settings saved", {k: s[k] for k in ("engine", "tta", "positive_at",
                                                               "blinded_first_read", "llm_enabled")},
                          category="system")
        self.caption.setText("saved · " + paths.data_directory())

    def _test(self):
        from msx.explain import LLMNarrator

        ok = LLMNarrator(self.llm_url.text(), self.llm_model.text(), True).available()
        self.llm_state.setText("reachable, model present" if ok else
                               "not reachable or model not pulled - template text will be used")

    def _wipe(self):
        if QMessageBox.question(self, "Delete all studies",
                                "Delete every study, decision and image? The audit trail is kept."
                                ) != QMessageBox.StandardButton.Yes:
            return
        rows = self.services.store.studies()
        for row in rows:
            self.services.store.delete(row["id"])
        self.services.log("all studies deleted", {"count": len(rows)}, category="system")
        self.services.studies_changed.emit()
