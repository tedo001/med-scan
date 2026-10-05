"""Model Training - train, compare and roll back the feedback model; fit calibration and fairness.

Admin workspace. Three things are trained here:

* the **feedback model** (:mod:`msx.learner`) - from doctors' decisions and
  ground truth in the study database and / or a labelled image folder; every
  run is versioned, the active one can be rolled back;
* **confidence calibration** - per-group temperatures fitted on a labelled folder;
* **fairness thresholds** - per-subgroup operating points that close sensitivity
  gaps (:func:`msx.bias.mitigate`).

Everything long runs in the background with progress and a log; nothing is
applied to the analyser until the admin presses an Apply / Activate button.
"""

from __future__ import annotations

import os
from datetime import datetime

from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
                             QHeaderView, QLineEdit, QPlainTextEdit, QProgressBar, QTableWidget,
                             QTableWidgetItem)

from msx import bias, evaluation, paths, uncertainty
from msx.learner import MIN_SAMPLES, collect, collect_analyses

from .. import theme
from ..components import Card, HBarChart, Page, StatStrip, button, clear, hbox, label
from ..jobs import Job


def _f(value, digits=3):
    return "—" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


FEATURE_NAMES = {"measure_logit": "measurement", "deep_logit": "deep model", "has_deep": "deep present",
                 "decisiveness": "decisiveness", "stability": "stability", "agreement": "agreement",
                 "quality": "scan quality", "ap_view": "AP view"}


def real_samples_folder() -> str:
    return os.path.join(paths.data_directory(), "real_samples")


class TrainingPage(Page):
    def __init__(self, services):
        super().__init__("Model Training", "feedback model (ML) · confidence calibration · "
                                           "fairness thresholds — versioned, applied only on your say")
        self.services = services
        self.job = None
        self.fit = None                     # last calibration / fairness fit
        self.train_button = self.add_action(button("Train feedback model", "Primary", self._train))

        top = QGridLayout()
        top.setSpacing(12)
        data = Card("1 · Training data")
        self.use_db = QCheckBox("Doctor decisions and ground truth in the study database")
        self.use_db.setChecked(True)
        self.db_count = label("", "Faint")
        self.use_folder = QCheckBox("Labelled image folder (analysed now; needs labels.csv)")
        self.folder = QLineEdit(paths.SAMPLES_DIR)
        self.engine = QComboBox()
        self.engine.addItems(["builtin", "hybrid"])
        self.l2 = QDoubleSpinBox()
        self.l2.setRange(0.0, 2.0)
        self.l2.setSingleStep(0.05)
        self.l2.setValue(0.05)
        self.note = QLineEdit()
        self.note.setPlaceholderText("note for this run (optional)")
        data.body.addWidget(self.use_db)
        data.body.addWidget(self.db_count)
        data.body.addWidget(self.use_folder)
        data.body.addLayout(hbox(self.folder, button("…", "", self._choose),
                                 button("Demo phantoms", "", lambda: self.folder.setText(paths.SAMPLES_DIR)),
                                 button("Real samples", "", lambda: self.folder.setText(real_samples_folder()))))
        data.body.addLayout(hbox(label("Engine for folder", "Small"), self.engine,
                                 label("L2 regularisation", "Small"), self.l2, None))
        data.body.addWidget(self.note)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.status = label("", "MonoSmall", wrap=True)
        data.body.addWidget(self.progress)
        data.body.addWidget(self.status)
        top.addWidget(data, 0, 0)

        active = Card("Active model", right=button("Deactivate", "", self._deactivate))
        self.active_stats = StatStrip([("Trained", "—", ""), ("Labelled findings", "0", ""),
                                       ("CV accuracy", "—", "5-fold"), ("CV Brier", "—", "lower is better"),
                                       ("Blend weight", "0.00", "share of each probability")])
        active.body.addWidget(self.active_stats)
        active.body.addWidget(label(f"The model is used once it has ≥ {MIN_SAMPLES} labelled findings; "
                                    "its weight grows to 0.5 at 100 findings. It appears as the "
                                    "'learned' source on every finding card.", "Faint", wrap=True))
        top.addWidget(active, 0, 1)
        top.setColumnStretch(0, 3)
        top.setColumnStretch(1, 2)
        self.body.addLayout(top)

        mid = QGridLayout()
        mid.setSpacing(12)
        runs = Card("2 · Training runs", "every run is kept - select one to roll back",
                    right=button("Activate selected", "Primary", self._activate))
        self.runs = QTableWidget(0, 9)
        self.runs.setHorizontalHeaderLabels(["Version", "Trained", "Sources", "n", "Positives",
                                             "CV acc", "Brier", "L2", "Active"])
        self.runs.verticalHeader().setVisible(False)
        self.runs.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.runs.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.runs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.runs.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.runs.setMinimumHeight(260)
        runs.body.addWidget(self.runs)
        mid.addWidget(runs, 0, 0)
        weights = Card("Feature weights", "active model · navy raises, red lowers the probability")
        self.weights = HBarChart(260)
        weights.body.addWidget(self.weights)
        mid.addWidget(weights, 0, 1)
        mid.setColumnStretch(0, 3)
        mid.setColumnStretch(1, 2)
        self.body.addLayout(mid)

        calib = Card("3 · Calibration and fairness from a labelled folder",
                     "uses the folder above; nothing changes until you apply",
                     right=button("Fit on folder", "Primary", self._fit))
        self.calib_grid = QGridLayout()
        self.calib_grid.setSpacing(12)
        calib.body.addLayout(self.calib_grid)
        self.apply_temps = button("Apply temperatures", "", self._apply_temperatures)
        self.apply_thresholds = button("Apply fairness thresholds", "", self._apply_thresholds)
        self.apply_temps.setEnabled(False)
        self.apply_thresholds.setEnabled(False)
        calib.body.addLayout(hbox(None, self.apply_temps, self.apply_thresholds))
        self.body.addWidget(calib)

        log = Card("Log")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(140)
        self.log.setStyleSheet(f"font-family: {theme.MONO}; font-size: 11px;")
        log.body.addWidget(self.log)
        self.body.addWidget(log)
        self.body.addStretch(1)

    # ------------------------------------------------------------- helpers ---
    def _say(self, text: str) -> None:
        self.log.appendPlainText(f"{datetime.now():%H:%M:%S}  {text}")

    def _choose(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Labelled image folder", self.folder.text())
        if folder:
            self.folder.setText(folder)

    def _busy(self, on: bool) -> None:
        self.progress.setVisible(on)
        for b in (self.train_button,):
            b.setEnabled(not on)

    def _run(self, fn, done, title: str) -> None:
        if self.job is not None and self.job.running:
            return
        self._busy(True)
        self._say(title + " …")
        self.job = Job(fn, self)
        self.job.progress.connect(self._progress)
        self.job.finished.connect(lambda result: (self._busy(False), done(result)))
        self.job.failed.connect(self._failed)
        self.job.start()

    def _progress(self, done: int, total: int, message: str) -> None:
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        self.status.setText(f"{done}/{total} · {message}")

    def _failed(self, message: str) -> None:
        self._busy(False)
        self.status.setText(message)
        self._say("FAILED: " + message)

    def _folder_analyses(self, progress):
        """Analyse the labelled folder (adaptive) - for training rows, temperatures and fairness."""
        folder = self.folder.text().strip()
        if not os.path.isfile(os.path.join(folder, "labels.csv")):
            raise FileNotFoundError(f"no labels.csv in {folder}")
        keep = []
        report = evaluation.benchmark(folder, self.engine.currentText(), modes=("adaptive",),
                                      progress=progress, simulate_runs=20, keep_analyses=keep)
        return report, keep

    # -------------------------------------------------------------- train ----
    def _train(self) -> None:
        use_db, use_folder = self.use_db.isChecked(), self.use_folder.isChecked()
        if not use_db and not use_folder:
            self.status.setText("Choose at least one source of training data.")
            return
        l2, note = self.l2.value(), self.note.text().strip()
        store = self.services.store if use_db else None
        learner = self.services.learner

        def work(progress):
            extra = None
            if use_folder:
                _, keep = self._folder_analyses(progress)
                extra = collect_analyses(keep)
            progress(1, 1, "fitting")
            return learner.train(store, extra, l2=l2, note=note)

        self._run(work, self._trained, "training feedback model")

    def _trained(self, info) -> None:
        self.services._engine = None                       # the analyser picks up the new model
        self.services.log("feedback model trained", {k: info.get(k) for k in
                                                     ("version", "n", "cv_accuracy", "cv_brier",
                                                      "sources", "l2", "status")}, category="system")
        self.status.setText(str(info.get("status")))
        self._say(f"{info.get('status')} · n={info.get('n')} · CV acc {_f(info.get('cv_accuracy'))} · "
                  f"Brier {_f(info.get('cv_brier'))} · {info.get('version', '')}")
        self.refresh()

    def _activate(self) -> None:
        row = self.runs.currentRow()
        if row < 0:
            return
        version = self.runs.item(row, 0).text()
        info = self.services.learner.activate(version)
        self.services._engine = None
        self.services.log("feedback model activated", {"version": version}, category="system")
        self._say(f"activated {version} (n={info.get('n')})")
        self.refresh()

    def _deactivate(self) -> None:
        self.services.learner.reset()
        self.services._engine = None
        self.services.log("feedback model deactivated", category="system")
        self._say("feedback model deactivated - analyser uses measurements + deep model only")
        self.refresh()

    # -------------------------------------------------- calibration / bias ---
    def _fit(self) -> None:
        def work(progress):
            report, _ = self._folder_analyses(progress)
            return report

        self._run(work, self._fitted, "fitting calibration and fairness on folder")

    def _fitted(self, report) -> None:
        pairs = report["modes"]["adaptive"].get("pairs", {})
        temperatures = {g: uncertainty.fit_temperature([p for p, _ in v], [y for _, y in v])
                        for g, v in pairs.items() if len({y for _, y in v}) == 2}
        labels = [l for l in evaluation.EVAL_LABELS if any(l in r["truth"] for r in report["records"])]
        mitigation = bias.mitigate(report["records"], labels, float(self.services.settings["positive_at"]))
        self.fit = {"temperatures": temperatures, "mitigation": mitigation,
                    "folder": report["folder"], "images": report["images"]}
        clear(self.calib_grid)
        current = self.services.settings.get("temperatures", {})
        t_table = QTableWidget(len(temperatures), 3)
        t_table.setHorizontalHeaderLabels(["Group", "Current T", "Fitted T"])
        t_table.verticalHeader().setVisible(False)
        t_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for r, (g, t) in enumerate(sorted(temperatures.items())):
            for c, v in enumerate((g, _f(current.get(g, 1.0), 2), _f(t, 2))):
                t_table.setItem(r, c, QTableWidgetItem(v))
        t_table.setMinimumHeight(60 + 32 * max(1, len(temperatures)))
        self.calib_grid.addWidget(label("Confidence calibration (temperature scaling)", "SectionLabel"), 0, 0)
        self.calib_grid.addWidget(t_table, 1, 0)
        rows = mitigation["table"]
        b_table = QTableWidget(len(rows), 4)
        b_table.setHorizontalHeaderLabels(["Subgroup", "Threshold", "Sensitivity before → after",
                                           "Specificity before → after"])
        b_table.verticalHeader().setVisible(False)
        b_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for r, t in enumerate(rows):
            values = (t["group"], f"{t['threshold']:.2f}",
                      f"{_f(t['sens_before'])} → {_f(t['sens_after'])}",
                      f"{_f(t['spec_before'])} → {_f(t['spec_after'])}")
            for c, v in enumerate(values):
                b_table.setItem(r, c, QTableWidgetItem(v))
        b_table.setMinimumHeight(60 + 32 * max(1, len(rows)))
        self.calib_grid.addWidget(label("Fairness thresholds", "SectionLabel"), 0, 1)
        self.calib_grid.addWidget(b_table, 1, 1)
        self.calib_grid.addWidget(label(
            f"{report['images']} images · largest subgroup sensitivity gap "
            f"{_f(mitigation['max_gap_before'])} → {_f(mitigation['max_gap_after'])}"
            + ("" if mitigation["thresholds"] else " · no subgroup needs a change"), "Faint", wrap=True), 2, 0, 1, 2)
        self.calib_grid.setColumnStretch(0, 1)
        self.calib_grid.setColumnStretch(1, 2)
        self.apply_temps.setEnabled(bool(temperatures))
        self.apply_thresholds.setEnabled(True)
        self._say(f"fitted on {report['images']} images: temperatures {temperatures}; "
                  f"thresholds {mitigation['thresholds'] or 'none'}")

    def _apply_temperatures(self) -> None:
        if not self.fit:
            return
        self.services.settings["temperatures"] = self.fit["temperatures"]
        self.services.save_settings()
        self.services.log("confidence calibrated", {"temperatures": self.fit["temperatures"],
                                                    "folder": self.fit["folder"]}, category="system")
        self._say("temperatures applied")

    def _apply_thresholds(self) -> None:
        if not self.fit:
            return
        m = self.fit["mitigation"]
        self.services.settings["subgroup_thresholds"] = m["thresholds"]
        self.services.save_settings()
        self.services.log("bias thresholds applied", {"thresholds": m["thresholds"],
                                                      "gap_before": m["max_gap_before"],
                                                      "gap_after": m["max_gap_after"]}, category="system")
        self._say(f"fairness thresholds applied: {m['thresholds'] or 'none (defaults)'}")

    # ------------------------------------------------------------- refresh ---
    def refresh(self) -> None:
        X, y, _ = collect(self.services.store)
        self.db_count.setText(f"{len(y)} labelled findings available in the database "
                              f"({int(y.sum()) if len(y) else 0} positive)")
        learner = self.services.learner
        info = learner.info or {}
        active = info.get("status") == "trained"
        self.active_stats.set(0, (info.get("trained_at", "") or "")[5:16].replace("T", " ") if active else "—",
                              info.get("version", "") if active else "")
        self.active_stats.set(1, str(info.get("n", 0)) if active else "0")
        self.active_stats.set(2, _f(info.get("cv_accuracy")) if active else "—")
        self.active_stats.set(3, _f(info.get("cv_brier")) if active else "—")
        self.active_stats.set(4, f"{learner.weight:.2f}")
        history = learner.history()
        self.runs.setRowCount(len(history))
        for r, run in enumerate(history):
            sources = ", ".join(f"{k} {v}" for k, v in (run.get("sources") or {}).items())
            values = (run.get("version", ""), (run.get("trained_at", "") or "")[:16].replace("T", " "),
                      sources + (f" · {run['note']}" if run.get("note") else ""), str(run.get("n", 0)),
                      str(run.get("positives", 0)), _f(run.get("cv_accuracy")), _f(run.get("cv_brier")),
                      _f(run.get("l2"), 2), "✓" if run.get("active") else "")
            for c, v in enumerate(values):
                self.runs.setItem(r, c, QTableWidgetItem(v))
        coefficients = learner.coefficients()[:12] if active else []
        self.weights.set([(f"{FEATURE_NAMES.get(n, n.replace('label=', ''))} {'+' if w >= 0 else '−'}", abs(w))
                          for n, w in coefficients],
                         [theme.NAVY if w >= 0 else theme.RED for _, w in coefficients])
