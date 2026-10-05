"""Benchmark - run, keep, compare and inspect evaluations of the analyser.

Admin workspace. A benchmark analyses every image of a labelled folder with the
fixed pipeline and / or adaptive routing and scores it (per-label AUC,
sensitivity, specificity, quality-gate accuracy, time, modules, model calls),
plus the Doctor + AI simulation. Every run is saved under
``<data>/evaluation/benchmark_*.json`` - the same files the Evaluation page reads.

* run history on the left; select a run to inspect it
* compare it with any earlier run (Δ column) - e.g. built-in vs hybrid, before
  vs after training or calibration
* per-image results with errors-only filtering, CSV / JSON export
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QHeaderView,
                             QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar,
                             QSpinBox, QTableWidget, QTableWidgetItem)

from msx import evaluation, paths
from msx.router import CONTEXTS

from .. import theme
from ..components import Card, ColumnChart, Page, StatStrip, button, hbox, label
from ..jobs import Job
from .training_page import real_samples_folder


def runs_folder() -> str:
    folder = os.path.join(paths.data_directory(), "evaluation")
    os.makedirs(folder, exist_ok=True)
    return folder


def _f(value, pct=False):
    if value is None:
        return "—"
    if pct:
        return f"{value:.0%}"
    return f"{value:.3f}" if isinstance(value, float) and abs(value) <= 1 else (
        f"{value:,.1f}" if isinstance(value, float) else str(value))


def _delta(a, b):
    if a is None or b is None:
        return "—"
    d = a - b
    return "0" if abs(d) < 1e-9 else f"{d:+.3f}" if abs(d) <= 1 else f"{d:+,.1f}"


def mean_auc(mode: Dict[str, object]) -> Optional[float]:
    values = [v.get("auc") for v in mode.get("per_label", {}).values() if v.get("auc") is not None]
    return sum(values) / len(values) if values else None


class BenchmarkPage(Page):
    def __init__(self, services):
        super().__init__("Benchmark", "fixed vs adaptive · per-label accuracy · cost · Doctor + AI "
                                      "simulation — every run kept and comparable")
        self.services = services
        self.job = None
        self.report: Optional[Dict[str, object]] = None
        self.reports: Dict[str, Dict[str, object]] = {}

        config = Card("New run")
        self.folder = QLineEdit(paths.SAMPLES_DIR)
        self.engine = QComboBox()
        self.engine.addItems(["builtin", "hybrid"])
        self.context = QComboBox()
        self.context.addItems([c for c in CONTEXTS if c != "Teaching"])
        self.fixed = QCheckBox("Fixed")
        self.fixed.setChecked(True)
        self.adaptive = QCheckBox("Adaptive")
        self.adaptive.setChecked(True)
        self.sim_runs = QSpinBox()
        self.sim_runs.setRange(0, 1000)
        self.sim_runs.setValue(200)
        self.run_button = button("Run benchmark", "Primary", self._run)
        services.benchmark_folder.connect(self._use_folder)
        self.fetch_button = button("Fetch real samples", "", self._fetch,
                                   tip="Download 50 public chest X-rays (COVID-19 Image Data Collection)")
        config.body.addLayout(hbox(label("Folder", "Small"), self.folder, button("…", "", self._choose),
                                   button("Demo phantoms", "", lambda: self.folder.setText(paths.SAMPLES_DIR)),
                                   button("Real samples", "", self._use_real), self.fetch_button))
        config.body.addLayout(hbox(label("Engine", "Small"), self.engine, label("Context", "Small"),
                                   self.context, label("Pipelines", "Small"), self.fixed, self.adaptive,
                                   label("Simulation runs", "Small"), self.sim_runs, None, self.run_button))
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.status = label("Folder needs a labels.csv (file,labels[,quality,sex,age,site,view]).",
                            "MonoSmall", wrap=True)
        config.body.addWidget(self.progress)
        config.body.addWidget(self.status)
        self.body.addWidget(config)

        grid = QGridLayout()
        grid.setSpacing(12)
        history = Card("Runs", "newest first")
        self.list = QListWidget()
        self.list.setMinimumWidth(300)
        self.list.setWordWrap(True)
        self.list.setMinimumHeight(420)
        self.list.currentItemChanged.connect(lambda item, _: self._select(item))
        history.body.addWidget(self.list)
        history.body.addLayout(hbox(button("Export JSON", "", self._export_json),
                                    button("Delete", "Danger", self._delete), None))
        grid.addWidget(history, 0, 0)

        result = Card("Result", "", right=None)
        self.compare = QComboBox()
        self.compare.currentIndexChanged.connect(lambda _: self._show())
        result.body.addLayout(hbox(label("Compare with", "Small"), self.compare, None))
        self.stats = StatStrip([("Images", "—", ""), ("Sensitivity", "—", "any abnormality"),
                                ("Specificity", "—", "any abnormality"), ("Mean AUC", "—", "per label"),
                                ("Time saved", "—", "adaptive vs fixed"), ("Modules saved", "—", "")])
        result.body.addWidget(self.stats)
        self.metrics = QTableWidget(0, 4)
        self.metrics.setHorizontalHeaderLabels(["Metric", "Fixed", "Adaptive", "Δ vs compared"])
        self.metrics.verticalHeader().setVisible(False)
        self.metrics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics.setMinimumHeight(420)
        charts = QGridLayout()
        self.auc_chart = ColumnChart(240)
        self.cost_chart = ColumnChart(240)
        auc_card = Card("Per-label AUC")
        auc_card.body.addWidget(self.auc_chart)
        cost_card = Card("Cost per scan")
        cost_card.body.addWidget(self.cost_chart)
        charts.addWidget(auc_card, 0, 0)
        charts.addWidget(cost_card, 0, 1)
        result.body.addWidget(self.metrics)
        result.body.addLayout(charts)
        self.result_card = result
        grid.addWidget(result, 0, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 3)
        self.body.addLayout(grid)

        images = Card("Per-image results", "adaptive pipeline of the selected run",
                      right=button("Export CSV", "", self._export_csv))
        self.errors_only = QCheckBox("Errors only")
        self.errors_only.toggled.connect(lambda _: self._images())
        images.body.addWidget(self.errors_only)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["File", "Truth", "AI positive", "Status", "Correct",
                                              "Modules", "Model calls", "ms"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(360)
        images.body.addWidget(self.table)
        self.body.addWidget(images)
        self.body.addStretch(1)

    # ------------------------------------------------------------- run -------
    def _use_folder(self, folder: str) -> None:
        """A test set built on Model Training: preselect it (hybrid, the dataset model needs DenseNet)."""
        self.folder.setText(folder)
        self.engine.setCurrentText("hybrid")
        self.status.setText(f"Test set from Model Training: {folder}")

    def _choose(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Labelled image folder", self.folder.text())
        if folder:
            self.folder.setText(folder)

    def _use_real(self) -> None:
        folder = real_samples_folder()
        if not os.path.isfile(os.path.join(folder, "labels.csv")):
            self.status.setText("No real samples yet - press “Fetch real samples” first.")
            return
        self.folder.setText(folder)

    def _start(self, fn, done, title):
        if self.job is not None and self.job.running:
            return False
        self.progress.setVisible(True)
        self.run_button.setEnabled(False)
        self.fetch_button.setEnabled(False)
        self.status.setText(title + " …")
        self.job = Job(fn, self)
        self.job.progress.connect(self._progress)
        self.job.finished.connect(lambda r: (self._idle(), done(r)))
        self.job.failed.connect(lambda m: (self._idle(), self.status.setText("Failed: " + m)))
        self.job.start()
        return True

    def _idle(self) -> None:
        self.progress.setVisible(False)
        self.run_button.setEnabled(True)
        self.fetch_button.setEnabled(True)

    def _progress(self, done, total, message) -> None:
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        self.status.setText(f"{done}/{total} · {message}")

    def _run(self) -> None:
        modes = tuple(m for m, box in (("fixed", self.fixed), ("adaptive", self.adaptive)) if box.isChecked())
        folder = self.folder.text().strip()
        if not modes:
            self.status.setText("Choose at least one pipeline.")
            return
        if not os.path.isfile(os.path.join(folder, "labels.csv")):
            self.status.setText(f"No labels.csv in {folder}")
            return
        engine, context, runs = self.engine.currentText(), self.context.currentText(), self.sim_runs.value()
        self.services.log("benchmark started", {"folder": folder, "engine": engine, "context": context,
                                                "modes": list(modes)})
        self._start(lambda progress: evaluation.benchmark(folder, engine, context, modes, progress,
                                                          simulate_runs=runs),
                    self._finished, f"benchmarking {folder}")

    def _finished(self, report) -> None:
        report["run_at"] = datetime.now().isoformat(timespec="seconds")
        report["requested_engine"] = self.engine.currentText()
        name = datetime.now().strftime("benchmark_%Y%m%d_%H%M%S.json")
        with open(os.path.join(runs_folder(), name), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
        self.services.log("benchmark finished", {"report": name, "images": report["images"],
                                                 "comparison": report.get("comparison", {})})
        self.status.setText(f"saved {name}")
        self.refresh(select=name)

    def _fetch(self) -> None:
        def work(progress):
            root = os.path.dirname(paths.PACKAGE_DIR)
            sys.path.insert(0, os.path.join(root, "tools"))
            import importlib

            module = importlib.import_module("fetch_real_samples")
            progress(0, 1, "downloading - this can take a few minutes")
            module.main(40)
            return real_samples_folder()

        self._start(work, lambda folder: (self.folder.setText(folder),
                                          self.status.setText(f"real samples ready in {folder}")),
                    "fetching real chest X-rays")

    # ---------------------------------------------------------- history ------
    def refresh(self, select: str = "") -> None:
        self.reports = {}
        for name in sorted(os.listdir(runs_folder()), reverse=True):
            if name.startswith("benchmark_") and name.endswith(".json"):
                try:
                    with open(os.path.join(runs_folder(), name), encoding="utf-8") as handle:
                        self.reports[name] = json.load(handle)
                except (OSError, ValueError):
                    continue
        current = select or (self.list.currentItem().data(Qt.ItemDataRole.UserRole)
                             if self.list.currentItem() else "")
        self.list.blockSignals(True)
        self.list.clear()
        for name, report in self.reports.items():
            stamp = name[10:25].replace("_", " ")
            folder = os.path.basename(str(report.get("folder", "")).split(" (")[0].rstrip("/\\")) or "?"
            item = QListWidgetItem(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]}\n"
                                   f"{folder} · {report.get('images', 0)} images · "
                                   f"{report.get('engine', '')} · {report.get('context', '')}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.list.addItem(item)
        self.list.blockSignals(False)
        self.compare.blockSignals(True)
        self.compare.clear()
        self.compare.addItem("— none —", "")
        for i in range(self.list.count()):
            self.compare.addItem(self.list.item(i).text().replace("\n", " · "),
                                 self.list.item(i).data(Qt.ItemDataRole.UserRole))
        self.compare.blockSignals(False)
        target = next((i for i in range(self.list.count())
                       if self.list.item(i).data(Qt.ItemDataRole.UserRole) == current), 0)
        if self.list.count():
            self.list.setCurrentRow(target)
            self._select(self.list.item(target))
        else:
            self.report = None
            self._show()

    def _select(self, item) -> None:
        if item is None:
            return
        self.report = self.reports.get(item.data(Qt.ItemDataRole.UserRole))
        self._show()

    def _delete(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        if QMessageBox.question(self, "Delete run", f"Delete {name}?") != QMessageBox.StandardButton.Yes:
            return
        os.remove(os.path.join(runs_folder(), name))
        self.services.log("benchmark deleted", {"report": name}, category="system")
        self.refresh()

    def _export_json(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        path, _ = QFileDialog.getSaveFileName(self, "Export run", name, "JSON (*.json)")
        if path:
            shutil.copy(os.path.join(runs_folder(), name), path)

    # ------------------------------------------------------------ display ----
    def _show(self) -> None:
        report = self.report
        self.metrics.setRowCount(0)
        if not report:
            self.result_card.set_caption("no run selected")
            for i in range(6):
                self.stats.set(i, "—")
            self.auc_chart.set([], [])
            self.cost_chart.set([], [])
            self.table.setRowCount(0)
            return
        modes = report["modes"]
        main = modes.get("adaptive") or next(iter(modes.values()))
        other = self.reports.get(self.compare.currentData() or "")
        other_main = (other["modes"].get("adaptive") or next(iter(other["modes"].values()))) if other else None
        self.result_card.set_caption(f"{report.get('engine', '')} · {report.get('context', '')} · "
                                     f"{report.get('folder', '')}"[:140])
        c = report.get("comparison") or {}
        self.stats.set(0, str(report.get("images", 0)))
        self.stats.set(1, _f(main["any"]["sensitivity"]))
        self.stats.set(2, _f(main["any"]["specificity"]))
        self.stats.set(3, _f(mean_auc(main)))
        self.stats.set(4, f"{c['time_saved_pct']:.0f}%" if c else "—")
        self.stats.set(5, f"{c['modules_saved_pct']:.0f}%" if c else "—")

        rows = [("Any-abnormal sensitivity", lambda m: m["any"]["sensitivity"]),
                ("Any-abnormal specificity", lambda m: m["any"]["specificity"]),
                ("Mean AUC", mean_auc)]
        labels = sorted({l for m in modes.values() for l in m.get("per_label", {})})
        for l in labels:
            rows.append((f"{l} AUC", lambda m, l=l: m.get("per_label", {}).get(l, {}).get("auc")))
            rows.append((f"{l} sensitivity", lambda m, l=l: m.get("per_label", {}).get(l, {}).get("sensitivity")))
        rows += [("Quality-gate accuracy", lambda m: m["quality_gate"]["accuracy"]),
                 ("Mean ms per scan", lambda m: m["mean_ms"]),
                 ("Mean modules (of 4)", lambda m: m["mean_modules"]),
                 ("Mean model calls", lambda m: m["mean_calls"]),
                 ("Early-exit rate", lambda m: m["early_exit_rate"]),
                 ("Findings abstained", lambda m: m["abstained"])]
        self.metrics.setRowCount(len(rows))
        for r, (name, get) in enumerate(rows):
            values = [name]
            for mode in ("fixed", "adaptive"):
                values.append(_f(get(modes[mode])) if mode in modes else "—")
            values.append(_delta(get(main), get(other_main)) if other_main else "—")
            for col, v in enumerate(values):
                item = QTableWidgetItem(v)
                if col == 3 and v not in ("—", "0"):
                    better = v.startswith("+") != ("ms" in name or "modules" in name or "calls" in name
                                                   or "abstained" in name)
                    item.setForeground(QColor(theme.GREEN if better else theme.RED))
                self.metrics.setItem(r, col, item)

        series = []
        for mode, colour in (("fixed", theme.G[400]), ("adaptive", theme.NAVY)):
            if mode in modes:
                series.append((mode, [modes[mode].get("per_label", {}).get(l, {}).get("auc") or 0
                                      for l in labels], colour))
        if other_main:
            series.append(("compared", [other_main.get("per_label", {}).get(l, {}).get("auc") or 0
                                        for l in labels], theme.TEAL))
        self.auc_chart.set(labels, series, fmt="{:.2f}", ymax=1.0)
        cost = []
        for mode, colour in (("fixed", theme.G[400]), ("adaptive", theme.NAVY)):
            if mode in modes:
                m = modes[mode]
                cost.append((mode, [m["mean_ms"] / 1000, m["mean_modules"], m["mean_calls"] / 10], colour))
        self.cost_chart.set(["seconds / scan", "modules / scan", "calls / 10"], cost, fmt="{:.1f}")
        self._images()

    def _records(self) -> List[Dict[str, object]]:
        return list((self.report or {}).get("records", []))

    @staticmethod
    def _correct(record) -> Optional[bool]:
        if "called" not in record:
            return None
        if record.get("held"):
            return None
        return set(record["called"]) == set(record["truth"])

    def _images(self) -> None:
        records = self._records()
        if self.errors_only.isChecked():
            records = [r for r in records if self._correct(r) is False]
        self.table.setRowCount(len(records))
        for row, r in enumerate(records):
            ok = self._correct(r)
            values = (r["file"], ", ".join(r["truth"]) or "No Finding",
                      ", ".join(r.get("called", [])) or "—", r.get("status", "held" if r["held"] else ""),
                      "—" if ok is None else ("✓" if ok else "✕"), str(r.get("modules", "")),
                      str(r.get("calls", "")), _f(r.get("ms")))
            for col, v in enumerate(values):
                item = QTableWidgetItem(v)
                if col == 4 and ok is not None:
                    item.setForeground(QColor(theme.GREEN if ok else theme.RED))
                self.table.setItem(row, col, item)

    def _export_csv(self) -> None:
        records = self._records()
        if not records:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export per-image results", "benchmark_images.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "truth", "ai_positive", "status", "correct", "modules", "calls", "ms",
                             "sex", "age_band"])
            for r in records:
                writer.writerow([r["file"], ";".join(r["truth"]), ";".join(r.get("called", [])),
                                 r.get("status", ""), self._correct(r), r.get("modules", ""),
                                 r.get("calls", ""), r.get("ms", ""), r.get("sex", ""), r.get("age_band", "")])
        self.services.log("benchmark images exported", {"rows": len(records)})
