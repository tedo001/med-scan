"""Evaluation - fixed vs adaptive routing, and Doctor vs AI vs Doctor + AI."""

from __future__ import annotations

import json
import os
from datetime import datetime

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QGridLayout, QHeaderView, QLineEdit,
                             QProgressBar, QTableWidget, QTableWidgetItem)

from msx import bias, evaluation, paths, uncertainty
from msx.router import CONTEXTS

from .. import theme
from ..components import Card, ColumnChart, Page, button, hbox, label


class BenchWorker(QObject):
    progress = pyqtSignal(int, int, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, folder, engine, context):
        super().__init__()
        self.folder, self.engine, self.context = folder, engine, context

    def run(self):
        try:
            report = evaluation.benchmark(self.folder, self.engine, self.context,
                                          progress=lambda d, t, m: self.progress.emit(d, t, m))
            self.done.emit(report)
        except Exception as error:  # noqa: BLE001
            self.failed.emit(f"{type(error).__name__}: {error}")


def _table(headers, rows) -> QTableWidget:
    table = QTableWidget(len(rows), len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table.setItem(r, c, QTableWidgetItem(value))
    table.setMinimumHeight(min(420, 40 + 32 * len(rows)))
    return table


def _f(value, pct=False) -> str:
    if value is None:
        return "—"
    if pct:
        return f"{value:.0%}"
    return f"{value:.3f}" if isinstance(value, float) and value <= 1 else f"{value:,.1f}" \
        if isinstance(value, float) else str(value)


class EvaluationPage(Page):
    def __init__(self, services):
        super().__init__("Evaluation", "does adaptive routing cost accuracy? does Doctor + AI beat "
                                       "either alone?")
        self.services = services
        self.thread = None
        self.report = None

        bench = Card("1 · Fixed vs adaptive pipeline", "same scans, same engine - only the router differs")
        self.folder = QLineEdit(paths.SAMPLES_DIR)
        self.engine = QComboBox()
        self.engine.addItems(["builtin", "hybrid"])
        self.context = QComboBox()
        self.context.addItems([c for c in CONTEXTS if c != "Teaching"])
        self.run_button = button("Run benchmark", "Primary", self._run)
        bench.body.addLayout(hbox(label("Folder", "Small"), self.folder,
                                  button("…", "", self._choose), label("Engine", "Small"),
                                  self.engine, label("Context", "Small"), self.context,
                                  self.run_button))
        bench.body.addWidget(label("The folder needs a labels.csv (file,labels[,quality,sex,age,"
                                   "site,view]). The demo phantoms are synthetic - use CheXpert / "
                                   "MIMIC-CXR validation sets for real performance figures.",
                                   "Faint", wrap=True))
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.bar_note = label("", "MonoSmall")
        bench.body.addWidget(self.bar)
        bench.body.addWidget(self.bar_note)
        self.bench_grid = QGridLayout()
        self.bench_grid.setSpacing(12)
        bench.body.addLayout(self.bench_grid)
        self.calibrate = button("Calibrate confidence from this run (temperature scaling)", "",
                                self._calibrate)
        self.calibrate.setEnabled(False)
        bench.body.addLayout(hbox(None, self.calibrate))
        self.body.addWidget(bench)

        self.reader = Card("2 · Doctor alone vs AI alone vs Doctor + AI",
                           "from signed studies with a blinded first read and ground truth")
        self.reader_grid = QGridLayout()
        self.reader.body.addLayout(self.reader_grid)
        self.body.addWidget(self.reader)

        self.sim = Card("3 · Doctor + AI simulation study",
                        "simulated readers × trust styles, on the AI's real outputs from the last run")
        self.sim_grid = QGridLayout()
        self.sim_grid.setSpacing(12)
        self.sim.body.addLayout(self.sim_grid)
        self.body.addWidget(self.sim)

        self.bias_card = Card("4 · Reduce bias", "per-subgroup operating thresholds that close "
                                                 "sensitivity gaps - never raised, only lowered",
                              right=button("Apply thresholds", "Primary", self._apply_bias))
        self.bias_grid = QGridLayout()
        self.bias_card.body.addLayout(self.bias_grid)
        self.body.addWidget(self.bias_card)

        self.learn_card = Card("5 · Feedback-trained model (machine learning)",
                               "logistic-regression stacker trained on doctors' decisions + ground truth",
                               right=button("Retrain now", "Primary", self._retrain))
        self.learn_body = QGridLayout()
        self.learn_card.body.addLayout(self.learn_body)
        self.body.addWidget(self.learn_card)
        self.body.addStretch(1)
        self.mitigation = None
        self._load_last()

    def _choose(self):
        folder = QFileDialog.getExistingDirectory(self, "Labelled image folder", self.folder.text())
        if folder:
            self.folder.setText(folder)

    def _run(self):
        if self.thread is not None:
            return
        self.thread = QThread()
        self.worker = BenchWorker(self.folder.text(), self.engine.currentText(),
                                  self.context.currentText())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.bar.setVisible(True)
        self.run_button.setEnabled(False)
        self.services.log("benchmark started", {"folder": self.folder.text(),
                                                "engine": self.engine.currentText()})
        self.thread.start()

    def _progress(self, done, total, message):
        self.bar.setMaximum(total)
        self.bar.setValue(done)
        self.bar_note.setText(f"{done}/{total} · {message}")

    def _finish_thread(self):
        self.thread.quit()
        self.thread.wait()
        self.thread = None
        self.run_button.setEnabled(True)
        self.bar.setVisible(False)

    def _failed(self, error):
        self._finish_thread()
        self.bar_note.setText(error)

    def _done(self, report):
        self._finish_thread()
        self.report = report
        folder = os.path.join(paths.data_directory(), "evaluation")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, datetime.now().strftime("benchmark_%Y%m%d_%H%M%S.json"))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
        self.bar_note.setText(f"saved {path}")
        self.services.log("benchmark finished", {"comparison": report.get("comparison", {}),
                                                 "report": os.path.basename(path)})
        self._show_bench(report)

    def _load_last(self):
        folder = os.path.join(paths.data_directory(), "evaluation")
        try:
            files = sorted(f for f in os.listdir(folder) if f.startswith("benchmark_"))
        except OSError:
            files = []
        if files:
            with open(os.path.join(folder, files[-1]), encoding="utf-8") as handle:
                self.report = json.load(handle)
            self.bar_note.setText(f"last run: {files[-1]}")
            self._show_bench(self.report)

    def _show_bench(self, report):
        while self.bench_grid.count():
            item = self.bench_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        modes = report["modes"]
        names = list(modes)
        rows = [["Any-abnormal sensitivity"] + [_f(modes[m]["any"]["sensitivity"]) for m in names],
                ["Any-abnormal specificity"] + [_f(modes[m]["any"]["specificity"]) for m in names]]
        for lab in evaluation.EVAL_LABELS:
            if any(lab in modes[m]["per_label"] for m in names):
                rows.append([f"{lab} AUC / sens"] + [
                    f"{_f(modes[m]['per_label'].get(lab, {}).get('auc'))} / "
                    f"{_f(modes[m]['per_label'].get(lab, {}).get('sensitivity'))}" for m in names])
        rows += [["Quality gate accuracy"] + [_f(modes[m]["quality_gate"]["accuracy"]) for m in names],
                 ["Mean ms per scan"] + [f"{modes[m]['mean_ms']:,.0f}" for m in names],
                 ["Mean modules activated (of 4)"] + [f"{modes[m]['mean_modules']:.2f}" for m in names],
                 ["Mean model calls"] + [f"{modes[m]['mean_calls']:.1f}" for m in names],
                 ["Early-exit rate"] + [_f(modes[m]["early_exit_rate"], True) for m in names],
                 ["Findings abstained"] + [str(modes[m]["abstained"]) for m in names]]
        table = _table(["Metric"] + names, rows)
        self.bench_grid.addWidget(table, 0, 0)
        chart = ColumnChart(260)
        f, a = modes.get("fixed"), modes.get("adaptive")
        if f and a:
            chart.set(["seconds / scan", "modules / scan", "model calls / 10"],
                      [("fixed", [f["mean_ms"] / 1000, f["mean_modules"], f["mean_calls"] / 10], theme.G[400]),
                       ("adaptive", [a["mean_ms"] / 1000, a["mean_modules"], a["mean_calls"] / 10], theme.NAVY)],
                      fmt="{:.1f}")
        holder = Card("Cost per scan", report.get("engine", ""))
        holder.body.addWidget(chart)
        c = report.get("comparison")
        if c:
            holder.body.addWidget(label(
                f"Adaptive routing: {c['time_saved_pct']}% less time, {c['modules_saved_pct']}% fewer "
                f"modules, {c['calls_saved_pct']}% fewer model calls; sensitivity change "
                f"{_f(c['sensitivity_change'])}, specificity change {_f(c['specificity_change'])}.",
                "Body", wrap=True))
        self.bench_grid.addWidget(holder, 0, 1)
        self.bench_grid.setColumnStretch(0, 3)
        self.bench_grid.setColumnStretch(1, 2)
        self.calibrate.setEnabled("adaptive" in modes and bool(modes["adaptive"].get("pairs")))
        self._show_simulation(report.get("simulation"))
        self._show_bias(report)

    @staticmethod
    def _clear(grid):
        while grid.count():
            item = grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _show_simulation(self, sim):
        self._clear(self.sim_grid)
        if not sim:
            self.sim_grid.addWidget(label("Run the benchmark to simulate readers.", "Small"), 0, 0)
            return
        for col, (key, title) in enumerate((("clean", "AI as measured"),
                                            ("stressed", "AI degraded to real-world error (15%)"))):
            data = sim[key]
            rows = []
            for r in data["readers"]:
                rows.append([r["skill"], r["trust"], _f(r["doctor"]["accuracy"]),
                             _f(r["ai"]["accuracy"]), _f(r["team"]["accuracy"]),
                             _f(r["automation_bias"]), _f(r["rescue_rate"]),
                             "✓" if r["team_beats_both"] else "–"])
            card = Card(title, f"{data['cases']} cases · {data['runs']} runs")
            card.body.addWidget(_table(["Skill", "Trust", "Doctor", "AI", "Doctor+AI", "Auto-bias",
                                        "Rescue", "Beats both"], rows))
            chart = ColumnChart(220)
            calibrated = [r for r in data["readers"] if r["trust"] == "Calibrated"]
            chart.set([r["skill"] for r in calibrated],
                      [("Doctor alone", [r["doctor"]["accuracy"] for r in calibrated], theme.G[400]),
                       ("AI alone", [r["ai"]["accuracy"] for r in calibrated], theme.TEAL),
                       ("Doctor + AI", [r["team"]["accuracy"] for r in calibrated], theme.NAVY)],
                      fmt="{:.2f}", ymax=1.0)
            card.body.addWidget(label("Calibrated readers - follow the AI only when its confidence "
                                      "is ≥ 0.75:", "Faint"))
            card.body.addWidget(chart)
            self.sim_grid.addWidget(card, 0, col)
        self.sim_grid.addWidget(label(
            "Auto-bias = share of wrong AI calls adopted by a doctor who was right; Rescue = share of "
            "the doctor's own errors fixed by a correct AI call. Simulated readers, not a clinical "
            "study - the same page scores real signed reads in section 2.", "Faint", wrap=True), 1, 0, 1, 2)

    def _show_bias(self, report):
        self._clear(self.bias_grid)
        records = report.get("records")
        if not records:
            self.bias_grid.addWidget(label("Run the benchmark first.", "Small"), 0, 0)
            return
        labels = [l for l in evaluation.EVAL_LABELS if any(l in r["truth"] for r in records)]
        self.mitigation = bias.mitigate(records, labels, float(self.services.settings["positive_at"]))
        m = self.mitigation
        rows = [[t["group"], f"{t['threshold']:.2f}", _f(t["sens_before"]), _f(t["sens_after"]),
                 _f(t["spec_before"]), _f(t["spec_after"])] for t in m["table"]]
        self.bias_grid.addWidget(_table(["Subgroup", "Threshold", "Sens before", "Sens after",
                                         "Spec before", "Spec after"], rows), 0, 0)
        applied = self.services.settings.get("subgroup_thresholds") or {}
        self.bias_grid.addWidget(label(
            f"Overall sensitivity {_f(m['overall_sensitivity'])}. Largest subgroup gap: "
            f"{_f(m['max_gap_before'])} before → {_f(m['max_gap_after'])} after. "
            + ("Proposed thresholds: " + ", ".join(f"{k} → {v}" for k, v in m["thresholds"].items())
               if m["thresholds"] else "No subgroup needs a change on this data.")
            + (f"  Currently applied: {applied}" if applied else ""), "Body", wrap=True), 1, 0)

    def _apply_bias(self):
        if not self.mitigation:
            return
        self.services.settings["subgroup_thresholds"] = self.mitigation["thresholds"]
        self.services.save_settings()
        self.services.log("bias thresholds applied", {"thresholds": self.mitigation["thresholds"],
                                                      "gap_before": self.mitigation["max_gap_before"],
                                                      "gap_after": self.mitigation["max_gap_after"]},
                          category="system")
        self._show_bias(self.report)

    def _retrain(self):
        self.services.retrain()
        self._show_learner()

    def _show_learner(self):
        self._clear(self.learn_body)
        info = self.services.learner.info or {}
        if not info:
            self.learn_body.addWidget(label("Not trained yet - press Retrain (it also retrains "
                                            "after every sign-off).", "Small"), 0, 0)
            return
        cells = [("Labelled findings", str(info.get("n", 0))), ("Positives", str(info.get("positives", 0))),
                 ("CV accuracy", _f(info.get("cv_accuracy"))), ("CV Brier", _f(info.get("cv_brier"))),
                 ("Blend weight", f"{self.services.learner.weight:.2f}"),
                 ("Trained", str(info.get("trained_at", ""))[:16].replace("T", " "))]
        for i, (k, v) in enumerate(cells):
            self.learn_body.addWidget(label(k, "StatLabel"), 0, i)
            self.learn_body.addWidget(label(v, "Headline"), 1, i)
        self.learn_body.addWidget(label(
            f"{info.get('status', '')} · sources: " + ", ".join(f"{k} {v}" for k, v in
                                                              (info.get("sources") or {}).items())
            + " · its probability appears as the 'learned' source on each finding.", "Faint",
            wrap=True), 2, 0, 1, 6)

    def _calibrate(self):
        pairs = self.report["modes"]["adaptive"].get("pairs", {})
        temperatures = {}
        for group, values in pairs.items():
            if len({y for _, y in values}) == 2:
                temperatures[group] = uncertainty.fit_temperature([p for p, _ in values],
                                                                  [y for _, y in values])
        self.services.settings["temperatures"] = temperatures
        self.services.save_settings()
        self.services.log("confidence calibrated", {"temperatures": temperatures}, category="system")
        self.bar_note.setText("temperatures saved: " + ", ".join(
            f"{g} T={t}" for g, t in temperatures.items()) + " - applied to new analyses")

    def refresh(self):
        self._show_learner()
        while self.reader_grid.count():
            item = self.reader_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        study = evaluation.reader_study(self.services.store)
        if not study["studies"]:
            self.reader_grid.addWidget(label(
                "No eligible studies yet. In Review, keep “blinded first read” on (Settings), record "
                "your impression, decide the findings and sign off. Studies from the demo samples "
                "carry ground truth automatically.", "Small", wrap=True), 0, 0)
            return
        names = (("doctor", "Doctor alone"), ("ai", "AI alone"), ("doctor_ai", "Doctor + AI"))
        rows = []
        for metric in ("sensitivity", "specificity", "accuracy"):
            rows.append([metric.capitalize()] + [_f((study[k] or {}).get(metric)) for k, _ in names])
        rows.append(["Mean read time (s)", _f(study["mean_first_read_s"]), "—",
                     _f(study["mean_final_read_s"])])
        self.reader_grid.addWidget(_table(["Metric"] + [n for _, n in names], rows), 0, 0)
        chart = ColumnChart(240)
        chart.set([n for _, n in names],
                  [("sensitivity", [(study[k] or {}).get("sensitivity") or 0 for k, _ in names], theme.NAVY),
                   ("specificity", [(study[k] or {}).get("specificity") or 0 for k, _ in names], theme.TEAL)],
                  fmt="{:.2f}", ymax=1.0)
        self.reader_grid.addWidget(chart, 0, 1)
        self.reader_grid.addWidget(label(f"{study['studies']} signed studies with ground truth, scored "
                                         f"over {len(evaluation.EVAL_LABELS)} labels each.", "Faint"), 1, 0)
