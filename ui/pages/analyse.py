"""Analyse - bring scans in, choose the clinical context, watch the pipeline run."""

from __future__ import annotations

import csv
import glob
import os
import traceback
from typing import Dict, List, Optional

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QFrame, QGridLayout,
                             QHeaderView, QLineEdit, QSpinBox, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from msx import imaging, paths
from msx.pipeline import STAGES
from msx.router import CONTEXTS

from .. import theme
from ..components import (Card, Page, Pill, Segmented, button, clear, hbox, label, vbox)


class Worker(QObject):
    started_file = pyqtSignal(int)
    finished_file = pyqtSignal(int, object)
    failed_file = pyqtSignal(int, str)
    done = pyqtSignal()

    def __init__(self, engine, jobs: List[Dict[str, object]], context: str, mode: str):
        super().__init__()
        self.engine, self.jobs, self.context, self.mode = engine, jobs, context, mode
        self.stop = False

    def run(self) -> None:
        for index, job in enumerate(self.jobs):
            if self.stop:
                break
            self.started_file.emit(index)
            try:
                analysis = self.engine.analyse_file(job["path"], context=self.context,
                                                    mode=self.mode, **job["facts"])
                self.finished_file.emit(index, analysis)
            except Exception as error:  # noqa: BLE001 - report per file, keep going
                self.failed_file.emit(index, f"{type(error).__name__}: {error}")
                traceback.print_exc()
        self.done.emit()


class DropZone(QFrame):
    dropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)
        col = QVBoxLayout(self)
        col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon = label("⇪")
        icon.setStyleSheet(f"font-size: 30px; color: {theme.G[400]};")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = label("Drop chest X-rays here", "Headline")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note = label("PNG · JPEG · TIFF · DICOM — identifiers are stripped on load", "Faint")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col.addWidget(icon)
        col.addWidget(title)
        col.addWidget(note)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("hover", "true")
            self.style().polish(self)

    def dragLeaveEvent(self, event):
        self.setProperty("hover", "false")
        self.style().polish(self)

    def dropEvent(self, event):
        self.setProperty("hover", "false")
        self.style().polish(self)
        files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                files += [f for f in glob.glob(os.path.join(path, "*"))
                          if f.lower().endswith(imaging.SUPPORTED)]
            elif path.lower().endswith(imaging.SUPPORTED):
                files.append(path)
        self.dropped.emit(sorted(files))


class StageChip(QFrame):
    def __init__(self, number: int, name: str):
        super().__init__()
        self.setObjectName("StageChip")
        col = QVBoxLayout(self)
        col.setContentsMargins(10, 8, 10, 8)
        col.setSpacing(2)
        self.title = label(f"{number}. {name}", "Small")
        self.title.setStyleSheet("font-weight: 600;")
        self.time = label("—", "MonoSmall")
        self.note = label("", "Faint", wrap=True)
        col.addWidget(self.title)
        col.addWidget(self.time)
        col.addWidget(self.note)
        self.set("idle")

    def set(self, state: str, ms: Optional[float] = None, note: str = "") -> None:
        self.setProperty("state", state)
        self.style().polish(self)
        self.time.setText("running…" if state == "running" else
                          "—" if ms is None else f"{ms:,.0f} ms · {state}")
        self.note.setText(note)


class AnalysePage(Page):
    navigate = pyqtSignal(str)

    def __init__(self, services):
        super().__init__("Analyse", "load scans · choose the clinical context · the router decides "
                                    "how deep each scan is analysed")
        self.services = services
        self.jobs: List[Dict[str, object]] = []
        self.thread: Optional[QThread] = None
        self.last_id = ""

        grid = QGridLayout()
        grid.setSpacing(12)

        # -- left: input -----------------------------------------------------
        source = Card("1 · Scans", "")
        self.drop = DropZone()
        self.drop.dropped.connect(self.add_files)
        source.body.addWidget(self.drop)
        source.body.addLayout(hbox(button("Choose files…", "", self._choose),
                                   button("Load demo samples", "", self._demo,
                                          tip="Synthetic phantoms with known ground truth"),
                                   None, button("Clear", "", self._clear)))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["File", "Facts", "Status", "Result"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(220)
        self.table.cellDoubleClicked.connect(self._open_row)
        source.body.addWidget(self.table)

        options = Card("2 · Clinical context")
        self.context = Segmented(list(CONTEXTS), services.settings["default_context"])
        self.context.changed.connect(self._context_note)
        options.body.addWidget(self.context)
        self.context_note = label("", "Small", wrap=True)
        options.body.addWidget(self.context_note)
        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)
        self.mode = QComboBox()
        self.mode.addItems(["Adaptive routing", "Fixed pipeline (all modules)"])
        self.sex = QComboBox()
        self.sex.addItems(["from file", "F", "M"])
        self.age = QSpinBox()
        self.age.setRange(0, 110)
        self.age.setSpecialValueText("from file")
        self.view = QComboBox()
        self.view.addItems(["from file", "PA", "AP"])
        self.site = QLineEdit(services.settings["site"])
        for r, (name, widget) in enumerate((("Pipeline", self.mode), ("Sex", self.sex),
                                            ("Age", self.age), ("View", self.view),
                                            ("Site", self.site))):
            form.addWidget(label(name, "Small"), r, 0)
            form.addWidget(widget, r, 1)
        options.body.addLayout(form)
        options.body.addWidget(label("Sex and age are used only for bias monitoring; they never "
                                     "change a finding.", "Faint", wrap=True))
        self.run_button = button("Analyse", "Primary", self._run)
        self.run_button.setMinimumHeight(34)
        self.progress = label("", "MonoSmall")
        options.body.addWidget(self.run_button)
        options.body.addWidget(self.progress)

        # -- right: pipeline ---------------------------------------------------
        pipeline = Card("3 · Pipeline", "8 stages · timed")
        chips = QGridLayout()
        chips.setSpacing(6)
        self.chips = []
        for i, name in enumerate(STAGES):
            chip = StageChip(i + 1, name)
            self.chips.append(chip)
            chips.addWidget(chip, i // 4, i % 4)
        for c in range(4):
            chips.setColumnStretch(c, 1)
        pipeline.body.addLayout(chips)

        self.result = Card("Result", "")
        self.result_body = QVBoxLayout()
        self.result.body.addLayout(self.result_body)
        self.open_button = button("Open in Review →", "Primary",
                                  lambda: self.services.open_study.emit(self.last_id))
        self.open_button.setEnabled(False)
        self.result.body.addLayout(hbox(None, self.open_button))

        left = vbox(source, spacing=12)
        right = vbox(options, pipeline, self.result, None, spacing=12)
        grid.addLayout(left, 0, 0)
        grid.addLayout(right, 0, 1)
        grid.setColumnStretch(0, 5)
        grid.setColumnStretch(1, 6)
        self.body.addLayout(grid)
        self._context_note(self.context.value)
        self._show_result(None)

    # ------------------------------------------------------------- input ----
    def _context_note(self, name: str) -> None:
        c = CONTEXTS[name]
        self.context_note.setText(
            f"{c['note'].capitalize()}. Route ≥ {c['route']:.2f}"
            + (f", early exit < {c['exit']:.2f}" if c["exit"] >= 0 else "")
            + f", explanation depth: {c['depth']}.")

    def _choose(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Choose chest X-rays", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff "
                                             "*.dcm *.dicom);;All files (*)")
        self.add_files(files)

    def _demo(self) -> None:
        self.add_files(sorted(glob.glob(os.path.join(paths.SAMPLES_DIR, "cxr_*"))))

    def _truth(self, path: str) -> Dict[str, str]:
        labels = os.path.join(os.path.dirname(path), "labels.csv")
        if not os.path.isfile(labels):
            return {}
        with open(labels, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("file") == os.path.basename(path):
                    return row
        return {}

    def add_files(self, files: List[str]) -> None:
        known = {j["path"] for j in self.jobs}
        for path in files:
            if path in known:
                continue
            row = self._truth(path)
            facts: Dict[str, object] = {}
            if row.get("sex"):
                facts["sex"] = row["sex"]
            if row.get("age"):
                facts["age_band"] = imaging.age_band(float(row["age"]))
            job = {"path": path, "facts": facts, "truth": row.get("labels", ""),
                   "site": row.get("site", ""), "status": "queued"}
            self.jobs.append(job)
        self._render_table()

    def _clear(self) -> None:
        if self.thread is None:
            self.jobs = []
            self._render_table()

    def _render_table(self) -> None:
        self.table.setRowCount(len(self.jobs))
        for r, job in enumerate(self.jobs):
            facts = " ".join(str(v) for v in job["facts"].values())
            items = [os.path.basename(job["path"]), facts or "—", job["status"],
                     job.get("result", "")]
            for c, text in enumerate(items):
                item = QTableWidgetItem(text)
                if c == 2:
                    colour = {"done": "#15803D", "failed": "#B91C1C", "running": theme.NAVY,
                              "held": "#B45309"}.get(job["status"])
                    if colour:
                        item.setForeground(QColor(colour))
                self.table.setItem(r, c, item)
        self.run_button.setText(f"Analyse {sum(1 for j in self.jobs if j['status'] == 'queued')} scan(s)")

    def _open_row(self, row: int, _col: int) -> None:
        study = self.jobs[row].get("study")
        if study:
            self.services.open_study.emit(study)

    # --------------------------------------------------------------- run ----
    def _facts_override(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        if self.sex.currentIndex():
            out["sex"] = self.sex.currentText()
        if self.age.value():
            out["age_band"] = imaging.age_band(self.age.value())
        if self.view.currentIndex():
            out["view"] = self.view.currentText()
        return out

    def _run(self) -> None:
        queued = [j for j in self.jobs if j["status"] == "queued"]
        if not queued or self.thread is not None:
            return
        override = self._facts_override()
        for job in queued:
            job["facts"] = dict(job["facts"], **override)
        context = self.context.value
        mode = "fixed" if self.mode.currentIndex() == 1 else "adaptive"
        self.services.log("analysis batch started", {"files": len(queued), "context": context,
                                                     "mode": mode})
        self.thread = QThread()
        self.worker = Worker(self.services.engine, queued, context, mode)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.started_file.connect(lambda i: self._started(queued[i]))
        self.worker.finished_file.connect(lambda i, a: self._finished(queued[i], a))
        self.worker.failed_file.connect(lambda i, e: self._failed(queued[i], e))
        self.worker.done.connect(self._done)
        self.run_button.setEnabled(False)
        self.thread.start()

    def _started(self, job) -> None:
        job["status"] = "running"
        self._render_table()
        self.progress.setText(f"analysing {os.path.basename(job['path'])} …")
        for chip in self.chips:
            chip.set("running")

    def _finished(self, job, analysis) -> None:
        site = self.site.text().strip() or job.get("site", "")
        self.services.store.save_analysis(analysis, site=job.get("site") or site,
                                          ground_truth=job.get("truth", ""),
                                          user=self.services.user.username if self.services.user else "")
        self.services.log("scan analysed", {
            "source": analysis.scan.get("source"), "status": analysis.status,
            "context": analysis.context, "mode": analysis.mode, "engine": analysis.engine,
            "modules": analysis.modules_activated, "ms": round(analysis.total_ms),
            "identifiers_removed": len(analysis.scan.get("removed_tags", [])),
            "findings": [f"{f.title} {f.probability:.2f}" for f in analysis.shown]}, analysis.id)
        job["status"] = "held" if analysis.status == "quality-hold" else "done"
        job["study"] = analysis.id
        top = analysis.top
        job["result"] = ("quality hold" if analysis.status == "quality-hold" else
                         f"{top.title} {top.probability:.0%}" if top else "no significant finding")
        self.last_id = analysis.id
        for chip, stage in zip(self.chips, analysis.stages):
            chip.set(stage.status, stage.ms, stage.note)
        self._show_result(analysis)
        self._render_table()
        self.services.studies_changed.emit()

    def _failed(self, job, error: str) -> None:
        job["status"] = "failed"
        job["result"] = error
        self._render_table()
        self.services.log("analysis failed", {"source": os.path.basename(job["path"]),
                                              "error": error}, category="system")

    def _done(self) -> None:
        self.thread.quit()
        self.thread.wait()
        self.thread = None
        self.run_button.setEnabled(True)
        self.progress.setText("batch complete - double-click a row to open it")
        self._render_table()

    def _show_result(self, analysis) -> None:
        clear(self.result_body)
        if analysis is None:
            self.result_body.addWidget(label("Run an analysis to see the result here.", "Note"))
            self.open_button.setEnabled(False)
            return
        self.open_button.setEnabled(True)
        status = {"quality-hold": ("Quality hold - human review", "fail"),
                  "findings": ("Findings", "warn"), "uncertain": ("Engine abstained", "info"),
                  "no-finding": ("No significant finding", "ok")}[analysis.status]
        self.result.set_caption(f"{analysis.id} · {analysis.total_ms:,.0f} ms · "
                                f"{analysis.modules_activated}/4 modules · {analysis.engine}")
        self.result_body.addLayout(hbox(Pill(status[0], status[1]),
                                        Pill(analysis.context, "grey"),
                                        Pill("URGENT", "fail") if analysis.urgent else None, None))
        self.result_body.addWidget(label(analysis.explanation.get("headline", ""), "Headline", wrap=True))
        for f in analysis.shown:
            self.result_body.addWidget(label(
                f"• {f.title} — {f.probability:.0%}, {f.confidence_level.lower()} confidence"
                + (" (abstained)" if f.status == "uncertain" else ""), "Body"))
        routes = analysis.routing.get("routes", [])
        if routes:
            self.result_body.addWidget(label("Router: " + "; ".join(
                f"{r['module']} {'✓' if r['run'] else '–'}" for r in routes), "MonoSmall", wrap=True))

    def refresh(self) -> None:
        pass
