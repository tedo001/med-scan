"""Review - the doctor's bench: see the evidence, decide every finding, sign off.

Left, the worklist (urgent first). Right, one case:

* the scan with the AI's heatmap, finding regions, lung outline and measurement
  lines - each a toggle;
* scan quality and the router's decision (which modules ran, and why);
* an optional **blinded first read** - the doctor records an impression before
  any AI output is shown, which is what lets the Evaluation page compare
  Doctor alone, AI alone and Doctor + AI;
* one card per finding: probability, confidence (with its breakdown), the
  evidence, the limitations, and **Accept / Correct / Reject / Question**;
* the explanation at Brief / Standard / Detailed depth with cited evidence;
* questions and answers; then **Sign off** or **Request second read**.

Every click is written to the audit trail and the decisions table.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFrame, QGridLayout,
                             QHBoxLayout, QHeaderView, QInputDialog, QLineEdit, QMenu,
                             QMessageBox, QScrollArea, QSizePolicy, QSlider, QSplitter,
                             QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from msx import explain
from msx.datastore import LABELS, OPEN, SECOND_READ, SIGNED
from msx.findings import NEGATIVE, POSITIVE, POSSIBLE, UNCERTAIN, Finding

from .. import theme
from ..components import (Card, Pill, ProbBar, Segmented, XrayViewer, button, clear, hbox, label,
                          vbox)

STATUS_TONE = {POSITIVE: ("Positive", "fail"), POSSIBLE: ("Possible", "warn"),
               UNCERTAIN: ("AI abstains", "info"), NEGATIVE: ("Not found", "grey")}
STUDY_TONE = {"quality-hold": ("Quality hold", "#B45309"), "findings": ("Findings", "#B91C1C"),
              "uncertain": ("Abstained", "#7C3AED"), "no-finding": ("Normal", "#15803D")}
ACTION_TEXT = {"accept": "Accepted", "reject": "Rejected", "correct": "Corrected",
               "add": "Added by doctor"}


class FindingCard(QFrame):
    decided = pyqtSignal(str, str, str)      # key, action, corrected label
    asked = pyqtSignal(str)
    selected = pyqtSignal(str)

    def __init__(self, finding: Finding, decision: Optional[Dict[str, object]], locked: bool):
        super().__init__()
        self.finding = finding
        self.setObjectName("FindingCard")
        self.setProperty("status", finding.status)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        col = QVBoxLayout(self)
        col.setContentsMargins(14, 10, 14, 10)
        col.setSpacing(6)
        text, tone = STATUS_TONE[finding.status]
        conf_tone = {"High": "ok", "Moderate": "warn", "Low": "fail"}.get(finding.confidence_level, "grey")
        head = hbox(label(finding.title, "FindingTitle"), Pill(text, tone), None,
                    ProbBar(finding.probability, 130),
                    Pill(f"{finding.confidence_level} confidence {finding.confidence:.2f}", conf_tone))
        col.addLayout(head)
        meta = f"{finding.module} module"
        if finding.zones:
            meta += " · " + ", ".join(finding.zones) + " zone"
        meta += " · " + " · ".join(f"{k} {v:.2f}" for k, v in finding.sources.items())
        col.addWidget(label(meta, "MonoSmall"))
        for line in finding.evidence:
            col.addWidget(label(f"• {line}", "Body", wrap=True))
        u = finding.uncertainty
        if u:
            col.addWidget(label(
                f"confidence = decisiveness {u.get('decisiveness', 0):.2f} · stability "
                f"{u.get('stability', 0):.2f} (TTA σ {u.get('std', 0):.3f}, n={u.get('n', 1)}) · "
                f"agreement {u.get('agreement', 1):.2f} · quality {u.get('quality', 1):.2f}",
                "MonoSmall", wrap=True))
        if finding.limitations:
            lim = label("Limitations: " + " · ".join(finding.limitations), "Small", wrap=True)
            lim.setStyleSheet(f"color: {theme.G[500]};")
            col.addWidget(lim)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.accept = button("✓ Accept", "Accept", checkable=True)
        self.correct = button("✎ Correct ▾", "Correct", checkable=True)
        self.reject = button("✕ Reject", "Reject", checkable=True)
        ask = button("? Question", "Question")
        menu = QMenu(self.correct)
        for name in LABELS + ("No Finding",):
            if name != finding.label:
                menu.addAction(name, lambda n=name: self._decide("correct", n))
        self.correct.clicked.connect(lambda: (self.correct.setChecked(
            bool(decision and decision["action"] == "correct")), menu.exec(
            self.correct.mapToGlobal(self.correct.rect().bottomLeft()))))
        self.accept.clicked.connect(lambda: self._decide("accept"))
        self.reject.clicked.connect(lambda: self._decide("reject"))
        ask.clicked.connect(lambda: self.asked.emit(finding.key))
        for b in (self.accept, self.correct, self.reject):
            b.setEnabled(not locked)
            row.addWidget(b)
        row.addWidget(ask)
        row.addStretch(1)
        if decision:
            action = decision["action"]
            {"accept": self.accept, "reject": self.reject, "correct": self.correct}.get(
                action, self.accept).setChecked(action in ("accept", "reject", "correct"))
            text = ACTION_TEXT.get(action, action)
            if action == "correct":
                text += f" → {decision['corrected_label']}"
            row.addWidget(label(f"{text} by {decision['user']} · {decision['at'][11:16]}",
                                "MonoSmall"))
        col.addLayout(row)

    def _decide(self, action: str, corrected: str = "") -> None:
        self.decided.emit(self.finding.key, action, corrected)

    def mousePressEvent(self, event):
        self.selected.emit(self.finding.key)
        super().mousePressEvent(event)

    def set_selected(self, on: bool) -> None:
        self.setProperty("selected", "true" if on else "false")
        self.style().polish(self)


class ReviewPage(QWidget):
    navigate = pyqtSignal(str)

    def __init__(self, services):
        super().__init__()
        self.setObjectName("Page")
        self.services = services
        self.study_id = ""
        self.analysis = None
        self.row: Dict[str, object] = {}
        self.cards: Dict[str, FindingCard] = {}
        self.opened_at = time.monotonic()
        self.selected_key = ""
        self.depth_override: Optional[str] = None

        outer = QHBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 0)
        outer.setSpacing(12)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(10)
        splitter.addWidget(self._worklist())
        splitter.addWidget(self._case())
        splitter.setSizes([340, 1100])
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter)

    # ---------------------------------------------------------- worklist ----
    def _worklist(self) -> QWidget:
        panel = QWidget()
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 0, 12)
        col.setSpacing(8)
        col.addWidget(label("Review", "PageTitle"))
        self.caption = label("", "PageCaption")
        col.addWidget(self.caption)
        self.state = QComboBox()
        self.state.addItems(["Open", "Second read", "Signed", "All"])
        self.state.currentIndexChanged.connect(self.refresh)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search ref, file, finding")
        self.search.textChanged.connect(self.refresh)
        col.addLayout(hbox(self.state, self.search))
        card = QFrame()
        card.setObjectName("Card")
        inner = QVBoxLayout(card)
        inner.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Study", "Top finding", "Prob"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setShowGrid(False)
        self.table.itemSelectionChanged.connect(self._picked)
        inner.addWidget(self.table)
        col.addWidget(card, 1)
        return panel

    def refresh(self) -> None:
        state = {0: OPEN, 1: SECOND_READ, 2: SIGNED, 3: None}[self.state.currentIndex()]
        rows = self.services.store.studies(state=state, search=self.search.text().strip())
        counts = self.services.store.counts()
        self.caption.setText(f"{counts['open']} open · {counts['second']} second read · "
                             f"{counts['signed']} signed · {counts['urgent']} urgent")
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        keep = -1
        for r, row in enumerate(rows):
            status, colour = STUDY_TONE.get(row["status"], (row["status"], theme.G[500]))
            ref = QTableWidgetItem(f"{row['id']}\n{row['created'][5:16].replace('T', ' ')}")
            ref.setData(Qt.ItemDataRole.UserRole, row["id"])
            ref.setFont(self._mono())
            top = QTableWidgetItem((("▲ URGENT  " if row["urgent"] else "") +
                                    (row["top_label"] or status)) + f"\n{row['source'][:28]}")
            top.setForeground(QColor(colour))
            prob = QTableWidgetItem(f"{row['top_prob']:.0%}" if row["top_label"] else "—")
            prob.setForeground(QColor(theme.prob_colour(row["top_prob"] or 0)))
            prob.setFont(self._mono(True))
            for c, item in enumerate((ref, top, prob)):
                self.table.setItem(r, c, item)
            self.table.setRowHeight(r, 46)
            if row["id"] == self.study_id:
                keep = r
        self.table.blockSignals(False)
        if keep >= 0:
            self.table.selectRow(keep)
        elif rows and not self.study_id:
            self.table.selectRow(0)
        elif not rows and not self.study_id:
            self._show_empty()

    @staticmethod
    def _mono(bold: bool = False):
        from PyQt6.QtGui import QFont

        font = QFont("JetBrains Mono", 9)
        if bold:
            font.setWeight(QFont.Weight.DemiBold)
        return font

    def _picked(self) -> None:
        items = self.table.selectedItems()
        if items:
            study = self.table.item(items[0].row(), 0).data(Qt.ItemDataRole.UserRole)
            if study != self.study_id:
                self.load(study)

    def select(self, study_id: str) -> None:
        """Open ``study_id`` (from Analyse, Home or the bell) and highlight it in the list."""
        for widget in (self.state, self.search):
            widget.blockSignals(True)
        self.state.setCurrentIndex(3)
        self.search.clear()
        for widget in (self.state, self.search):
            widget.blockSignals(False)
        self.load(study_id)
        self.refresh()

    # -------------------------------------------------------------- case ----
    def _case(self) -> QWidget:
        holder = QWidget()
        col = QVBoxLayout(holder)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        scroll = self.case_scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("PageBody")
        self.case = QVBoxLayout(body)
        self.case.setContentsMargins(0, 0, 8, 16)
        self.case.setSpacing(12)
        scroll.setWidget(body)
        col.addWidget(scroll, 1)

        # header
        self.ref = label("", "CaseRef")
        self.meta = label("", "Small", wrap=True)
        self.pills = QHBoxLayout()
        self.pills.setSpacing(6)
        self.case.addLayout(hbox(vbox(self.ref, self.meta, spacing=2), None, self.pills))

        # viewer + side panels
        view_card = Card("Scan", "", padding=10)
        toggles = QHBoxLayout()
        toggles.setSpacing(4)
        self.toggles = {}
        for key, text in (("heat", "Heatmap"), ("regions", "Regions"), ("lungs", "Lung outline"),
                          ("measure", "Measurements")):
            t = button(text, "Toggle", checkable=True)
            t.setChecked(key != "lungs")
            t.toggled.connect(lambda on, k=key: self.viewer.set_layer(k, on))
            self.toggles[key] = t
            toggles.addWidget(t)
        toggles.addStretch(1)
        toggles.addWidget(label("overlay", "Faint"))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(10, 100)
        slider.setValue(55)
        slider.setFixedWidth(110)
        slider.valueChanged.connect(lambda v: self.viewer.set_opacity(v / 100))
        toggles.addWidget(slider)
        view_card.body.addLayout(toggles)
        self.viewer = XrayViewer()
        self.viewer.setMinimumHeight(520)
        view_card.body.addWidget(self.viewer, 1)

        side = QVBoxLayout()
        side.setSpacing(12)
        self.quality_card = Card("Scan quality", "")
        self.router_card = Card("Adaptive router", "")
        self.stage_card = Card("Pipeline", "")
        side.addWidget(self.quality_card)
        side.addWidget(self.router_card)
        side.addWidget(self.stage_card)
        side.addStretch(1)
        top = QGridLayout()
        top.setSpacing(12)
        top.addWidget(view_card, 0, 0)
        top.addLayout(side, 0, 1)
        top.setColumnStretch(0, 3)
        top.setColumnStretch(1, 2)
        self.case.addLayout(top)

        # first read / headline / findings / explanation / Q&A
        self.first_read = QFrame()
        self.first_read.setObjectName("InfoPanel")
        self.case.addWidget(self.first_read)
        self.headline = QFrame()
        self.case.addWidget(self.headline)
        self.findings_card = Card("AI findings", "")
        self.case.addWidget(self.findings_card)
        self.explain_card = Card("Explanation", "")
        self.depth = Segmented(["Brief", "Standard", "Detailed"])
        self.depth.changed.connect(self._depth_changed)
        self.explain_card.body.addWidget(self.depth)
        self.explain_body = QVBoxLayout()
        self.explain_card.body.addLayout(self.explain_body)
        self.case.addWidget(self.explain_card)
        self.qa_card = Card("Questions", "ask about a finding - answered from its evidence and the "
                                         "knowledge base")
        self.question_target = QComboBox()
        self.question = QLineEdit()
        self.question.setPlaceholderText("e.g. Why do you think this?  How sure are you?  "
                                         "What could mimic it?  What next?")
        self.question.returnPressed.connect(self._ask)
        self.qa_card.body.addLayout(hbox(self.question_target, self.question,
                                         button("Ask", "Primary", self._ask)))
        self.qa_body = QVBoxLayout()
        self.qa_card.body.addLayout(self.qa_body)
        self.case.addWidget(self.qa_card)
        self.case.addStretch(1)

        # decision bar
        bar = QFrame()
        bar.setObjectName("DecisionBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(14, 10, 14, 10)
        row.addWidget(label("HUMAN DECISION", "SectionLabel"))
        self.decision_note = label("", "Small")
        row.addWidget(self.decision_note, 1)
        self.second = button("Request second read", "", self._second_read)
        self.add_button = button("+ Add missed finding ▾", "")
        add_menu = QMenu(self.add_button)
        for name in LABELS:
            add_menu.addAction(name, lambda n=name: self._add_finding(n))
        self.add_button.setMenu(add_menu)
        self.sign = button("Sign off report", "Primary", self._sign_off)
        self.sign.setMinimumHeight(34)
        row.addWidget(self.add_button)
        row.addWidget(self.second)
        row.addWidget(self.sign)
        col.addWidget(bar)
        self._show_empty()
        return holder

    def _show_empty(self) -> None:
        self.ref.setText("No study selected")
        self.meta.setText("Analyse a scan, or pick one from the worklist.")
        self.viewer.set_scan(None)
        for b in (self.sign, self.second, self.add_button):
            b.setEnabled(False)

    # -------------------------------------------------------------- load ----
    def load(self, study_id: str) -> None:
        loaded = self.services.store.load(study_id)
        if loaded is None:
            return
        self.analysis, self.row = loaded
        self.study_id = study_id
        self.opened_at = time.monotonic()
        self.selected_key = ""
        self.depth_override = None
        self.services.log("study opened", study=study_id)
        self._render()

    @property
    def locked(self) -> bool:
        return self.row.get("review_state") == SIGNED

    @property
    def blinded(self) -> bool:
        return bool(self.services.settings.get("blinded_first_read")) and not self.row.get(
            "first_read") and not self.locked and self.analysis.status != "quality-hold"

    def _render(self) -> None:
        a, row = self.analysis, self.row
        scan = a.scan
        self.ref.setText(f"{a.id}  ·  {scan.get('source', '')}")
        facts = [scan.get("patient_ref", ""), scan.get("sex", ""), scan.get("age_band", ""),
                 scan.get("view", ""), row.get("site", ""), a.context,
                 f"{a.mode} routing" if a.mode == "adaptive" else "fixed pipeline",
                 datetime.fromisoformat(a.created).strftime("%d %b %Y %H:%M")]
        removed = scan.get("removed_tags", [])
        self.meta.setText(" · ".join(f for f in facts if f) +
                          (f" · {len(removed)} DICOM identifiers removed" if removed else ""))
        clear(self.pills)
        self.pills.addWidget(Pill(f"ENGINE · {a.engine}", "engine"))
        text, colour = STUDY_TONE.get(a.status, (a.status, theme.G[500]))
        tone = {"quality-hold": "warn", "findings": "fail", "uncertain": "info",
                "no-finding": "ok"}.get(a.status, "grey")
        self.pills.addWidget(Pill(text, tone))
        if a.urgent:
            self.pills.addWidget(Pill("▲ URGENT", "fail"))
        state = row.get("review_state", OPEN)
        self.pills.addWidget(Pill({"open": "Awaiting review", "signed": "Signed",
                                   "second-read": "Second read requested"}[state],
                                  "ok" if state == SIGNED else "info"))

        self.viewer.set_scan(a.work, a.heat, a.lung_mask)
        self.viewer.blind = self.blinded
        self._overlays()
        self._quality()
        self._router()
        self._stages()
        self._first_read()
        self._findings()
        self._explanation()
        self._questions()
        for b in (self.sign, self.second, self.add_button):
            b.setEnabled(not self.locked and not self.blinded)
        self._decision_note()

    def _overlays(self) -> None:
        regions, lines = [], []
        for f in self.analysis.shown:
            if f.bbox:
                colour = {POSITIVE: theme.RED, POSSIBLE: "#F59E0B", UNCERTAIN: "#7C3AED"}[f.status]
                regions.append({"key": f.key, "bbox": f.bbox, "colour": colour,
                                "text": f"{f.label} {f.probability:.0%}"})
            for line in f.measurements.get("lines", []) if isinstance(f.measurements, dict) else []:
                lines.append((*line, "#38BDF8"))
            for spot in f.measurements.get("spots", []) if isinstance(f.measurements, dict) else []:
                x, y, r = spot
                regions.append({"key": f.key, "bbox": (x - r, y - r, x + r, y + r),
                                "colour": "#F59E0B", "text": ""})
        self.viewer.set_overlays(regions, lines)

    def _quality(self) -> None:
        q = self.analysis.quality
        clear(self.quality_card.body)
        tone = {"good": "ok", "usable": "warn", "reject": "fail"}[q.get("grade", "good")]
        self.quality_card.body.addLayout(hbox(label(f"{q.get('score', 0)}", "StatValue"),
                                              label("/ 100", "Faint"), None,
                                              Pill(q.get("grade", "").upper(), tone)))
        for c in q.get("checks", []):
            mark = {"ok": "✓", "warn": "!", "fail": "✕"}[c["status"]]
            colour = {"ok": "#15803D", "warn": "#B45309", "fail": "#B91C1C"}[c["status"]]
            name = label(f"{mark}  {c['name']}", "Small")
            name.setStyleSheet(f"color: {colour}; font-weight: 600;")
            name.setFixedWidth(120)
            self.quality_card.body.addLayout(hbox(name, label(c["detail"], "Small", wrap=True),
                                                  spacing=6))

    def _router(self) -> None:
        r = self.analysis.routing
        clear(self.router_card.body)
        if not r:
            self.router_card.body.addWidget(label("Not routed - the scan stopped at the quality "
                                                  "gate and was sent for human review.", "Small",
                                                  wrap=True))
            self.router_card.set_caption("")
            return
        t = r.get("thresholds", {})
        self.router_card.set_caption(f"{r.get('context')} · route ≥ {t.get('route', 0):.2f}")
        if r.get("early_exit"):
            self.router_card.body.addWidget(label("Early exit: every screen score below "
                                                  f"{t.get('exit', 0):.2f} - no specialist needed.",
                                                  "Small", wrap=True))
        for route in r.get("routes", []):
            mark = label("●" if route["run"] else "○", "Small")
            mark.setStyleSheet(f"color: {theme.NAVY if route['run'] else theme.G[400]}; font-size: 14px;")
            name = label(route["module"], "Small")
            name.setStyleSheet("font-weight: 600;")
            name.setFixedWidth(78)
            bar = ProbBar(route["probability"], 96)
            self.router_card.body.addLayout(hbox(mark, name, bar, label(route["reason"], "MonoSmall",
                                                                        wrap=True), spacing=6))
        self.router_card.body.addWidget(label(
            f"{self.analysis.modules_activated} of 4 modules · {self.analysis.model_calls} model "
            f"calls", "Faint"))

    def _stages(self) -> None:
        clear(self.stage_card.body)
        self.stage_card.set_caption(f"{self.analysis.total_ms:,.0f} ms total")
        for s in self.analysis.stages:
            colour = {"ok": "#15803D", "skipped": theme.G[400], "stopped": "#B91C1C"}.get(s.status)
            name = label(s.name, "Small")
            name.setStyleSheet(f"color: {colour}; font-weight: 600;")
            name.setFixedWidth(88)
            self.stage_card.body.addLayout(hbox(name, label(f"{s.ms:>6,.0f} ms", "MonoSmall"),
                                                label(s.note, "Faint", wrap=True), spacing=6))

    # ------------------------------------------------------- first read -----
    def _first_read(self) -> None:
        panel = self.first_read
        if panel.layout() is None:
            QVBoxLayout(panel)
        clear(panel.layout())
        panel.setVisible(self.blinded or bool(self.row.get("first_read")))
        if not panel.isVisible():
            return
        lay = panel.layout()
        lay.setContentsMargins(14, 10, 14, 10)
        if not self.blinded:
            lay.addWidget(label(f"Your blinded first read: {self.row['first_read']}  "
                                f"({(self.row.get('first_read_ms') or 0) / 1000:.0f} s)", "Small"))
            return
        lay.addWidget(label("BLINDED FIRST READ", "SectionLabel"))
        lay.addWidget(label("Record your own impression before the AI's findings are shown. This is "
                            "how MEDSCAN measures whether Doctor + AI beats either alone.",
                            "Small", wrap=True))
        boxes = QGridLayout()
        self.first_boxes = {}
        for i, name in enumerate(LABELS + ("No Finding",)):
            box = QCheckBox(name)
            self.first_boxes[name] = box
            boxes.addWidget(box, i // 5, i % 5)
        lay.addLayout(boxes)
        lay.addLayout(hbox(None, button("Record impression and reveal AI", "Primary",
                                        self._record_first_read)))

    def _record_first_read(self) -> None:
        labels = [n for n, b in self.first_boxes.items() if b.isChecked() and n != "No Finding"]
        elapsed = (time.monotonic() - self.opened_at) * 1000
        self.services.store.record_first_read(self.study_id, labels, elapsed)
        self.services.log("blinded first read recorded", {"labels": labels or ["No Finding"],
                                                          "seconds": round(elapsed / 1000, 1)},
                          self.study_id)
        self.opened_at = time.monotonic()
        self.row = self.services.store.study(self.study_id)
        self._render()

    # ---------------------------------------------------------- findings ----
    def _findings(self) -> None:
        clear(self.findings_card.body)
        self.cards = {}
        a = self.analysis
        if self.headline.layout() is None:      # an empty QLayout is falsy - test for None
            QVBoxLayout(self.headline)
        lay = self.headline.layout()
        clear(lay)
        lay.setContentsMargins(14, 10, 14, 10)
        if self.blinded:
            self.headline.setVisible(False)
            self.findings_card.setVisible(False)
            return
        self.headline.setVisible(True)
        self.findings_card.setVisible(a.status != "quality-hold")
        name = {"quality-hold": "HoldPanel", "findings": "HoldPanel", "uncertain": "InfoPanel",
                "no-finding": "OkPanel"}[a.status]
        self.headline.setObjectName(name)
        self.headline.style().polish(self.headline)
        title = {"quality-hold": "QUALITY HOLD - HUMAN REVIEW REQUESTED",
                 "findings": "AI ASSESSMENT - NOT A DECISION",
                 "uncertain": "AI ABSTAINS ON ONE OR MORE FINDINGS",
                 "no-finding": "AI ASSESSMENT - NO SIGNIFICANT ABNORMALITY"}[a.status]
        lay.addWidget(label(title, "SectionLabel"))
        lay.addWidget(label(a.explanation.get("headline", ""), "Headline", wrap=True))
        if a.status == "quality-hold":
            lay.addWidget(label("The analyser did not produce findings for this scan. Re-acquire "
                                "or read it yourself; record any findings with “Add missed "
                                "finding”.", "Small", wrap=True))
        latest = self.services.store.latest_decisions(self.study_id)
        shown = a.shown
        self.findings_card.set_caption(f"{len(shown)} shown · {len(a.findings) - len(shown)} "
                                       "checked and not found")
        for f in shown:
            card = FindingCard(f, latest.get(f.key), self.locked)
            card.decided.connect(self._decide)
            card.asked.connect(self._focus_question)
            card.selected.connect(self._select)
            self.cards[f.key] = card
            self.findings_card.body.addWidget(card)
        checked = [f.label for f in a.findings if f.status == NEGATIVE]
        if checked:
            self.findings_card.body.addWidget(label("Looked for, not found: " + ", ".join(
                sorted(set(checked))), "Small", wrap=True))
        added = [d for d in latest.values() if d["action"] == "add"]
        for d in added:
            self.findings_card.body.addWidget(label(
                f"+ {d['label']} - added by {d['user']} (AI missed)", "Small"))
        if not shown and a.status != "quality-hold":
            self.findings_card.body.addWidget(label(
                "Nothing to decide. If you see something the AI missed, use “Add missed "
                "finding”; otherwise sign off as normal.", "Small", wrap=True))
        self.question_target.clear()
        for f in shown:
            self.question_target.addItem(f.title, f.key)
        self.question_target.addItem("General", "")

    def _select(self, key: str) -> None:
        self.selected_key = key
        self.viewer.selected = key
        self.viewer.update()
        for k, card in self.cards.items():
            card.set_selected(k == key)

    def _decide(self, key: str, action: str, corrected: str) -> None:
        if self.locked:
            return
        finding = next(f for f in self.analysis.findings if f.key == key)
        note = ""
        if action == "reject":
            note, ok = QInputDialog.getText(self, "Reject finding",
                                            f"Why reject “{finding.title}”? (optional)")
            if not ok:
                self._findings()
                return
        user = self.services.user
        self.services.store.record_decision(self.study_id, key, finding.label, action,
                                            user.username, user.role, corrected, note)
        self.services.log(f"finding {action}ed" if action != "correct" else "finding corrected",
                          {"finding": finding.title, "probability": finding.probability,
                           "confidence": finding.confidence, "corrected_to": corrected,
                           "note": note}, self.study_id)
        self._findings()
        self._select(key)
        self._decision_note()

    def _add_finding(self, name: str) -> None:
        user = self.services.user
        self.services.store.record_decision(self.study_id, f"ADD-{name}", name, "add",
                                            user.username, user.role)
        self.services.log("finding added by doctor", {"finding": name}, self.study_id)
        self._findings()
        self._decision_note()

    def _decision_note(self) -> None:
        if not self.analysis:
            return
        if self.locked:
            self.decision_note.setText(f"Signed by {self.row.get('signed_by')} at "
                                       f"{(self.row.get('signed_at') or '')[11:16]} · final read: "
                                       f"{self.row.get('final_read')}")
            return
        latest = self.services.store.latest_decisions(self.study_id)
        pending = [f for f in self.analysis.shown if f.key not in latest]
        self.decision_note.setText(
            ("Record your impression first." if self.blinded else
             f"{len(pending)} finding(s) undecided" if pending else "All findings decided")
            + " · the AI result above is a suggestion; your decision is recorded under "
              f"{self.services.user.display if self.services.user else ''}")

    # ------------------------------------------------------- explanation ----
    def _depth_changed(self, name: str) -> None:
        self.depth_override = name.lower()
        self._explanation()
        self.services.log("explanation depth changed", {"depth": name}, self.study_id)

    def _explanation(self) -> None:
        clear(self.explain_body)
        a = self.analysis
        self.explain_card.setVisible(not self.blinded)
        if self.blinded:
            return
        if self.depth_override:
            e = explain.build(a.findings, a.quality, a.routing, a.context, self.depth_override,
                              self.services.kb, auto_escalate=False)
        else:
            e = a.explanation
        self.depth.set(e.get("depth", "standard").capitalize())
        note = f"depth: {e.get('depth')}"
        if e.get("escalated"):
            note += f" (escalated from {e.get('requested_depth')} - the engine is less sure)"
        note += f" · context: {e.get('context')} · narrator: {e.get('narrator', 'template')}"
        self.explain_card.set_caption(note)
        if e.get("summary"):
            self.explain_body.addWidget(label(e["summary"], "Body", wrap=True))
        for section in e.get("sections", []):
            self.explain_body.addWidget(label(section["title"].upper(), "SectionLabel"))
            for line in section.get("lines", []):
                self.explain_body.addWidget(label(f"• {line}", "Body", wrap=True, selectable=True))
            for line in section.get("limitations", []) if e.get("depth") != "brief" else []:
                lim = label(f"! {line}", "Small", wrap=True)
                lim.setStyleSheet("color: #B45309;")
                self.explain_body.addWidget(lim)
            if section.get("action") and e.get("depth") != "brief":
                act = label(f"→ {section['action']}", "Small", wrap=True)
                act.setStyleSheet(f"color: {theme.NAVY}; font-weight: 600;")
                self.explain_body.addWidget(act)
            for ev in section.get("evidence", []):
                self.explain_body.addWidget(label(f"{ev['text']}\n— {ev['cite']}", "Evidence",
                                                  wrap=True, selectable=True))
            if section.get("cite"):
                self.explain_body.addWidget(label(f"— {section['cite']}", "Faint", wrap=True))

    # ----------------------------------------------------------- Q & A ------
    def _focus_question(self, key: str) -> None:
        index = self.question_target.findData(key)
        if index >= 0:
            self.question_target.setCurrentIndex(index)
        self.question.setFocus()
        self._select(key)

    def _ask(self) -> None:
        text = self.question.text().strip()
        if not text or not self.analysis or self.blinded:
            return
        key = self.question_target.currentData()
        finding = next((f for f in self.analysis.findings if f.key == key), None)
        result = explain.answer(text, finding, self.services.kb, self.services.narrator)
        user = self.services.user
        self.services.store.record_question(self.study_id, key or "", text, result["answer"],
                                            result["engine"], user.username if user else "")
        if finding is not None:
            self.services.store.record_decision(self.study_id, key, finding.label, "question",
                                                user.username, user.role, note=text)
        self.services.log("question asked", {"finding": finding.title if finding else "general",
                                             "intent": result["intent"],
                                             "engine": result["engine"]}, self.study_id)
        self.question.clear()
        self._questions()

    def _questions(self) -> None:
        clear(self.qa_body)
        self.qa_card.setVisible(not self.blinded)
        for q in reversed(self.services.store.questions(self.study_id)):
            self.qa_body.addWidget(label(f"Q · {q['user']} · {q['at'][11:16]}:  {q['question']}",
                                         "Small", wrap=True))
            self.qa_body.addWidget(label(q["answer"], "Evidence", wrap=True, selectable=True))

    # ----------------------------------------------------------- sign off ---
    def _second_read(self) -> None:
        self.services.store.set_state(self.study_id, SECOND_READ)
        self.services.log("second read requested", study=self.study_id)
        self.row = self.services.store.study(self.study_id)
        self.services.studies_changed.emit()
        self._render()

    def _sign_off(self) -> None:
        latest = self.services.store.latest_decisions(self.study_id)
        pending = [f for f in self.analysis.shown if f.key not in latest]
        if pending:
            answer = QMessageBox.question(
                self, "Sign off", f"{len(pending)} finding(s) have no decision: "
                + ", ".join(f.title for f in pending) + ".\n\nUndecided findings are left out of "
                "the final read. Sign off anyway?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        final = self.services.store.final_labels(self.study_id, self.analysis)
        elapsed = (time.monotonic() - self.opened_at) * 1000
        report = (f"Final read: {', '.join(final) or 'No significant abnormality'}. "
                  f"AI suggestion: {self.analysis.explanation.get('headline', '')}")
        user = self.services.user
        self.services.store.sign_off(self.study_id, user.username, final, elapsed, report)
        self.services.log("report signed", {"final": final or ["No Finding"],
                                            "seconds": round(elapsed / 1000, 1),
                                            "ai_positive": [f.title for f in self.analysis.findings
                                                            if f.status == POSITIVE]},
                          self.study_id)
        self.row = self.services.store.study(self.study_id)
        self.services.studies_changed.emit()
        self._render()
