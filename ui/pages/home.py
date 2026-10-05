"""Home - today at a glance, how the analyser works, and where to start."""

from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QGridLayout, QVBoxLayout

from msx.pipeline import STAGES

from .. import theme
from ..components import Card, Page, StatStrip, button, clear, hbox, label, vbox

STAGE_TEXT = {
    "Preprocess": "Load PNG/JPEG/DICOM, strip identifiers, letterbox to 512², find the lungs",
    "Quality": "7 checks - a failed scan stops here and goes to a human",
    "Screen": "Fast whole-image screen: measurements + DenseNet-121, fused",
    "Route": "Decide which specialist modules this scan needs",
    "Specialists": "Cardiac · Pleural · Parenchyma · Nodule - only the routed ones",
    "Confidence": "Calibration, test-time augmentation, abstention",
    "Explain": "Adaptive depth, evidence retrieved from guidelines and literature",
    "Record": "Your Accept / Correct / Reject / Question, hash-chained audit",
}


class HomePage(Page):
    navigate = pyqtSignal(str)

    def __init__(self, services):
        super().__init__("Home", "")
        self.services = services
        self.greeting = label("", "Headline")
        self.body.addWidget(self.greeting)
        self.stats = StatStrip([("Studies analysed", "0", "all time"),
                                ("Awaiting review", "0", "open + second read"),
                                ("Urgent", "0", "positive, high-acuity"),
                                ("Quality holds", "0", "sent to a human"),
                                ("AI agreement", "—", "accepted / decided")])
        self.body.addWidget(self.stats)

        actions = Card("Start here")
        actions.body.addLayout(hbox(
            button("Analyse a scan", "Primary", lambda: self.navigate.emit("Analyse")),
            button("Open review queue", "", lambda: self.navigate.emit("Review")),
            button("Run the evaluation", "", lambda: self.navigate.emit("Evaluation")), None))
        actions.body.addWidget(label(
            "New here? On Analyse press “Load demo samples” - 25 synthetic phantoms with known "
            "ground truth (normal, cardiomegaly, effusion, consolidation, pneumothorax, nodule, "
            "mass, and three deliberately poor scans) - then Analyse and open any result in "
            "Review.", "Small", wrap=True))
        self.body.addWidget(actions)

        how = Card("How the analyser works", "adaptive analysis routing")
        grid = QGridLayout()
        grid.setSpacing(8)
        for i, stage in enumerate(STAGES):
            cell = QFrame()
            cell.setObjectName("StageChip")
            cell.setProperty("state", "ok" if stage in ("Route",) else "skipped")
            col = QVBoxLayout(cell)
            col.setContentsMargins(10, 8, 10, 8)
            title = label(f"{i + 1}. {stage}", "Small")
            title.setStyleSheet(f"font-weight: 700; color: {theme.NAVY};")
            col.addWidget(title)
            col.addWidget(label(STAGE_TEXT[stage], "Faint", wrap=True))
            grid.addWidget(cell, i // 4, i % 4)
        how.body.addLayout(grid)
        how.body.addWidget(label(
            "Simple scans stop early; complex ones get deeper analysis. Every finding carries a "
            "probability, a confidence, its evidence and its limitations - and you decide.",
            "Small", wrap=True))
        self.body.addWidget(how)

        self.recent = Card("Recent studies", "")
        self.recent_body = QVBoxLayout()
        self.recent.body.addLayout(self.recent_body)
        self.body.addWidget(self.recent)
        self.body.addStretch(1)

    def refresh(self) -> None:
        user = self.services.user
        hour = datetime.now().hour
        part = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
        self.greeting.setText(f"Good {part}, {user.display if user else ''}.")
        self.caption.setText(datetime.now().strftime("%A %d %B %Y · ") + self.services.engine_status())
        c = self.services.store.counts()
        agreement = self.services.store.agreement()
        decided = sum(agreement.get(k, 0) for k in ("accept", "reject", "correct"))
        self.stats.set(0, str(c["total"]))
        self.stats.set(1, str(c["open"] + c["second"]))
        self.stats.set(2, str(c["urgent"]), colour=theme.RED if c["urgent"] else None)
        self.stats.set(3, str(c["held"]))
        self.stats.set(4, f"{agreement.get('accept', 0) / decided:.0%}" if decided else "—",
                       f"{decided} decisions")
        clear(self.recent_body)
        rows = self.services.store.studies(limit=6)
        if not rows:
            self.recent_body.addWidget(label("No studies yet.", "Note"))
        for row in rows:
            open_button = button("Open", "Link",
                                 lambda _=False, i=row["id"]: self.services.open_study.emit(i))
            self.recent_body.addLayout(hbox(
                label(row["id"], "Mono"), label(row["source"][:34], "Small"),
                label(("▲ " if row["urgent"] else "") + (row["top_label"] or row["status"]), "Small"),
                None, label(row["review_state"], "MonoSmall"), open_button))
