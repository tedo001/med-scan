"""Audit Trail - every action, chained by SHA-256, verifiable on screen."""

from __future__ import annotations

import json

from PyQt6.QtWidgets import (QComboBox, QFileDialog, QHeaderView, QLineEdit, QTableWidget,
                             QTableWidgetItem)

from ..components import Card, Page, Pill, button, clear, hbox, label


class AuditPage(Page):
    def __init__(self, services):
        super().__init__("Audit Trail", "append-only · hash-chained · no patient identifiers")
        self.services = services
        self.filter = QComboBox()
        self.filter.addItems(["All", "functionality", "system"])
        self.filter.currentIndexChanged.connect(self.refresh)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by study, actor or action")
        self.search.textChanged.connect(self.refresh)
        self.add_action(self.filter)
        self.add_action(self.search)
        self.add_action(button("Verify chain", "Primary", self._verify))
        self.add_action(button("Export CSV", "", self._export))
        self.chain = Card("Chain")
        self.chain_body = self.chain.body
        self.body.addWidget(self.chain)
        card = Card("Entries", "newest first")
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["When", "Actor", "Action", "Study", "Detail", "Hash"])
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        for column in (0, 1, 2, 3, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(520)
        card.body.addWidget(self.table)
        self.body.addWidget(card)

    def refresh(self):
        category = self.filter.currentText()
        needle = self.search.text().strip().lower()
        entries = self.services.audit.entries(1000)
        if category != "All":
            entries = [e for e in entries if e["category"] == category]
        if needle:
            entries = [e for e in entries if needle in json.dumps(e).lower()]
        self.table.setRowCount(len(entries))
        for r, e in enumerate(entries):
            values = [e["at"].replace("T", " "), f"{e['actor']} ({e['role'] or '-'})", e["action"],
                      e.get("study", ""), json.dumps(e.get("detail", {}), ensure_ascii=False)[:300],
                      e["hash"][:12] + "…"]
            for c, v in enumerate(values):
                self.table.setItem(r, c, QTableWidgetItem(v))
        self._verify(log=False)

    def _verify(self, log=True):
        report = self.services.audit.verify()
        clear(self.chain_body)
        if report.intact:
            self.chain_body.addLayout(hbox(Pill("CHAIN INTACT", "ok"),
                                           label(f"{report.entries} entries · head "
                                                 f"{report.head[:24]}…", "Mono"), None))
        else:
            self.chain_body.addLayout(hbox(Pill("CHAIN BROKEN", "fail"),
                                           label(f"at line {report.broken_at}: {report.reason}",
                                                 "Mono"), None))
        self.chain_body.addWidget(label("Each entry stores the SHA-256 of the one before it. Editing, "
                                        "inserting or deleting a line breaks the chain from that "
                                        "point. Note the head hash to detect lines removed from the "
                                        "end.", "Faint", wrap=True))
        if log:
            self.services.log("audit chain verified", {"intact": report.intact,
                                                       "entries": report.entries}, category="system")

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export audit trail", "medscan_audit.csv",
                                              "CSV (*.csv)")
        if path:
            n = self.services.audit.export_csv(path)
            self.services.log("audit exported", {"entries": n})
