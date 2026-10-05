"""Bias Monitor - the analyser's behaviour by sex, age band, site and view."""

from __future__ import annotations

from PyQt6.QtWidgets import QGridLayout, QHeaderView, QTableWidget, QTableWidgetItem

from msx import bias

from ..components import Card, Page, button, clear, label


def _p(v):
    return "—" if v is None else f"{v:.0%}"


class BiasPage(Page):
    def __init__(self, services):
        super().__init__("Bias Monitor", f"subgroups with n ≥ {bias.MIN_N} are flagged when "
                                         f"sensitivity or agreement trails overall by > "
                                         f"{bias.GAP:.0%}")
        self.services = services
        self.add_action(button("Record review in audit trail", "", self._record))
        self.flags = Card("Flags")
        self.body.addWidget(self.flags)
        self.grid = QGridLayout()
        self.grid.setSpacing(12)
        self.body.addLayout(self.grid)
        self.body.addStretch(1)
        self.data = None

    def refresh(self):
        self.data = bias.subgroups(self.services.store)
        clear(self.flags.body)
        o = self.data["overall"]
        self.flags.body.addWidget(label(
            f"Overall: {o.get('n', 0)} studies · abnormal {_p(o.get('abnormal_rate'))} · quality "
            f"hold {_p(o.get('hold_rate'))} · agreement {_p(o.get('agreement'))} · sensitivity "
            f"{_p(o.get('sensitivity'))}", "Body"))
        if self.data["flags"]:
            for flag in self.data["flags"]:
                item = label(f"▲ {flag}", "Body")
                item.setStyleSheet("color:#B91C1C; font-weight:600;")
                self.flags.body.addWidget(item)
        else:
            self.flags.body.addWidget(label("No subgroup disparity above the threshold.", "Small"))
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        for i, (title, table_data) in enumerate(self.data["dimensions"].items()):
            card = Card(title)
            headers = ["Group", "n", "Abnormal", "Held", "Mean conf", "Agreement", "Sensitivity"]
            table = QTableWidget(len(table_data), len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.verticalHeader().setVisible(False)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            table.horizontalHeader().setStretchLastSection(True)
            for r, (group, s) in enumerate(table_data.items()):
                values = [group, str(s["n"]), _p(s.get("abnormal_rate")), _p(s.get("hold_rate")),
                          "—" if s.get("mean_confidence") is None else f"{s['mean_confidence']:.2f}",
                          _p(s.get("agreement")), _p(s.get("sensitivity"))]
                for c, v in enumerate(values):
                    table.setItem(r, c, QTableWidgetItem(v))
            table.setMinimumHeight(60 + 32 * len(table_data))
            card.body.addWidget(table)
            self.grid.addWidget(card, i // 2, i % 2)

    def _record(self):
        if self.data:
            self.services.log("bias review", {"overall": self.data["overall"],
                                              "flags": self.data["flags"]})
