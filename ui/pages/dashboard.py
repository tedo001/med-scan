"""Dashboard - what the analyser and the doctors have been doing."""

from __future__ import annotations

import json
from collections import Counter

from PyQt6.QtWidgets import QGridLayout

from .. import theme
from ..components import (Card, ColumnChart, Donut, HBarChart, Page, StatStrip, button, label)


class DashboardPage(Page):
    def __init__(self, services):
        super().__init__("Dashboard", "all studies · adaptive routing · doctor decisions")
        self.services = services
        self.add_action(button("Refresh", "", self.refresh))
        self.stats = StatStrip([("Studies", "0", "analysed"),
                                ("Abnormal", "0", "with findings"),
                                ("Early exits", "0%", "no specialist needed"),
                                ("Modules / scan", "0", "of 4 - adaptive"),
                                ("Mean time", "0 ms", "per scan"),
                                ("AI agreement", "—", "accepted / decided")])
        self.body.addWidget(self.stats)
        grid = QGridLayout()
        grid.setSpacing(12)
        self.findings = HBarChart(240)
        self.status = Donut(240)
        self.modules = HBarChart(240, colour=theme.TEAL, percent=True)
        self.confidence = ColumnChart(240)
        self.decisions = Donut(240)
        self.contexts = ColumnChart(240)
        cards = [("Findings reported", "positive + possible", self.findings),
                 ("Study outcome", "", self.status),
                 ("How often each module ran", "share of analysed scans", self.modules),
                 ("Confidence of shown findings", "count by level", self.confidence),
                 ("Doctor decisions", "on AI findings", self.decisions),
                 ("Cost by clinical context", "mean modules · mean seconds", self.contexts)]
        for i, (title, caption, chart) in enumerate(cards):
            card = Card(title, caption)
            card.body.addWidget(chart)
            grid.addWidget(card, i // 3, i % 3)
        self.body.addLayout(grid)
        self.body.addStretch(1)

    def refresh(self) -> None:
        store = self.services.store
        rows = store.studies()
        counts = store.counts()
        analysed = [r for r in rows if r["status"] != "quality-hold"]
        findings, levels, module_runs = Counter(), Counter(), Counter()
        early = 0
        by_context = {}
        for row in rows:
            data = json.loads(store.study(row["id"])["analysis_json"])
            for f in data.get("findings", []):
                if f["status"] in ("positive", "possible"):
                    findings[f["label"]] += 1
                if f["status"] != "negative":
                    levels[f.get("confidence_level") or "Low"] += 1
            routing = data.get("routing", {})
            if routing.get("early_exit"):
                early += 1
            for r in routing.get("routes", []):
                if r.get("run"):
                    module_runs[r["module"]] += 1
            ctx = by_context.setdefault(row["context"], [0, 0.0, 0])
            ctx[0] += row["modules"] or 0
            ctx[1] += (row["total_ms"] or 0) / 1000
            ctx[2] += 1
        agreement = store.agreement()
        decided = sum(agreement.get(k, 0) for k in ("accept", "reject", "correct"))
        n = max(1, len(analysed))
        self.stats.set(0, str(counts["total"]), f"{counts['held']} held for quality")
        self.stats.set(1, str(counts["abnormal"]), f"{counts['urgent']} urgent")
        self.stats.set(2, f"{early / n:.0%}" if analysed else "—")
        self.stats.set(3, f"{sum(r['modules'] or 0 for r in analysed) / n:.2f}" if analysed else "—")
        self.stats.set(4, f"{sum(r['total_ms'] or 0 for r in rows) / max(1, len(rows)):,.0f} ms"
                       if rows else "—")
        self.stats.set(5, f"{agreement.get('accept', 0) / decided:.0%}" if decided else "—",
                       f"{decided} decisions · {agreement.get('question', 0)} questions")

        self.findings.set(findings.most_common(8))
        self.status.set([("Findings", counts["abnormal"], theme.RED),
                         ("Normal", counts["normal"], theme.GREEN),
                         ("Abstained", counts["uncertain"], "#7C3AED"),
                         ("Quality hold", counts["held"], theme.AMBER)], str(counts["total"]))
        self.modules.set([(m, module_runs.get(m, 0) / n) for m in
                          ("Cardiac", "Pleural", "Parenchyma", "Nodule")])
        self.confidence.set(["High", "Moderate", "Low"],
                            [("findings", [levels.get(k, 0) for k in ("High", "Moderate", "Low")],
                              theme.NAVY)])
        self.decisions.set([("Accepted", agreement.get("accept", 0), theme.GREEN),
                            ("Corrected", agreement.get("correct", 0), "#D97706"),
                            ("Rejected", agreement.get("reject", 0), theme.RED),
                            ("Added (AI missed)", agreement.get("add", 0), "#7C3AED")],
                           str(decided + agreement.get("add", 0)))
        names = sorted(by_context)
        self.contexts.set(names, [
            ("modules", [by_context[c][0] / by_context[c][2] for c in names], theme.NAVY),
            ("seconds", [by_context[c][1] / by_context[c][2] for c in names], theme.TEAL)],
            fmt="{:.1f}")
