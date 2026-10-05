"""Dashboard - the SENTRA layout, for chest X-ray studies.

Filter row (period, site, finding, clear, export) over a stat strip; a trend
card (abnormal and urgent studies over time, total volume on a second axis,
Today / Week / Month / Year / All, Line / Bar) beside a radar profile card
(findings, lung zones or modules); then breakdowns: findings reported, router
activity, doctor decisions, AI-doctor agreement per finding and recent studies.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                             QHeaderView, QTableWidget, QTableWidgetItem, QVBoxLayout)

from msx.datastore import LABELS

from .. import theme
from ..components import (Card, Donut, HBarChart, Page, RadarChart, Segmented, StatStrip,
                          TrendChart, button, clear, hbox, label)

PERIODS = {"Last 30 days": 30, "Last 90 days": 90, "12 months": 365}
ZONES = ("R upper", "R middle", "R lower", "L lower", "L middle", "L upper")


class DashboardPage(Page):
    def __init__(self, services):
        super().__init__("Dashboard", "")
        self.services = services
        self.rows: List[Dict[str, object]] = []
        self.data: Dict[str, dict] = {}
        self.updated = label("", "PageCaption")
        self.add_action(self.updated)
        self.period = Segmented(list(PERIODS), "Last 90 days")
        self.period.changed.connect(lambda _: self.refresh())
        self.add_action(self.period)
        self.site = QComboBox()
        self.site.setMinimumWidth(150)
        self.site.currentIndexChanged.connect(lambda _: self._filters_changed())
        self.finding = QComboBox()
        self.finding.setMinimumWidth(150)
        self.finding.addItems(["All findings"] + list(LABELS))
        self.finding.currentIndexChanged.connect(lambda _: self._filters_changed())
        self.add_action(self.site)
        self.add_action(self.finding)
        self.add_action(button("Clear filters", "", self._clear))
        self.add_action(button("Export CSV", "", self._export))

        self.stats = StatStrip([("Total studies", "0", "analysed"),
                                ("Abnormal", "0", "with findings"),
                                ("Mean confidence", "—", "shown findings, 0-1"),
                                ("Urgent", "0", "positive, high acuity"),
                                ("Awaiting review", "0", "open + second read")])
        self.body.addWidget(self.stats)

        top = QGridLayout()
        top.setSpacing(12)
        self.trend_range = Segmented(["Today", "Week", "Month", "Year", "All"], "Year")
        self.trend_range.changed.connect(lambda _: self._trend())
        self.trend_mode = Segmented(["↗ Line", "▮ Bar"], "↗ Line")
        self.trend_mode.changed.connect(lambda m: self.trend.set_mode("bar" if "Bar" in m else "line"))
        trend_card = Card("Finding trend — Abnormal & urgent studies", "",
                          right=self._pair(self.trend_range, self.trend_mode))
        self.trend = TrendChart(320)
        trend_card.body.addWidget(self.trend)
        self.trend_cells = StatStrip([("Peak abnormal / bucket", "0", ""),
                                      ("Average abnormal / bucket", "0", ""),
                                      ("Peak urgent / bucket", "0", ""),
                                      ("Total studies in period", "0", "")])
        self.trend_cells.setStyleSheet(f"QFrame#Card {{ background: {theme.G[50]}; }}")
        trend_card.body.addWidget(self.trend_cells)
        self.profile_kind = Segmented(["Findings", "Lung zones", "Modules"], "Findings")
        self.profile_kind.changed.connect(lambda _: self._profile())
        profile_card = Card("Finding profile", "", right=self.profile_kind)
        self.radar = RadarChart(400)
        profile_card.body.addWidget(self.radar)
        top.addWidget(trend_card, 0, 0)
        top.addWidget(profile_card, 0, 1)
        top.setColumnStretch(0, 2)
        top.setColumnStretch(1, 1)
        self.body.addLayout(top)

        grid = QGridLayout()
        grid.setSpacing(12)
        self.found = HBarChart(230)
        self.modules = HBarChart(230, colour=theme.TEAL, percent=True)
        self.decisions = Donut(230)
        self.agreement = HBarChart(230, colour=theme.GREEN, percent=True)
        for i, (title, caption, chart) in enumerate((
                ("Findings reported", "positive + possible", self.found),
                ("Router activity", "share of analysed scans", self.modules),
                ("Doctor decisions", "on AI findings", self.decisions),
                ("AI–doctor agreement", "accepted / decided, per finding", self.agreement))):
            card = Card(title, caption)
            card.body.addWidget(chart)
            grid.addWidget(card, i // 2, i % 2)
        self.body.addLayout(grid)

        recent = Card("Recent studies", "", right=button("Review queue", "Link",
                                                          lambda: self.services.open_study.emit(
                                                              self.rows[0]["id"]) if self.rows else None))
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Ref", "Analysed", "Site", "Top finding", "Prob", "Status"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(300)
        self.table.cellDoubleClicked.connect(
            lambda r, _: self.services.open_study.emit(self.table.item(r, 0).text()))
        recent.body.addWidget(self.table)
        self.body.addWidget(recent)
        self.body.addStretch(1)

    @staticmethod
    def _pair(*widgets):
        holder = QFrame()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        for w in widgets:
            row.addWidget(w)
        return holder

    # --------------------------------------------------------------- data ---
    def _load(self) -> None:
        store = self.services.store
        self.all_rows = store.studies(limit=100000)
        self.data = {}
        for row in self.all_rows:
            self.data[row["id"]] = json.loads(store.study(row["id"])["analysis_json"])
        sites = sorted({r["site"] for r in self.all_rows if r["site"]})
        current = self.site.currentText()
        self.site.blockSignals(True)
        self.site.clear()
        self.site.addItems(["All sites"] + sites)
        if current in sites:
            self.site.setCurrentText(current)
        self.site.blockSignals(False)

    def _filtered(self) -> List[Dict[str, object]]:
        days = PERIODS[self.period.value]
        since = datetime.now() - timedelta(days=days)
        site = self.site.currentText()
        finding = self.finding.currentText()
        out = []
        for row in self.all_rows:
            if datetime.fromisoformat(row["created"]) < since:
                continue
            if site not in ("", "All sites") and row["site"] != site:
                continue
            if finding != "All findings" and not any(
                    f["label"] == finding and f["status"] in ("positive", "possible")
                    for f in self.data[row["id"]].get("findings", [])):
                continue
            out.append(row)
        return out

    def _filters_changed(self) -> None:
        if hasattr(self, "all_rows"):
            self._render()

    def _clear(self) -> None:
        for w in (self.site, self.finding):
            w.blockSignals(True)
            w.setCurrentIndex(0)
            w.blockSignals(False)
        self.period.set("Last 90 days")
        self.refresh()

    def refresh(self) -> None:
        self._load()
        self._render()

    def _render(self) -> None:
        self.rows = self._filtered()
        days = PERIODS[self.period.value]
        start = (datetime.now() - timedelta(days=days)).strftime("%d %b %Y")
        self.caption.setText(f"{self.period.value} ({start} – {datetime.now():%d %b %Y}) · "
                             f"{len(self.rows)} studies · engine assessments")
        self.updated.setText(f"Updated {datetime.now():%d %b %Y, %H:%M}")
        self._stats()
        self._trend()
        self._profile()
        self._breakdowns()
        self._recent()

    # -------------------------------------------------------------- cards ---
    def _stats(self) -> None:
        rows = self.rows
        n = len(rows)
        abnormal = sum(1 for r in rows if r["status"] == "findings")
        urgent = sum(1 for r in rows if r["urgent"])
        pending = sum(1 for r in rows if r["review_state"] in ("open", "second-read"))
        confs = [f["confidence"] for r in rows for f in self.data[r["id"]].get("findings", [])
                 if f["status"] != "negative"]
        self.stats.set(0, str(n), f"{sum(1 for r in rows if r['status'] == 'quality-hold')} held for quality")
        self.stats.set(1, str(abnormal), f"{abnormal / n:.1%} · engine" if n else "—")
        self.stats.set(2, f"{sum(confs) / len(confs):.2f}" if confs else "—", "0-1, shown findings")
        self.stats.set(3, f"▲ {urgent}" if urgent else "0", "positive, high acuity",
                       colour=theme.RED if urgent else None)
        self.stats.set(4, str(pending), "+ second reads · now")

    def _buckets(self):
        kind = self.trend_range.value
        now = datetime.now()
        if kind == "Today":
            keys = [now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=h) for h in range(23, -1, -1)]
            key = lambda d: d.replace(minute=0, second=0, microsecond=0)  # noqa: E731
            fmt, unit = "%H:00", "hourly"
        elif kind in ("Week", "Month"):
            count = 7 if kind == "Week" else 30
            keys = [(now - timedelta(days=d)).date() for d in range(count - 1, -1, -1)]
            key = lambda d: d.date()  # noqa: E731
            fmt, unit = "%d %b", "daily"
        else:
            first = now.replace(day=1)
            months = 12
            if kind == "All" and self.all_rows:
                oldest = min(datetime.fromisoformat(r["created"]) for r in self.all_rows)
                months = max(1, (now.year - oldest.year) * 12 + now.month - oldest.month + 1)
            keys = []
            y, m = first.year, first.month
            for _ in range(months):
                keys.append((y, m))
                m -= 1
                if m == 0:
                    y, m = y - 1, 12
            keys.reverse()
            key = lambda d: (d.year, d.month)  # noqa: E731
            fmt, unit = "%b %y", "monthly"
        return keys, key, fmt, unit

    def _trend(self) -> None:
        keys, key, fmt, unit = self._buckets()
        rows = self.all_rows if self.trend_range.value == "All" else [
            r for r in self.all_rows if r in self.rows or self.trend_range.value != "Year"]
        site, finding = self.site.currentText(), self.finding.currentText()
        abnormal, urgent, total = Counter(), Counter(), Counter()
        for row in self.all_rows:
            if site not in ("", "All sites") and row["site"] != site:
                continue
            if finding != "All findings" and not any(
                    f["label"] == finding and f["status"] in ("positive", "possible")
                    for f in self.data[row["id"]].get("findings", [])):
                continue
            k = key(datetime.fromisoformat(row["created"]))
            total[k] += 1
            abnormal[k] += row["status"] == "findings"
            urgent[k] += bool(row["urgent"])
        labels = []
        for k in keys:
            if isinstance(k, tuple):
                labels.append(datetime(k[0], k[1], 1).strftime(fmt))
            elif isinstance(k, datetime):
                labels.append(k.strftime(fmt))
            else:
                labels.append(datetime(k.year, k.month, k.day).strftime(fmt))
        a = [abnormal[k] for k in keys]
        u = [urgent[k] for k in keys]
        t = [total[k] for k in keys]
        average = sum(a) / len(a) if a else 0
        first_label, last_label = (labels[0], labels[-1]) if labels else ("", "")
        self.trend.set(labels, [("Abnormal", a, theme.NAVY, "solid", "left"),
                                ("Urgent", u, theme.RED, "dash", "left"),
                                ("Total studies", t, theme.G[400], "dot", "right")],
                       average=average, left_title="Studies with findings", right_title="Studies",
                       caption=f"{first_label} → {last_label} · {unit}")
        peak_a = max(a) if a else 0
        peak_u = max(u) if u else 0
        self.trend_cells.set(0, str(peak_a), labels[a.index(peak_a)] if peak_a else "—")
        self.trend_cells.set(1, f"{average:.1f}", f"each {unit[:-2] if unit != 'daily' else 'day'} bucket")
        self.trend_cells.set(2, str(peak_u), labels[u.index(peak_u)] if peak_u else "—")
        self.trend_cells.set(3, str(sum(t)), "dated in this range")

    def _profile(self) -> None:
        kind = self.profile_kind.value
        rows = self.rows
        if kind == "Findings":
            cats = list(LABELS)
            pos, urg, chk = Counter(), Counter(), Counter()
            for r in rows:
                for f in self.data[r["id"]].get("findings", []):
                    chk[f["label"]] += 1
                    if f["status"] in ("positive", "possible"):
                        pos[f["label"]] += 1
                        if r["urgent"]:
                            urg[f["label"]] += 1
            series = [("Findings", [pos[c] for c in cats], theme.NAVY, "solid"),
                      ("In urgent studies", [urg[c] for c in cats], theme.RED, "dash"),
                      ("Checked", [chk[c] for c in cats], theme.G[400], "dot")]
        elif kind == "Lung zones":
            cats = list(ZONES)
            names = {("right", "upper"): 0, ("right", "middle"): 1, ("right", "lower"): 2,
                     ("left", "lower"): 3, ("left", "middle"): 4, ("left", "upper"): 5}
            pos, urg = [0] * 6, [0] * 6
            for r in rows:
                for f in self.data[r["id"]].get("findings", []):
                    if f["status"] not in ("positive", "possible") or f.get("side") not in ("right", "left"):
                        continue
                    for z in f.get("zones") or []:
                        idx = names.get((f["side"], z))
                        if idx is not None:
                            pos[idx] += 1
                            urg[idx] += bool(r["urgent"])
            series = [("Findings", pos, theme.NAVY, "solid"), ("In urgent studies", urg, theme.RED, "dash")]
        else:
            cats = ["Cardiac", "Pleural", "Parenchyma", "Nodule"]
            ran, found = Counter(), Counter()
            group_module = {"cardiac": "Cardiac", "pleural": "Pleural", "parenchymal": "Parenchyma",
                            "focal": "Nodule"}
            for r in rows:
                data = self.data[r["id"]]
                for route in data.get("routing", {}).get("routes", []):
                    if route.get("run"):
                        ran[route["module"]] += 1
                for f in data.get("findings", []):
                    if f["status"] == "positive":
                        found[group_module.get(f["group"], "")] += 1
            series = [("Module ran", [ran[c] for c in cats], theme.TEAL, "solid"),
                      ("Positive finding", [found[c] for c in cats], theme.RED, "dash")]
        self.radar.set(cats, series, caption=f"{self.period.value.lower()} · {len(rows)} studies")

    def _breakdowns(self) -> None:
        rows = self.rows
        analysed = [r for r in rows if r["status"] != "quality-hold"]
        found, ran = Counter(), Counter()
        early = 0
        for r in rows:
            data = self.data[r["id"]]
            for f in data.get("findings", []):
                if f["status"] in ("positive", "possible"):
                    found[f["label"]] += 1
            routing = data.get("routing", {})
            early += bool(routing.get("early_exit"))
            for route in routing.get("routes", []):
                if route.get("run"):
                    ran[route["module"]] += 1
        n = max(1, len(analysed))
        self.found.set(found.most_common(8))
        self.modules.set([(m, ran.get(m, 0) / n) for m in ("Cardiac", "Pleural", "Parenchyma", "Nodule")]
                         + [("Early exit", early / n)])
        ids = {r["id"] for r in rows}
        actions, per_label = Counter(), defaultdict(lambda: [0, 0])
        for d in self.services.store.decisions():
            if d["study_id"] not in ids:
                continue
            actions[d["action"]] += 1
            if d["action"] in ("accept", "reject", "correct"):
                per_label[d["label"]][1] += 1
                per_label[d["label"]][0] += d["action"] == "accept"
        decided = sum(actions[k] for k in ("accept", "reject", "correct"))
        self.decisions.set([("Accepted", actions["accept"], theme.GREEN),
                            ("Corrected", actions["correct"], "#D97706"),
                            ("Rejected", actions["reject"], theme.RED),
                            ("Added (AI missed)", actions["add"], "#7C3AED")],
                           str(decided + actions["add"]))
        self.agreement.set(sorted(((l, a / t) for l, (a, t) in per_label.items() if t),
                                  key=lambda x: -x[1]))

    def _recent(self) -> None:
        rows = self.rows[:40]
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [row["id"], row["created"][:16].replace("T", " "), row["site"] or "—",
                      ("▲ " if row["urgent"] else "") + (row["top_label"] or row["status"]),
                      f"{row['top_prob']:.0%}" if row["top_label"] else "—",
                      {"open": "Awaiting", "signed": "Signed", "second-read": "Second read"}[row["review_state"]]]
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                if c == 4 and row["top_label"]:
                    from PyQt6.QtGui import QColor

                    item.setForeground(QColor(theme.prob_colour(row["top_prob"])))
                self.table.setItem(r, c, item)

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export studies", "medscan_studies.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        keys = ["id", "created", "site", "sex", "age_band", "context", "status", "top_label",
                "top_prob", "top_conf", "urgent", "modules", "total_ms", "review_state", "final_read"]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(keys)
            for row in self.rows:
                writer.writerow([row.get(k, "") for k in keys])
        self.services.log("dashboard exported", {"rows": len(self.rows), "period": self.period.value})
