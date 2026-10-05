"""The study database (SQLite): analyses, doctor decisions, questions, sign-offs.

Three tables:

``studies``    one row per analysed scan - the full :class:`~msx.pipeline.Analysis`
               as JSON plus the columns the worklist and dashboard filter on,
               the de-identified image, its heatmap and lung mask as PNGs, the
               doctor's blinded first read and final signed read
``decisions``  every Accept / Correct / Reject / Add on a finding, with who and when
``questions``  every question asked of a finding and the answer given

The blinded first read is what makes the brief's last evaluation question
answerable: *does Doctor + AI beat either alone?* The doctor records an
impression before the AI is revealed; the AI's own call is in the analysis;
the signed final read is Doctor + AI. :mod:`msx.evaluation` compares all three
against ground truth where it is known.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from . import imaging, paths
from .findings import POSITIVE
from .pipeline import Analysis

__all__ = ["DataStore", "LABELS", "OPEN", "SIGNED", "SECOND_READ"]

LABELS = ("Cardiomegaly", "Effusion", "Pneumothorax", "Consolidation", "Nodule", "Mass",
          "Atelectasis", "Edema")
OPEN, SIGNED, SECOND_READ = "open", "signed", "second-read"

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
    id TEXT PRIMARY KEY, created TEXT, source TEXT, sha256 TEXT, patient_ref TEXT,
    sex TEXT, age_band TEXT, view TEXT, site TEXT, context TEXT, mode TEXT, engine TEXT,
    status TEXT, quality_score INTEGER, top_label TEXT, top_prob REAL, top_conf REAL,
    urgent INTEGER, modules INTEGER, model_calls INTEGER, total_ms REAL,
    analysis_json TEXT, image_path TEXT, heat_path TEXT, mask_path TEXT,
    ground_truth TEXT, review_state TEXT DEFAULT 'open', analysed_by TEXT,
    first_read TEXT, first_read_ms REAL, final_read TEXT, decision_ms REAL,
    signed_by TEXT, signed_at TEXT, report TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, study_id TEXT, finding_key TEXT, label TEXT,
    action TEXT, corrected_label TEXT, note TEXT, user TEXT, role TEXT, at TEXT
);
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, study_id TEXT, finding_key TEXT, question TEXT,
    answer TEXT, engine TEXT, user TEXT, at TEXT
);
CREATE INDEX IF NOT EXISTS ix_studies_created ON studies(created);
CREATE INDEX IF NOT EXISTS ix_decisions_study ON decisions(study_id);
"""


def _plain(value):
    """json.dumps fallback for numpy scalars and arrays inside measurements."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class DataStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(paths.data_directory(), "medscan.db")
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def _exec(self, sql: str, args: Iterable = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._db.execute(sql, tuple(args))
            self._db.commit()
            return cursor

    def _query(self, sql: str, args: Iterable = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, tuple(args)).fetchall()

    # ------------------------------------------------------------- studies ----
    def save_analysis(self, analysis: Analysis, site: str = "", ground_truth: str = "",
                      user: str = "") -> str:
        folder = paths.images_directory()
        image_path = heat_path = mask_path = ""
        if analysis.work is not None:
            image_path = imaging.save_png(analysis.work, os.path.join(folder, f"{analysis.id}.png"))
        if analysis.heat is not None:
            heat_path = imaging.save_png(analysis.heat, os.path.join(folder, f"{analysis.id}_heat.png"))
        if analysis.lung_mask is not None:
            mask_path = imaging.save_png(analysis.lung_mask.astype(np.float32),
                                         os.path.join(folder, f"{analysis.id}_mask.png"))
        top = analysis.top
        scan = analysis.scan
        self._exec(
            "INSERT OR REPLACE INTO studies (id, created, source, sha256, patient_ref, sex, "
            "age_band, view, site, context, mode, engine, status, quality_score, top_label, "
            "top_prob, top_conf, urgent, modules, model_calls, total_ms, analysis_json, "
            "image_path, heat_path, mask_path, ground_truth, review_state, analysed_by) VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (analysis.id, analysis.created, scan.get("source", ""), "", scan.get("patient_ref", ""),
             scan.get("sex", ""), scan.get("age_band", ""), scan.get("view", ""), site,
             analysis.context, analysis.mode, analysis.engine, analysis.status,
             int(analysis.quality.get("score", 0)), top.title if top else "",
             top.probability if top else 0.0, top.confidence if top else 0.0,
             int(analysis.urgent), analysis.modules_activated, analysis.model_calls,
             analysis.total_ms, json.dumps(analysis.to_dict(), default=_plain), image_path, heat_path, mask_path,
             ground_truth, OPEN, user))
        return analysis.id

    def studies(self, state: Optional[str] = None, search: str = "",
                limit: int = 1000) -> List[Dict[str, object]]:
        sql = ("SELECT id, created, source, patient_ref, sex, age_band, view, site, context, "
               "mode, engine, status, quality_score, top_label, top_prob, top_conf, urgent, "
               "modules, model_calls, total_ms, ground_truth, review_state, first_read, "
               "first_read_ms, final_read, decision_ms, signed_by, signed_at FROM studies")
        where, args = [], []
        if state:
            where.append("review_state = ?")
            args.append(state)
        if search:
            where.append("(id LIKE ? OR source LIKE ? OR top_label LIKE ? OR patient_ref LIKE ?)")
            args += [f"%{search}%"] * 4
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY urgent DESC, created DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self._query(sql, args)]

    def study(self, study_id: str) -> Optional[Dict[str, object]]:
        rows = self._query("SELECT * FROM studies WHERE id = ?", (study_id,))
        return dict(rows[0]) if rows else None

    def load(self, study_id: str) -> Optional[Tuple[Analysis, Dict[str, object]]]:
        """The analysis with its images restored, plus the raw row."""
        row = self.study(study_id)
        if row is None:
            return None
        analysis = Analysis.from_dict(json.loads(row["analysis_json"]))
        analysis.work = self._png(row["image_path"])
        heat = self._png(row["heat_path"])
        analysis.heat = heat if heat is not None and heat.max() > 0 else None
        mask = self._png(row["mask_path"])
        analysis.lung_mask = mask > 0.5 if mask is not None else None
        return analysis, row

    @staticmethod
    def _png(path: str) -> Optional[np.ndarray]:
        if not path or not os.path.isfile(path):
            return None
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("L")).astype(np.float32) / 255

    def delete(self, study_id: str) -> None:
        row = self.study(study_id)
        if row:
            for key in ("image_path", "heat_path", "mask_path"):
                if row[key] and os.path.isfile(row[key]):
                    os.remove(row[key])
        self._exec("DELETE FROM studies WHERE id = ?", (study_id,))
        self._exec("DELETE FROM decisions WHERE study_id = ?", (study_id,))
        self._exec("DELETE FROM questions WHERE study_id = ?", (study_id,))

    def set_created(self, study_id: str, when: str) -> None:
        """Re-date a study (demo seeding and imports of historical studies)."""
        self._exec("UPDATE studies SET created = ? WHERE id = ?", (when, study_id))

    def set_ground_truth(self, study_id: str, labels: str) -> None:
        self._exec("UPDATE studies SET ground_truth = ? WHERE id = ?", (labels, study_id))

    # ----------------------------------------------------------- decisions ----
    def record_first_read(self, study_id: str, labels: List[str], elapsed_ms: float) -> None:
        self._exec("UPDATE studies SET first_read = ?, first_read_ms = ? WHERE id = ?",
                   (";".join(labels) or "No Finding", elapsed_ms, study_id))

    def record_decision(self, study_id: str, finding_key: str, label: str, action: str,
                        user: str, role: str, corrected_label: str = "", note: str = "") -> int:
        cursor = self._exec(
            "INSERT INTO decisions (study_id, finding_key, label, action, corrected_label, note, "
            "user, role, at) VALUES (?,?,?,?,?,?,?,?,?)",
            (study_id, finding_key, label, action, corrected_label, note, user, role, _now()))
        return int(cursor.lastrowid)

    def undo_decision(self, decision_id: int) -> None:
        self._exec("DELETE FROM decisions WHERE id = ?", (decision_id,))

    def decisions(self, study_id: Optional[str] = None) -> List[Dict[str, object]]:
        if study_id:
            rows = self._query("SELECT * FROM decisions WHERE study_id = ? ORDER BY id", (study_id,))
        else:
            rows = self._query("SELECT * FROM decisions ORDER BY id")
        return [dict(r) for r in rows]

    def latest_decisions(self, study_id: str) -> Dict[str, Dict[str, object]]:
        """finding_key -> the most recent decision on it."""
        out: Dict[str, Dict[str, object]] = {}
        for d in self.decisions(study_id):
            if d["action"] != "question":
                out[d["finding_key"]] = d
        return out

    def record_question(self, study_id: str, finding_key: str, question: str, answer: str,
                        engine: str, user: str) -> None:
        self._exec("INSERT INTO questions (study_id, finding_key, question, answer, engine, user, "
                   "at) VALUES (?,?,?,?,?,?,?)",
                   (study_id, finding_key, question, answer, engine, user, _now()))

    def questions(self, study_id: str) -> List[Dict[str, object]]:
        return [dict(r) for r in self._query(
            "SELECT * FROM questions WHERE study_id = ? ORDER BY id", (study_id,))]

    def final_labels(self, study_id: str, analysis: Analysis) -> List[str]:
        """The Doctor + AI read: accepted AI findings, corrections and additions."""
        latest = self.latest_decisions(study_id)
        labels = set()
        for f in analysis.findings:
            d = latest.get(f.key)
            if d is None:
                continue
            if d["action"] == "accept":
                labels.add(f.label)
            elif d["action"] == "correct" and d["corrected_label"] not in ("", "No Finding"):
                labels.add(d["corrected_label"])
        for d in latest.values():
            if d["action"] == "add" and d["label"]:
                labels.add(d["label"])
        return sorted(labels)

    def sign_off(self, study_id: str, user: str, final: List[str], decision_ms: float,
                 report: str = "") -> None:
        self._exec("UPDATE studies SET review_state = ?, final_read = ?, decision_ms = ?, "
                   "signed_by = ?, signed_at = ?, report = ? WHERE id = ?",
                   (SIGNED, ";".join(final) or "No Finding", decision_ms, user, _now(), report,
                    study_id))

    def set_state(self, study_id: str, state: str) -> None:
        self._exec("UPDATE studies SET review_state = ? WHERE id = ?", (state, study_id))

    # --------------------------------------------------------------- stats ----
    def counts(self) -> Dict[str, int]:
        row = self._query(
            "SELECT COUNT(*) total, SUM(review_state = 'open') open, "
            "SUM(review_state = 'second-read') second, SUM(review_state = 'signed') signed, "
            "SUM(status = 'quality-hold') held, SUM(status = 'findings') abnormal, "
            "SUM(urgent) urgent, SUM(status = 'no-finding') normal, "
            "SUM(status = 'uncertain') uncertain FROM studies")[0]
        return {k: int(row[k] or 0) for k in row.keys()}

    def agreement(self) -> Dict[str, int]:
        rows = self._query("SELECT action, COUNT(*) n FROM decisions GROUP BY action")
        return {r["action"]: int(r["n"]) for r in rows}
