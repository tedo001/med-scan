"""Machine learning that learns from the doctors - the feedback loop.

The deep model and the measurements are fixed; the doctors' decisions are not.
This module trains a small **logistic-regression stacker** on every finding the
analyser has produced and the verdict it eventually got:

========================  ===================================================
target = 1                ground truth says present, or the doctor accepted
                          it, or the doctor *added* that label (an AI miss)
target = 0                ground truth says absent, or the doctor rejected it,
                          or corrected it to something else
========================  ===================================================

Features per finding: the measurement and deep-model log-odds, whether the
deep model spoke, decisiveness, test-time-augmentation stability, model-measure
agreement, scan quality score, AP projection, and a one-hot of the finding
label (so it learns, say, that this site's nodule calls are over-eager).

It is trained with L2-regularised gradient descent in NumPy (no extra
dependency), evaluated by 5-fold cross-validation, and saved as JSON beside
the data. In :mod:`msx.pipeline` its output is blended into each finding's
probability with a weight that grows with the amount of feedback
(``min(0.5, n / 200)``) - so a handful of decisions cannot swing the engine,
and its probability is shown as its own source on the finding card. It is
retrained after every sign-off and from the Evaluation page.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import paths

__all__ = ["Learner", "features", "collect", "LABELS", "MIN_SAMPLES"]

LABELS = ("Cardiomegaly", "Effusion", "Pneumothorax", "Consolidation", "Nodule", "Mass",
          "Atelectasis", "Edema")
MIN_SAMPLES = 20
NUMERIC = ("measure_logit", "deep_logit", "has_deep", "decisiveness", "stability", "agreement",
           "quality", "ap_view")


def _logit(p: Optional[float]) -> float:
    if p is None:
        return 0.0
    p = min(max(float(p), 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def features(finding: Dict[str, object], quality_score: float, view: str) -> np.ndarray:
    """Inputs available *before* confidence scoring, so training and use match."""
    sources = finding.get("sources", {}) or {}
    u = finding.get("uncertainty", {}) or {}
    single = float(u.get("single", finding.get("probability") or 0.5))
    agreement = 1.0
    if "deep" in sources and "measurement" in sources:
        agreement = 1 - 0.5 * abs(float(sources["deep"]) - float(sources["measurement"]))
    numeric = [_logit(sources.get("measurement")) / 5, _logit(sources.get("deep")) / 5,
               1.0 if "deep" in sources else 0.0, abs(2 * single - 1),
               max(0.0, 1 - float(u.get("std", 0.0)) / 0.2), agreement,
               quality_score / 100.0, 1.0 if view == "AP" else 0.0]
    onehot = [1.0 if finding.get("label") == l else 0.0 for l in LABELS]
    return np.array(numeric + onehot, dtype=np.float64)


def collect(store) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(X, y, sources) from every study with ground truth or doctor decisions."""
    X, y, why = [], [], []
    for row in store.studies():
        full = store.study(row["id"])
        data = json.loads(full["analysis_json"])
        if data.get("status") == "quality-hold":
            continue
        truth = None
        if row.get("ground_truth"):
            truth = {l for l in row["ground_truth"].split(";") if l and l != "No Finding"}
        latest = store.latest_decisions(row["id"])
        added = {d["label"] for d in latest.values() if d["action"] == "add"}
        quality = float(data.get("quality", {}).get("score", 100))
        view = data.get("scan", {}).get("view", "PA")
        for f in data.get("findings", []):
            if f.get("label") not in LABELS:
                continue
            target, source = None, ""
            if truth is not None:
                target, source = int(f["label"] in truth), "ground truth"
            else:
                d = latest.get(f.get("key"))
                if d is not None and d["action"] in ("accept", "reject", "correct"):
                    target, source = int(d["action"] == "accept"), "doctor"
                elif f["label"] in added:
                    target, source = 1, "doctor added"
            if target is None:
                continue
            X.append(features(f, quality, view))
            y.append(target)
            why.append(source)
    if not X:
        return np.zeros((0, len(NUMERIC) + len(LABELS))), np.zeros(0), []
    return np.vstack(X), np.array(y, dtype=np.float64), why


def _fit(X: np.ndarray, y: np.ndarray, l2: float = 0.05, steps: int = 600,
         rate: float = 0.5) -> np.ndarray:
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        grad = Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0]
        w -= rate * grad
    return w


def _predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((len(X), 1))])
    return 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))


class Learner:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(paths.data_directory(), "learner.json")
        self.weights: Optional[np.ndarray] = None
        self.info: Dict[str, object] = {}
        self.load()

    @property
    def ready(self) -> bool:
        return self.weights is not None and int(self.info.get("n", 0)) >= MIN_SAMPLES

    @property
    def weight(self) -> float:
        """How much the learned probability counts - grows with the feedback available."""
        return min(0.5, int(self.info.get("n", 0)) / 200) if self.ready else 0.0

    def load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as handle:
                data = json.load(handle)
            self.weights = np.array(data["weights"], dtype=np.float64)
            self.info = data.get("info", {})
        except (OSError, ValueError, KeyError):
            self.weights, self.info = None, {}

    def train(self, store) -> Dict[str, object]:
        X, y, why = collect(store)
        info: Dict[str, object] = {"n": int(len(y)), "positives": int(y.sum()) if len(y) else 0,
                                   "trained_at": datetime.now().isoformat(timespec="seconds"),
                                   "sources": {s: why.count(s) for s in set(why)}}
        if len(y) < MIN_SAMPLES or len(set(y.tolist())) < 2:
            info["status"] = f"need ≥ {MIN_SAMPLES} labelled findings with both outcomes (have {len(y)})"
            self.info = info
            return info
        # 5-fold cross-validated accuracy, then the final fit on everything
        rng = np.random.default_rng(0)
        order = rng.permutation(len(y))
        folds = np.array_split(order, 5)
        correct, brier = 0, 0.0
        for fold in folds:
            train = np.setdiff1d(order, fold)
            w = _fit(X[train], y[train])
            p = _predict(w, X[fold])
            correct += int(((p >= 0.5) == (y[fold] == 1)).sum())
            brier += float(((p - y[fold]) ** 2).sum())
        self.weights = _fit(X, y)
        info.update(status="trained", cv_accuracy=round(correct / len(y), 3),
                    cv_brier=round(brier / len(y), 3))
        self.info = info
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"weights": self.weights.tolist(), "info": info,
                       "features": list(NUMERIC) + [f"label={l}" for l in LABELS]}, handle, indent=1)
        return info

    def predict(self, finding: Dict[str, object], quality_score: float, view: str) -> Optional[float]:
        if not self.ready:
            return None
        return float(_predict(self.weights, features(finding, quality_score, view)[None])[0])
