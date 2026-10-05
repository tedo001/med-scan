"""The dataset model - a classifier head trained on your own labelled X-rays.

The DenseNet-121 is pretrained on eight public datasets; its 1024 learned image
features (the last layer before its classifier) describe a chest X-ray very
well. This module trains a new, small classifier on top of those features for
the labels your dataset has (e.g. pneumonia → Consolidation vs Normal) - the
standard "transfer learning" recipe, fast enough to run on a laptop CPU:

1. every sampled image → 512² working image → DenseNet features (frozen) → 1024 numbers
2. standardise them; train one L2-regularised logistic regression per label
   (one-vs-rest), scored by 5-fold cross-validated AUC, sensitivity and specificity
3. save the head as a version (rollback like the feedback model)

**Out-of-distribution guard.** A head trained on, say, Kaggle pneumonia films
should not vote on images clearly unlike them. Each scan's feature vector is
compared with a reference sample of the training images by cosine similarity;
if its nearest neighbour is further away than the 99th percentile seen within
the training set (leave-one-out, ×1.10), the head abstains and the analysis
says so. Honest limit: the DenseNet's features rate synthetic phantoms as
plausible chest films, so the guard does not catch them - what protects the
phantoms is that the head's vote is *weighted evidence* (log-odds, weight 0.7)
and only *corroborates*: it can raise a probability only when the measurements
already reach 0.35, so it never creates a finding on its own.

In the analyser the head's probability is added to the screen (so it can route
the right specialist) and to the strongest matching finding, in log-odds with
weight :data:`WEIGHT`, shown as the "dataset model" source on the finding card.
"""

from __future__ import annotations

import glob
import json
import math
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import paths
from .learner import _fit, _predict

__all__ = ["DatasetHead", "embed_folder", "WEIGHT"]

WEIGHT = 0.7
OOD_MARGIN = 1.10
REFERENCE_SIZE = 300
CORROBORATE = 0.35


def _auc(scores, truth) -> Optional[float]:
    pos = [s for s, t in zip(scores, truth) if t]
    neg = [s for s, t in zip(scores, truth) if not t]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def embed_folder(folder: str, deep, progress: Optional[Callable] = None
                 ) -> Tuple[np.ndarray, List[str], List[str]]:
    """(embeddings, label per image, files) for a labels.csv folder."""
    from . import evaluation, imaging

    rows = evaluation.load_labels(folder)
    vectors, labels, files = [], [], []
    for index, row in enumerate(rows, 1):
        try:
            scan = imaging.load_scan(row["path"])
            vectors.append(deep.embed(imaging.standardise(scan.pixels)))
            labels.append(next(iter(row["truth"]), "No Finding"))
            files.append(row["file"])
        except Exception as error:  # noqa: BLE001 - one unreadable image must not stop training
            if progress:
                progress(index, len(rows), f"skipped {row['file']}: {error}")
            continue
        if progress:
            progress(index, len(rows), f"features: {os.path.basename(row['file'])}")
    if not vectors:
        raise ValueError("no readable images")
    return np.vstack(vectors), labels, files


class DatasetHead:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(paths.data_directory(), "dataset_head.json")
        self.info: Dict[str, object] = {}
        self.weights: Dict[str, np.ndarray] = {}
        self.mean = self.std = None
        self.reference: Optional[np.ndarray] = None      # unit-normalised training features
        self.ood_threshold = 0.0
        self.load()

    @property
    def ready(self) -> bool:
        return bool(self.weights) and self.mean is not None and self.reference is not None

    @property
    def labels(self) -> List[str]:
        return sorted(self.weights)

    @property
    def runs_dir(self) -> str:
        folder = os.path.join(os.path.dirname(self.path), "dataset_head_runs")
        os.makedirs(folder, exist_ok=True)
        return folder

    def load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as handle:
                data = json.load(handle)
            self.weights = {k: np.array(v) for k, v in data["weights"].items()}
            self.mean, self.std = np.array(data["mean"]), np.array(data["std"])
            self.ood_threshold = float(data["ood_threshold"])
            self.reference = np.array(data["reference"], dtype=np.float64)
            self.info = data.get("info", {})
        except (OSError, ValueError, KeyError):
            self.weights, self.mean, self.std, self.info = {}, None, None, {}
            self.reference = None

    def train(self, X: np.ndarray, labels: Sequence[str], l2: float = 0.1, note: str = "",
              source: str = "") -> Dict[str, object]:
        labels = list(labels)
        targets = sorted({l for l in labels if l != "No Finding"})
        info: Dict[str, object] = {"n": len(labels), "classes": {l: labels.count(l) for l in set(labels)},
                                   "trained_at": datetime.now().isoformat(timespec="seconds"),
                                   "l2": l2, "note": note, "source": source, "per_label": {}}
        if len(labels) < 20 or not targets or len(set(labels)) < 2:
            info["status"] = "need ≥ 20 images and at least two classes (one of them a finding)"
            return info
        mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6
        Z = (X - mean) / std
        rng = np.random.default_rng(0)
        order = rng.permutation(len(labels))
        folds = np.array_split(order, 5)
        weights = {}
        for target in targets:
            y = np.array([1.0 if l == target else 0.0 for l in labels])
            if y.sum() < 5 or (1 - y).sum() < 5:
                continue
            scores = np.zeros(len(y))
            for fold in folds:
                train = np.setdiff1d(order, fold)
                w = _fit(Z[train], y[train], l2=l2, steps=400, rate=0.3)
                scores[fold] = _predict(w, Z[fold])
            predicted = scores >= 0.5
            tp = int((predicted & (y == 1)).sum())
            tn = int((~predicted & (y == 0)).sum())
            info["per_label"][target] = {
                "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                "cv_auc": _auc(scores.tolist(), y.tolist()),
                "cv_sensitivity": round(float(tp / max(1, y.sum())), 3),
                "cv_specificity": round(float(tn / max(1, (1 - y).sum())), 3)}
            weights[target] = _fit(Z, y, l2=l2, steps=400, rate=0.3)
        if not weights:
            info["status"] = "every label needs ≥ 5 positive and ≥ 5 negative images"
            return info
        unit = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        pick = rng.permutation(len(unit))[:REFERENCE_SIZE]
        reference = unit[pick]
        similarity = unit @ reference.T
        similarity[pick, np.arange(len(pick))] = -1.0          # leave-one-out
        nearest = 1 - similarity.max(axis=1)
        self.weights, self.mean, self.std, self.reference = weights, mean, std, reference
        self.ood_threshold = float(np.percentile(nearest, 99) * OOD_MARGIN)
        version = datetime.now().strftime("v%Y%m%d-%H%M%S-%f")[:-3]
        info.update(status="trained", version=version, ood_threshold=round(self.ood_threshold, 3))
        self.info = info
        payload = {"weights": {k: v.tolist() for k, v in weights.items()}, "mean": mean.tolist(),
                   "std": std.tolist(), "ood_threshold": self.ood_threshold,
                   "reference": np.round(reference, 5).tolist(), "info": info}
        for path in (self.path, os.path.join(self.runs_dir, version + ".json")):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        return info

    def predict(self, embedding: np.ndarray) -> Tuple[Dict[str, float], bool, float]:
        """(label -> probability, in distribution?, distance)."""
        if not self.ready:
            return {}, False, 0.0
        embedding = np.asarray(embedding, dtype=np.float64)
        z = (embedding - self.mean) / self.std
        unit = embedding / (np.linalg.norm(embedding) + 1e-9)
        distance = float(1 - (self.reference @ unit).max())
        probs = {label: float(_predict(w, z[None])[0]) for label, w in self.weights.items()}
        return probs, distance <= self.ood_threshold, round(distance, 4)

    def history(self) -> List[Dict[str, object]]:
        runs = []
        for path in glob.glob(os.path.join(self.runs_dir, "v*.json")):
            try:
                with open(path, encoding="utf-8") as handle:
                    info = json.load(handle).get("info", {})
            except (OSError, ValueError):
                continue
            runs.append(dict(info, active=info.get("version") == self.info.get("version")))
        return sorted(runs, key=lambda r: r.get("version", ""), reverse=True)

    def activate(self, version: str) -> Dict[str, object]:
        with open(os.path.join(self.runs_dir, version + ".json"), encoding="utf-8") as handle:
            payload = json.load(handle)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        self.load()
        return self.info

    def reset(self) -> None:
        if os.path.isfile(self.path):
            os.remove(self.path)
        self.weights, self.mean, self.std, self.info = {}, None, None, {}
        self.reference = None


def fuse_head(probability: float, head: float, weight: float = WEIGHT,
              floor: float = CORROBORATE) -> float:
    """Add the head's evidence in log-odds (weighted) to an existing probability.

    The head may always argue *against* a finding, but it may argue *for* one only
    when the measurements already point that way (probability ≥ ``floor``): it
    corroborates, it never creates a finding on its own.
    """
    if head > 0.5 and probability < floor:
        return probability
    clip = lambda p: min(max(p, 1e-4), 1 - 1e-4)  # noqa: E731
    logit = lambda p: math.log(clip(p) / (1 - clip(p)))  # noqa: E731
    z = logit(probability) + weight * max(-5.0, min(5.0, logit(head)))
    return 1 / (1 + math.exp(-z))
