"""Evaluation - the two questions the deck promises to answer.

1. **Fixed vs adaptive pipeline** (:func:`benchmark`). Every image in a labelled
   folder is analysed twice: once with every specialist module forced on
   (fixed), once routed (adaptive). Reported per mode: per-label sensitivity,
   specificity and AUC, "any abnormality" sensitivity / specificity, quality
   gate accuracy, mean time per scan, mean modules activated, model calls,
   early-exit rate. Same scans, same engine - the difference is the router.

2. **Doctor vs AI vs Doctor + AI** (:func:`reader_study`). From signed studies
   with ground truth: the doctor's blinded first read, the AI's own positives,
   and the signed final read, each scored at label level, plus decision times.

Labels come from ``labels.csv`` beside the images: ``file,labels`` with labels
separated by ``;`` and ``No Finding`` for normal (extra columns - sex, age,
site, view, quality - are used when present).

Command line::

    python -m msx.evaluation samples            # benchmark the phantoms
    python -m msx.evaluation /data/chexpert/val --engine hybrid --context Emergency
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from typing import Dict, List, Optional, Sequence

from . import imaging
from .findings import POSITIVE
from .pipeline import AnalysisEngine

__all__ = ["EVAL_LABELS", "load_labels", "benchmark", "reader_study", "auc", "rates"]

EVAL_LABELS = ("Cardiomegaly", "Effusion", "Pneumothorax", "Consolidation", "Nodule", "Mass")


def load_labels(folder: str) -> List[Dict[str, object]]:
    path = os.path.join(folder, "labels.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found - expected columns file,labels")
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels = {l.strip() for l in (row.get("labels") or "").split(";") if l.strip()}
            labels.discard("No Finding")
            row["truth"] = labels
            row["path"] = os.path.join(folder, row["file"])
            rows.append(row)
    return rows


def auc(scores: Sequence[float], truth: Sequence[int]) -> Optional[float]:
    """Area under the ROC curve (Mann-Whitney), None if one class is missing."""
    positives = [s for s, t in zip(scores, truth) if t]
    negatives = [s for s, t in zip(scores, truth) if not t]
    if not positives or not negatives:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return round(wins / (len(positives) * len(negatives)), 3)


def rates(predicted: Sequence[int], truth: Sequence[int]) -> Dict[str, Optional[float]]:
    tp = sum(1 for p, t in zip(predicted, truth) if p and t)
    tn = sum(1 for p, t in zip(predicted, truth) if not p and not t)
    fp = sum(1 for p, t in zip(predicted, truth) if p and not t)
    fn = sum(1 for p, t in zip(predicted, truth) if not p and t)
    return {"sensitivity": round(tp / (tp + fn), 3) if tp + fn else None,
            "specificity": round(tn / (tn + fp), 3) if tn + fp else None,
            "accuracy": round((tp + tn) / max(1, tp + tn + fp + fn), 3),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _label_score(analysis, label: str) -> float:
    values = [f.probability for f in analysis.findings if f.label == label]
    return max(values) if values else 0.0


def _label_positive(analysis, label: str) -> int:
    return int(any(f.label == label and f.status == POSITIVE for f in analysis.findings))


def benchmark(folder: str, engine: str = "builtin", context: str = "Routine OPD",
              modes: Sequence[str] = ("fixed", "adaptive"), progress=None) -> Dict[str, object]:
    rows = load_labels(folder)
    analyser = AnalysisEngine(engine=engine)
    report: Dict[str, object] = {"folder": os.path.abspath(folder), "engine": analyser.engine_label,
                                 "context": context, "images": len(rows), "modes": {}}
    total = len(rows) * len(modes)
    done = 0
    for mode in modes:
        results = []
        for row in rows:
            facts = {"sex": row.get("sex", ""), "view": row.get("view", "") or None}
            if row.get("age"):
                try:
                    facts["age_band"] = imaging.age_band(float(row["age"]))
                except ValueError:
                    pass
            started = time.perf_counter()
            analysis = analyser.analyse_file(row["path"], context=context, mode=mode, **facts)
            results.append((row, analysis, (time.perf_counter() - started) * 1000))
            done += 1
            if progress:
                progress(done, total, f"{mode}: {row['file']}")
        report["modes"][mode] = _score(results)
    if "fixed" in report["modes"] and "adaptive" in report["modes"]:
        f, a = report["modes"]["fixed"], report["modes"]["adaptive"]
        report["comparison"] = {
            "time_saved_pct": round(100 * (1 - a["mean_ms"] / f["mean_ms"]), 1) if f["mean_ms"] else 0,
            "modules_saved_pct": round(100 * (1 - a["mean_modules"] / f["mean_modules"]), 1)
            if f["mean_modules"] else 0,
            "calls_saved_pct": round(100 * (1 - a["mean_calls"] / f["mean_calls"]), 1)
            if f["mean_calls"] else 0,
            "sensitivity_change": _delta(a["any"]["sensitivity"], f["any"]["sensitivity"]),
            "specificity_change": _delta(a["any"]["specificity"], f["any"]["specificity"]),
        }
    return report


def _delta(a, b):
    return round(a - b, 3) if a is not None and b is not None else None


def _score(results) -> Dict[str, object]:
    analysable = [(r, a, ms) for r, a, ms in results if a.status != "quality-hold"]
    per_label = {}
    for label in EVAL_LABELS:
        truth = [int(label in r["truth"]) for r, _, _ in analysable]
        if not any(truth):
            continue
        scores = [_label_score(a, label) for _, a, _ in analysable]
        predicted = [_label_positive(a, label) for _, a, _ in analysable]
        per_label[label] = dict(rates(predicted, truth), auc=auc(scores, truth), n=sum(truth))
    any_truth = [int(bool(r["truth"] & set(EVAL_LABELS))) for r, _, _ in analysable]
    any_pred = [int(any(f.status == POSITIVE for f in a.findings)) for _, a, _ in analysable]
    quality_truth = [int((r.get("quality") or "ok") != "ok") for r, _, _ in results]
    quality_pred = [int(a.status == "quality-hold") for _, a, _ in results]
    ms = [m for _, _, m in results]
    group_of = {"Cardiomegaly": "cardiac", "Effusion": "pleural", "Pneumothorax": "pleural",
                "Consolidation": "parenchymal", "Nodule": "focal", "Mass": "focal"}
    pairs: Dict[str, List[List[float]]] = {}
    for r, a, _ in analysable:            # (probability, truth) per group, for calibration
        for label, group in group_of.items():
            if any(f.label == label for f in a.findings):
                pairs.setdefault(group, []).append([_label_score(a, label), int(label in r["truth"])])
    return {
        "pairs": pairs,
        "per_label": per_label,
        "any": rates(any_pred, any_truth),
        "quality_gate": rates(quality_pred, quality_truth),
        "mean_ms": round(statistics.mean(ms), 1) if ms else 0.0,
        "median_ms": round(statistics.median(ms), 1) if ms else 0.0,
        "mean_modules": round(statistics.mean(a.modules_activated for _, a, _ in results), 2),
        "mean_calls": round(statistics.mean(a.model_calls for _, a, _ in results), 2),
        "early_exit_rate": round(sum(1 for _, a, _ in results if a.routing.get("early_exit"))
                                 / max(1, len(results)), 3),
        "abstained": sum(1 for _, a, _ in results for f in a.findings if f.status == "uncertain"),
    }


def reader_study(store) -> Dict[str, object]:
    """Doctor alone (blinded first read) vs AI alone vs Doctor + AI (signed read)."""
    from .datastore import DataStore  # noqa: F401 - type only

    import json as _json

    from .pipeline import Analysis

    reads = {"doctor": ([], []), "ai": ([], []), "doctor_ai": ([], [])}
    times_first, times_final, n = [], [], 0
    for row in store.studies(state="signed"):
        if not row.get("ground_truth") or not row.get("first_read"):
            continue
        full = store.study(row["id"])
        analysis = Analysis.from_dict(_json.loads(full["analysis_json"]))
        truth = {l for l in row["ground_truth"].split(";") if l and l != "No Finding"}
        first = {l for l in row["first_read"].split(";") if l and l != "No Finding"}
        final = {l for l in (row["final_read"] or "").split(";") if l and l != "No Finding"}
        ai = {f.label for f in analysis.findings if f.status == POSITIVE}
        n += 1
        for label in EVAL_LABELS:
            t = int(label in truth)
            for key, called in (("doctor", first), ("ai", ai), ("doctor_ai", final)):
                reads[key][0].append(int(label in called))
                reads[key][1].append(t)
        if row.get("first_read_ms"):
            times_first.append(row["first_read_ms"] / 1000)
        if row.get("decision_ms"):
            times_final.append(row["decision_ms"] / 1000)
    out = {"studies": n}
    for key, (predicted, truth) in reads.items():
        out[key] = rates(predicted, truth) if predicted else None
    out["mean_first_read_s"] = round(statistics.mean(times_first), 1) if times_first else None
    out["mean_final_read_s"] = round(statistics.mean(times_final), 1) if times_final else None
    return out


def _print(report: Dict[str, object]) -> None:
    print(f"\nMEDSCAN benchmark - {report['images']} images - {report['engine']} - "
          f"{report['context']}\n")
    modes = report["modes"]
    header = f"{'':28s}" + "".join(f"{m:>14s}" for m in modes)
    print(header)

    def line(name, getter):
        print(f"{name:28s}" + "".join(f"{_fmt(getter(modes[m])):>14s}" for m in modes))

    line("any-abnormal sensitivity", lambda m: m["any"]["sensitivity"])
    line("any-abnormal specificity", lambda m: m["any"]["specificity"])
    for label in EVAL_LABELS:
        if any(label in m["per_label"] for m in modes.values()):
            line(f"{label} AUC", lambda m, l=label: m["per_label"].get(l, {}).get("auc"))
    line("quality gate accuracy", lambda m: m["quality_gate"]["accuracy"])
    line("mean ms / scan", lambda m: m["mean_ms"])
    line("mean modules activated", lambda m: m["mean_modules"])
    line("mean model calls", lambda m: m["mean_calls"])
    line("early-exit rate", lambda m: m["early_exit_rate"])
    if "comparison" in report:
        c = report["comparison"]
        print(f"\nadaptive vs fixed: {c['time_saved_pct']}% less time, {c['modules_saved_pct']}% "
              f"fewer modules, {c['calls_saved_pct']}% fewer model calls; sensitivity change "
              f"{c['sensitivity_change']}, specificity change {c['specificity_change']}")


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}" if value <= 1 else f"{value:.1f}"
    return str(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark fixed vs adaptive routing.")
    parser.add_argument("folder")
    parser.add_argument("--engine", default="builtin", choices=("builtin", "hybrid"))
    parser.add_argument("--context", default="Routine OPD")
    parser.add_argument("--json", help="write the full report here")
    args = parser.parse_args(argv)
    report = benchmark(args.folder, args.engine, args.context)
    _print(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
