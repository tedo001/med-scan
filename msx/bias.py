"""Bias monitoring - does the analyser behave the same for everyone?

Chest X-ray AI is known to under-diagnose some groups even when overall
accuracy looks good (Seyyed-Kalantari et al., Nat Med 2021). This module
slices every stored study by **sex, age band, site and view** and reports per
subgroup:

* n, abnormal-flag rate, quality-hold rate, mean top-finding confidence
* AI-doctor agreement: accepted / (accepted + corrected + rejected) findings
* sensitivity against ground truth, where ground truth is recorded

A subgroup is flagged when it has at least :data:`MIN_N` studies and its
sensitivity or agreement is more than :data:`GAP` below the overall figure -
the Bias Monitor page shows the flag and the audit trail records each review.
"""

from __future__ import annotations

import json
from typing import Dict, List

from .findings import POSITIVE

__all__ = ["subgroups", "MIN_N", "GAP", "DIMENSIONS"]

MIN_N = 3
GAP = 0.10
DIMENSIONS = (("sex", "Sex"), ("age_band", "Age band"), ("site", "Site"), ("view", "View"))


def _stats(rows, decisions_by_study, analyses) -> Dict[str, object]:
    n = len(rows)
    if not n:
        return {"n": 0}
    abnormal = sum(1 for r in rows if r["status"] == "findings")
    held = sum(1 for r in rows if r["status"] == "quality-hold")
    confs = [r["top_conf"] for r in rows if r["top_conf"]]
    acc = cor = rej = 0
    for r in rows:
        for d in decisions_by_study.get(r["id"], []):
            acc += d["action"] == "accept"
            cor += d["action"] == "correct"
            rej += d["action"] == "reject"
    tp = fn = 0
    for r in rows:
        truth = {l for l in (r.get("ground_truth") or "").split(";") if l and l != "No Finding"}
        if not truth or r["id"] not in analyses:
            continue
        called = analyses[r["id"]]
        for label in truth:
            tp += label in called
            fn += label not in called
    return {"n": n, "abnormal_rate": round(abnormal / n, 3), "hold_rate": round(held / n, 3),
            "mean_confidence": round(sum(confs) / len(confs), 3) if confs else None,
            "agreement": round(acc / (acc + cor + rej), 3) if acc + cor + rej else None,
            "sensitivity": round(tp / (tp + fn), 3) if tp + fn else None,
            "decisions": acc + cor + rej}


def subgroups(store) -> Dict[str, object]:
    rows = store.studies()
    decisions: Dict[str, List[Dict[str, object]]] = {}
    for d in store.decisions():
        decisions.setdefault(d["study_id"], []).append(d)
    analyses = {}
    for r in rows:
        if r.get("ground_truth"):
            full = store.study(r["id"])
            data = json.loads(full["analysis_json"])
            analyses[r["id"]] = {f["label"] for f in data.get("findings", [])
                                 if f.get("status") == POSITIVE}
    overall = _stats(rows, decisions, analyses)
    out = {"overall": overall, "dimensions": {}, "flags": []}
    for key, title in DIMENSIONS:
        groups: Dict[str, List[Dict[str, object]]] = {}
        for r in rows:
            groups.setdefault(r.get(key) or "unknown", []).append(r)
        table = {}
        for value, members in sorted(groups.items()):
            stats = _stats(members, decisions, analyses)
            table[value] = stats
            for metric in ("sensitivity", "agreement"):
                mine, all_ = stats.get(metric), overall.get(metric)
                if stats["n"] >= MIN_N and mine is not None and all_ is not None and \
                        all_ - mine > GAP:
                    out["flags"].append(f"{title} = {value}: {metric} {mine:.0%} vs "
                                        f"{all_:.0%} overall")
        out["dimensions"][title] = table
    return out


# ------------------------------------------------------------- mitigation -----
def _rates(pairs, threshold):
    tp = sum(1 for p, t in pairs if t and p >= threshold)
    fn = sum(1 for p, t in pairs if t and p < threshold)
    tn = sum(1 for p, t in pairs if not t and p < threshold)
    fp = sum(1 for p, t in pairs if not t and p >= threshold)
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    return sens, spec


def mitigate(records, labels, default: float = 0.5, max_spec_drop: float = 0.10) -> dict:
    """Per-subgroup operating thresholds that close sensitivity gaps - *reducing* bias.

    ``records`` come from :func:`msx.evaluation.benchmark` (scores and truth per
    image, with sex and age band). For each subgroup (sex, age band) whose
    sensitivity at the default threshold trails the overall figure by more than
    :data:`GAP`, the threshold is lowered in 0.02 steps (never below 0.30) until
    the gap closes or the subgroup's specificity has dropped by more than
    ``max_spec_drop``. Thresholds are only ever *lowered*: a subgroup is never
    made less sensitive to equalise numbers. Returns the thresholds and a
    before / after table; :class:`msx.pipeline.AnalysisEngine` applies them.
    """
    def pairs_for(rows):
        return [(r["scores"].get(l, 0.0), int(l in r["truth"])) for r in rows if not r["held"]
                for l in labels]

    usable = [r for r in records if not r.get("held")]
    overall_sens, overall_spec = _rates(pairs_for(usable), default)
    table, thresholds = [], {}
    for key, title in (("sex", "Sex"), ("age_band", "Age band")):
        groups = {}
        for r in usable:
            groups.setdefault(r.get(key) or "unknown", []).append(r)
        for value, rows in sorted(groups.items()):
            pairs = pairs_for(rows)
            if sum(t for _, t in pairs) < MIN_N or value == "unknown":
                continue
            before_sens, before_spec = _rates(pairs, default)
            chosen = default
            if overall_sens is not None and before_sens is not None and \
                    overall_sens - before_sens > GAP:
                t = default
                while t > 0.30:
                    t = round(t - 0.02, 2)
                    sens, spec = _rates(pairs, t)
                    if before_spec is not None and spec is not None and before_spec - spec > max_spec_drop:
                        break
                    chosen = t
                    if overall_sens - sens <= GAP / 2:
                        break
            after_sens, after_spec = _rates(pairs, chosen)
            if chosen != default:
                thresholds[f"{key}:{value}"] = chosen
            table.append({"group": f"{title} = {value}", "threshold": chosen,
                          "sens_before": before_sens, "sens_after": after_sens,
                          "spec_before": before_spec, "spec_after": after_spec})
    gaps_before = [overall_sens - t["sens_before"] for t in table
                   if overall_sens is not None and t["sens_before"] is not None]
    gaps_after = [overall_sens - t["sens_after"] for t in table
                  if overall_sens is not None and t["sens_after"] is not None]
    return {"overall_sensitivity": overall_sens, "overall_specificity": overall_spec,
            "thresholds": thresholds, "table": table,
            "max_gap_before": round(max(gaps_before), 3) if gaps_before else None,
            "max_gap_after": round(max(gaps_after), 3) if gaps_after else None}


def threshold_for(scan_facts: dict, thresholds: dict, default: float) -> float:
    """The lowest applicable subgroup threshold for this scan (or the default)."""
    candidates = [thresholds[k] for k in (f"sex:{scan_facts.get('sex', '')}",
                                          f"age_band:{scan_facts.get('age_band', '')}")
                  if k in thresholds]
    return min(candidates + [default])
