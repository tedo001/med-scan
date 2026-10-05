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
