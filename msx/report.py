"""Generative AI - a draft radiology report the doctor edits and signs.

The draft has the shape radiologists write in:

    EXAMINATION / TECHNIQUE / FINDINGS / IMPRESSION / RECOMMENDATIONS

It is built in two steps:

1. **Grounded skeleton** - every sentence comes from the analysis itself: the
   positive and possible findings with side, zone and measurement, the
   pertinent negatives that were checked, scan-quality caveats, and the
   recommendation engine's next steps.
2. **Generation** - when a local LLM is reachable (Ollama, e.g. ``gemma2``),
   it rewrites FINDINGS and IMPRESSION into fluent report prose. Its output is
   fact-checked by :meth:`msx.explain.LLMNarrator.check`: a number that is not
   in the skeleton, or a finding that was not reported, rejects the rewrite and
   the skeleton is kept. The report says which engine wrote it.

Either way the report is labelled *AI-drafted - requires clinician review*; it
is saved only when the doctor signs off, with whatever edits they made.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .findings import NEGATIVE, POSITIVE, POSSIBLE, UNCERTAIN, Finding
from .recommend import Recommendation, recommend

__all__ = ["draft", "DISCLAIMER"]

DISCLAIMER = "AI-drafted by MEDSCAN - requires clinician review and signature. Not a diagnosis."


def _finding_sentence(f: Finding) -> str:
    where = ""
    if f.side and f.side != "bilateral":
        where = f" in the {f.side}" + (f" {'/'.join(f.zones)} zone" if f.zones else " hemithorax")
    m = f.measurements if isinstance(f.measurements, dict) else {}
    detail = ""
    if f.label == "Cardiomegaly" and m.get("ctr"):
        detail = f", cardiothoracic ratio {m['ctr']:.2f}"
    elif f.label in ("Nodule", "Mass") and m.get("diameter_mm"):
        detail = f", approximately {m['diameter_mm']:.0f} mm"
    elif f.label == "Effusion" and m.get("asymmetry"):
        detail = f", lung height reduced by {m['asymmetry']:.0%}"
    hedge = {POSITIVE: "", POSSIBLE: "Possible ", UNCERTAIN: "Indeterminate "}[f.status]
    label = f.label.lower() if hedge else f.label
    return f"{hedge}{label}{where}{detail} (AI probability {f.probability:.0%}, " \
           f"{f.confidence_level.lower()} confidence)."


def draft(analysis, recommendations: Optional[List[Recommendation]] = None,
          narrator=None) -> Dict[str, object]:
    findings: List[Finding] = analysis.findings
    recommendations = recommendations if recommendations is not None else recommend(
        findings, analysis.context, analysis.quality)
    scan = analysis.scan
    quality = analysis.quality
    problems = [c for c in quality.get("checks", []) if c["status"] != "ok"]
    shown = [f for f in findings if f.status in (POSITIVE, POSSIBLE, UNCERTAIN)]
    negatives = sorted({f.label.lower() for f in findings if f.status == NEGATIVE})

    exam = f"Chest radiograph, {scan.get('view') or 'PA'} projection."
    technique = ("Diagnostic quality." if not problems else
                 "Limited: " + "; ".join(c["detail"] for c in problems) + ".")
    if quality.get("grade") == "reject":
        findings_text = "Not assessed by AI - the image failed the quality gate."
        impression = ["Non-diagnostic study for AI analysis; human read required."]
    else:
        lines = [_finding_sentence(f) for f in shown]
        if negatives:
            lines.append("No " + ", ".join(negatives) + " identified.")
        findings_text = " ".join(lines) or "No significant abnormality identified."
        impression = [f"{f.label}" + (f", {f.side}" if f.side and f.side != "bilateral" else "")
                      + ("" if f.status == POSITIVE else " - " + f.status)
                      for f in shown if f.status != POSSIBLE] or ["No acute cardiopulmonary abnormality "
                                                                 "identified by AI."]
    recs = [f"[{r.tier}] {r.action}." for r in recommendations]

    engine = "template"
    note = ""
    if narrator is not None and getattr(narrator, "enabled", False) and shown:
        grounded = f"FINDINGS: {findings_text}\nIMPRESSION: " + "; ".join(impression)
        prose = narrator.rewrite("Write the FINDINGS and IMPRESSION of a chest radiograph report "
                                 "in concise radiology style. Keep the two headings.",
                                 grounded, {}, [f.label for f in shown])
        if prose and "IMPRESSION" in prose.upper():
            head, _, tail = prose.partition("IMPRESSION")
            findings_text = head.replace("FINDINGS:", "").strip(" :\n") or findings_text
            impression = [l.strip(" -:•\t") for l in tail.strip(" :\n").splitlines() if l.strip(" -:•\t")] \
                or impression
            engine = narrator.model
        else:
            note = narrator.last_rejection or "LLM unavailable - grounded template used"

    sections = [("EXAMINATION", exam), ("TECHNIQUE", technique), ("FINDINGS", findings_text),
                ("IMPRESSION", "\n".join(f"{i + 1}. {line}" for i, line in enumerate(impression))),
                ("RECOMMENDATIONS", "\n".join(recs) or "None.")]
    text = "\n\n".join(f"{h}:\n{b}" for h, b in sections) + f"\n\n{DISCLAIMER}"
    return {"text": text, "sections": sections, "engine": engine, "note": note,
            "recommendations": [r.to_dict() for r in recommendations]}
