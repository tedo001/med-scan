"""The recommendation engine - what to do next, how soon, and why.

For every shown finding it proposes an urgency tier, the next test, a follow-up
interval and a referral, then adjusts them:

* **confidence** - a low-confidence or abstained finding is recommended for
  confirmation (second read, the confirming test) rather than treatment;
* **size** - nodules follow Fleischner 2017 size bands when DICOM pixel spacing
  gives a real size; otherwise CT characterisation is recommended;
* **clinical context** - Emergency moves "same day" items to "now";
* **patterns across findings** - cardiomegaly with effusion (or oedema) suggests
  heart failure and proposes that work-up once, instead of three separate ones;
  consolidation with effusion raises a parapneumonic effusion / empyema check;
* **scan quality** - a quality hold recommends re-acquisition.

Each recommendation carries its reason and the guideline it comes from, so it
can be questioned like a finding. Recommendations never order anything - the
doctor accepts or ignores them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .findings import POSITIVE, POSSIBLE, UNCERTAIN, Finding

__all__ = ["Recommendation", "recommend", "TIERS"]

TIERS = ("Now", "Same day", "Within 1 week", "Routine")


@dataclass
class Recommendation:
    tier: str
    action: str
    reason: str
    source: str
    finding: str = ""

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


RULES = {
    "Pneumothorax": ("Same day", "Clinical assessment for tension; repeat film or ultrasound; "
                     "chest-drain decision by size and symptoms", "BTS Pleural Disease Guideline 2023"),
    "Effusion": ("Same day", "Thoracic ultrasound; diagnostic aspiration if unilateral and "
                 "unexplained (protein, LDH, pH, cytology)", "BTS Pleural Disease Guideline 2023"),
    "Consolidation": ("Same day", "Correlate with fever, cough, CRP / WBC; CURB-65; antibiotics "
                      "per local pneumonia pathway; repeat film at 6 weeks if > 50 or smoker",
                      "NICE NG138 / BTS CAP guideline"),
    "Cardiomegaly": ("Within 1 week", "ECG, BNP / NT-proBNP and echocardiography if new",
                     "ESC Heart Failure Guidelines 2021"),
    "Nodule": ("Within 1 week", "CT chest to characterise; compare with any prior imaging",
               "Fleischner Society 2017"),
    "Mass": ("Same day", "Contrast CT chest and upper abdomen; urgent suspected-lung-cancer "
             "referral", "NICE NG122"),
    "Atelectasis": ("Routine", "Check inspiration and post-operative status; CT or bronchoscopy "
                    "if lobar collapse", "Fleischner glossary"),
    "Edema": ("Same day", "Assess fluid status; BNP; echocardiography; review diuretics",
              "ESC Heart Failure Guidelines 2021"),
}


def _earlier(tier: str) -> str:
    return TIERS[max(0, TIERS.index(tier) - 1)]


def _nodule(f: Finding) -> Optional[Recommendation]:
    size = f.measurements.get("diameter_mm") if isinstance(f.measurements, dict) else None
    if not size:
        return None
    if size < 6:
        return Recommendation("Routine", f"Nodule ~{size:.0f} mm: no routine follow-up if low risk; "
                              "optional CT at 12 months if high risk", "Size < 6 mm (Fleischner)",
                              "Fleischner Society 2017", f.title)
    if size <= 8:
        return Recommendation("Within 1 week", f"Nodule ~{size:.0f} mm: CT at 6-12 months",
                              "Size 6-8 mm (Fleischner)", "Fleischner Society 2017", f.title)
    return Recommendation("Within 1 week", f"Nodule ~{size:.0f} mm: CT at 3 months, PET-CT or "
                          "tissue sampling", "Size > 8 mm (Fleischner)", "Fleischner Society 2017",
                          f.title)


def recommend(findings: List[Finding], context: str = "Routine OPD",
              quality: Optional[Dict[str, object]] = None) -> List[Recommendation]:
    out: List[Recommendation] = []
    if quality and quality.get("grade") == "reject":
        problems = ", ".join(c["name"].lower() for c in quality.get("checks", [])
                             if c["status"] == "fail")
        return [Recommendation("Same day", "Re-acquire the radiograph, or read it without AI support",
                               f"Quality gate failed: {problems}", "Technical adequacy criteria")]
    shown = [f for f in findings if f.status in (POSITIVE, POSSIBLE, UNCERTAIN)]
    labels = {f.label for f in shown if f.status == POSITIVE}
    handled = set()

    if "Cardiomegaly" in labels and labels & {"Effusion", "Edema"}:
        out.append(Recommendation("Same day", "Heart-failure work-up: BNP / NT-proBNP, ECG, "
                                  "echocardiography; one pathway for the combined findings",
                                  "Cardiomegaly with " + " and ".join(sorted(labels & {"Effusion", "Edema"})).lower(),
                                  "ESC Heart Failure Guidelines 2021", "Pattern"))
        handled |= {"Cardiomegaly", "Edema"}
    if "Consolidation" in labels and "Effusion" in labels:
        out.append(Recommendation("Same day", "Ultrasound the effusion and consider aspiration to "
                                  "exclude a parapneumonic effusion / empyema",
                                  "Consolidation with effusion", "BTS Pleural Disease Guideline 2023",
                                  "Pattern"))

    for f in shown:
        if f.label in handled or f.label not in RULES:
            continue
        tier, action, source = RULES[f.label]
        if f.status == UNCERTAIN or f.confidence_level == "Low" or f.status == POSSIBLE:
            out.append(Recommendation(
                "Routine" if f.status == POSSIBLE else tier,
                "Confirm before acting: second read" + (" and " + action.split(";")[0].lower()
                                                       if f.status != POSSIBLE else ""),
                f"{f.status.capitalize()} finding, confidence {f.confidence:.2f}", source, f.title))
            continue
        if f.label == "Nodule":
            sized = _nodule(f)
            if sized:
                out.append(sized)
                continue
        if context == "Emergency" and tier == "Same day":
            tier = _earlier(tier)
        out.append(Recommendation(tier, action, f"{f.title} at {f.probability:.0%}", source, f.title))
    out.sort(key=lambda r: TIERS.index(r.tier))
    return out
