"""Stage 7 - adaptive, evidence-grounded explanation (Medical RAG + generation).

The brief asks to "adapt the level of explanation, evidence and guidance to the
clinical situation". Three depths:

=========  ====================================================================
brief      one line per finding, most urgent first, the action - for an
           emergency department reading at speed
standard   each finding with its measurements, confidence and limitations, and
           the most relevant guideline passage - routine reporting
detailed   everything: why each module ran or was skipped, every measurement,
           the confidence breakdown, two or three literature passages, mimics -
           teaching, audit, second opinion
=========  ====================================================================

The context picks the starting depth (see :data:`msx.router.CONTEXTS`). The
depth then **escalates by one** when the engine is less sure - any finding it
abstained on, or a positive finding at low confidence - because that is
exactly when a doctor needs more evidence, not less.

**Grounding.** Text is assembled from the findings' own numbers and from
passages retrieved from :mod:`msx.knowledge`; every passage shown carries its
citation. An optional local LLM (Ollama, e.g. ``gemma2``) can rewrite the
summary into fluent prose, but its output is *checked*: it may only mention
findings that exist and numbers that appear in the facts it was given. If it
invents either, the template text is used instead and the audit trail says so.
That is the "hallucination risk" on the deck's limitations slide, handled.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Dict, List, Optional

from .findings import NEGATIVE, POSITIVE, POSSIBLE, UNCERTAIN, Finding
from .knowledge import KnowledgeBase, default_kb

__all__ = ["DEPTHS", "build", "answer", "LLMNarrator", "escalate"]

DEPTHS = ("brief", "standard", "detailed")
URGENT = {"Pneumothorax": 3, "Effusion": 2, "Consolidation": 2, "Mass": 2, "Edema": 2,
          "Cardiomegaly": 1, "Nodule": 1, "Atelectasis": 1}
ACTIONS = {
    "Pneumothorax": "Assess clinically for tension now; size and symptoms guide management.",
    "Effusion": "Confirm with ultrasound; unexplained unilateral effusion merits aspiration.",
    "Consolidation": "Correlate with fever, cough, inflammatory markers; treat per pneumonia pathway if fitting.",
    "Cardiomegaly": "Correlate with heart-failure symptoms; echocardiography if new.",
    "Nodule": "Characterise with CT; Fleischner 2017 guides follow-up by size and risk.",
    "Mass": "Contrast CT chest and upper abdomen; suspected lung cancer pathway.",
    "Atelectasis": "Check inspiration and post-operative status; look for an obstructing cause if lobar.",
    "Edema": "Correlate with fluid status and cardiac function.",
}


def escalate(depth: str, findings: List[Finding]) -> str:
    """One step deeper when the engine is unsure."""
    unsure = any(f.status == UNCERTAIN for f in findings) or any(
        f.status == POSITIVE and f.confidence_level == "Low" for f in findings)
    if unsure and depth != "detailed":
        return DEPTHS[DEPTHS.index(depth) + 1]
    return depth


def _ranked(findings: List[Finding]) -> List[Finding]:
    order = {POSITIVE: 0, UNCERTAIN: 1, POSSIBLE: 2, NEGATIVE: 3}
    return sorted(findings, key=lambda f: (order[f.status], -URGENT.get(f.label, 0),
                                           -f.probability))


def build(findings: List[Finding], quality: Dict[str, object], routing: Dict[str, object],
          context: str, depth: str, kb: Optional[KnowledgeBase] = None,
          auto_escalate: bool = True) -> Dict[str, object]:
    """The explanation as structured sections plus a plain-text rendering."""
    kb = kb or default_kb()
    requested = depth
    if auto_escalate:
        depth = escalate(depth, findings)
    shown = [f for f in _ranked(findings) if f.shown]
    checked = sorted({f.label for f in findings if f.status == NEGATIVE} - {f.label for f in shown})
    sections: List[Dict[str, object]] = []
    citations: List[str] = []

    if quality.get("grade") == "reject":
        problems = [c["detail"] for c in quality.get("checks", []) if c["status"] == "fail"]
        headline = "Scan not analysed - quality insufficient. Human review requested."
        sections.append({"title": "Why the analysis stopped", "lines": problems})
        passage = kb.search("chest radiograph quality inspiration rotation exposure", "Quality", 1)
        if passage:
            sections.append({"title": "What makes a film adequate",
                             "lines": [passage[0].text], "cite": passage[0].cite()})
            citations.append(passage[0].cite())
        return _pack(headline, sections, citations, depth, requested, context)

    if not shown:
        headline = ("No significant abnormality detected" +
                    (" (early exit after screening)." if routing.get("early_exit") else "."))
    else:
        tops = [f"{f.title} ({f.probability:.0%}, {f.confidence_level.lower()} confidence)"
                if f.status != UNCERTAIN else f"{f.title} - uncertain, needs your read"
                for f in shown[:3]]
        headline = "; ".join(tops) + ("" if len(shown) <= 3 else f"; +{len(shown) - 3} more")

    if depth == "brief":
        lines = []
        for f in shown:
            action = ACTIONS.get(f.label, "")
            lines.append(f"{f.title}: {f.probability:.0%} ({f.confidence_level}). {action}"
                         if f.status != UNCERTAIN else f"{f.title}: engine abstains - read it yourself.")
        sections.append({"title": "Findings", "lines": lines or ["Nothing requiring action."]})
    else:
        for f in shown:
            lines = list(f.evidence)
            lines.append(f"Probability {f.probability:.0%}; confidence {f.confidence:.2f} "
                         f"({f.confidence_level})" + (
                             f" - sources: " + ", ".join(f"{k} {v:.2f}" for k, v in f.sources.items())
                             if depth == "detailed" else ""))
            if depth == "detailed" and f.uncertainty:
                u = f.uncertainty
                lines.append(f"Confidence breakdown: decisiveness {u.get('decisiveness', 0):.2f}, "
                             f"stability {u.get('stability', 0):.2f} (TTA std {u.get('std', 0):.3f}, "
                             f"n={u.get('n', 1)}), agreement {u.get('agreement', 1):.2f}, "
                             f"quality {u.get('quality', 1):.2f}")
            section = {"title": f.title, "lines": lines, "limitations": f.limitations,
                       "action": ACTIONS.get(f.label, "")}
            passages = kb.search(f"{f.label} {' '.join(f.zones)} {f.side} signs management",
                                 f.label, 2 if depth == "detailed" else 1)
            section["evidence"] = [{"title": p.title, "text": p.text, "cite": p.cite()}
                                   for p in passages]
            citations += [p.cite() for p in passages]
            sections.append(section)
        if checked:
            sections.append({"title": "Looked for, not found", "lines": [", ".join(checked)]})
        problems = [c for c in quality.get("checks", []) if c["status"] != "ok"]
        if problems:
            sections.append({"title": "Scan quality caveats",
                             "lines": [f"{c['name']}: {c['detail']}" for c in problems]})
        if depth == "detailed":
            sections.append({"title": "How the analysis was routed", "lines": [
                f"{r['module']}: {'ran' if r['run'] else 'skipped'} - {r['reason']}"
                for r in routing.get("routes", [])]})
            general = kb.search("radiologist AI collaboration confidence explanation", "General", 1)
            if general:
                sections.append({"title": "Using this result", "lines": [general[0].text],
                                 "cite": general[0].cite()})
                citations.append(general[0].cite())
    return _pack(headline, sections, citations, depth, requested, context)


def _pack(headline, sections, citations, depth, requested, context):
    seen, unique = set(), []
    for c in citations:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    text = [headline, ""]
    for s in sections:
        text.append(s["title"].upper())
        text += [f"  - {line}" for line in s.get("lines", [])]
        if s.get("limitations"):
            text += [f"  ! {line}" for line in s["limitations"]]
        if s.get("action"):
            text.append(f"  > {s['action']}")
        for e in s.get("evidence", []):
            text.append(f"  [evidence] {e['text']} ({e['cite']})")
        text.append("")
    return {"headline": headline, "sections": sections, "citations": unique, "depth": depth,
            "requested_depth": requested, "escalated": depth != requested, "context": context,
            "text": "\n".join(text).strip(), "narrator": "template"}


# ------------------------------------------------------------------ Q & A ------
INTENTS = (
    ("why", ("why", "evidence", "basis", "how did", "what made", "show")),
    ("confidence", ("sure", "confident", "confidence", "certain", "trust", "reliable", "accuracy")),
    ("mimic", ("mimic", "differential", "else", "could it be", "artefact", "artifact", "false")),
    ("next", ("next", "do now", "should", "manage", "follow", "action", "refer", "treat")),
    ("size", ("size", "big", "large", "measure", "mm", "ratio", "how much")),
)


def _intent(question: str) -> str:
    q = question.lower()
    for name, words in INTENTS:
        if any(w in q for w in words):
            return name
    return "why"


def answer(question: str, finding: Optional[Finding], kb: Optional[KnowledgeBase] = None,
           narrator: Optional["LLMNarrator"] = None) -> Dict[str, object]:
    """Answer a doctor's question about one finding, from its facts and the knowledge base."""
    kb = kb or default_kb()
    intent = _intent(question)
    label = finding.label if finding else ""
    passages = kb.search(question + " " + label, label, 2)
    lines: List[str] = []
    if finding is None:
        lines.append("Select a finding to ask about; general guidance follows.")
    elif intent == "why":
        lines += [f"The {finding.module} module reported this because:"] + [
            f"- {e}" for e in finding.evidence]
    elif intent == "confidence":
        u = finding.uncertainty
        lines.append(f"Probability {finding.probability:.0%}, confidence {finding.confidence:.2f} "
                     f"({finding.confidence_level}).")
        lines.append(f"- decisiveness {u.get('decisiveness', 0):.2f}: distance from a 50/50 call")
        lines.append(f"- stability {u.get('stability', 0):.2f}: held across {u.get('n', 1)} "
                     f"perturbed copies (std {u.get('std', 0):.3f})")
        if "agreement" in u:
            lines.append(f"- agreement {u['agreement']:.2f} between measurement and deep model")
        lines.append(f"- quality factor {u.get('quality', 1):.2f}")
    elif intent == "mimic":
        lines += ["Known limitations and mimics for this finding:"] + [
            f"- {l}" for l in finding.limitations]
    elif intent == "next":
        lines.append(ACTIONS.get(finding.label, "Correlate clinically."))
    elif intent == "size":
        lines += [f"- {k}: {v}" for k, v in finding.measurements.items()
                  if not isinstance(v, (list, dict))] or ["No size measurement for this finding."]
    for p in passages:
        lines.append(f"Evidence: {p.text} ({p.cite()})")
    result = {"question": question, "intent": intent, "answer": "\n".join(lines),
              "citations": [p.cite() for p in passages], "engine": "template"}
    if narrator and narrator.enabled and finding is not None:
        facts = {"finding": finding.to_dict(), "passages": [p.text for p in passages]}
        prose = narrator.rewrite(question, "\n".join(lines), facts, [finding.label])
        if prose:
            result.update(answer=prose, engine=narrator.model)
    return result


# ----------------------------------------------------------------- narrator ----
class LLMNarrator:
    """Optional local LLM (Ollama) that rewrites grounded text - and is checked."""

    def __init__(self, url: str = "http://localhost:11434", model: str = "gemma2:latest",
                 enabled: bool = False, timeout: float = 30.0):
        self.url, self.model, self.enabled, self.timeout = url.rstrip("/"), model, enabled, timeout
        self.last_rejection: Optional[str] = None

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/api/tags", timeout=2) as response:
                names = [m.get("name") for m in json.load(response).get("models", [])]
            return self.model in names
        except Exception:  # noqa: BLE001 - not running, not installed
            return False

    def _generate(self, prompt: str) -> Optional[str]:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0.1}}).encode()
        request = urllib.request.Request(self.url + "/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response).get("response", "").strip()
        except Exception:  # noqa: BLE001
            return None

    def rewrite(self, task: str, grounded: str, facts: Dict[str, object],
                allowed_labels: List[str]) -> Optional[str]:
        prompt = ("You are assisting a doctor reading a chest X-ray. Rewrite the GROUNDED TEXT "
                  "as a short, clear answer to the TASK. Use only facts in the GROUNDED TEXT. Do "
                  "not add findings, numbers, or references. Do not give a diagnosis.\n\n"
                  f"TASK: {task}\n\nGROUNDED TEXT:\n{grounded}\n\nANSWER:")
        text = self._generate(prompt)
        if not text:
            return None
        problem = self.check(text, grounded, allowed_labels)
        if problem:
            self.last_rejection = problem
            return None
        return text

    @staticmethod
    def check(text: str, grounded: str, allowed_labels: List[str]) -> Optional[str]:
        """Reject output that names an unknown finding or a number not in the facts."""
        known = set(re.findall(r"\d+(?:\.\d+)?", grounded))
        for number in re.findall(r"\d+(?:\.\d+)?", text):
            if number not in known and number.rstrip("0").rstrip(".") not in known:
                return f"number {number} not in the facts"
        vocabulary = ("Cardiomegaly", "Effusion", "Pneumothorax", "Consolidation", "Nodule",
                      "Mass", "Atelectasis", "Edema", "Emphysema", "Fibrosis", "Fracture",
                      "Pneumonia", "Hernia")
        lowered = text.lower()
        grounded_lower = grounded.lower()
        for word in vocabulary:
            if word.lower() in lowered and word not in allowed_labels and \
                    word.lower() not in grounded_lower:
                return f"mentions {word}, which was not reported"
        return None

    def narrate(self, explanation: Dict[str, object], findings: List[Finding]) -> Dict[str, object]:
        if not self.enabled:
            return explanation
        labels = [f.label for f in findings if f.shown]
        prose = self.rewrite("Summarise these chest X-ray findings for the reporting doctor.",
                             explanation["text"], {}, labels)
        if prose:
            explanation = dict(explanation, summary=prose, narrator=self.model)
        else:
            explanation = dict(explanation, narrator="template",
                               narrator_note=self.last_rejection or "LLM unavailable")
        return explanation
