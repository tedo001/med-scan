"""The medical knowledge base and its retriever - the "R" in Medical RAG.

A curated set of short passages (``knowledge/kb.json``), each tied to a finding
and carrying its source: Fleischner Society, BTS, NICE, ESC guidance and the
radiology literature the TECHgium brief cites. Explanations and answers quote
these passages - they never invent a reference.

Retrieval is Okapi BM25 over title + text, boosted for passages tagged with
the finding being explained. It is pure Python, deterministic and instant;
when ``sentence-transformers`` is installed a dense encoder can re-rank, but
for ~25 curated passages lexical retrieval is already exact.

Add passages by editing ``kb.json`` - the Engines page shows how many are loaded.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from . import paths

__all__ = ["Passage", "KnowledgeBase", "default_kb"]

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an and are as at be by for from has have in is it its of on or that the "
            "this to was were with which may can not no on so than then there these they "
            "usually often".split())


def tokens(text: str) -> List[str]:
    out = []
    for word in _WORD.findall(text.lower()):
        if word in _STOP:
            continue
        if len(word) > 4 and word.endswith("s"):
            word = word[:-1]          # crude plural folding: nodules -> nodule
        out.append(word)
    return out


@dataclass
class Passage:
    id: str
    finding: str
    title: str
    text: str
    source: str
    score: float = 0.0

    def cite(self) -> str:
        return f"{self.title} - {self.source}"


class KnowledgeBase:
    def __init__(self, passages: List[Passage]):
        self.passages = passages
        self._docs = [tokens(p.title + " " + p.text + " " + p.finding) for p in passages]
        self._lengths = [len(d) for d in self._docs]
        self._avg = sum(self._lengths) / max(1, len(self._lengths))
        frequency: Counter = Counter()
        for doc in self._docs:
            frequency.update(set(doc))
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in frequency.items()}
        self._counts = [Counter(d) for d in self._docs]

    @classmethod
    def load(cls, path: Optional[str] = None) -> "KnowledgeBase":
        path = path or os.path.join(paths.KNOWLEDGE_DIR, "kb.json")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return cls([Passage(**{k: v for k, v in item.items() if k != "score"}) for item in data])

    def search(self, query: str, finding: str = "", k: int = 3,
               k1: float = 1.5, b: float = 0.75) -> List[Passage]:
        """The ``k`` best passages for ``query``; passages about ``finding`` get a boost."""
        terms = tokens(query)
        scored = []
        for index, passage in enumerate(self.passages):
            counts, length = self._counts[index], self._lengths[index]
            s = 0.0
            for term in terms:
                f = counts.get(term, 0)
                if f:
                    s += self._idf.get(term, 0) * f * (k1 + 1) / (
                        f + k1 * (1 - b + b * length / self._avg))
            if finding and passage.finding.lower() == finding.lower():
                s += 2.5
            if s > 0:
                scored.append((s, index))
        scored.sort(reverse=True)
        out = []
        for s, index in scored[:k]:
            p = self.passages[index]
            out.append(Passage(p.id, p.finding, p.title, p.text, p.source, round(s, 2)))
        return out

    def for_finding(self, finding: str) -> List[Passage]:
        return [p for p in self.passages if p.finding.lower() == finding.lower()]


_DEFAULT: Optional[KnowledgeBase] = None


def default_kb() -> KnowledgeBase:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = KnowledgeBase.load()
    return _DEFAULT
