"""The Finding - what every specialist module hands back, and what the doctor judges."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = ["Finding", "POSITIVE", "POSSIBLE", "UNCERTAIN", "NEGATIVE"]

POSITIVE = "positive"      # probability >= 0.5 and the engine is confident enough to say so
POSSIBLE = "possible"      # 0.35-0.5: worth a look, not asserted
UNCERTAIN = "uncertain"    # the engine abstains: unstable or contradictory evidence
NEGATIVE = "negative"      # looked for and not found (kept so the doctor sees it was checked)


@dataclass
class Finding:
    label: str
    group: str
    module: str
    probability: float
    side: str = ""                                   # "right", "left", "bilateral", ""
    zones: List[str] = field(default_factory=list)
    bbox: Optional[Tuple[int, int, int, int]] = None  # 512 working-image pixels
    measurements: Dict[str, object] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    sources: Dict[str, float] = field(default_factory=dict)   # engine -> probability
    confidence: float = 0.0
    confidence_level: str = ""
    uncertainty: Dict[str, float] = field(default_factory=dict)
    status: str = NEGATIVE
    key: str = ""                                    # stable id within one analysis

    @property
    def title(self) -> str:
        where = f" - {self.side}" if self.side and self.side != "bilateral" else (
            " - bilateral" if self.side == "bilateral" else "")
        return f"{self.label}{where}"

    @property
    def shown(self) -> bool:
        """Positive, possible and abstained findings are shown; negatives are listed as checked."""
        return self.status != NEGATIVE

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["bbox"] = list(self.bbox) if self.bbox else None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Finding":
        data = dict(data)
        if data.get("bbox"):
            data["bbox"] = tuple(data["bbox"])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})
