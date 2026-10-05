"""Engines - what is installed, what is loaded, and how each stage decides."""

from __future__ import annotations

import platform

from PyQt6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem

from msx import __version__, screening
from msx.router import CONTEXTS
from msx.screening import CURVES

from ..components import Card, Page, Pill, button, clear, hbox, label


def _version(module: str) -> str:
    try:
        return getattr(__import__(module), "__version__", "installed")
    except Exception:  # noqa: BLE001
        return ""


class EnginesPage(Page):
    def __init__(self, services):
        super().__init__("Engines", f"MEDSCAN {__version__} · Python {platform.python_version()}")
        self.services = services
        self.add_action(button("Load DenseNet now", "Primary", self._load))
        self.status = Card("Stages")
        self.body.addWidget(self.status)
        self.curves = Card("Measurement curves", "logistic: p = 1 / (1 + e^-(x - centre)/scale)")
        self.body.addWidget(self.curves)
        self.routes = Card("Router thresholds by clinical context")
        self.body.addWidget(self.routes)
        self.stack = Card("Software stack")
        self.body.addWidget(self.stack)
        self.body.addStretch(1)

    def _load(self):
        deep = screening.get_deep()
        self.services.log("deep engine load", {"ok": deep is not None,
                                               "error": screening.deep_error()}, category="system")
        self.refresh()

    def refresh(self):
        deep_ok = screening.deep_available()
        loaded = screening._DEEP is not None
        rows = [
            ("1 Preprocess", "OpenCV + pydicom", "ready", "letterbox 512², DICOM de-identification, classical lung segmentation"),
            ("1 Preprocess", "PSPNet lung + heart segmenter (TorchXRayVision)",
             "loaded" if screening._SEG is not None else "available" if deep_ok else "not installed",
             "hybrid mode: learned masks first, classical fallback when implausible"),
            ("2 Quality", "built-in", "ready", "7 checks: resolution, sharpness, contrast, exposure, lung fields, coverage, symmetry"),
            ("3 Screen", "built-in measurements", "ready", "CTR, lung asymmetry, CP blunting, zone density, lucency, coarse LoG"),
            ("3 Screen", "DenseNet-121 (TorchXRayVision, densenet121-res224-all)",
             "loaded" if loaded else "available" if deep_ok else "not installed",
             "18 pathologies, 8 public datasets; Grad-CAM for routed findings"),
            ("4 Route", "Adaptive Analysis Router", "ready", "context thresholds, early exit, borderline band"),
            ("5 Specialists", "Cardiac · Pleural · Parenchyma · Nodule", "ready", "measurement + deep fusion in log-odds, TTA ×" + str(self.services.settings["tta"])),
            ("5b Learning", "Feedback-trained logistic stacker (ML)",
             "trained" if self.services.learner.ready else "collecting feedback",
             f"{self.services.learner.info.get('n', 0)} labelled findings · blend weight {self.services.learner.weight:.2f}"),
            ("6 Confidence", "calibration + TTA + abstention", "ready",
             "temperatures: " + (", ".join(f"{k}={v}" for k, v in self.services.settings["temperatures"].items()) or "none (T=1)")
             + " · subgroup thresholds: " + (", ".join(f"{k}={v}" for k, v in self.services.settings["subgroup_thresholds"].items()) or "none")),
            ("7 Explain", f"BM25 RAG over {len(self.services.kb.passages)} cited passages", "ready",
             "LLM narrator: " + (f"{self.services.settings['llm_model']} (guarded)" if self.services.settings["llm_enabled"] else "off - template text")),
            ("7b Recommend", "Recommendation engine", "ready", "urgency tier, next test, follow-up, patterns across findings"),
            ("7c Report", "Generative draft report", "ready",
             "LLM " + (self.services.settings["llm_model"] if self.services.settings["llm_enabled"] else "off")
             + " · grounded template fallback · fact-checked"),
            ("8 Record", "SQLite + hash-chained JSONL", "ready", self.services.store.path),
        ]
        clear(self.status.body)
        table = QTableWidget(len(rows), 4)
        table.setHorizontalHeaderLabels(["Stage", "Engine", "State", "Detail"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                table.setItem(r, c, QTableWidgetItem(v))
        table.setMinimumHeight(60 + 32 * len(rows))
        self.status.body.addWidget(table)
        if screening.deep_error():
            self.status.body.addWidget(label(f"Deep engine error: {screening.deep_error()}", "Small"))

        clear(self.curves.body)
        for name, (centre, scale) in CURVES.items():
            self.curves.body.addLayout(hbox(label(name, "Mono"), None,
                                            label(f"centre {centre}  scale {scale}", "MonoSmall")))
        clear(self.routes.body)
        for name, c in CONTEXTS.items():
            self.routes.body.addLayout(hbox(label(name, "Small"), None, label(
                f"route ≥ {c['route']:.2f} · exit < {c['exit']:.2f} · band {c['band']:.2f} · "
                f"depth {c['depth']}", "MonoSmall")))
        clear(self.stack.body)
        for name, module in (("PyQt6", "PyQt6.QtCore"), ("NumPy", "numpy"), ("SciPy", "scipy"),
                             ("OpenCV", "cv2"), ("pydicom", "pydicom"), ("Pillow", "PIL"),
                             ("PyTorch", "torch"), ("TorchXRayVision", "torchxrayvision")):
            v = _version(module.split(".")[0]) if module != "PyQt6.QtCore" else __import__(
                "PyQt6.QtCore", fromlist=["PYQT_VERSION_STR"]).PYQT_VERSION_STR
            self.stack.body.addLayout(hbox(label(name, "Small"), None,
                                           Pill(v or "not installed", "ok" if v else "grey")))
