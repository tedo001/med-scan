"""Stage 3 - the fast whole-image screen.

One quick look at the scan that answers a single question for the router:
*which groups of abnormality might be here?* The four groups map one-to-one
onto the specialist modules:

==============  ==============================================  ==================
group           findings                                        specialist
==============  ==============================================  ==================
cardiac         Cardiomegaly, Enlarged Cardiomediastinum         CardiacModule
pleural         Effusion, Pneumothorax, Pleural thickening       PleuralModule
parenchymal     Consolidation, Pneumonia, Lung Opacity,          ParenchymaModule
                Atelectasis, Edema, Infiltration
focal           Nodule, Mass, Lung Lesion                        NoduleModule
==============  ==============================================  ==================

Two engines implement the screen, chosen in Settings:

* :class:`MeasurementScreen` (built in, always available). Deterministic,
  explainable image measurements turned into probabilities with logistic
  curves: cardiothoracic ratio, lung-height asymmetry and costophrenic
  blunting, contralateral zone density, peripheral lucency, a coarse blob
  search. No model, no network, ~60 ms.
* :class:`DeepScreen` (optional: ``pip install torch torchxrayvision``).
  TorchXRayVision's DenseNet-121 trained on eight public chest X-ray datasets
  (NIH, CheXpert, MIMIC-CXR, PadChest, ...) - 18 pathology outputs calibrated
  so 0.5 is each label's operating point. One 224 x 224 forward pass,
  ~0.1-0.3 s on a CPU. It also provides Grad-CAM heatmaps, but only the
  specialists that are routed ask for them - that is the adaptive saving.

The screen deliberately does *not* produce findings. It only routes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import measures
from .anatomy import Anatomy

__all__ = ["GROUPS", "GROUP_LABELS", "ScreenResult", "MeasurementScreen", "DeepScreen",
           "logistic", "deep_available", "get_deep"]

GROUPS = ("cardiac", "pleural", "parenchymal", "focal")
GROUP_LABELS = {
    "cardiac": ("Cardiomegaly", "Enlarged Cardiomediastinum"),
    "pleural": ("Effusion", "Pneumothorax", "Pleural_Thickening"),
    "parenchymal": ("Consolidation", "Pneumonia", "Lung Opacity", "Atelectasis", "Edema",
                    "Infiltration"),
    "focal": ("Nodule", "Mass", "Lung Lesion"),
}


def logistic(x: float, centre: float, scale: float) -> float:
    """A smooth 0-1 step: 0.5 at ``centre``, 0.73 one ``scale`` above it."""
    z = (x - centre) / scale
    return 1 / (1 + math.exp(-max(-40.0, min(40.0, z))))


@dataclass
class ScreenResult:
    engine: str
    groups: Dict[str, float]                       # group -> probability
    labels: Dict[str, float] = field(default_factory=dict)   # label -> probability
    features: Dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    @property
    def abnormality(self) -> float:
        return max(self.groups.values()) if self.groups else 0.0


# ---------------------------------------------------------------- built-in -----
#: Logistic curves for the measurement screen: (centre, scale). Chosen so a
#: textbook normal sits below 0.05 and a textbook abnormal above 0.9; they are
#: published thresholds where one exists (CTR 0.5 on a PA film).
CURVES = {
    "ctr": (0.52, 0.025),
    "ctr_ap": (0.56, 0.025),        # AP films magnify the heart
    "effusion": (0.10, 0.025),
    "pneumothorax": (0.12, 0.025),
    "opacity": (0.50, 0.07),
    "focal": (0.14, 0.02),
}


class MeasurementScreen:
    """The built-in screen: cheap global measurements only."""

    name = "Built-in measurements"
    short = "builtin"

    def screen(self, work: np.ndarray, anatomy: Anatomy, view: str = "PA") -> ScreenResult:
        started = time.perf_counter()
        f: Dict[str, float] = {}
        ctr = anatomy.cardiothoracic_ratio() or 0.0
        f["ctr"] = round(ctr, 3)
        cardiac = logistic(ctr, *CURVES["ctr_ap" if view == "AP" else "ctr"])

        pleural_scores = []
        for side in ("right", "left"):
            asym = anatomy.height_asymmetry(side)
            blunt = anatomy.costophrenic_blunting(side)
            f[f"asymmetry_{side}"], f[f"blunting_{side}"] = round(asym, 3), round(blunt, 3)
            pleural_scores.append(logistic(max(asym, blunt), *CURVES["effusion"]))
        rim = measures.pneumothorax_rim(anatomy)
        for side, values in rim.items():
            f[f"lucency_{side}"] = values["lucency"]
            pleural_scores.append(logistic(values["lucency"], *CURVES["pneumothorax"]))
        pleural = max(pleural_scores)

        zones = measures.zone_opacity(anatomy)
        opacity = max((z.excess_density + 2 * z.dense_fraction for z in zones), default=0.0)
        f["opacity_index"] = round(opacity, 3)
        parenchymal = logistic(opacity, *CURVES["opacity"])

        focal_contrast = self._coarse_focal(anatomy)
        f["focal_contrast"] = round(focal_contrast, 3)
        focal = logistic(focal_contrast, *CURVES["focal"])

        groups = {"cardiac": cardiac, "pleural": pleural, "parenchymal": parenchymal,
                  "focal": focal}
        return ScreenResult(self.short, {k: round(v, 4) for k, v in groups.items()},
                            features=f, elapsed_ms=(time.perf_counter() - started) * 1000)

    @staticmethod
    def _coarse_focal(anatomy: Anatomy) -> float:
        """Strongest round dense spot at two scales on a half-size image."""
        from scipy import ndimage

        small = anatomy.image[::2, ::2]
        lungs = ndimage.binary_erosion(anatomy.lung_mask[::2, ::2], iterations=4)
        if not lungs.any():
            return 0.0
        best = 0.0
        for sigma in (4.0, 8.0):
            response = -(sigma ** 2) * ndimage.gaussian_laplace(small, sigma)
            best = max(best, float(response[lungs].max()) * 1.6)
        return best


# ------------------------------------------------------------------ deep -------
_DEEP = None
_DEEP_ERROR: Optional[str] = None


def deep_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchxrayvision  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def get_deep() -> Optional["DeepScreen"]:
    """The shared DenseNet, loaded on first use; None if the deep stack is missing."""
    global _DEEP, _DEEP_ERROR
    if _DEEP is None and _DEEP_ERROR is None:
        try:
            _DEEP = DeepScreen()
        except Exception as error:  # noqa: BLE001 - missing package, no weights, no network
            _DEEP_ERROR = f"{type(error).__name__}: {error}"
    return _DEEP


def deep_error() -> Optional[str]:
    return _DEEP_ERROR


class DeepScreen:
    """TorchXRayVision DenseNet-121 (``densenet121-res224-all``)."""

    name = "DenseNet-121 (TorchXRayVision, 8 datasets)"
    short = "densenet121-all"
    weights = "densenet121-res224-all"

    def __init__(self) -> None:
        import torch
        import torchxrayvision as xrv

        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self.torch, self.xrv = torch, xrv
        self.model = xrv.models.DenseNet(weights=self.weights)
        self.model.eval()
        self.labels = [p for p in self.model.pathologies if p]

    def _tensor(self, work: np.ndarray):
        """512 working image -> the model's input: 224 x 224 in [-1024, 1024]."""
        import cv2

        image = cv2.resize(work.astype(np.float32), (224, 224), interpolation=cv2.INTER_AREA)
        image = (image * 2 - 1) * 1024
        return self.torch.from_numpy(image)[None, None]

    def predict(self, work: np.ndarray) -> Dict[str, float]:
        with self.torch.no_grad():
            output = self.model(self._tensor(work))[0].numpy()
        return {label: float(np.clip(value, 0, 1))
                for label, value in zip(self.model.pathologies, output) if label}

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, float]]:
        batch = self.torch.cat([self._tensor(image) for image in images])
        with self.torch.no_grad():
            output = self.model(batch).numpy()
        return [{label: float(np.clip(v, 0, 1)) for label, v in zip(self.model.pathologies, row)
                 if label} for row in output]

    def screen(self, work: np.ndarray, anatomy: Anatomy, view: str = "PA") -> ScreenResult:
        started = time.perf_counter()
        labels = self.predict(work)
        groups = {group: max(labels.get(label, 0.0) for label in names)
                  for group, names in GROUP_LABELS.items()}
        return ScreenResult(self.short, {k: round(v, 4) for k, v in groups.items()},
                            labels={k: round(v, 4) for k, v in labels.items()},
                            elapsed_ms=(time.perf_counter() - started) * 1000)

    def gradcam(self, work: np.ndarray, label: str) -> Optional[np.ndarray]:
        """Grad-CAM for ``label`` on the last dense block, upsampled to 512 x 512, in [0, 1].

        The gradient of the label's output with respect to the final feature
        maps weights each map; their positive sum shows where the evidence
        for that label sits in the image.
        """
        import cv2

        if label not in self.model.pathologies:
            return None
        index = list(self.model.pathologies).index(label)
        tensor = self._tensor(work).requires_grad_(False)
        features = self.model.features(tensor)
        features.retain_grad()
        pooled = self.torch.nn.functional.adaptive_avg_pool2d(
            self.torch.nn.functional.relu(features), (1, 1)).flatten(1)
        logits = self.model.classifier(pooled)
        self.model.zero_grad()
        logits[0, index].backward()
        weights = features.grad.mean(dim=(2, 3), keepdim=True)
        cam = self.torch.relu((weights * features).sum(dim=1))[0].detach().numpy()
        if cam.max() <= 0:
            return np.zeros(work.shape, dtype=np.float32)
        cam = cv2.resize(cam / cam.max(), work.shape[::-1], interpolation=cv2.INTER_CUBIC)
        return np.clip(cam, 0, 1).astype(np.float32)


# ------------------------------------------------------------ segmenter ---------
_SEG = None
_SEG_ERROR: Optional[str] = None


def get_segmenter() -> Optional["LungSegmenter"]:
    """TorchXRayVision's PSPNet chest segmenter, loaded on first use; None if unavailable."""
    global _SEG, _SEG_ERROR
    if _SEG is None and _SEG_ERROR is None:
        try:
            _SEG = LungSegmenter()
        except Exception as error:  # noqa: BLE001
            _SEG_ERROR = f"{type(error).__name__}: {error}"
    return _SEG


class LungSegmenter:
    """PSPNet trained on ChestX-Det: 14 structures; we use both lungs and the heart.

    Classical thresholding fails on tightly cropped, scanned or post-processed
    films where the lungs merge with the image edge. The learned segmenter does
    not, so the hybrid engine uses it first and falls back to the classical
    method when its masks are implausible (e.g. on synthetic phantoms).
    """

    name = "PSPNet (TorchXRayVision, ChestX-Det)"

    def __init__(self) -> None:
        import torch
        import torchxrayvision as xrv

        self.torch = torch
        self.model = xrv.baseline_models.chestx_det.PSPNet()
        self.model.eval()
        self.targets = list(self.model.targets)

    def masks(self, work: np.ndarray) -> Dict[str, np.ndarray]:
        tensor = self.torch.from_numpy(((work * 2 - 1) * 1024).astype(np.float32))[None, None]
        with self.torch.no_grad():
            out = self.model(tensor)[0].numpy()
        pick = lambda name: out[self.targets.index(name)] > 0  # noqa: E731
        # "Left Lung" is the patient's left - image right
        return {"right": pick("Right Lung"), "left": pick("Left Lung"), "heart": pick("Heart")}
