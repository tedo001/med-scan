"""Small Kaggle-style chest X-ray datasets for tests, made from the demo phantoms.

Layout as in "Lungs Disease Dataset (4 types)": <root>/<split>/<class>/<image>.
Each image is a jittered copy (shift, gamma, noise) of a phantom, so classes differ
the way the phantoms do. A FakeDeep gives 1024 features without PyTorch.
"""

from __future__ import annotations

import os
import zipfile

import numpy as np
from PIL import Image

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
SOURCES = {"Normal": ("cxr_01_normal.png", "cxr_02_normal.png", "cxr_03_normal.png"),
           "Bacterial Pneumonia": ("cxr_13_consolidation.png", "cxr_14_consolidation.png",
                                   "cxr_15_consolidation.png"),
           "Tuberculosis": ("cxr_18_nodule.png",)}


def make_dataset(root: str, per_split=(("train", 12), ("test", 4)), seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    base = os.path.join(root, "Lung Disease Dataset")
    for split, count in per_split:
        for cls, names in SOURCES.items():
            folder = os.path.join(base, split, cls)
            os.makedirs(folder, exist_ok=True)
            for i in range(count):
                src = np.asarray(Image.open(os.path.join(SAMPLES, names[i % len(names)])).convert("L"),
                                 dtype=np.float64) / 255
                src = np.roll(src, (int(rng.integers(-6, 7)), int(rng.integers(-6, 7))), axis=(0, 1))
                src = np.clip(src ** rng.uniform(0.9, 1.1) + rng.normal(0, 0.01, src.shape), 0, 1)
                Image.fromarray((src * 255).astype(np.uint8)).save(
                    os.path.join(folder, f"{cls[:3].lower()}_{split}_{i:03d}.png"))
    return base


def make_zip(root: str, folder: str) -> str:
    path = os.path.join(root, "lungs-disease-dataset-4-types.zip")
    with zipfile.ZipFile(path, "w") as archive:
        for dirpath, _, files in os.walk(folder):
            for name in files:
                full = os.path.join(dirpath, name)
                archive.write(full, os.path.relpath(full, os.path.dirname(folder)))
    return path


class FakeDeep:
    """1024 image features (32 × 32 thumbnail) - stands in for the DenseNet in fast tests."""

    def embed(self, work):
        img = Image.fromarray((np.clip(work, 0, 1) * 255).astype(np.uint8)).resize((32, 32))
        return np.asarray(img, dtype=np.float64).ravel() / 255 + 0.05
