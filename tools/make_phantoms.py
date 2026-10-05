"""Synthetic chest X-ray phantoms with known ground truth.

    python tools/make_phantoms.py            # writes samples/*.png, samples/labels.csv

Real chest X-rays are patient data; a demo and a test suite should not depend
on them. These phantoms are drawn from simple anatomy - body, two lung fields
with ribs and vessel texture, heart, spine, diaphragm domes - with one finding
painted in where the label says so. They are not realistic enough to train or
validate a clinical model, and every image carries a SYNTHETIC mark. They are
realistic enough that the built-in analyser's measurements (cardiothoracic
ratio, costophrenic angles, zone density, blob detection, quality checks) have
something real to measure, and that the labels let the evaluation page score
fixed against adaptive routing.

Use CheXpert, MIMIC-CXR or NIH ChestX-ray14 for genuine evaluation.
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

SIZE = 768
OUT = os.path.join(os.path.dirname(HERE), "samples")


def _grid():
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    return x, y


def _ellipse(x, y, cx, cy, ax, ay, soft=0.01):
    d = np.sqrt(((x - cx) / ax) ** 2 + ((y - cy) / ay) ** 2)
    return np.clip((1 - d) / soft * min(ax, ay) + 0.5, 0, 1)


def _disc(x, y, cx, cy, radius, edge=0.25):
    """A filled circle whose edge fades over ``edge`` of its radius."""
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / radius
    return np.clip((1 - d) / edge + 0.5, 0, 1)


def _smooth_noise(rng, sigma):
    from scipy.ndimage import gaussian_filter

    field = gaussian_filter(rng.standard_normal((SIZE, SIZE)).astype(np.float32), sigma)
    return field / (field.std() + 1e-6)


def _lung(x, y, cx, top, bottom, half_width, medial_sign):
    """A lung field: rounded apex, broad base, diaphragm dome at the bottom."""
    t = np.clip((y - top) / (bottom - top), 0, 1)
    width = half_width * (0.35 + 0.65 * np.sqrt(np.clip(t * 1.6, 0, 1)))
    inside_x = np.abs(x - cx) < width
    dome = bottom - 0.035 * np.cos((x - cx) / half_width * 1.3)
    inside_y = (y > top) & (y < dome)
    mask = (inside_x & inside_y).astype(np.float32)
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(mask, 2.0)


def phantom(rng, findings=(), cardiomegaly=False, quality=None):
    """One image and the facts about it."""
    from scipy.ndimage import gaussian_filter

    x, y = _grid()
    jitter = lambda s: rng.uniform(-s, s)  # noqa: E731
    image = np.full((SIZE, SIZE), 0.03, dtype=np.float32)

    body = _ellipse(x, y, 0.5 + jitter(0.01), 0.58, 0.47, 0.56, soft=0.04)
    shoulders = _ellipse(x, y, 0.5, 0.17, 0.5, 0.12, soft=0.05)
    body = np.maximum(body, shoulders * (y > 0.08))
    image += body * 0.52

    top = 0.17 + jitter(0.015)
    bottom = 0.74 + jitter(0.02)
    right = _lung(x, y, 0.315 + jitter(0.01), top, bottom, 0.135 + jitter(0.006), +1)
    left = _lung(x, y, 0.685 + jitter(0.01), top + 0.01, bottom + 0.015, 0.13 + jitter(0.006), -1)

    heart_w = (0.185 if cardiomegaly else 0.115) + jitter(0.008)
    heart = _ellipse(x, y, 0.535, 0.6 + (0.02 if cardiomegaly else 0), heart_w, 0.13 + (0.03 if cardiomegaly else 0), soft=0.05)
    spine = np.clip(1 - np.abs(x - 0.5) / 0.04, 0, 1) ** 0.5 * (y > 0.05)
    mediastinum = np.clip(1 - np.abs(x - 0.5) / 0.065, 0, 1) * (y < 0.5) * (y > 0.1)

    lungs = np.clip(right + left, 0, 1) * (1 - np.clip(heart * 1.4, 0, 1))
    image -= lungs * 0.36

    # ribs: bright arcs across both lung fields, posterior ribs slope downwards laterally
    ribs = np.zeros_like(image)
    for k in range(10):
        level = top + 0.02 + k * (bottom - top) / 10.5
        for side in (-1, 1):
            curve = level + 0.06 * ((x - 0.5) * side * 2) ** 2 - 0.02 * ((x - 0.5) * side * 2)
            band = np.exp(-((y - curve) / 0.008) ** 2) * ((x - 0.5) * side > 0.03)
            ribs += band
    image += ribs * lungs * 0.10

    # vessels: smooth branching texture, densest at the hila and fading to the periphery
    vessels = np.clip(_smooth_noise(rng, 3) * 0.5 + _smooth_noise(rng, 1.2) * 0.5, 0, None)
    hilar = np.exp(-(((np.abs(x - 0.5) - 0.12) / 0.15) ** 2 + ((y - 0.45) / 0.25) ** 2))
    texture = vessels * (0.025 + 0.05 * hilar)

    image += heart * 0.26 + spine * 0.22 + mediastinum * 0.15
    clavicles = sum(np.exp(-((y - (0.16 + 0.12 * np.abs(x - 0.5) * 1.2 - 0.02 * s)) / 0.01) ** 2)
                    * (np.abs(x - 0.5) < 0.3) * (np.abs(x - 0.5) > 0.03) for s in (0,))
    image += clavicles * 0.12

    truth = []
    lung_masks = {"right": right, "left": left}

    for finding in findings:
        kind, side = finding.split(":") if ":" in finding else (finding, "right")
        mask = lung_masks[side]
        cx = 0.315 if side == "right" else 0.685
        if kind == "effusion":
            level = bottom - 0.12 - rng.uniform(0, 0.06)
            meniscus = level - 0.05 * ((x - cx) * (1 if side == "left" else -1) / 0.13)
            fluid = gaussian_filter(((y > meniscus) * (mask > 0.2)).astype(np.float32), 4)
            image += fluid * 0.33
            truth.append("Effusion")
        elif kind == "consolidation":
            zone_y = rng.choice([0.35, 0.5, 0.62])
            patch = _disc(x, y, cx + jitter(0.03), zone_y, 0.075, edge=0.6)
            blotch = np.clip(_smooth_noise(rng, 6) + 0.8, 0, None)
            image += np.clip(patch * blotch, 0, 1) * mask * 0.28
            truth.append("Consolidation")
        elif kind == "pneumothorax":
            edge = cx + (-1 if side == "right" else 1) * 0.07
            pleural = (x < edge) if side == "right" else (x > edge)
            region = (pleural & (y < bottom - 0.12)).astype(np.float32) * (mask > 0.3)
            region = gaussian_filter(region, 1.5)
            texture = texture * (1 - region)
            image -= region * 0.08
            line = np.exp(-((x - edge) / 0.0025) ** 2) * (y < bottom - 0.12) * (mask > 0.3)
            image += line * 0.08
            truth.append("Pneumothorax")
        elif kind in ("nodule", "mass"):
            radius = rng.uniform(0.017, 0.024) if kind == "nodule" else rng.uniform(0.045, 0.055)
            ny = rng.uniform(top + 0.12, bottom - 0.15)
            nx = cx + jitter(0.05)
            blob = _disc(x, y, nx, ny, radius, edge=0.3)
            image += blob * 0.22
            truth.append("Nodule" if kind == "nodule" else "Mass")
    if cardiomegaly:
        truth.append("Cardiomegaly")

    image += texture * lungs
    image += rng.standard_normal(image.shape).astype(np.float32) * 0.008
    image = gaussian_filter(image, 0.8)

    if quality == "blur":
        image = gaussian_filter(image, 12)
        image = 0.45 + (image - image.mean()) * 0.35
    elif quality == "cutoff":
        image = image[: int(SIZE * 0.55)]
    elif quality == "overexposed":
        image = np.clip(image * 2.4 + 0.25, 0, 1)
    image = np.clip(image, 0, 1)
    return image, truth


def _stamp(image):
    """Burn a small SYNTHETIC mark into the bottom-right corner."""
    from PIL import Image, ImageDraw

    canvas = Image.fromarray((image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    draw.text((w - 150, h - 22), "SYNTHETIC PHANTOM", fill=150)
    return canvas


CASES = [
    ("normal", [], False, None), ("normal", [], False, None), ("normal", [], False, None),
    ("normal", [], False, None), ("normal", [], False, None), ("normal", [], False, None),
    ("cardiomegaly", [], True, None), ("cardiomegaly", [], True, None),
    ("cardiomegaly", [], True, None),
    ("effusion", ["effusion:right"], False, None), ("effusion", ["effusion:left"], False, None),
    ("effusion", ["effusion:right"], True, None),
    ("consolidation", ["consolidation:right"], False, None),
    ("consolidation", ["consolidation:left"], False, None),
    ("consolidation", ["consolidation:right"], False, None),
    ("pneumothorax", ["pneumothorax:right"], False, None),
    ("pneumothorax", ["pneumothorax:left"], False, None),
    ("nodule", ["nodule:right"], False, None), ("nodule", ["nodule:left"], False, None),
    ("nodule", ["nodule:left"], False, None), ("mass", ["mass:right"], False, None),
    ("quality", [], False, "blur"), ("quality", [], False, "cutoff"),
    ("quality", [], False, "overexposed"),
]
SITES = ("Chennai GH", "Coimbatore MC", "Madurai RH")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(2026)
    rows = []
    for index, (name, findings, cardio, quality) in enumerate(CASES, 1):
        image, truth = phantom(rng, findings, cardio, quality)
        file_name = f"cxr_{index:02d}_{name}.png"
        _stamp(image).save(os.path.join(OUT, file_name))
        rows.append({"file": file_name, "labels": ";".join(truth) or "No Finding",
                     "quality": quality or "ok", "sex": rng.choice(["M", "F"]),
                     "age": int(rng.integers(19, 84)), "site": SITES[index % 3], "view": "PA"})
    with open(os.path.join(OUT, "labels.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_dicom(rng)
    print(f"wrote {len(rows)} phantoms and 1 DICOM to {OUT}")
    return 0


def _write_dicom(rng):
    """One DICOM with a fake patient header, to show de-identification on load."""
    try:
        import pydicom
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    except ImportError:
        return
    image, _ = phantom(rng, ["effusion:left"], False, None)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName, ds.PatientID = "PHANTOM^DEMO", "TG-0001"
    ds.PatientBirthDate, ds.PatientSex, ds.PatientAge = "19700101", "F", "056Y"
    ds.InstitutionName, ds.ReferringPhysicianName = "Synthetic Hospital", "DOE^JANE"
    ds.Modality, ds.ViewPosition = "DX", "PA"
    ds.PhotometricInterpretation, ds.SamplesPerPixel = "MONOCHROME2", 1
    ds.Rows, ds.Columns = image.shape
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelSpacing = [0.5, 0.5]
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.PixelData = (image * 4095).astype(np.uint16).tobytes()
    ds.save_as(os.path.join(OUT, "cxr_25_effusion_dicom.dcm"), enforce_file_format=True)
    with open(os.path.join(OUT, "labels.csv"), "a", newline="", encoding="utf-8") as handle:
        handle.write("cxr_25_effusion_dicom.dcm,Effusion,ok,F,56,Chennai GH,PA\n")


if __name__ == "__main__":
    raise SystemExit(main())
