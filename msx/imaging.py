"""Stage 1 - load a chest X-ray, strip identifiers, normalise it.

Accepts PNG, JPEG, BMP, TIFF and DICOM. Whatever comes in leaves as a
:class:`Scan`: a float32 image in [0, 1] where *bright means dense* (bone,
fluid, heart) and *dark means air* (lungs), plus the few facts the analyser and
the bias monitor need - view, sex, age band, pixel spacing.

**Privacy.** A DICOM header names the patient. Every tag in
:data:`PHI_TAGS` is dropped on load and its *name* (never its value) is listed
in ``Scan.removed_tags`` so the audit trail can show de-identification
happened. The patient ID is replaced by a pseudonym - an HMAC under a key kept
on this machine - so two scans of one patient still link up without the ID
being stored anywhere.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import paths

__all__ = ["Scan", "load_scan", "from_array", "standardise", "SUPPORTED", "PHI_TAGS",
           "pseudonym", "save_png"]

SUPPORTED = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom")

#: DICOM attributes that identify a person, an institution or an operator.
PHI_TAGS = (
    "PatientName", "PatientID", "PatientBirthDate", "PatientBirthTime", "PatientAddress",
    "PatientTelephoneNumbers", "OtherPatientIDs", "OtherPatientNames", "PatientMotherBirthName",
    "MedicalRecordLocator", "ReferringPhysicianName", "PerformingPhysicianName",
    "OperatorsName", "InstitutionName", "InstitutionAddress", "StationName",
    "AccessionNumber", "StudyID", "IssuerOfPatientID", "RequestingPhysician",
)

#: Working resolution for every stage after loading.
WORK_SIZE = 512


@dataclass
class Scan:
    """One de-identified chest X-ray, ready for analysis."""

    pixels: np.ndarray                 # float32, HxW, [0, 1], bright = dense
    source: str = ""                   # file name only, never the folder
    sha256: str = ""                   # of the original file's bytes
    view: str = "PA"                   # PA, AP or LATERAL
    sex: str = ""                      # M / F / "" (for bias monitoring only)
    age_band: str = ""                 # "0-17", "18-39", "40-64", "65+"
    pixel_spacing_mm: Optional[float] = None
    patient_ref: str = ""              # pseudonym, never the real ID
    removed_tags: List[str] = field(default_factory=list)
    original_shape: Tuple[int, int] = (0, 0)
    bit_depth: int = 8
    modality: str = ""

    @property
    def mm_per_work_pixel(self) -> Optional[float]:
        """Pixel spacing at the 512 working size, if the file gave one."""
        if not self.pixel_spacing_mm or not self.original_shape[0]:
            return None
        return self.pixel_spacing_mm * max(self.original_shape) / WORK_SIZE

    def facts(self) -> Dict[str, object]:
        return {"source": self.source, "view": self.view, "sex": self.sex,
                "age_band": self.age_band, "patient_ref": self.patient_ref,
                "original_shape": list(self.original_shape), "bit_depth": self.bit_depth,
                "pixel_spacing_mm": self.pixel_spacing_mm, "removed_tags": list(self.removed_tags)}


def _key() -> bytes:
    """The local pseudonymisation key, made once per installation."""
    path = os.path.join(paths.data_directory(), "pseudonym.key")
    try:
        with open(path, "rb") as handle:
            key = handle.read()
        if len(key) >= 32:
            return key
    except OSError:
        pass
    key = secrets.token_bytes(32)
    try:
        with open(path, "wb") as handle:
            handle.write(key)
    except OSError:
        pass
    return key


def pseudonym(patient_id: str) -> str:
    """A stable, non-reversible reference for ``patient_id`` on this machine."""
    if not patient_id:
        return ""
    digest = hmac.new(_key(), patient_id.strip().encode("utf-8"), hashlib.sha256).hexdigest()
    return "PT-" + digest[:10].upper()


def age_band(age: Optional[float]) -> str:
    if age is None:
        return ""
    if age < 18:
        return "0-17"
    if age < 40:
        return "18-39"
    if age < 65:
        return "40-64"
    return "65+"


def _dicom_age(value: str) -> Optional[float]:
    """'045Y' -> 45.0; months, weeks and days scaled to years."""
    try:
        number, unit = float(value[:-1]), value[-1].upper()
    except (TypeError, ValueError, IndexError):
        return None
    return number / {"Y": 1, "M": 12, "W": 52, "D": 365}.get(unit, 1)


def _to_unit(array: np.ndarray) -> np.ndarray:
    """Robust min-max to [0, 1] on the 0.5-99.5 percentile window."""
    array = array.astype(np.float32)
    low, high = np.percentile(array, (0.5, 99.5))
    if high - low < 1e-6:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def _load_dicom(path: str) -> Scan:
    import pydicom

    dataset = pydicom.dcmread(path)
    removed = [name for name in PHI_TAGS if name in dataset]
    patient_id = str(getattr(dataset, "PatientID", "") or "")
    sex = str(getattr(dataset, "PatientSex", "") or "").upper()[:1]
    age = _dicom_age(str(getattr(dataset, "PatientAge", "") or ""))
    view = str(getattr(dataset, "ViewPosition", "") or "PA").upper()
    spacing = getattr(dataset, "PixelSpacing", None) or getattr(dataset, "ImagerPixelSpacing", None)
    for name in removed:
        delattr(dataset, name)
    pixels = dataset.pixel_array.astype(np.float32)
    if pixels.ndim == 3:  # multi-frame or colour: take the first plane
        pixels = pixels[0] if pixels.shape[0] < pixels.shape[-1] else pixels[..., 0]
    pixels = pixels * float(getattr(dataset, "RescaleSlope", 1) or 1) + float(
        getattr(dataset, "RescaleIntercept", 0) or 0)
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        pixels = pixels.max() - pixels  # MONOCHROME1 stores air as bright
    return Scan(pixels=_to_unit(pixels), view=view if view in ("PA", "AP", "LL", "LATERAL") else "PA",
                sex=sex if sex in ("M", "F") else "", age_band=age_band(age),
                pixel_spacing_mm=float(spacing[0]) if spacing else None,
                patient_ref=pseudonym(patient_id), removed_tags=removed,
                original_shape=tuple(pixels.shape), bit_depth=int(getattr(dataset, "BitsStored", 12)),
                modality=str(getattr(dataset, "Modality", "") or ""))


def _load_raster(path: str) -> Scan:
    from PIL import Image

    with Image.open(path) as image:
        depth = 16 if image.mode in ("I;16", "I;16B", "I") else 8
        if image.mode not in ("L", "I;16", "I;16B", "I", "F"):
            image = image.convert("L")
        pixels = np.asarray(image).astype(np.float32)
    return Scan(pixels=_to_unit(pixels), original_shape=tuple(pixels.shape), bit_depth=depth)


def load_scan(path: str, **facts) -> Scan:
    """Read ``path`` into a de-identified :class:`Scan`; ``facts`` override header values."""
    extension = os.path.splitext(path)[1].lower()
    if extension not in SUPPORTED:
        raise ValueError(f"Unsupported file type {extension or '(none)'}; use "
                         + ", ".join(SUPPORTED))
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    scan = _load_dicom(path) if extension in (".dcm", ".dicom") else _load_raster(path)
    scan.source, scan.sha256 = os.path.basename(path), digest
    for key, value in facts.items():
        if value not in (None, "") and hasattr(scan, key):
            setattr(scan, key, value)
    if not scan.patient_ref:
        scan.patient_ref = pseudonym(digest)  # one scan, one reference
    return scan


def from_array(pixels: np.ndarray, source: str = "array", **facts) -> Scan:
    """A :class:`Scan` from an in-memory image (tests, augmentation)."""
    scan = Scan(pixels=_to_unit(np.asarray(pixels)), source=source,
                original_shape=tuple(np.asarray(pixels).shape))
    scan.sha256 = hashlib.sha256(np.ascontiguousarray(scan.pixels).tobytes()).hexdigest()
    for key, value in facts.items():
        setattr(scan, key, value)
    return scan


def standardise(pixels: np.ndarray, size: int = WORK_SIZE) -> np.ndarray:
    """Letterbox ``pixels`` to ``size`` x ``size`` (aspect kept, black padding)."""
    import cv2

    height, width = pixels.shape
    scale = size / max(height, width)
    resized = cv2.resize(pixels.astype(np.float32), (max(1, round(width * scale)),
                                                     max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((size, size), dtype=np.float32)
    top, left = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def clahe(pixels: np.ndarray) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation, for display and texture."""
    import cv2

    eight = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(eight).astype(np.float32) / 255


def save_png(pixels: np.ndarray, path: str) -> str:
    """Write a [0, 1] image as an 8-bit PNG - the de-identified copy we keep."""
    from PIL import Image

    Image.fromarray((np.clip(pixels, 0, 1) * 255).astype(np.uint8)).save(path)
    return path
