"""Import labelled chest X-ray datasets - class folders, zips, or labels.csv.

Most public datasets (Kaggle and others) label images by **folder name**::

    Lung Disease Dataset/
        train/ Bacterial Pneumonia/ Corona Virus Disease/ Normal/ Tuberculosis/ Viral Pneumonia/
        val/   ...
        test/  ...

This module reads that layout from a folder **or straight from the downloaded
.zip** (nothing is unpacked except the sampled images), maps each class to a
MEDSCAN finding, draws a balanced sample per class from the chosen splits, and
writes a ``labels.csv`` manifest that every training and benchmark tool reads.

Class mapping is by keywords and is editable in the UI:

======================================  =================================
class name contains                     MEDSCAN label
======================================  =================================
pneumonia, covid, corona, opacity,      Consolidation
consolidation, infiltrat
normal, no finding, healthy             No Finding
effusion / cardiomegaly / pneumothorax  same name
/ nodule / mass
tuberculosis, tb                        (skipped - no TB finding in MEDSCAN)
======================================  =================================
"""

from __future__ import annotations

import csv
import os
import random
import re
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import paths

__all__ = ["DatasetInfo", "inspect", "build_manifest", "default_label", "TARGETS", "SKIP",
           "IMAGE_EXTENSIONS"]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm")
SPLITS = ("train", "val", "valid", "validation", "test")
SKIP = "Skip"
TARGETS = ("Consolidation", "No Finding", "Cardiomegaly", "Effusion", "Pneumothorax", "Nodule",
           "Mass", "Atelectasis", "Edema", SKIP)
RULES: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("tubercul", "tb"), SKIP),
    (("normal", "no finding", "no_finding", "healthy"), "No Finding"),
    (("pneumonia", "covid", "corona", "opacity", "consolidation", "infiltrat"), "Consolidation"),
    (("effusion",), "Effusion"), (("cardiomegaly",), "Cardiomegaly"),
    (("pneumothorax",), "Pneumothorax"), (("nodule",), "Nodule"), (("mass",), "Mass"),
    (("atelectasis",), "Atelectasis"), (("edema", "oedema"), "Edema"),
)


def default_label(class_name: str) -> str:
    name = class_name.lower().replace("_", " ").replace("-", " ")
    words = set(re.findall(r"[a-z]+", name))
    for keys, label in RULES:
        for key in keys:
            if (len(key) <= 2 and key in words) or (len(key) > 2 and key in name):
                return label
    return SKIP


@dataclass
class DatasetInfo:
    source: str
    kind: str                                   # "class-folders", "zip", "labels.csv"
    classes: Dict[str, Dict[str, int]] = field(default_factory=dict)   # class -> split -> count
    files: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)  # (class, split) -> paths
    total: int = 0

    @property
    def splits(self) -> List[str]:
        return sorted({s for c in self.classes.values() for s in c})

    def mapping(self) -> Dict[str, str]:
        return {c: default_label(c) for c in self.classes}


def _classify(parts: List[str]) -> Optional[Tuple[str, str]]:
    """(class, split) from a path's directory parts: the image's parent folder is the class."""
    if len(parts) < 2:
        return None
    cls = parts[-2]
    split = "all"
    for p in parts[:-2]:
        if p.lower() in SPLITS:
            split = {"valid": "val", "validation": "val"}.get(p.lower(), p.lower())
    if cls.lower() in SPLITS:
        return None
    return cls, split


def inspect(source: str, progress: Optional[Callable] = None) -> DatasetInfo:
    """Find classes, splits and image counts in a folder, a .zip, or a labels.csv folder."""
    if os.path.isfile(os.path.join(source, "labels.csv")):
        info = DatasetInfo(source, "labels.csv")
        with open(os.path.join(source, "labels.csv"), newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                cls = (row.get("labels") or "No Finding").split(";")[0] or "No Finding"
                info.classes.setdefault(cls, {}).setdefault("all", 0)
                info.classes[cls]["all"] += 1
                info.files.setdefault((cls, "all"), []).append(os.path.join(source, row["file"]))
                info.total += 1
        return info
    if zipfile.is_zipfile(source):
        info = DatasetInfo(source, "zip")
        with zipfile.ZipFile(source) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(IMAGE_EXTENSIONS)
                     and not n.startswith("__MACOSX")]
        entries = [(n, n.replace("\\", "/").split("/")) for n in names]
    elif os.path.isdir(source):
        info = DatasetInfo(source, "class-folders")
        entries = []
        for root, _, files in os.walk(source):
            for name in files:
                if name.lower().endswith(IMAGE_EXTENSIONS):
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, source).replace("\\", "/").split("/")
                    entries.append((full, [os.path.basename(source)] + rel))
            if progress:
                progress(len(entries), 0, f"scanning {root}")
    else:
        raise FileNotFoundError(f"{source} is not a folder or a .zip")
    for path, parts in entries:
        found = _classify(parts)
        if found is None:
            continue
        cls, split = found
        info.classes.setdefault(cls, {}).setdefault(split, 0)
        info.classes[cls][split] += 1
        info.files.setdefault((cls, split), []).append(path)
        info.total += 1
    if not info.classes:
        raise ValueError("no images found in class folders (expected <split>/<class>/<image>)")
    return info


def build_manifest(info: DatasetInfo, mapping: Dict[str, str], splits: List[str],
                   per_class: int, name: str, seed: int = 2026,
                   progress: Optional[Callable] = None) -> str:
    """Sample up to ``per_class`` images per class from ``splits``; write labels.csv.

    Zip members are extracted (only the sampled ones). Returns the manifest folder:
    ``<data>/datasets/<name>/`` with ``labels.csv`` (absolute image paths for folders,
    extracted copies for zips).
    """
    out = os.path.join(paths.data_directory(), "datasets", re.sub(r"[^A-Za-z0-9_.-]+", "_", name))
    os.makedirs(out, exist_ok=True)
    rng = random.Random(seed)
    chosen: List[Tuple[str, str, str]] = []                 # (path, class, label)
    for cls, label in mapping.items():
        if label == SKIP:
            continue
        pool = [p for s in splits for p in info.files.get((cls, s), [])]
        rng.shuffle(pool)
        chosen += [(p, cls, label) for p in pool[:per_class]]
    rng.shuffle(chosen)
    rows = []
    archive = zipfile.ZipFile(info.source) if info.kind == "zip" else None
    try:
        for index, (path, cls, label) in enumerate(chosen, 1):
            if archive is not None:
                target = os.path.join(out, "images", f"{index:05d}{os.path.splitext(path)[1].lower()}")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                if not os.path.isfile(target):
                    with archive.open(path) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                file_ref = os.path.relpath(target, out)
            else:
                file_ref = os.path.abspath(path)
            rows.append({"file": file_ref, "labels": label, "quality": "ok", "sex": "", "age": "",
                         "site": os.path.basename(info.source.rstrip("/\\"))[:40], "view": "PA",
                         "source_class": cls})
            if progress:
                progress(index, len(chosen), f"{cls} → {label}")
    finally:
        if archive is not None:
            archive.close()
    if not rows:
        raise ValueError("nothing to sample - every class is mapped to Skip or the splits are empty")
    with open(os.path.join(out, "labels.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out
