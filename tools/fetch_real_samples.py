"""Download a small *real* chest X-ray evaluation set (pneumonia vs no finding).

    python tools/fetch_real_samples.py                 # -> ~/.medscan/real_samples
    python -m msx.evaluation ~/.medscan/real_samples --engine hybrid

Source: the public COVID-19 Image Data Collection (Cohen et al., 2020,
https://github.com/ieee8023/covid-chestxray-dataset). Only frontal PA films are
taken: every "No Finding" film and a sample of pneumonia films (viral,
bacterial, fungal). Pneumonia is labelled **Consolidation** - MEDSCAN's
airspace-opacity finding - since that is what it looks like on a radiograph.

The images keep their original licences (listed per image in the dataset's
metadata) and are **not** redistributed in this repository; this script fetches
them into your own data folder with a labels.csv the evaluation reads.

This is a small, imbalanced, convenience set - good for a sanity check on real
films, not a clinical validation. Use CheXpert / MIMIC-CXR / NIH ChestX-ray14
(credentialed downloads) for that, with the same labels.csv format.
"""

from __future__ import annotations

import csv
import io
import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msx import paths  # noqa: E402

BASE = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main(pneumonia: int = 40) -> int:
    out = os.path.join(paths.data_directory(), "real_samples")
    os.makedirs(out, exist_ok=True)
    rows = list(csv.DictReader(io.StringIO(fetch(BASE + "metadata.csv").decode("utf-8"))))
    frontal = [r for r in rows if r["modality"] == "X-ray" and r["view"] == "PA"
               and r["folder"] == "images" and r["filename"].lower().endswith((".jpg", ".jpeg", ".png"))]
    normal = [r for r in frontal if r["finding"] == "No Finding"]
    sick = [r for r in frontal if r["finding"].startswith("Pneumonia")]
    random.Random(2026).shuffle(sick)
    chosen = [(r, "No Finding") for r in normal] + [(r, "Consolidation") for r in sick[:pneumonia]]
    labels = []
    for index, (row, label) in enumerate(chosen, 1):
        name = f"real_{index:03d}{os.path.splitext(row['filename'])[1].lower()}"
        target = os.path.join(out, name)
        if not os.path.isfile(target):
            try:
                data = fetch(BASE + "images/" + urllib.request.quote(row["filename"]))
            except Exception as error:  # noqa: BLE001
                print("skip", row["filename"], error)
                continue
            with open(target, "wb") as handle:
                handle.write(data)
        labels.append({"file": name, "labels": label, "quality": "ok", "sex": row["sex"],
                       "age": row["age"] or "", "site": (row["location"] or "unknown")[:40],
                       "view": "PA", "source": row["filename"], "finding": row["finding"]})
        print(f"{index:3d}/{len(chosen)} {label:13s} {row['finding']}")
    with open(os.path.join(out, "labels.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)
    print(f"wrote {len(labels)} images to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 40))
