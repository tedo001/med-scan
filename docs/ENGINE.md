# The MEDSCAN analyser engine (`msx`): what it does, how, and why

MEDSCAN is a human-in-the-loop chest X-ray assistant. The analyser engine is the
`msx` package. The desktop app (`ui/`) only shows its results and records the
doctor's decisions.

The engine's job is the TECHgium challenge in one sentence: **find and highlight
possible abnormalities, say how sure it is and why, refuse to guess on bad
scans, and leave the decision to the doctor.** Its design idea is the one the
deck proposes: **adaptive analysis routing**. Every scan gets a cheap screen.
Only the scans that need it get expensive specialist analysis.

```
 file ─▶ 1 Preprocess ─▶ 2 Quality ──fail──▶ HUMAN REVIEW (no findings)
                              │ pass
                              ▼
                         3 Screen (fast)  ── 4 groups: cardiac · pleural · parenchymal · focal
                              │
                              ▼
                         4 Router ──all low──▶ EARLY EXIT "no significant abnormality"
                              │ some high
                              ▼
                 5 Specialists (only the routed ones, each × test-time augmentation)
                   Cardiac · Pleural · Parenchyma · Nodule
                              │
                              ▼
                 6 Confidence: calibration · stability · agreement · quality → abstain?
                              │
                              ▼
                 7 Explain: brief / standard / detailed + cited evidence (RAG)
                              │
                              ▼
                 8 Doctor: Accept · Correct · Reject · Question → sign off → audit chain
```

---

## Stage by stage

### 1 · Preprocess: `msx/imaging.py`, `msx/anatomy.py`

| What | How | Why |
|---|---|---|
| Load PNG, JPEG, TIFF or DICOM | Pillow; for DICOM, pydicom applies rescale slope/intercept and inverts MONOCHROME1 | Hospitals send DICOM; teaching sets are PNG |
| **De-identify** | Drops 20 identifying DICOM tags (name, ID, birth date, institution, physicians…) and records only the *names* of the removed tags. The patient ID becomes a pseudonym: an HMAC-SHA256 under a key that never leaves the machine | Patient privacy is in the brief. Pseudonyms still let two scans of the same patient link up |
| Normalise | Robust 0.5–99.5 percentile window to [0, 1]; letterbox to 512 × 512 | Every later stage sees the same geometry and intensity scale |
| **Find the lungs** | Gaussian smoothing, then Otsu threshold. Dark regions touching the image border are outside air and are dropped. The largest remaining region on each side is a lung. If the lungs bleed into soft tissue, the threshold steps down (×0.92 … ×0.62) until both lungs stand apart. The **envelope** is the aerated lung with holes filled and gaps up to about 25 px closed: where lung *should* be | The quality gate, CTR, costophrenic angles and lung zones all need the lungs. Classical segmentation is deterministic, about 50 ms, needs no network, and fails visibly: no lungs means a quality hold |

Image-left is the **patient's right**, as radiographs are always viewed, so
findings are named anatomically.

### 2 · Quality gate: `msx/quality.py`

Seven measured checks, each pass, warn or fail with a sentence of reasoning:

| Check | Measure | Fail when |
|---|---|---|
| Resolution | shortest side of the original | < 256 px |
| Sharpness | 99th-percentile gradient ÷ contrast, after a 1-px blur so noise doesn't count | < 0.33 (motion, defocus) |
| Contrast | 5th–95th percentile spread inside the body | < 0.25 |
| Exposure | share of body pixels clipped to white or black | > 30 % |
| Lung fields | were both lungs found? | either missing |
| Field coverage | do the lungs touch the image edge (apex or angles cut off)? | 2+ edges cut |
| Symmetry | lung area ratio | warn only (< 0.6: rotation or opaque lung) |

**Any fail stops the pipeline.** The study goes to the review queue as
*Quality hold – human review* with **no findings**. This is the brief's
"detect poor-quality or incomplete scans and request human review instead of
producing unreliable results". Warnings let analysis continue but reduce every
finding's confidence.

### 3 · Screen: `msx/screening.py`

One fast look that produces **four group probabilities**, nothing more:

* **Built-in measurement screen** (always available, about 100 ms). Interpretable
  measurements turned into probabilities with logistic curves calibrated so a
  textbook normal sits below 0.05:
  * cardiothoracic ratio
  * lung-height asymmetry and costophrenic blunting
  * contralateral zone density
  * lateral-band lucency
  * a coarse blob search
* **DenseNet-121** from TorchXRayVision (`densenet121-res224-all`, optional).
  Trained on NIH, CheXpert, MIMIC-CXR, PadChest and four other public datasets.
  It has 18 outputs calibrated so that 0.5 is each label's operating point, and
  takes one 224 × 224 forward pass.

In **hybrid** mode the two are fused per group in log-odds space (see stage 5).

### 4 · Adaptive Analysis Router: `msx/router.py`

The centre of the design. Thresholds move with the **clinical context**,
because a missed finding costs more in some settings than others:

| Context | Route at | Early exit below | Explanation |
|---|---|---|---|
| Emergency | 0.20 | 0.08 | brief, action first |
| Routine OPD | 0.30 | 0.12 | standard |
| Screening camp | 0.40 | 0.18 | brief |
| Teaching | all modules | – | detailed |

* **Early exit:** every group is below the exit threshold on a clean scan, so no
  specialist runs.
* **Route:** a group at or above the threshold sends the scan to its module.
* **Borderline band:** a group just under the threshold is routed anyway in
  Emergency, or when the scan has quality warnings.
* Every decision is stored with its reason; the Review page shows it.
  `mode="fixed"` runs all four modules and is the baseline for evaluation.

### 5 · Specialist modules: `msx/specialists.py`, `msx/measures.py`

| Module | Region | Measurement |
|---|---|---|
| **Cardiac** | heart between the lungs | **CTR** = widest gap between the medial lung margins (lower half, above the domes) ÷ outer thoracic width. Above 0.50 on PA (0.56 on AP) is cardiomegaly. Measurement lines are drawn on the image |
| **Pleural** | bases, lateral and apical rim | **Effusion:** lung shorter than the other side, and/or lateral costophrenic recess raised above the medial base (meniscus). **Pneumothorax:** lateral band more lucent than the other side's, with lung-marking density as support |
| **Parenchyma** | six zones (upper, middle, lower × 2) | each zone's density in SDs above its mirror zone, plus the share of expected lung that is no longer aerated |
| **Nodule** | inside the lungs | multi-scale Laplacian-of-Gaussian blobs with a local-contrast test. Blobs are rejected when they sit on a lung edge, are faint and in the hilum, or are in an already-dense or crowded zone. Larger than 30 mm is a mass; real size is reported when DICOM pixel spacing is known |

**Fusion with the deep model:**

```
logit(p) = 0.8 · logit(p_deep) + 0.8 · logit(p_measured)
```

Two independent sources of evidence add up. A model sitting on the fence (0.5)
adds nothing, which matters on images unlike its training data. When the two
sources disagree, the answer is pulled back towards 0.5, and the agreement term
in stage 6 lowers confidence. The measurement says *where* and *how much*; the
model adds pattern knowledge from about 800,000 training films. For routed
findings, Grad-CAM on the DenseNet's last dense block gives the heatmap,
restricted to the lungs.

**Test-time augmentation:** each routed module re-measures on 4 slightly shifted,
scaled and re-exposed copies of the scan; the DenseNet scores the same copies in
one batch. The spread across copies is the finding's **stability**.

### 6 · Confidence and abstention: `msx/uncertainty.py`

```
confidence = (0.5·decisiveness + 0.5·stability) × agreement × quality
```

| Term | Meaning |
|---|---|
| decisiveness | `|2p − 1|`: distance from a 50/50 call |
| stability | `1 − σ_TTA / 0.2`: does the finding survive small perturbations? |
| agreement | `1 − 0.5·|p_deep − p_measured|` when the model is informative |
| quality | `1 − 0.08` per quality warning (floor 0.6) |

The four terms combine into an overall level: High ≥ 0.75, Moderate ≥ 0.5,
otherwise Low. If confidence < 0.40 while the probability is in the grey zone
(0.3–0.7), the engine **abstains**: the finding is marked *AI abstains – human
read required*. A probability backed only by the deep model never raises a
finding on its own below the positive threshold.

**Calibration:** temperature scaling per group, `p' = σ(logit(p)/T)`. T is fitted
by minimising log loss on a labelled benchmark (Evaluation → *Calibrate*) and
applied to every later analysis.

### 7 · Adaptive, evidence-grounded explanation: `msx/explain.py`, `msx/knowledge.py`

* **Depth** follows the context (brief / standard / detailed) and **escalates one
  level when the engine is unsure**: an abstention, or a positive at low
  confidence. That is when a doctor needs more evidence, not less.
* **Medical RAG:** Okapi BM25 over 22 curated passages. Each carries its source:
  * Fleischner 2017
  * BTS Pleural 2023
  * NICE NG122 and NG138
  * ESC Heart Failure 2021
  * the CheXpert, AI-explanation and bias papers the deck cites

  Explanations quote the retrieved passages with citations; nothing is invented.
* **Doctor Q&A:** "Why?", "How sure?", "What could mimic it?", "What next?",
  "How big?" Each question is answered from the finding's own evidence,
  confidence breakdown, limitations and retrieved guidance.
* **Optional local LLM** (Ollama, e.g. `gemma2`) can rewrite the text. Its output
  is **checked**: any number not in the facts, or any finding that wasn't
  reported, rejects it and the template text is used instead. This addresses
  the deck's *hallucination risk*.

### 8 · Record: `msx/datastore.py`, `msx/audit.py`, `msx/evaluation.py`, `msx/bias.py`

* **SQLite** holds the full analysis, the de-identified image, heatmap and lung
  mask, every Accept / Correct / Reject / Add and Question, the blinded first
  read and the signed final read.
* **Audit trail:** append-only JSONL. Each entry carries the SHA-256 of the one
  before, so an edit, insertion or deletion breaks the chain at that line, and
  *Verify chain* says where.
* **Evaluation:**
  1. Fixed vs adaptive pipeline on a labelled folder: AUC, sensitivity,
     specificity, time, modules and model calls.
  2. **Doctor alone** (blinded first read) vs **AI alone** vs **Doctor + AI**
     (signed read), with decision times.
* **Bias monitor:** sex, age band, site and view subgroups. A subgroup is flagged
  when sensitivity or doctor agreement trails the overall figure by more than
  10 points.

---

## Adapting to the doctor, learning from the doctor

The stages above adapt to the **clinical situation** and to the engine's own
uncertainty. These pieces adapt to the **doctor**, learn from them, and close
the loop with recommendations and a report. They map directly onto the
problem statement.

| Problem statement asks for | Module | What it does |
|---|---|---|
| Adjust help to different doctors *without assuming ability* | `msx/support.py` | Four **doctor-chosen** support modes: Guided, Concise, Second opinion, Evidence-first. A mode is saved on the doctor's own account and switchable on any case. It is never set from title or seniority |
| Explain why a finding may matter | `msx/explain.py` | A clinical significance, urgency and "if missed" block per finding, plus a "how to read this" checklist in Guided mode |
| Recommendations (deliverable) | `msx/recommend.py` | Urgency tier (Now / Same day / Within 1 week / Routine), next test, follow-up and referral. Adjusted for confidence (low → "confirm before acting"), nodule size (Fleischner bands), context (Emergency brings items forward) and **patterns across findings** (cardiomegaly + effusion → one heart-failure work-up) |
| Generative AI (deliverable) | `msx/report.py` | A draft report (EXAMINATION / TECHNIQUE / FINDINGS / IMPRESSION / RECOMMENDATIONS) that the doctor edits and signs. A local LLM writes the prose when available; output is fact-checked against the findings; a grounded template is the fallback |
| Machine learning (deliverable) and learning from doctors | `msx/learner.py` | A logistic-regression stacker trained on every finding's eventual verdict: ground truth, Accept / Reject / Correct, or Add (an AI miss). 5-fold cross-validated, blended into each finding with weight `min(0.5, n/200)`, retrained after every sign-off |
| Deep learning on **your own image dataset** | `msx/datasets.py`, `msx/deephead.py` | Transfer learning. A labelled dataset (class folders or the downloaded .zip, e.g. Kaggle's *Lungs Disease Dataset (4 types)*) is mapped class → finding (pneumonia / COVID → Consolidation, Normal → No Finding, Tuberculosis skipped), sampled per class, passed through the frozen DenseNet-121 (1024 features per image), and a new one-vs-rest logistic classifier is trained on top, scored by 5-fold CV AUC / sensitivity / specificity and versioned. In the analyser it adds "dataset model" evidence in log-odds (weight 0.7) to the matching finding. It **only corroborates** (it can raise a probability that is already ≥ 0.35, and can always lower one), and abstains on images whose features are further than the training set's own 99th-percentile nearest-neighbour distance |
| Robust anatomy on real films | `msx/screening.py` `LungSegmenter` | TorchXRayVision **PSPNet** lung and heart segmentation, used in hybrid mode as a **fallback** when classical segmentation fails (tightly cropped or post-processed films). The test-time-augmentation copies reuse the warped masks |
| Test Doctor + AI vs either alone, with *simulations* | `msx/simulation.py` | Simulated readers: 3 skill levels × 3 trust styles (sceptical, calibrated, over-trusting) on the AI's real outputs. Reports accuracy for Doctor alone, AI alone and Doctor + AI, **automation bias** and **rescue rate**, with the AI as measured and degraded to 15 % error |
| *Reduce* bias, not only monitor it | `msx/bias.py` `mitigate` | Per-subgroup operating thresholds (sex, age band). A threshold is only ever lowered, and only while specificity holds, to close a sensitivity gap. Before / after table on the Evaluation page; the analyser applies the thresholds |

**What the simulation shows** (phantom set, AI degraded to 15 % error): a
*calibrated* reader, who follows the AI only when its confidence is ≥ 0.75,
beats both the AI and their own solo read:

| Reader | Doctor alone | AI alone | Doctor + AI |
|---|---|---|---|
| Trainee | 0.86 | 0.85 | **0.93** |
| Registrar | 0.90 | 0.84 | **0.94** |

An *over-trusting* reader loses that gain, with automation bias of about 0.85.
That is the case for showing confidence and for the blinded first read.

## Results on real chest X-rays

```
python tools/fetch_real_samples.py
python -m msx.evaluation ~/.medscan/real_samples --engine hybrid
```

The set is 50 PA films from the public COVID-19 Image Data Collection: 40
pneumonia, labelled as consolidation, and 10 no-finding. The images are not redistributed. The full report is in
`docs/real_eval_hybrid.json`.

| Metric | Value |
|---|---|
| Quality gate: films accepted | 50 / 50 |
| Consolidation (pneumonia) sensitivity | **0.80** |
| Consolidation specificity | 0.50 (only 10 normal films) |
| Consolidation AUC | **0.82** |
| Adaptive vs fixed | 25 % fewer modules, 20 % fewer model calls; no change in sensitivity or specificity |

This is a small, imbalanced convenience set, so it is a sanity check, not
validation. Specificity is the weak point: published journal figures are often
cropped and post-processed. The next step is CheXpert and NIH ChestX-ray14 with
the same `labels.csv` format.

## Results on the synthetic phantom set

```
python -m msx.evaluation samples
```

25 images, built-in engine, Routine OPD:

| Metric | Fixed | Adaptive |
|---|---|---|
| Any-abnormal sensitivity / specificity | 1.00 / 1.00 | 1.00 / 1.00 |
| Per-label AUC (6 labels) | 0.965–1.00 | 0.965–1.00 |
| Quality gate accuracy | 1.00 | 1.00 |
| Mean time / scan | 3.2 s | **1.6 s (−50 %)** |
| Modules activated / scan | 3.5 | **0.88 (−75 %)** |
| Model calls / scan | 17.6 | **4.4 (−75 %)** |
| Early-exit rate | 0 % | 24 % |

**Honest caveat:** these are *synthetic phantoms*, drawn so the measurements have
something real to measure and the labels are known. They show that the routing
saves compute without losing what the full pipeline finds. They are **not**
clinical performance. For that, run the same command on a CheXpert, MIMIC-CXR or
NIH ChestX-ray14 validation folder with a `labels.csv`, using `--engine hybrid`.

## Known limitations

* The built-in measurements are heuristic and tuned on phantoms. On real films
  they are a transparent second opinion to the DenseNet, not a substitute; the
  nodule search in particular is prone to vessel and rib false positives.
* Lung segmentation is classical; heavily opaque lungs (white-out) fail the gate
  rather than being analysed.
* The dataset model is only as good as its dataset. Many Kaggle sets are
  augmented copies of a few thousand films, with the same patient in train and
  test; trust the **test split** benchmark, not the cross-validation figure.
  Its out-of-distribution guard catches very different images (noise, other
  modalities) but rates the synthetic phantoms as plausible chest films.
* The DenseNet has no laterality. Side comes from the measurements or the CAM.
* Not a medical device; not validated for clinical use.
