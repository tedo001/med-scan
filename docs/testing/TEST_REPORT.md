# MEDSCAN: UI and functional test report

**Date:** 05 Oct 2026 · **Branch:** `tedo` · **Python** 3.11 · **PyQt6** offscreen (`QT_QPA_PLATFORM=offscreen`)

```
python -m pytest tests                    # everything (≈ 7 min)
python -m pytest tests/gui                # UI tests
python -m pytest tests/functional         # functional / end-to-end tests
MEDSCAN_UI_SHOTS=out python -m pytest tests/gui -k visual_sweep   # keep the page renders
```

## Result

| Suite | Tests | Passed | Failed |
|---|---|---|---|
| Original unit tests (`tests/test_*.py`) | 40 | 40 | 0 |
| **UI tests** (`tests/gui`) | 70 | 70 | 0 |
| **Functional tests** (`tests/functional`) | 21 | 21 | 0 |
| **Total** | **131** | **131** | **0** |

The 11 newest UI tests cover the **Model Training** and **Benchmark** admin pages:

* training from a labelled folder, versioned runs, rollback, deactivate, feature weights;
* refusal with no data source, the "need ≥ 20 findings" message, a missing labels.csv;
* fitting and applying calibration and fairness thresholds;
* benchmark runs saved and shown, comparison Δ between two runs, validation messages;
* CSV / JSON export and delete; the Evaluation page reading the same runs;
* Engines-page shortcuts to both new pages.

The machine-readable results are in `docs/testing/junit.xml`.

## How the tests avoid affecting the original application

* **Isolated data.** Every test gets its own temporary `MEDSCAN_HOME`: database,
  audit trail, settings, accounts, learner and images. The operator's real
  `~/.medscan` was fingerprinted (SHA-256 of every file) before and after the
  run, and it is **unchanged**.
* **No dialogs or network.** Dialogs are answered by monkeypatching
  `QMessageBox` / `QInputDialog` / `QFileDialog` for the duration of one test
  only. The LLM check points at a closed local port. Benchmarks run on small
  copies of the demo phantoms.
* **Test code only in `tests/`.** No test helper was added to the application.
  The app changes listed below are fixes for defects the tests found. Each has
  its own regression test and is in a separate commit.

## What the UI tests cover (59)

| Area | Tests |
|---|---|
| Sign-in | Portal toggle (button text, credential card); Fill in; successful sign-in audited; wrong password shows an error and stays open; clinician rejected on the Admin portal; show / hide password; **Enter signs in exactly once** |
| Main window | Clinician tabs (5) and admin tabs (6); every tab switches the page; Review / bell badges match the database; engine pill; sign-out signal and audit; `open_study` jumps to Review |
| Analyse | Demo samples fill the queue with facts from labels.csv, no duplicates, Clear; context buttons update thresholds; **background worker** analyses, saves, audits, enables "Open in Review"; fixed-pipeline option; sex / age / view overrides; an unreadable file is reported without stopping the batch; **no stage left "running" after a quality hold** |
| Review | Worklist (urgent first) and search; row selection loads the case; **blinded first read** hides findings, recommendations and report until recorded; viewer layers, wheel zoom, double-click reset; regions and CTR lines drawn; **Accept / Reject (with note) / Correct / Add**; cancelling a reject records nothing; **support modes** (Guided checklist and "why it matters", Concise, Evidence-first); Make default saves on the account; **Second opinion** blinds, then marks agree / differ; recommendations and draft report rendered; the doctor's report edits survive refresh, Regenerate replaces them; depth buttons; Q&A logged; **sign-off** locks the case, saves the edited report and retrains the ML model; undecided findings ask first; second read; quality-hold case |
| Dashboard | Period buttons (30 / 90 / 365 days); site and finding filters; Clear filters; trend ranges Today / Week / Month / Year / All with the right number of buckets; Line / Bar; radar Findings / Lung zones / Modules; CSV export; double-click opens a study |
| Evaluation | Benchmark runs in the background and renders comparison, simulation and bias sections; Calibrate and Apply bias thresholds persist to settings; Retrain shows the learner state; empty reader-study message |
| Admin | Audit Trail lists entries newest first, search, **detects a tampered line**, exports CSV; Bias Monitor renders 4 dimensions and records a review; Engines lists every stage; Accounts create (password rule), reset, remove, and an admin cannot remove themself; Settings save persists and rebuilds the engine; LLM test reports unreachable; Delete all studies keeps the audit chain intact |
| Widgets | Every chart paints empty and with data; X-ray viewer with no scan, with overlays, blinded |
| **Visual sweep** | Every page of both workspaces renders at **1440 × 900** with real content, and **no page forces the window wider than a 1440-px screen** |

## What the functional tests cover (21)

| Area | Tests |
|---|---|
| Clinical workflow | Analyse → store → blinded read → accept → final read → signed report → **audit chain intact (4 entries)** → reader study (Doctor 0.5 → Doctor + AI 1.0 sensitivity) |
| Safety | Quality hold: no findings, no routing, "re-acquire" recommendation, non-diagnostic report. **Black, white, noise and tiny images are held, never guessed** |
| Accuracy | Every phantom class detected on the correct side, and every poor scan held; context changes routing and depth, not the findings; **adaptive equals fixed accuracy with fewer modules and calls**; the evaluation CLI writes a JSON report |
| **Privacy** | After analysing the DICOM with a fake patient header, **no identifier (name, ID, birth date, institution, physician) appears in any file** in the data folder; pseudonyms link the same patient only |
| Robustness | Unsupported or corrupt files are rejected cleanly; RGB JPEG, 16-bit PNG and a 2304-px image all work; results are deterministic; the store survives 6 concurrent writer threads (150 decisions) |
| Persistence | Settings round-trip and fall back to defaults when corrupt; accounts persist and never store passwords; audit detects a deleted middle line and **export survives a damaged trail**; the ML learner reaches ≥ 0.9 cross-validated accuracy on the phantoms |
| Deep engine | Hybrid (default) engine uses DenseNet; **hybrid gives the same calls as built-in on the phantoms** and falls back to the learned segmenter only when classical segmentation fails |

## Defects found and fixed

The first run had 5 failures in the UI and functional suites (the 40 original
tests passed throughout). Two were mistakes in my own tests and are corrected.
The rest, plus one found while investigating, were **real application defects**.
Each is fixed minimally (25 lines added, 12 removed, in 7 files) and guarded by a
regression test:

| # | Severity | Defect | Fix |
|---|---|---|---|
| 1 | **High** | In **hybrid mode (the default engine)** the learned PSPNet segmenter, introduced in the previous change, replaced classical segmentation whenever its masks *looked* plausible. On the synthetic demo phantoms the masks were wrong, so **every phantom (including normals) got false consolidation and effusion, and cardiomegaly was missed**. The earlier test suites used the built-in engine, which hid this | `msx/pipeline.py`: the learned segmenter is now a **fallback**, used only when classical segmentation fails, as on tightly cropped real films. Behaviour on images the classical method handles is back to the original |
| 2 | Medium | The **Review** decision bar used a long, unwrapped status line (629 px), so once a case was open the window could not shrink below **1544 px**. On a 1366 / 1440-px laptop screen the window overflowed on every page | `ui/pages/review.py`: the line wraps |
| 3 | Medium | The **Dashboard** filter row forced a minimum width of 1497 px, with the same effect | `ui/pages/dashboard.py`: "Updated …" moved into the caption, filter combos narrower; minimum now 1312 px |
| 4 | Medium | **Enter in the password box did not sign in**: the default button lost its default status once another button had focus | `ui/login.py`: Enter in either field signs in, guarded so it runs only once |
| 5 | Low | After a **quality-hold** scan, the Analyse page's last pipeline chip ("Record") stayed on **"running…"** forever, because that path reports 7 of 8 stages | `ui/pages/analyse.py`: unreported stages show "skipped · not reached" |
| 6 | Low | **Audit CSV export crashed** (`KeyError`) on a damaged trail, exactly when an auditor needs it | `msx/audit.py`: export reads fields tolerantly |

**Effect of fix 1 on the real-image check** (50 public films, hybrid engine,
re-run after the fix): AUC **0.74 → 0.82**, specificity **0.44 → 0.50**,
sensitivity unchanged at 0.80, quality holds **1 → 0**. Adaptive vs fixed now
shows no loss of sensitivity (previously −0.025). `docs/real_eval_hybrid.json`
and `docs/ENGINE.md` are updated.

## Not covered

* Real user interaction on a physical display (tests run offscreen with
  programmatic clicks and keys); final visual judgement still needs a human.
* The Ollama LLM path (no model is available here; the fact-check guard is
  unit-tested).
* Performance at scale (thousands of studies).
* Windows / macOS packaging (PyInstaller build).
