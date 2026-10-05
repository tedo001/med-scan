"""MEDSCAN - human-in-the-loop chest X-ray analysis with adaptive analysis routing.

TECHgium® 10th Edition - MedTech challenge.

    python medscan.py                    # sign in, open the workspace the account belongs to
    python medscan.py --present          # everything 1.5x larger, for projectors and slides
    python medscan.py --screenshots docs # render every screen to PNG (no display needed)
    MEDSCAN_SMOKE=1 python medscan.py    # build every page with demo data, then exit

One sign-in; the account decides the workspace.

* **Clinical workspace** - Home, Analyse, Review, Dashboard, Evaluation.
* **Administration** - Dashboard, Bias Monitor, Audit Trail, Engines, Accounts, Settings.

The analyser engine is :mod:`msx`; this file only starts the desktop app.
Research prototype - not a medical device, not for clinical use.
"""

from __future__ import annotations

import os
import sys

WINDOW_TITLE = "MEDSCAN"


def _require_pyqt6() -> None:
    try:
        import PyQt6.QtWidgets  # noqa: F401
    except ImportError:
        sys.stderr.write("MEDSCAN needs PyQt6:  pip install -r requirements-app.txt\n")
        raise SystemExit(2)


def create_application(argv):
    from PyQt6.QtWidgets import QApplication

    from ui import theme

    application = QApplication.instance() or QApplication(argv)
    application.setApplicationName(WINDOW_TITLE)
    application.setApplicationDisplayName(WINDOW_TITLE)
    theme.apply(application)
    return application


def run_signed_in(application) -> int:
    """Sign in, show the workspace; signing out returns to the sign-in screen."""
    from ui.login import LoginDialog
    from ui.services import Services
    from ui.window import MainWindow

    services = Services()
    services.log("application started", category="system")
    while True:
        dialog = LoginDialog(services.accounts, services.settings["show_demo_credentials"],
                             services.audit)
        if dialog.exec() != LoginDialog.DialogCode.Accepted:
            services.log("application closed", category="system")
            return 0
        services.user = dialog.account
        window = MainWindow(services)
        signed_out = {"value": False}
        window.signed_out.connect(lambda: signed_out.update(value=True))
        window.showMaximized() if os.environ.get("MEDSCAN_MAXIMISE") else window.show()
        application.exec()
        if not signed_out["value"]:
            services.log("application closed", category="system")
            return 0


def seed_demo(services, decide: bool = True, passes: int = 1, spread_days: int = 0) -> None:
    """Analyse the demo phantoms into ``services`` and, optionally, review some.

    ``passes`` analyses the set under several clinical contexts; ``spread_days``
    re-dates the studies across that many past days so the dashboard trend has a
    timeline. Demo data only - the dates and the simulated reviews are labelled
    as such in the README.
    """
    import csv
    import glob
    import random
    from datetime import datetime, timedelta

    from msx import imaging, paths

    labels = {}
    with open(os.path.join(paths.SAMPLES_DIR, "labels.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["file"]] = row
    contexts = ("Routine OPD", "Emergency", "Screening camp")
    engine = services.engine
    rng = random.Random(11)
    files = sorted(glob.glob(os.path.join(paths.SAMPLES_DIR, "cxr_*")))
    for p_ in range(passes):
        for index, path in enumerate(files):
            row = labels.get(os.path.basename(path), {})
            facts = {"sex": row.get("sex") or None}
            if row.get("age"):
                facts["age_band"] = imaging.age_band(float(row["age"]))
            analysis = engine.analyse_file(path, context=contexts[(index + p_) % 3], **facts)
            services.store.save_analysis(analysis, row.get("site", ""), row.get("labels", ""),
                                         services.user.username if services.user else "")
            if spread_days:
                # more recent months busier, a winter peak - a plausible demo timeline
                day = int(abs(rng.gauss(0, spread_days / 2.2))) % spread_days
                if rng.random() < 0.25:
                    day = rng.randint(250, 290) % spread_days
                when = datetime.now() - timedelta(days=day, hours=rng.randint(0, 9),
                                                  minutes=rng.randint(0, 59))
                services.store.set_created(analysis.id, when.isoformat(timespec="seconds"))
            services.log("scan analysed", {"source": analysis.scan["source"],
                                           "status": analysis.status}, analysis.id)
    if not decide:
        return
    for row in services.store.studies()[:max(14, len(files) * passes // 2)]:
        loaded = services.store.load(row["id"])
        if loaded is None:
            continue
        analysis, _ = loaded
        truth = {l for l in row["ground_truth"].split(";") if l and l != "No Finding"}
        first = sorted(t for t in truth if rng.random() > 0.3)          # the doctor misses some
        services.store.record_first_read(row["id"], first, rng.uniform(25000, 70000))
        user = services.user
        for f in analysis.shown:
            action = "accept" if f.label in truth else "reject"
            services.store.record_decision(row["id"], f.key, f.label, action, user.username,
                                           user.role)
        final = services.store.final_labels(row["id"], analysis)
        services.store.sign_off(row["id"], user.username, final, rng.uniform(15000, 45000))
    services.retrain()


def screenshots(folder: str) -> int:
    """Render every screen to ``folder`` with demo data, offscreen."""
    import tempfile

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    home = tempfile.mkdtemp(prefix="medscan-shots-")
    os.environ["MEDSCAN_HOME"] = home
    application = create_application(sys.argv[:1])
    from PyQt6.QtCore import QSize

    from ui.login import LoginDialog
    from ui.services import Services
    from ui.window import MainWindow

    os.makedirs(folder, exist_ok=True)
    services = Services()
    size = QSize(1440, 900)

    def grab(widget, name):
        widget.resize(size)
        widget.show()
        for _ in range(4):
            application.processEvents()
        widget.grab().save(os.path.join(folder, name))
        print("wrote", name)

    login = LoginDialog(services.accounts)
    login._fill()
    grab(login, "01-signin.png")
    services.user = services.accounts.get("doctor")
    engine_choice = services.settings["engine"]
    services.settings["engine"], services._engine = "builtin", None   # fast demo seeding
    seed_demo(services, passes=3, spread_days=365)
    services.settings["engine"], services._engine = engine_choice, None
    window = MainWindow(services)
    window.resize(size)
    for page, name in (("Home", "02-home.png"), ("Analyse", "03-analyse.png"),
                       ("Dashboard", "06-dashboard.png")):
        window.show_page(page)
        grab(window, name)
    window.pages["Dashboard"].scroll.verticalScrollBar().setValue(760)
    window.show_page("Dashboard")
    window.pages["Dashboard"].scroll.verticalScrollBar().setValue(760)
    grab(window, "06b-dashboard-breakdowns.png")
    # analyse page with a finished result
    analyse = window.pages["Analyse"]
    analyse._demo()
    jobs = [j for j in analyse.jobs if "effusion_dicom" in j["path"]]
    if jobs:
        job = jobs[0]
        analysis = services.engine.analyse_file(job["path"], context="Emergency", **job["facts"])
        analyse._started(job)
        analyse._finished(job, analysis)
        window.show_page("Analyse")
        grab(window, "03-analyse.png")
    review = window.pages["Review"]
    window.show_page("Review")
    open_rows = services.store.studies(state="open")
    target = next((r for r in open_rows if "effusion" in r["source"] and r["status"] == "findings"),
                  open_rows[0] if open_rows else None)
    if target:
        services.settings["blinded_first_read"] = False
        review.select(target["id"])
        grab(window, "04-review.png")
        review.depth.set("Detailed")
        review._depth_changed("Detailed")
        review.question.setText("How sure are you?")
        review._ask()
        review.case_scroll.verticalScrollBar().setValue(860)
        grab(window, "05-review-evidence.png")
        review._mode_changed("Guided")
        review.case_scroll.verticalScrollBar().setValue(900)
        grab(window, "05b-review-guided.png")
        bar = review.case_scroll.verticalScrollBar()
        bar.setValue(review.rec_card.y() - 20)
        grab(window, "05c-review-report.png")
        review._mode_changed("Concise")
        services.settings["blinded_first_read"] = True
        blind = next((r for r in services.store.studies(state="open")
                      if r["status"] == "findings" and r["id"] != target["id"]), None)
        if blind:
            review.select(blind["id"])
            grab(window, "04b-review-blinded.png")
    window.show_page("Evaluation")
    evaluation_page = window.pages["Evaluation"]
    from msx import evaluation as ev

    report = ev.benchmark(os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples"),
                          "builtin")
    evaluation_page.report = report
    evaluation_page._show_bench(report)
    evaluation_page.refresh()
    grab(window, "07-evaluation.png")
    evaluation_page.scroll.verticalScrollBar().setValue(evaluation_page.sim.y() - 10)
    grab(window, "07b-simulation.png")
    evaluation_page.scroll.verticalScrollBar().setValue(evaluation_page.bias_card.y() - 10)
    grab(window, "07c-bias-learning.png")
    window.close()

    services.user = services.accounts.get("admin")
    admin = MainWindow(services)
    for page, name in (("Bias Monitor", "08-bias.png"), ("Audit Trail", "09-audit.png"),
                       ("Engines", "10-engines.png"), ("Settings", "11-settings.png")):
        admin.show_page(page)
        grab(admin, name)
    import json as _json

    from ui.pages.benchmark_page import runs_folder

    with open(os.path.join(runs_folder(), "benchmark_20261005_090000.json"), "w") as handle:
        _json.dump(dict(report, run_at="2026-10-05T09:00:00"), handle)
    admin.show_page("Model Training")
    grab(admin, "12-model-training.png")
    admin.show_page("Benchmark")
    grab(admin, "13-benchmark.png")
    bench_page = admin.pages["Benchmark"]
    bench_page.scroll.verticalScrollBar().setValue(bench_page.table.parentWidget().y() - 10)
    grab(admin, "13b-benchmark-images.png")
    return 0


def smoke_test() -> int:
    """MEDSCAN_SMOKE=1: build both workspaces with demo data and exit - for checking a build."""
    import tempfile

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["MEDSCAN_HOME"] = tempfile.mkdtemp(prefix="medscan-smoke-")
    application = create_application(sys.argv[:1])
    from ui.services import Services
    from ui.window import MainWindow

    services = Services()
    services.settings["engine"] = "builtin"
    services.user = services.accounts.get("doctor")
    seed_demo(services)
    pages = []
    for username in ("doctor", "admin"):
        services.user = services.accounts.get(username)
        window = MainWindow(services)
        for name in window.pages:
            window.show_page(name)
            application.processEvents()
            pages.append(name)
        window.close()
    print(f"MEDSCAN smoke OK - {services.store.counts()['total']} studies, {len(pages)} pages: "
          + ", ".join(pages))
    return 0


def main(argv=None) -> int:
    argv = list(argv if argv is not None else sys.argv)
    _require_pyqt6()
    if os.environ.get("MEDSCAN_SMOKE"):
        return smoke_test()
    if "--screenshots" in argv:
        index = argv.index("--screenshots")
        return screenshots(argv[index + 1] if len(argv) > index + 1 else "docs")
    if "--present" in argv:
        os.environ.setdefault("QT_SCALE_FACTOR", "1.5")
        argv.remove("--present")
    application = create_application(argv)
    return run_signed_in(application)


if __name__ == "__main__":
    raise SystemExit(main())
