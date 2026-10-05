"""UI: Review bench - worklist, viewer, blinded read, decisions, modes, report, Q&A, sign-off."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QInputDialog, QLabel, QMessageBox

from msx.datastore import SECOND_READ, SIGNED

from gui_helpers import pump


@pytest.fixture
def review(qapp, services, seeded):
    from ui.pages.review import ReviewPage

    page = ReviewPage(services)
    page.resize(1440, 900)
    page.show()
    page.refresh()
    pump(qapp)
    return page


def texts(widget):
    return " ".join(l.text() for l in widget.findChildren(QLabel))


def open_unblinded(review, services, seeded, name="cxr_12_effusion.png"):
    services.settings["blinded_first_read"] = False
    review.select(seeded[name])
    return review


def test_worklist_lists_all_and_urgent_first(review, services, seeded):
    review.state.setCurrentIndex(3)                          # All
    assert review.table.rowCount() == len(seeded)
    urgent_flags = [review.table.item(r, 1).text().startswith("▲") for r in range(review.table.rowCount())]
    assert urgent_flags == sorted(urgent_flags, reverse=True)


def test_search_filters_worklist(review, seeded):
    review.state.setCurrentIndex(3)
    QTest.keyClicks(review.search, "pneumothorax")
    assert review.table.rowCount() == 1


def test_selecting_a_row_loads_the_case(qapp, review, seeded):
    review.state.setCurrentIndex(3)
    review.table.selectRow(1)
    pump(qapp)
    assert review.study_id and review.ref.text().startswith(review.study_id)
    assert review.viewer.base is not None


def test_blinded_first_read_hides_ai_until_recorded(qapp, review, services, seeded):
    services.settings["blinded_first_read"] = True
    review.select(seeded["cxr_10_effusion.png"])
    assert review.blinded and review.viewer.blind
    assert not review.findings_card.isVisibleTo(review)
    assert not review.report_card.isVisibleTo(review) and not review.rec_card.isVisibleTo(review)
    assert not review.sign.isEnabled()
    review.first_boxes["Effusion"].setChecked(True)
    review._record_first_read()
    pump(qapp)
    assert not review.blinded and not review.viewer.blind
    assert review.findings_card.isVisibleTo(review)
    assert services.store.study(review.study_id)["first_read"] == "Effusion"


def test_viewer_layers_zoom_and_reset(qapp, review, services, seeded):
    open_unblinded(review, services, seeded)
    viewer = review.viewer
    review.toggles["heat"].click()
    assert viewer.show["heat"] is False
    review.toggles["lungs"].click()
    assert viewer.show["lungs"] is True
    centre = QPointF(viewer.width() / 2, viewer.height() / 2)
    event = QWheelEvent(centre, viewer.mapToGlobal(centre), QPoint(0, 0), QPoint(0, 120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    viewer.wheelEvent(event)
    assert viewer.zoom > 1.0
    QTest.mouseDClick(viewer, Qt.MouseButton.LeftButton)
    assert viewer.zoom == 1.0
    assert viewer.grab().width() > 0


def test_regions_drawn_for_findings(review, services, seeded):
    open_unblinded(review, services, seeded)
    assert review.viewer.regions, "finding regions should be overlaid"
    assert any(line for line in review.viewer.lines), "CTR measurement lines expected"


def test_accept_reject_correct_and_add(qapp, review, services, seeded, monkeypatch):
    open_unblinded(review, services, seeded)
    keys = list(review.cards)
    assert len(keys) >= 2
    review.cards[keys[0]].accept.click()
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("looks like fat pad", True)))
    review.cards[keys[1]]._decide("reject")
    pump(qapp)
    latest = services.store.latest_decisions(review.study_id)
    assert latest[keys[0]]["action"] == "accept"
    assert latest[keys[1]]["action"] == "reject" and latest[keys[1]]["note"] == "looks like fat pad"
    review.cards[keys[1]]._decide("correct", "Mass")
    assert services.store.latest_decisions(review.study_id)[keys[1]]["corrected_label"] == "Mass"
    review._add_finding("Nodule")
    assert any(d["action"] == "add" for d in services.store.decisions(review.study_id))
    assert "All findings decided" in review.decision_note.text()


def test_cancelling_reject_records_nothing(review, services, seeded, monkeypatch):
    open_unblinded(review, services, seeded)
    key = next(iter(review.cards))
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("", False)))
    review.cards[key]._decide("reject")
    assert key not in services.store.latest_decisions(review.study_id)


def test_support_modes_change_presentation(qapp, review, services, seeded):
    open_unblinded(review, services, seeded)
    QTest.mouseClick(review.mode_switch.buttons["Guided"], Qt.MouseButton.LeftButton)
    pump(qapp)
    assert review.mode.key == "guided"
    assert "HOW TO READ THIS" in texts(review.findings_card)
    assert "Why it may matter" in texts(review.findings_card)
    QTest.mouseClick(review.mode_switch.buttons["Concise"], Qt.MouseButton.LeftButton)
    pump(qapp)
    assert "HOW TO READ THIS" not in texts(review.findings_card)
    QTest.mouseClick(review.mode_switch.buttons["Evidence-first"], Qt.MouseButton.LeftButton)
    pump(qapp)
    assert "GUIDELINE EVIDENCE" in texts(review.explain_card)


def test_make_default_mode_saves_on_account(review, services, seeded):
    open_unblinded(review, services, seeded)
    review._mode_changed("Guided")
    assert review.mode_default.isVisibleTo(review)
    review._mode_default()
    assert services.accounts.get("doctor").support_mode == "guided"
    assert not review.mode_default.isVisibleTo(review)


def test_second_opinion_mode_blinds_then_shows_disagreements(qapp, review, services, seeded):
    services.settings["blinded_first_read"] = False
    review.select(seeded["cxr_12_effusion.png"])
    review._mode_changed("Second opinion")
    assert review.blinded                      # the mode itself asks for the doctor's read first
    review.first_boxes["Effusion"].setChecked(True)
    review._record_first_read()
    pump(qapp)
    labels = texts(review.findings_card)
    assert "Agrees with your first read" in labels and "Differs from your first read" in labels


def test_recommendations_and_report_rendered(review, services, seeded):
    open_unblinded(review, services, seeded)
    assert "Heart-failure work-up" in texts(review.rec_card)
    report = review.report_edit.toPlainText()
    for heading in ("EXAMINATION:", "FINDINGS:", "IMPRESSION:", "RECOMMENDATIONS:"):
        assert heading in report
    review.report_edit.setPlainText(report + "\nEdited by the doctor.")
    review._report()                           # must not overwrite the doctor's edits
    assert "Edited by the doctor." in review.report_edit.toPlainText()
    review._report(force=True)                 # explicit regenerate does
    assert "Edited by the doctor." not in review.report_edit.toPlainText()


def test_depth_buttons_change_explanation(qapp, review, services, seeded):
    open_unblinded(review, services, seeded)
    QTest.mouseClick(review.depth.buttons["Brief"], Qt.MouseButton.LeftButton)
    pump(qapp)
    brief = texts(review.explain_card)
    QTest.mouseClick(review.depth.buttons["Detailed"], Qt.MouseButton.LeftButton)
    pump(qapp)
    detailed = texts(review.explain_card)
    assert len(detailed) > len(brief) and "Confidence breakdown" in detailed


def test_question_and_answer_logged(qapp, review, services, seeded):
    open_unblinded(review, services, seeded)
    QTest.keyClicks(review.question, "How sure are you?")
    QTest.keyClick(review.question, Qt.Key.Key_Return)
    pump(qapp)
    qa = services.store.questions(review.study_id)
    assert qa and "confidence" in qa[0]["answer"].lower()
    assert review.question.text() == ""


def test_sign_off_locks_case_saves_report_and_retrains(qapp, review, services, seeded, monkeypatch):
    open_unblinded(review, services, seeded)
    for card in list(review.cards.values()):
        card.accept.click()
    review.report_edit.setPlainText("MY SIGNED REPORT")
    review._sign_off()
    pump(qapp)
    row = services.store.study(review.study_id)
    assert row["review_state"] == SIGNED and row["report"] == "MY SIGNED REPORT"
    assert review.locked and not review.sign.isEnabled()
    assert all(not c.accept.isEnabled() for c in review.cards.values())
    assert review.report_edit.isReadOnly()
    assert any(e["action"] == "feedback model retrained" for e in services.audit.entries(10))


def test_sign_off_with_undecided_asks_first(review, services, seeded, monkeypatch):
    open_unblinded(review, services, seeded)
    asked = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda *a, **k: asked.append(a) or QMessageBox.StandardButton.No))
    review._sign_off()
    assert asked and services.store.study(review.study_id)["review_state"] != SIGNED


def test_second_read_request(review, services, seeded):
    open_unblinded(review, services, seeded)
    review._second_read()
    assert services.store.study(review.study_id)["review_state"] == SECOND_READ


def test_quality_hold_case_shows_hold_and_no_cards(review, services, seeded):
    open_unblinded(review, services, seeded, "cxr_22_quality.png")
    assert review.cards == {}
    assert "QUALITY HOLD" in texts(review.headline)
    assert "Re-acquire" in texts(review.rec_card)
