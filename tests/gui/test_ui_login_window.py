"""UI: sign-in dialog and main-window shell (header, tabs, workspaces, sign-out)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from msx import accounts as accounts_mod

from gui_helpers import pump


def make_login(services):
    from ui.login import LoginDialog

    dialog = LoginDialog(services.accounts, True, services.audit)
    dialog.show()
    return dialog


def test_portal_toggle_updates_button_and_credentials(qapp, services):
    dialog = make_login(services)
    assert dialog.role == accounts_mod.CLINICIAN
    assert "Clinical Workspace" in dialog.submit.text()
    assert dialog.card_title.text() == "CLINICIAN LOGIN"
    QTest.mouseClick(dialog.portal.button(1), Qt.MouseButton.LeftButton)
    pump(qapp)
    assert dialog.role == accounts_mod.ADMIN
    assert "Administration" in dialog.submit.text()
    assert dialog.card_title.text() == "ADMIN LOGIN"
    assert "admin" in dialog.card_user.text()


def test_fill_in_and_sign_in_succeeds_and_is_audited(qapp, services):
    dialog = make_login(services)
    dialog._fill()
    assert dialog.username.text() == "doctor" and dialog.password.text() == "medscan"
    QTest.mouseClick(dialog.submit, Qt.MouseButton.LeftButton)
    pump(qapp)
    assert dialog.result() == dialog.DialogCode.Accepted
    assert dialog.account.username == "doctor"
    assert services.audit.entries(5)[0]["action"] == "signed in"


def test_wrong_password_shows_error_and_stays_open(qapp, services):
    dialog = make_login(services)
    QTest.keyClicks(dialog.username, "doctor")
    QTest.keyClicks(dialog.password, "nope")
    QTest.keyClick(dialog.password, Qt.Key.Key_Return)
    pump(qapp)
    assert dialog.account is None
    assert "wrong password" in dialog.error.text()
    assert dialog.isVisible()
    assert services.audit.entries(5)[0]["action"] == "sign-in failed"


def test_clinician_account_rejected_on_admin_portal(qapp, services):
    dialog = make_login(services)
    QTest.mouseClick(dialog.portal.button(1), Qt.MouseButton.LeftButton)
    dialog.username.setText("doctor")
    dialog.password.setText("medscan")
    dialog._sign_in()
    assert dialog.account is None and "Clinician Login" in dialog.error.text()


def test_show_password_toggle(qapp, services):
    from PyQt6.QtWidgets import QLineEdit, QPushButton

    dialog = make_login(services)
    show = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Show")
    show.click()
    assert dialog.password.echoMode() == QLineEdit.EchoMode.Normal
    show.click()
    assert dialog.password.echoMode() == QLineEdit.EchoMode.Password


def test_clinician_workspace_tabs_and_navigation(qapp, services, seeded):
    from ui.window import CLINICIAN_TABS, MainWindow

    window = MainWindow(services)
    window.show()
    pump(qapp)
    assert tuple(window.tabs) == CLINICIAN_TABS
    for name in CLINICIAN_TABS:
        QTest.mouseClick(window.tabs[name], Qt.MouseButton.LeftButton)
        pump(qapp)
        assert window.stack.currentWidget() is window.pages[name]
        assert window.tabs[name].isChecked()
    counts = services.store.counts()
    assert window.tab_badges["Review"].text() == str(counts["open"] + counts["second"])
    assert window.badge.text() == str(counts["urgent"])
    assert "built-in" in window.engine_pill.text()
    window.close()


def test_admin_workspace_has_admin_tabs_only(qapp, services):
    from ui.window import ADMIN_TABS, MainWindow

    services.user = services.accounts.get("admin")
    window = MainWindow(services)
    assert tuple(window.tabs) == ADMIN_TABS
    assert "Review" not in window.pages and "Analyse" not in window.pages
    window.close()


def test_sign_out_emits_and_logs(qapp, services):
    from ui.window import MainWindow

    window = MainWindow(services)
    window.show()
    fired = []
    window.signed_out.connect(lambda: fired.append(True))
    window._sign_out()
    pump(qapp)
    assert fired and services.user is None
    assert any(e["action"] == "signed out" for e in services.audit.entries(5))


def test_open_study_signal_switches_to_review(qapp, services, seeded):
    from ui.window import MainWindow

    window = MainWindow(services)
    window.show()
    study = seeded["cxr_10_effusion.png"]
    services.open_study.emit(study)
    pump(qapp)
    assert window.stack.currentWidget() is window.pages["Review"]
    assert window.pages["Review"].study_id == study
    window.close()


def test_enter_signs_in_exactly_once(qapp, services):
    dialog = make_login(services)
    dialog.username.setText("doctor")
    dialog.password.setText("medscan")
    QTest.keyClick(dialog.password, Qt.Key.Key_Return)
    dialog.submit.click()                               # a second trigger must be ignored
    pump(qapp)
    assert dialog.result() == dialog.DialogCode.Accepted
    assert [e["action"] for e in services.audit.entries(5)].count("signed in") == 1
