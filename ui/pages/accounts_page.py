"""Accounts - who can sign in, and as what."""

from __future__ import annotations

from PyQt6.QtWidgets import (QComboBox, QHeaderView, QInputDialog, QLineEdit, QMessageBox,
                             QTableWidget, QTableWidgetItem)

from msx import accounts as accounts_mod

from ..components import Card, Page, button, hbox, label


class AccountsPage(Page):
    def __init__(self, services):
        super().__init__("Accounts", "PBKDF2-SHA256 · lockout after 5 failures")
        self.services = services
        card = Card("Accounts")
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Username", "Name", "Role", "Title", "Support mode (their choice)",
                                              "Last sign-in"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(260)
        card.body.addWidget(self.table)
        card.body.addLayout(hbox(None, button("Reset password…", "", self._reset),
                                 button("Remove", "Danger", self._remove)))
        self.body.addWidget(card)

        new = Card("New account")
        self.username = QLineEdit()
        self.username.setPlaceholderText("username")
        self.display = QLineEdit()
        self.display.setPlaceholderText("Dr. Full Name")
        self.title_edit = QLineEdit("Radiologist")
        self.role = QComboBox()
        self.role.addItems([accounts_mod.CLINICIAN, accounts_mod.ADMIN])
        self.password = QLineEdit()
        self.password.setPlaceholderText("password (6+)")
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        new.body.addLayout(hbox(self.username, self.display, self.title_edit, self.role,
                                self.password, button("Create", "Primary", self._create)))
        self.note = label("", "Small")
        new.body.addWidget(self.note)
        self.body.addWidget(new)
        self.body.addStretch(1)

    def refresh(self):
        accounts = self.services.accounts.all()
        self.table.setRowCount(len(accounts))
        for r, a in enumerate(accounts):
            from msx import support

            for c, v in enumerate((a.username, a.display, a.role, a.title,
                                   support.get(a.support_mode).name, a.last_login or "—")):
                self.table.setItem(r, c, QTableWidgetItem(v))

    def _selected(self):
        row = self.table.currentRow()
        return self.table.item(row, 0).text() if row >= 0 else None

    def _create(self):
        try:
            self.services.accounts.add(self.username.text(), self.password.text(),
                                       self.display.text(), self.role.currentText(),
                                       self.title_edit.text())
        except ValueError as error:
            self.note.setText(str(error))
            return
        self.services.log("account created", {"username": self.username.text().strip().lower(),
                                              "role": self.role.currentText()}, category="system")
        self.note.setText("created")
        self.username.clear()
        self.password.clear()
        self.refresh()

    def _reset(self):
        username = self._selected()
        if not username:
            return
        password, ok = QInputDialog.getText(self, "Reset password", f"New password for {username}",
                                            QLineEdit.EchoMode.Password)
        if ok:
            try:
                self.services.accounts.set_password(username, password)
                self.services.log("password reset", {"username": username}, category="system")
            except ValueError as error:
                QMessageBox.warning(self, "Reset password", str(error))

    def _remove(self):
        username = self._selected()
        if not username or (self.services.user and username == self.services.user.username):
            return
        if QMessageBox.question(self, "Remove", f"Remove {username}?") == QMessageBox.StandardButton.Yes:
            self.services.accounts.remove(username)
            self.services.log("account removed", {"username": username}, category="system")
            self.refresh()
