"""Sign-in: a white form on the left, the navy wordmark panel on the right (as SENTRA)."""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QFrame, QHBoxLayout, QLineEdit,
                             QVBoxLayout, QWidget)

from msx import accounts as accounts_mod, screening

from .components import button, hbox, label, logo_pixmap, vbox


class GradientPanel(QWidget):
    """Navy-to-teal gradient with a faint grid, the wordmark and three lines."""

    def __init__(self, status: str):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addStretch(3)
        word = label("MEDSCAN", "GradientWordmark")
        word.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(word)
        tag = label("Screen · Explain · Decide", "GradientTag")
        tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(tag)
        layout.addSpacing(44)
        for line in ("Safer Reads", "Clearer Evidence", "Doctor in Control"):
            item = label(line, "GradientLine")
            item.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(item)
            layout.addSpacing(10)
        layout.addSpacing(40)
        pill = label(f"●  {status}", "GradientStatus")
        layout.addLayout(hbox(None, pill, None))
        layout.addStretch(4)

    def paintEvent(self, event):
        p = QPainter(self)
        gradient = QLinearGradient(QPointF(0, 0), QPointF(self.width(), self.height()))
        gradient.setColorAt(0, QColor("#0F1B2D"))
        gradient.setColorAt(0.55, QColor("#1E3A5F"))
        gradient.setColorAt(1, QColor("#0F3B3A"))
        p.fillRect(self.rect(), gradient)
        p.setPen(QPen(QColor(255, 255, 255, 12), 1))
        for x in range(0, self.width(), 40):
            p.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 40):
            p.drawLine(0, y, self.width(), y)
        # a faint rib-cage curve, the product's subject
        p.setPen(QPen(QColor(255, 255, 255, 18), 2))
        cx, cy = self.width() / 2, self.height() * 0.5
        for k in range(7):
            w, h = 120 + k * 34, 40 + k * 10
            p.drawArc(QRectF(cx - w, cy - 220 + k * 62, w * 2, h * 2), 20 * 16, 140 * 16)
        p.end()


class LoginDialog(QDialog):
    def __init__(self, accounts: accounts_mod.Accounts, show_credentials: bool = True, audit=None):
        super().__init__()
        self.accounts, self.audit = accounts, audit
        self.account = None
        self.setWindowTitle("MEDSCAN - Sign in")
        self.resize(1280, 800)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        form = QFrame()
        form.setObjectName("LoginForm")
        col = QVBoxLayout(form)
        col.setContentsMargins(40, 36, 40, 28)
        logo = label()
        logo.setPixmap(logo_pixmap(34, dark=False))
        col.addLayout(hbox(logo, vbox(label("TECHgium® 10th Edition", "FormOrg"),
                                      label("MedTech · Chest X-ray Decision Support", "FormDept"),
                                      spacing=0), None, spacing=12))
        col.addStretch(2)

        inner = QVBoxLayout()
        inner.setSpacing(8)
        inner.addWidget(label("MEDSCAN", "FormWordmark"))
        inner.addWidget(label("Screen · Explain · Decide", "FormTag"))
        inner.addSpacing(22)
        self.portal = QButtonGroup(self)
        clinician = button("Clinician Login", "Portal", checkable=True)
        clinician.setProperty("side", "left")
        admin = button("Admin Login", "Portal", checkable=True)
        admin.setProperty("side", "right")
        clinician.setChecked(True)
        self.portal.addButton(clinician, 0)
        self.portal.addButton(admin, 1)
        inner.addLayout(hbox(clinician, admin, spacing=0))
        inner.addSpacing(12)
        inner.addWidget(label("Username", "FieldLabel"))
        self.username = QLineEdit()
        self.username.setPlaceholderText("Username")
        inner.addWidget(self.username)
        inner.addWidget(label("Password", "FieldLabel"))
        self.password = QLineEdit()
        self.password.setPlaceholderText("Password")
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        show = button("Show", "", checkable=True)
        show.toggled.connect(lambda on: self.password.setEchoMode(
            QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        inner.addLayout(hbox(self.password, show, spacing=0))
        self.remember = QCheckBox("Remember me on this workstation")
        self.remember.setChecked(True)
        inner.addWidget(self.remember)
        self.error = label("", "LoginError", wrap=True)
        inner.addWidget(self.error)
        self.submit = button("Sign in to Clinical Workspace", "LoginPrimary", self._sign_in)
        self.submit.setDefault(True)
        # Enter in either field signs in (the default-button route is lost once another
        # button in the dialog has taken focus)
        self.username.returnPressed.connect(self._sign_in)
        self.password.returnPressed.connect(self._sign_in)
        inner.addWidget(self.submit)
        inner.addSpacing(10)

        self.card = QFrame()
        self.card.setObjectName("CredentialCard")
        card_col = QVBoxLayout(self.card)
        card_col.setContentsMargins(14, 10, 14, 10)
        card_col.setSpacing(2)
        self.card_title = label("", "CredentialTitle")
        fill = button("Fill in", "Link", self._fill)
        card_col.addLayout(hbox(self.card_title, None, fill))
        self.card_user = label("", "CredentialLine")
        self.card_pass = label("", "CredentialLine")
        card_col.addWidget(self.card_user)
        card_col.addWidget(self.card_pass)
        inner.addWidget(self.card)
        self.card.setVisible(show_credentials)
        inner.addSpacing(8)
        inner.addWidget(label("After 5 failed attempts the account is locked for 5 minutes. "
                              "Every sign-in is written to the audit trail.", "LoginNote", wrap=True))
        holder = QWidget()
        holder.setMaximumWidth(360)
        holder.setLayout(inner)
        col.addLayout(hbox(None, holder, None))
        col.addStretch(3)
        col.addWidget(label("TECHgium MedTech  |  Research prototype - not a medical device",
                            "FormFoot"))

        status = ("DenseNet-121 ready" if screening.deep_available() else "Built-in analyser ready")
        row.addWidget(form, 44)
        row.addWidget(GradientPanel(f"{status} · system operational"), 56)
        self.portal.idToggled.connect(lambda *_: self._portal_changed())
        self._portal_changed()
        self.username.setFocus()

    @property
    def role(self) -> str:
        return accounts_mod.ADMIN if self.portal.checkedId() == 1 else accounts_mod.CLINICIAN

    def _portal_changed(self) -> None:
        admin = self.role == accounts_mod.ADMIN
        demo = accounts_mod.DEMO[1 if admin else 0]
        self.card_title.setText("ADMIN LOGIN" if admin else "CLINICIAN LOGIN")
        self.card_user.setText(f"Username  {demo[0]}")
        self.card_pass.setText(f"Password  {demo[1]}")
        self.submit.setText("Sign in to Administration" if admin else "Sign in to Clinical Workspace")
        self.error.setText("")

    def _fill(self) -> None:
        demo = accounts_mod.DEMO[1 if self.role == accounts_mod.ADMIN else 0]
        self.username.setText(demo[0])
        self.password.setText(demo[1])

    def _sign_in(self) -> None:
        if self.result() == QDialog.DialogCode.Accepted:   # Enter and the default button both fired
            return
        try:
            self.account = self.accounts.authenticate(self.username.text(), self.password.text(),
                                                      self.role)
        except accounts_mod.AuthError as error:
            self.error.setText(str(error))
            if self.audit:
                self.audit.record("sign-in failed", {"username": self.username.text().strip()},
                                  category="system")
            return
        if self.audit:
            self.audit.record("signed in", {"workspace": self.account.role},
                              self.account.username, self.account.role, "system")
        self.accept()
