"""The main window: title row, tab row, and the workspace's pages."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QMainWindow, QMenu,
                             QMessageBox, QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from msx import __version__, screening

from .components import button, hbox, label, logo_pixmap, vbox
from .services import Services

CLINICIAN_TABS = ("Home", "Analyse", "Review", "Dashboard", "Evaluation")
ADMIN_TABS = ("Dashboard", "Bias Monitor", "Audit Trail", "Engines", "Accounts", "Settings")


class MainWindow(QMainWindow):
    signed_out = pyqtSignal()

    def __init__(self, services: Services):
        super().__init__()
        self.services = services
        self.setWindowTitle("MEDSCAN")
        self.resize(1440, 900)
        admin = services.is_admin
        self.workspace = "admin" if admin else "clinician"

        root = QWidget()
        column = QVBoxLayout(root)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self._header())
        column.addWidget(self._tab_row(ADMIN_TABS if admin else CLINICIAN_TABS))
        self.stack = QStackedWidget()
        column.addWidget(self.stack, 1)
        self.setCentralWidget(root)

        self.pages: Dict[str, QWidget] = {}
        for name in (ADMIN_TABS if admin else CLINICIAN_TABS):
            page = self._make_page(name)
            self.pages[name] = page
            self.stack.addWidget(page)
        services.studies_changed.connect(self._studies_changed)
        services.settings_changed.connect(self._update_engine_pill)
        services.open_study.connect(self._open_study)
        self.show_page(next(iter(self.pages)))
        self._studies_changed()

    # ------------------------------------------------------------ header ----
    def _header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("Header")
        frame.setProperty("workspace", self.workspace)
        frame.setFixedHeight(52)
        row = QHBoxLayout(frame)
        row.setContentsMargins(16, 0, 16, 0)
        row.setSpacing(10)
        logo = label()
        logo.setPixmap(logo_pixmap(28))
        row.addWidget(logo)
        row.addLayout(vbox(label("TECHgium® 10th Edition", "OrgName"),
                           label("MedTech · Chest X-ray", "OrgPlace"), spacing=0))
        row.addStretch(1)
        row.addWidget(label("MEDSCAN", "Wordmark"))
        tag = label("ADMINISTRATION" if self.workspace == "admin" else "CLINICAL WORKSPACE",
                    "WorkspaceTag")
        tag.setProperty("workspace", self.workspace)
        tag.setFixedHeight(20)
        row.addWidget(tag, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)
        row.addWidget(label(f"v{__version__}", "ProjectCode"))

        self.engine_pill = QToolButton()
        self.engine_pill.setObjectName("EnginePill")
        self.engine_pill.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.engine_pill)
        group = QActionGroup(menu)
        self.engine_actions = {}
        for key, text in (("hybrid", "Hybrid - DenseNet-121 + measurements"),
                          ("builtin", "Built-in measurements only")):
            action = QAction(text, menu, checkable=True)
            action.setEnabled(key == "builtin" or screening.deep_available())
            action.triggered.connect(lambda _=False, k=key: self._set_engine(k))
            group.addAction(action)
            menu.addAction(action)
            self.engine_actions[key] = action
        self.engine_pill.setMenu(menu)
        row.addWidget(self.engine_pill)
        self._update_engine_pill()

        self.bell = QToolButton()
        self.bell.setObjectName("HeaderIcon")
        self.bell.setText("🔔")
        self.bell.setToolTip("Urgent studies awaiting review")
        self.bell.clicked.connect(lambda: self.show_page("Review") if "Review" in self.pages else None)
        self.badge = label("0", "Badge")
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setFixedHeight(16)
        self.badge.setMinimumWidth(16)
        bell_row = hbox(self.bell, self.badge, spacing=0)
        bell_row.setAlignment(self.badge, Qt.AlignmentFlag.AlignTop)
        row.addLayout(bell_row)

        user = self.services.user
        avatar = label(user.initials if user else "?", "Avatar")
        avatar.setFixedSize(30, 30)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(avatar)
        row.addLayout(vbox(label(user.display if user else "", "UserName"),
                           label(user.title if user else "", "UserRole"), spacing=0))
        menu_button = QToolButton()
        menu_button.setObjectName("UserMenu")
        menu_button.setText("▾")
        menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        user_menu = QMenu(menu_button)
        user_menu.addAction("About MEDSCAN", self._about)
        user_menu.addSeparator()
        user_menu.addAction("Sign out", self._sign_out)
        menu_button.setMenu(user_menu)
        row.addWidget(menu_button)
        return frame

    def _tab_row(self, tabs) -> QFrame:
        frame = QFrame()
        frame.setObjectName("TabRow")
        frame.setProperty("workspace", self.workspace)
        row = QHBoxLayout(frame)
        row.setContentsMargins(12, 0, 16, 0)
        row.setSpacing(0)
        self.tab_group = QButtonGroup(self)
        self.tabs: Dict[str, object] = {}
        self.tab_badges: Dict[str, object] = {}
        for name in tabs:
            tab = button(name, "Tab", checkable=True)
            tab.clicked.connect(lambda _=False, n=name: self.show_page(n))
            self.tab_group.addButton(tab)
            self.tabs[name] = tab
            row.addWidget(tab)
            if name in ("Review", "Bias Monitor"):
                badge = label("", "TabBadge")
                badge.setVisible(False)
                self.tab_badges[name] = badge
                row.addWidget(badge)
                row.addSpacing(8)
        row.addStretch(1)
        self.tab_note = label("", "TabNote")
        row.addWidget(self.tab_note)
        return frame

    def _make_page(self, name: str) -> QWidget:
        from .pages import (accounts_page, analyse, audit_page, bias_page, dashboard,
                            engines_page, evaluation_page, home, review, settings_page)

        factory = {"Home": home.HomePage, "Analyse": analyse.AnalysePage,
                   "Review": review.ReviewPage, "Dashboard": dashboard.DashboardPage,
                   "Evaluation": evaluation_page.EvaluationPage,
                   "Bias Monitor": bias_page.BiasPage, "Audit Trail": audit_page.AuditPage,
                   "Engines": engines_page.EnginesPage, "Accounts": accounts_page.AccountsPage,
                   "Settings": settings_page.SettingsPage}[name]
        page = factory(self.services)
        if hasattr(page, "navigate"):
            page.navigate.connect(self.show_page)
        return page

    # ------------------------------------------------------------ actions ---
    def show_page(self, name: str) -> None:
        if name not in self.pages:
            return
        self.tabs[name].setChecked(True)
        page = self.pages[name]
        self.stack.setCurrentWidget(page)
        if hasattr(page, "refresh"):
            page.refresh()

    def _open_study(self, study_id: str) -> None:
        if "Review" in self.pages:
            self.show_page("Review")
            self.pages["Review"].select(study_id)

    def _studies_changed(self) -> None:
        counts = self.services.store.counts()
        pending = counts["open"] + counts["second"]
        if "Review" in self.tab_badges:
            badge = self.tab_badges["Review"]
            badge.setText(str(pending))
            badge.setVisible(pending > 0)
        self.badge.setText(str(counts["urgent"]))
        self.badge.setVisible(counts["urgent"] > 0)
        latest = self.services.store.studies(limit=1)
        if latest:
            when = datetime.fromisoformat(latest[0]["created"]).strftime("%d %b %Y, %H:%M")
            self.tab_note.setText(f"Last analysis {when}")
        current = self.stack.currentWidget()
        if current is not None and hasattr(current, "refresh"):
            current.refresh()

    def _update_engine_pill(self) -> None:
        key = self.services.settings["engine"]
        if key == "hybrid" and not screening.deep_available():
            key = "builtin"
        self.engine_actions[key].setChecked(True)
        self.engine_pill.setText(f"● {self.services.engine_status()}")

    def _set_engine(self, key: str) -> None:
        self.services.settings["engine"] = key
        self.services.save_settings()
        self.services.log("engine changed", {"engine": key}, category="system")

    def _about(self) -> None:
        QMessageBox.about(self, "About MEDSCAN",
                          f"<b>MEDSCAN {__version__}</b><br>Human-in-the-loop chest X-ray analysis "
                          "with adaptive analysis routing.<br><br>TECHgium® 10th Edition - MedTech."
                          "<br>Research prototype. Not a medical device; not for clinical use.")

    def _sign_out(self) -> None:
        self.services.log("signed out", category="system")
        self.services.user = None
        self.signed_out.emit()
        self.close()
