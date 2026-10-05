"""The MEDSCAN look - SENTRA's design language, re-pointed at a clinic.

A near-black title row over a dark tab row with a white underline under the
current page; Tailwind greys on a warm page (#F0EFEB); navy #1E3A5F for anything
pressable; Inter for text and JetBrains Mono for anything a machine wrote.
Status pills are tinted blue, green, amber and red; probability bars are red
from 0.5, amber from 0.35, grey below. Clinical teal marks the AI's own voice.
"""

from __future__ import annotations

import os

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

NAVY = "#1E3A5F"
NAVY_DARK = "#16305A"
PAGE = "#F0EFEB"
HEADER = "#18181B"
TABS = "#222226"
AMBER = "#CD881A"
RED = "#DC2626"
GREEN = "#16A34A"
TEAL = "#0F766E"
FONT = '"Inter", "Segoe UI", "DejaVu Sans", Arial, sans-serif'
MONO = '"JetBrains Mono", "Consolas", "DejaVu Sans Mono", monospace'
G = {50: "#F9FAFB", 100: "#F3F4F6", 200: "#E5E7EB", 300: "#D1D5DB", 400: "#9CA3AF",
     500: "#6B7280", 600: "#4B5563", 700: "#374151", 800: "#1F2937", 900: "#111827"}
TONES = {"ok": ("#F0FDF4", "#15803D", "#BBF7D0"), "warn": ("#FFFBEB", "#B45309", "#FDE68A"),
         "fail": ("#FEF2F2", "#B91C1C", "#FECACA"), "info": ("#EFF6FF", "#1D4ED8", "#BFDBFE"),
         "grey": (G[100], G[600], G[100]), "engine": (G[800], "#FFFFFF", G[800]),
         "navy": (NAVY, "#FFFFFF", NAVY), "teal": ("#F0FDFA", TEAL, "#99F6E4")}
#: series colours for charts (validated for contrast on white)
SERIES = ("#1E3A5F", "#0F766E", "#CD881A", "#DC2626", "#7C3AED", "#2563EB", "#64748B", "#DB2777")


def prob_colour(p: float) -> str:
    return RED if p >= 0.5 else "#F59E0B" if p >= 0.35 else G[400]


STYLESHEET = f"""
QWidget {{ font-family: {FONT}; font-size: 13px; color: {G[900]}; }}
QMainWindow, QWidget#Page, QScrollArea#PageScroll, QWidget#PageBody {{ background: {PAGE}; }}
QDialog {{ background: #FFFFFF; }}
QToolTip {{ background: {G[900]}; color: #FFFFFF; border: none; padding: 4px 8px; font-size: 12px; }}

/* title row */
QFrame#Header {{ background: {HEADER}; }}
QFrame#Header[workspace="admin"] {{ border-top: 3px solid {AMBER}; }}
QFrame#Header QLabel {{ color: #FFFFFF; background: transparent; }}
QLabel#OrgName {{ font-size: 12px; font-weight: 500; }}
QLabel#OrgPlace {{ font-size: 10px; color: {G[300]}; }}
QLabel#Wordmark {{ font-size: 15px; font-weight: 700; letter-spacing: 5px; }}
QLabel#WorkspaceTag {{ font-size: 9px; font-weight: 500; letter-spacing: 1.5px;
    color: rgba(255,255,255,0.6); border: 1px solid rgba(255,255,255,0.3); padding: 2px 8px; }}
QLabel#WorkspaceTag[workspace="admin"] {{ color: #F2B84B; border-color: {AMBER}; }}
QFrame#Header QLabel#ProjectCode {{ font-family: {MONO}; font-size: 12px; color: {G[400]}; }}
QFrame#Header QLabel#Avatar {{ background: #475569; border-radius: 15px; font-size: 11px; font-weight: 600; }}
QFrame#Header QLabel#UserName {{ font-size: 12px; font-weight: 500; }}
QFrame#Header QLabel#UserRole {{ font-size: 10px; color: {G[400]}; }}
QFrame#Header QLabel#Badge {{ background: #EF4444; border-radius: 8px; font-size: 9px; font-weight: 700; padding: 0 4px; }}
QToolButton#EnginePill {{ font-family: {MONO}; font-size: 12px; color: #E4E4E7;
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.14);
    border-radius: 14px; padding: 3px 12px; }}
QToolButton#EnginePill:hover {{ background: rgba(255,255,255,0.12); }}
QToolButton#HeaderIcon, QToolButton#UserMenu {{ background: transparent; border: none; color: #FFFFFF; padding: 2px; }}
QToolButton#UserMenu::menu-indicator {{ image: none; }}

/* tab row */
QFrame#TabRow {{ background: {TABS}; border-bottom: 1px solid rgba(255,255,255,0.10); }}
QPushButton#Tab {{ color: {G[400]}; font-size: 14px; background: transparent; border: none;
    border-bottom: 2px solid transparent; border-radius: 0; padding: 10px 16px 9px 16px; }}
QPushButton#Tab:hover {{ color: {G[200]}; background: transparent; }}
QPushButton#Tab:checked {{ color: #FFFFFF; font-weight: 500; border-bottom: 2px solid #FFFFFF; }}
QFrame#TabRow[workspace="admin"] QPushButton#Tab:checked {{ border-bottom-color: {AMBER}; }}
QLabel#TabBadge {{ background: {NAVY}; color: #FFFFFF; border-radius: 2px; font-family: {MONO};
    font-size: 10px; font-weight: 500; padding: 1px 5px; }}
QLabel#TabNote {{ font-family: {MONO}; font-size: 10px; color: {G[500]}; background: transparent; }}

/* pages */
QLabel#PageTitle {{ font-size: 20px; font-weight: 600; }}
QLabel#PageCaption {{ font-family: {MONO}; font-size: 11px; color: {G[400]}; }}
QLabel#Note {{ font-size: 12px; color: {G[500]}; }}
QLabel#Mono {{ font-family: {MONO}; font-size: 12px; color: {G[700]}; }}
QLabel#MonoSmall {{ font-family: {MONO}; font-size: 11px; color: {G[500]}; }}
QFrame#Card {{ background: #FFFFFF; border: 1px solid {G[200]}; border-radius: 4px; }}
QFrame#Card QLabel {{ background: transparent; }}
QFrame#CardHead {{ border: none; border-bottom: 1px solid {G[100]}; background: transparent; }}
QLabel#CardTitle {{ font-size: 14px; font-weight: 600; }}
QLabel#CardCaption {{ font-size: 11px; color: {G[400]}; }}
QFrame#StatCell[first="false"] {{ border-left: 1px solid {G[200]}; }}
QLabel#StatLabel {{ font-size: 11px; color: {G[500]}; }}
QLabel#StatValue {{ font-size: 30px; font-weight: 700; }}
QLabel#StatNote {{ font-size: 11px; color: {G[400]}; }}
QLabel#Pill {{ border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 500; }}
QLabel#SectionLabel {{ font-family: {MONO}; font-size: 11px; letter-spacing: 1px; color: {G[600]}; font-weight: 600; }}
QLabel#Body {{ font-size: 13px; color: {G[800]}; }}
QLabel#Small {{ font-size: 12px; color: {G[600]}; }}
QLabel#Faint {{ font-size: 11px; color: {G[400]}; }}
QLabel#Headline {{ font-size: 15px; font-weight: 600; color: {G[900]}; }}
QLabel#CaseRef {{ font-family: {MONO}; font-size: 18px; font-weight: 700; }}
QLabel#Evidence {{ font-size: 12px; color: {G[700]}; background: {G[50]}; border: 1px solid {G[200]};
    border-left: 3px solid {TEAL}; border-radius: 3px; padding: 6px 8px; }}

QPushButton {{ background: #FFFFFF; border: 1px solid {G[300]}; border-radius: 4px; padding: 5px 12px; font-size: 12px; }}
QPushButton:hover {{ background: {G[50]}; }}
QPushButton:disabled {{ color: {G[400]}; background: {G[50]}; border-color: {G[200]}; }}
QPushButton#Primary {{ background: {NAVY}; border-color: {NAVY}; color: #FFFFFF; font-weight: 500; }}
QPushButton#Primary:hover {{ background: {NAVY_DARK}; }}
QPushButton#Primary:disabled {{ background: {G[300]}; border-color: {G[300]}; }}
QPushButton#Danger {{ background: {RED}; border-color: {RED}; color: #FFFFFF; font-weight: 500; }}
QPushButton#Danger:hover {{ background: #B91C1C; }}
QPushButton#Accept {{ color: #15803D; border-color: #BBF7D0; background: #F0FDF4; font-weight: 500; }}
QPushButton#Accept:checked {{ background: #16A34A; color: #FFFFFF; border-color: #16A34A; }}
QPushButton#Reject {{ color: #B91C1C; border-color: #FECACA; background: #FEF2F2; font-weight: 500; }}
QPushButton#Reject:checked {{ background: {RED}; color: #FFFFFF; border-color: {RED}; }}
QPushButton#Correct {{ color: #B45309; border-color: #FDE68A; background: #FFFBEB; font-weight: 500; }}
QPushButton#Correct:checked {{ background: #D97706; color: #FFFFFF; border-color: #D97706; }}
QPushButton#Question {{ color: #1D4ED8; border-color: #BFDBFE; background: #EFF6FF; font-weight: 500; }}
QPushButton#Seg {{ padding: 6px 12px; }}
QPushButton#Seg:checked {{ background: {NAVY}; border-color: {NAVY}; color: #FFFFFF; }}
QPushButton#Link {{ border: none; background: transparent; color: {NAVY}; padding: 0; font-size: 12px; }}
QPushButton#Link:hover {{ text-decoration: underline; }}
QPushButton#Toggle {{ padding: 4px 10px; font-size: 11px; }}
QPushButton#Toggle:checked {{ background: {G[800]}; border-color: {G[800]}; color: #FFFFFF; }}
QPushButton#FlagButton {{ text-align: left; padding: 8px 10px; font-size: 12px; }}
QPushButton#FlagButton:checked {{ background: #EFF6FF; border-color: {NAVY}; color: {NAVY}; font-weight: 600; }}

QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: #FFFFFF; border: 1px solid {G[300]}; border-radius: 4px; padding: 5px 8px; font-size: 12px; }}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{ border-color: {NAVY}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{ background: #FFFFFF; border: 1px solid {G[200]};
    selection-background-color: #EFF6FF; selection-color: {NAVY}; }}
QCheckBox {{ spacing: 8px; font-size: 12px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {G[300]}; border-radius: 3px; background: #FFFFFF; }}
QCheckBox::indicator:checked {{ background: {NAVY}; border-color: {NAVY}; image: url({ASSETS.replace(os.sep, '/')}/check.svg); }}
QSlider::groove:horizontal {{ height: 4px; background: {G[200]}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {NAVY}; width: 14px; margin: -5px 0; border-radius: 7px; }}
QProgressBar {{ border: 1px solid {G[200]}; border-radius: 3px; background: {G[100]}; height: 8px;
    text-align: center; font-size: 10px; }}
QProgressBar::chunk {{ background: {NAVY}; border-radius: 3px; }}

QTableWidget {{ background: #FFFFFF; border: none; font-size: 12px; gridline-color: transparent;
    selection-background-color: #EFF6FF; selection-color: {G[900]}; alternate-background-color: {G[50]}; }}
QTableWidget::item {{ border-bottom: 1px solid {G[100]}; padding: 4px 8px; }}
QTableWidget::item:selected {{ background: #EFF6FF; color: {G[900]}; }}
QHeaderView::section {{ background: #FFFFFF; color: {G[500]}; font-size: 12px; font-weight: 500;
    border: none; border-bottom: 1px solid {G[200]}; padding: 8px 8px; }}
QListWidget {{ background: #FFFFFF; border: none; font-size: 12px; }}
QListWidget::item {{ padding: 8px 10px; border-bottom: 1px solid {G[100]}; }}
QListWidget::item:selected {{ background: #EFF6FF; color: {NAVY}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ width: 8px; background: transparent; }}
QScrollBar::handle:vertical {{ background: {G[300]}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ height: 8px; background: transparent; }}
QScrollBar::handle:horizontal {{ background: {G[300]}; border-radius: 4px; }}
QMenu {{ background: #FFFFFF; border: 1px solid {G[200]}; border-radius: 6px; padding: 4px; }}
QMenu::item {{ padding: 8px 28px 8px 14px; border-radius: 4px; font-size: 13px; }}
QMenu::item:selected {{ background: #EFF6FF; color: {NAVY}; }}
QMenu::separator {{ height: 1px; background: {G[200]}; margin: 4px 6px; }}

/* panels inside the review case */
QFrame#FindingCard {{ background: #FFFFFF; border: 1px solid {G[200]}; border-left: 4px solid {G[300]}; border-radius: 4px; }}
QFrame#FindingCard[status="positive"] {{ border-left-color: {RED}; }}
QFrame#FindingCard[status="possible"] {{ border-left-color: #F59E0B; }}
QFrame#FindingCard[status="uncertain"] {{ border-left-color: #7C3AED; background: #FAF5FF; }}
QFrame#FindingCard[selected="true"] {{ border-color: {NAVY}; border-left-width: 4px; }}
QFrame#FindingCard QLabel {{ background: transparent; }}
QLabel#FindingTitle {{ font-size: 14px; font-weight: 600; }}
QFrame#HoldPanel {{ background: #FEF2F2; border: 1px solid #FECACA; border-left: 4px solid {RED}; border-radius: 6px; }}
QFrame#HoldPanel QLabel, QFrame#OkPanel QLabel, QFrame#InfoPanel QLabel {{ background: transparent; }}
QFrame#OkPanel {{ background: #F0FDF4; border: 1px solid #BBF7D0; border-left: 4px solid {GREEN}; border-radius: 6px; }}
QFrame#InfoPanel {{ background: #EFF6FF; border: 1px solid #BFDBFE; border-left: 4px solid {NAVY}; border-radius: 6px; }}
QFrame#DecisionBar {{ background: #FFFFFF; border-top: 1px solid {G[200]}; }}
QFrame#DecisionBar QLabel {{ background: transparent; }}
QFrame#DropZone {{ border: 2px dashed {G[300]}; border-radius: 6px; background: #FFFFFF; }}
QFrame#DropZone[hover="true"] {{ border-color: {NAVY}; background: #EFF6FF; }}
QFrame#DropZone QLabel {{ background: transparent; }}
QFrame#StageChip {{ background: #FFFFFF; border: 1px solid {G[200]}; border-radius: 4px; }}
QFrame#StageChip[state="ok"] {{ border-color: #BBF7D0; background: #F0FDF4; }}
QFrame#StageChip[state="skipped"] {{ border-color: {G[200]}; background: {G[50]}; }}
QFrame#StageChip[state="stopped"] {{ border-color: #FECACA; background: #FEF2F2; }}
QFrame#StageChip[state="running"] {{ border-color: {NAVY}; background: #EFF6FF; }}
QFrame#StageChip QLabel {{ background: transparent; }}

/* sign-in */
QFrame#LoginForm {{ background: #FFFFFF; }}
QFrame#LoginForm QLabel {{ background: transparent; }}
QLabel#FormOrg {{ font-size: 14px; font-weight: 700; }}
QLabel#FormDept {{ font-size: 12px; color: {G[500]}; }}
QLabel#FormWordmark {{ font-size: 30px; font-weight: 700; letter-spacing: 4px; }}
QLabel#FormTag {{ font-size: 12px; letter-spacing: 1.5px; color: {G[400]}; }}
QLabel#FieldLabel {{ font-size: 12px; color: {G[500]}; }}
QFrame#LoginForm QLineEdit {{ padding: 8px 10px; font-size: 13px; min-height: 22px; }}
QLabel#LoginNote {{ font-size: 11px; color: {G[400]}; }}
QLabel#LoginError {{ font-size: 12px; color: #B91C1C; }}
QLabel#FormFoot {{ font-family: {MONO}; font-size: 10px; color: {G[400]}; }}
QPushButton#LoginPrimary {{ background: {NAVY}; border-color: {NAVY}; color: #FFFFFF; font-size: 13px; font-weight: 600; padding: 9px 16px; }}
QPushButton#LoginPrimary:hover {{ background: {NAVY_DARK}; }}
QPushButton#Portal {{ background: #FFFFFF; color: {G[700]}; border: 1px solid {G[300]}; padding: 9px 0; font-size: 13px; font-weight: 600; border-radius: 0; }}
QPushButton#Portal[side="left"] {{ border-top-left-radius: 4px; border-bottom-left-radius: 4px; border-right: none; }}
QPushButton#Portal[side="right"] {{ border-top-right-radius: 4px; border-bottom-right-radius: 4px; }}
QPushButton#Portal:checked {{ background: {NAVY}; color: #FFFFFF; border-color: {NAVY}; }}
QFrame#CredentialCard {{ background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 6px; }}
QFrame#CredentialCard QLabel {{ background: transparent; }}
QLabel#CredentialTitle {{ font-family: {MONO}; font-size: 11px; letter-spacing: 1px; color: {NAVY}; font-weight: 600; }}
QLabel#CredentialLine {{ font-family: {MONO}; font-size: 12px; color: {G[800]}; }}
QLabel#GradientWordmark {{ font-size: 60px; font-weight: 700; letter-spacing: 9px; color: #FFFFFF; background: transparent; }}
QLabel#GradientTag {{ font-size: 12px; letter-spacing: 1.5px; color: {G[400]}; background: transparent; }}
QLabel#GradientLine {{ font-size: 18px; font-weight: 300; color: rgba(255,255,255,0.8); background: transparent; }}
QLabel#GradientStatus {{ font-family: {MONO}; font-size: 12px; color: rgba(255,255,255,0.75);
    background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.20); border-radius: 16px; padding: 7px 20px; }}
"""

FONT_FILES = ("Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf", "Inter-Bold.ttf",
              "JetBrainsMono-Regular.ttf", "JetBrainsMono-Medium.ttf",
              "JetBrainsMono-SemiBold.ttf", "JetBrainsMono-Bold.ttf")
_loaded = False


def load_fonts() -> None:
    """Register the bundled Inter and JetBrains Mono (SIL OFL), once."""
    global _loaded
    if _loaded:
        return
    from PyQt6.QtGui import QFontDatabase

    for name in FONT_FILES:
        path = os.path.join(ASSETS, "fonts", name)
        if os.path.isfile(path):
            QFontDatabase.addApplicationFont(path)
    _loaded = True


def apply(app) -> None:
    from PyQt6.QtGui import QFont

    load_fonts()
    app.setStyle("Fusion")
    font = QFont("Inter", 10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)
    app.setStyleSheet(STYLESHEET)
