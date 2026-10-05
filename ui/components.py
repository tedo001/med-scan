"""Shared widgets: cards, pills, stat strips, charts and the X-ray viewer."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap,
                         QPolygonF)
from PyQt6.QtWidgets import (QButtonGroup, QFrame, QGridLayout, QHBoxLayout, QLabel,
                             QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from . import theme
from .theme import G

# ------------------------------------------------------------------ basics -----


def label(text: str = "", name: str = "", wrap: bool = False, selectable: bool = False) -> QLabel:
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(wrap)
    if selectable:
        widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return widget


def button(text: str, name: str = "", slot: Optional[Callable] = None,
           checkable: bool = False, tip: str = "") -> QPushButton:
    widget = QPushButton(text)
    if name:
        widget.setObjectName(name)
    widget.setCheckable(checkable)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    if tip:
        widget.setToolTip(tip)
    if slot:
        widget.clicked.connect(slot)
    return widget


def hbox(*items, spacing: int = 8, margins=(0, 0, 0, 0)) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)
    for item in items:
        _add(layout, item)
    return layout


def vbox(*items, spacing: int = 8, margins=(0, 0, 0, 0)) -> QVBoxLayout:
    layout = QVBoxLayout()
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)
    for item in items:
        _add(layout, item)
    return layout


def _add(layout, item) -> None:
    if item is None:
        layout.addStretch(1)
    elif isinstance(item, int):
        layout.addSpacing(item)
    elif isinstance(item, (QHBoxLayout, QVBoxLayout, QGridLayout)):
        layout.addLayout(item)
    else:
        layout.addWidget(item)


def clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        elif item.layout() is not None:
            clear(item.layout())


class Pill(QLabel):
    def __init__(self, text: str = "", tone: str = "grey"):
        super().__init__(text)
        self.setObjectName("Pill")
        self.set(text, tone)

    def set(self, text: str, tone: str = "grey") -> None:
        bg, fg, border = theme.TONES.get(tone, theme.TONES["grey"])
        self.setText(text)
        self.setStyleSheet(f"background:{bg}; color:{fg}; border:1px solid {border};")
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)


class Card(QFrame):
    """White card with an optional title row."""

    def __init__(self, title: str = "", caption: str = "", right: Optional[QWidget] = None,
                 padding: int = 14):
        super().__init__()
        self.setObjectName("Card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.title_label = self.caption_label = None
        if title:
            head = QFrame()
            head.setObjectName("CardHead")
            row = QHBoxLayout(head)
            row.setContentsMargins(padding, 10, padding, 10)
            self.title_label = label(title, "CardTitle")
            row.addWidget(self.title_label)
            row.addStretch(1)
            if caption:
                self.caption_label = label(caption, "CardCaption")
                row.addWidget(self.caption_label)
            if right is not None:
                row.addWidget(right)
            outer.addWidget(head)
        self.body = QVBoxLayout()
        self.body.setContentsMargins(padding, padding, padding, padding)
        self.body.setSpacing(8)
        outer.addLayout(self.body)

    def set_caption(self, text: str) -> None:
        if self.caption_label is not None:
            self.caption_label.setText(text)


class StatStrip(QFrame):
    """A row of big numbers: label, value, note."""

    def __init__(self, cells: Sequence[Tuple[str, str, str]]):
        super().__init__()
        self.setObjectName("Card")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.values: List[QLabel] = []
        self.notes: List[QLabel] = []
        for index, (name, value, note) in enumerate(cells):
            cell = QFrame()
            cell.setObjectName("StatCell")
            cell.setProperty("first", "true" if index == 0 else "false")
            col = QVBoxLayout(cell)
            col.setContentsMargins(16, 12, 16, 12)
            col.setSpacing(2)
            col.addWidget(label(name, "StatLabel"))
            v = label(value, "StatValue")
            n = label(note, "StatNote")
            col.addWidget(v)
            col.addWidget(n)
            self.values.append(v)
            self.notes.append(n)
            row.addWidget(cell, 1)

    def set(self, index: int, value: str, note: Optional[str] = None,
            colour: Optional[str] = None) -> None:
        self.values[index].setText(value)
        self.values[index].setStyleSheet(f"color:{colour};" if colour else "")
        if note is not None:
            self.notes[index].setText(note)


class Segmented(QWidget):
    changed = pyqtSignal(str)

    def __init__(self, options: Sequence[str], current: str = ""):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: Dict[str, QPushButton] = {}
        for option in options:
            b = button(option, "Seg", checkable=True)
            self.group.addButton(b)
            self.buttons[option] = b
            row.addWidget(b)
            b.clicked.connect(lambda _=False, o=option: self.changed.emit(o))
        self.set(current or options[0])

    def set(self, option: str) -> None:
        if option in self.buttons:
            self.buttons[option].setChecked(True)

    @property
    def value(self) -> str:
        for option, b in self.buttons.items():
            if b.isChecked():
                return option
        return ""


class Page(QWidget):
    """Title row with actions, then a scrolling body."""

    def __init__(self, title: str, caption: str = "", scroll: bool = True):
        super().__init__()
        self.setObjectName("Page")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 0)
        outer.setSpacing(10)
        head = QHBoxLayout()
        head.setSpacing(8)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        self.title = label(title, "PageTitle")
        self.caption = label(caption, "PageCaption")
        titles.addWidget(self.title)
        titles.addWidget(self.caption)
        head.addLayout(titles)
        head.addStretch(1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(6)
        head.addLayout(self.actions)
        outer.addLayout(head)
        body = QWidget()
        body.setObjectName("PageBody")
        self.body = QVBoxLayout(body)
        self.body.setContentsMargins(0, 0, 0, 16)
        self.body.setSpacing(12)
        if scroll:
            area = QScrollArea()
            area.setObjectName("PageScroll")
            area.setWidgetResizable(True)
            area.setWidget(body)
            area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            outer.addWidget(area, 1)
        else:
            self.body.setContentsMargins(0, 0, 0, 0)
            outer.addWidget(body, 1)

    def add_action(self, widget: QWidget) -> QWidget:
        self.actions.addWidget(widget)
        return widget

    def refresh(self) -> None:  # pages override
        pass


def logo_pixmap(size: int = 30, dark: bool = True) -> QPixmap:
    """A rounded square with a medical cross over a lung curve - drawn, not a bitmap."""
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.GlobalColor.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor("#FFFFFF" if dark else theme.NAVY))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(0, 0, size, size), size * 0.22, size * 0.22)
    p.setBrush(QColor(theme.RED))
    c, w, l = size / 2, size * 0.16, size * 0.56
    p.drawRect(QRectF(c - w / 2, c - l / 2, w, l))
    p.drawRect(QRectF(c - l / 2, c - w / 2, l, w))
    p.end()
    return pixmap


# --------------------------------------------------------------- prob bar ------
class ProbBar(QWidget):
    """A thin probability bar with its number, coloured by threshold."""

    def __init__(self, value: float = 0.0, width: int = 120):
        super().__init__()
        self.value = value
        self.setFixedSize(width, 16)

    def set(self, value: float) -> None:
        self.value = value
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bar_w = self.width() - 40
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(G[200]))
        p.drawRoundedRect(QRectF(0, 6, bar_w, 4), 2, 2)
        p.setBrush(QColor(theme.prob_colour(self.value)))
        p.drawRoundedRect(QRectF(0, 6, bar_w * max(0.0, min(1.0, self.value)), 4), 2, 2)
        p.setPen(QColor(G[700]))
        font = QFont("JetBrains Mono", 8)
        font.setWeight(QFont.Weight.DemiBold)
        p.setFont(font)
        p.drawText(QRectF(bar_w + 4, 0, 36, 16), Qt.AlignmentFlag.AlignVCenter,
                   f"{self.value:.0%}")
        p.end()


# ------------------------------------------------------------------ charts -----
class Chart(QWidget):
    """Base: white background, margins, an empty-state message."""

    empty_text = "No data yet"

    def __init__(self, height: int = 220):
        super().__init__()
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _empty(self, p: QPainter) -> None:
        p.setPen(QColor(G[400]))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_text)

    @staticmethod
    def _font(size: int = 8, mono: bool = False, bold: bool = False) -> QFont:
        font = QFont("JetBrains Mono" if mono else "Inter", size)
        if bold:
            font.setWeight(QFont.Weight.DemiBold)
        return font


class HBarChart(Chart):
    """Horizontal bars: [(label, value)], optional colour per bar."""

    def __init__(self, height: int = 220, colour: str = theme.NAVY, percent: bool = False):
        super().__init__(height)
        self.items: List[Tuple[str, float]] = []
        self.colours: List[str] = []
        self.colour, self.percent = colour, percent

    def set(self, items: Sequence[Tuple[str, float]], colours: Optional[Sequence[str]] = None):
        self.items, self.colours = list(items), list(colours or [])
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.items:
            self._empty(p)
            return
        left, right = 130, 46
        top = 6
        row_h = min(28, (self.height() - top * 2) / len(self.items))
        peak = max(v for _, v in self.items) or 1
        if self.percent:
            peak = max(1.0, peak)
        for i, (name, value) in enumerate(self.items):
            y = top + i * row_h
            p.setPen(QColor(G[600]))
            p.setFont(self._font(9))
            p.drawText(QRectF(0, y, left - 10, row_h), Qt.AlignmentFlag.AlignRight |
                       Qt.AlignmentFlag.AlignVCenter, name)
            width = (self.width() - left - right) * (value / peak)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(G[100]))
            p.drawRect(QRectF(left, y + row_h * 0.25, self.width() - left - right, row_h * 0.5))
            p.setBrush(QColor(self.colours[i] if i < len(self.colours) else self.colour))
            p.drawRect(QRectF(left, y + row_h * 0.25, max(1.0, width), row_h * 0.5))
            p.setPen(QColor(G[800]))
            p.setFont(self._font(8, mono=True, bold=True))
            text = f"{value:.0%}" if self.percent else (f"{value:g}" if value < 1000 else f"{value:.0f}")
            p.drawText(QRectF(left + width + 6, y, right + 40, row_h), Qt.AlignmentFlag.AlignVCenter,
                       text)
        p.end()


class Donut(Chart):
    """A ring of shares with a centre total and a legend."""

    def __init__(self, height: int = 220):
        super().__init__(height)
        self.items: List[Tuple[str, float, str]] = []
        self.centre = ""

    def set(self, items: Sequence[Tuple[str, float, str]], centre: str = "") -> None:
        self.items, self.centre = [i for i in items if i[1] > 0], centre
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        total = sum(v for _, v, _ in self.items)
        if not total:
            self._empty(p)
            return
        size = min(self.height() - 20, self.width() * 0.5)
        rect = QRectF(10, (self.height() - size) / 2, size, size)
        angle = 90 * 16
        for name, value, colour in self.items:
            span = -int(360 * 16 * value / total)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(colour))
            p.drawPie(rect, angle, span)
            angle += span
        hole = rect.adjusted(size * 0.22, size * 0.22, -size * 0.22, -size * 0.22)
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(hole)
        p.setPen(QColor(G[900]))
        p.setFont(self._font(15, bold=True))
        p.drawText(hole, Qt.AlignmentFlag.AlignCenter, self.centre or f"{total:g}")
        x = rect.right() + 18
        y = rect.top() + 6
        for name, value, colour in self.items:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(colour))
            p.drawRoundedRect(QRectF(x, y + 4, 10, 10), 2, 2)
            p.setPen(QColor(G[700]))
            p.setFont(self._font(9))
            p.drawText(QRectF(x + 16, y, self.width() - x - 16, 18), Qt.AlignmentFlag.AlignVCenter,
                       f"{name}  {value:g} ({value / total:.0%})")
            y += 22
        p.end()


class ColumnChart(Chart):
    """Vertical columns, optionally grouped: categories x series."""

    def __init__(self, height: int = 220):
        super().__init__(height)
        self.categories: List[str] = []
        self.series: List[Tuple[str, List[float], str]] = []
        self.fmt = "{:g}"
        self.ymax: Optional[float] = None

    def set(self, categories: Sequence[str], series: Sequence[Tuple[str, Sequence[float], str]],
            fmt: str = "{:g}", ymax: Optional[float] = None) -> None:
        self.categories = list(categories)
        self.series = [(n, list(v), c) for n, v, c in series]
        self.fmt, self.ymax = fmt, ymax
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.categories or not self.series:
            self._empty(p)
            return
        left, bottom, top = 36, 36, 26
        h = self.height() - bottom - top
        w = self.width() - left - 10
        peak = self.ymax or max([max(v) for _, v, _ in self.series if v] + [1e-9])
        p.setFont(self._font(8, mono=True))
        for k in range(5):
            y = top + h * (1 - k / 4)
            p.setPen(QPen(QColor(G[100]), 1))
            p.drawLine(QPointF(left, y), QPointF(left + w, y))
            p.setPen(QColor(G[400]))
            p.drawText(QRectF(0, y - 8, left - 6, 16), Qt.AlignmentFlag.AlignRight |
                       Qt.AlignmentFlag.AlignVCenter, self.fmt.format(peak * k / 4))
        slot = w / len(self.categories)
        bar = slot * 0.7 / len(self.series)
        for ci, category in enumerate(self.categories):
            for si, (name, values, colour) in enumerate(self.series):
                value = values[ci] if ci < len(values) else 0
                bh = h * (value / peak) if peak else 0
                x = left + ci * slot + slot * 0.15 + si * bar
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(colour))
                p.drawRect(QRectF(x, top + h - bh, bar - 2, bh))
                if len(self.series) <= 3 and bar > 22:
                    p.setPen(QColor(G[700]))
                    p.setFont(self._font(7, mono=True))
                    p.drawText(QRectF(x - 6, top + h - bh - 14, bar + 10, 12),
                               Qt.AlignmentFlag.AlignCenter, self.fmt.format(value))
            p.setPen(QColor(G[600]))
            p.setFont(self._font(8))
            p.drawText(QRectF(left + ci * slot, top + h + 4, slot, 30),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                       category)
        x = left
        p.setFont(self._font(8))
        for name, _, colour in self.series:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(colour))
            p.drawRect(QRectF(x, 6, 10, 10))
            p.setPen(QColor(G[600]))
            p.drawText(QPointF(x + 14, 15), name)
            x += 24 + p.fontMetrics().horizontalAdvance(name)
        p.end()


# ------------------------------------------------------------------ viewer -----
def to_qimage(array: np.ndarray) -> QImage:
    eight = np.ascontiguousarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
    image = QImage(eight.data, eight.shape[1], eight.shape[0], eight.shape[1],
                   QImage.Format.Format_Grayscale8)
    return image.copy()


def heat_qimage(heat: np.ndarray) -> QImage:
    """Transparent -> yellow -> red overlay, alpha following the heat."""
    h = np.clip(heat, 0, 1)
    rgba = np.zeros(h.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = (np.clip(1.6 - 1.6 * h, 0, 1) * 220).astype(np.uint8)
    rgba[..., 2] = (np.clip(0.6 - h, 0, 1) * 60).astype(np.uint8)
    rgba[..., 3] = (np.clip((h - 0.15) / 0.85, 0, 1) ** 0.8 * 255).astype(np.uint8)
    rgba = np.ascontiguousarray(rgba)
    image = QImage(rgba.data, h.shape[1], h.shape[0], h.shape[1] * 4, QImage.Format.Format_RGBA8888)
    return image.copy()


class XrayViewer(QWidget):
    """The scan, with toggleable heatmap, regions, lung outline and measurements.

    Mouse wheel zooms around the cursor, drag pans, double-click resets.
    """

    def __init__(self):
        super().__init__()
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.base: Optional[QImage] = None
        self.heat: Optional[QImage] = None
        self.outline: List[QPolygonF] = []
        self.regions: List[Dict[str, object]] = []
        self.lines: List[Tuple[float, float, float, float, str]] = []
        self.show = {"heat": True, "regions": True, "lungs": False, "measure": True}
        self.opacity = 0.55
        self.selected = ""
        self.zoom, self.offset, self._drag = 1.0, QPointF(0, 0), None
        self.message = "No scan selected"
        self.blind = False

    def set_scan(self, work: Optional[np.ndarray], heat: Optional[np.ndarray] = None,
                 lungs: Optional[np.ndarray] = None) -> None:
        self.base = to_qimage(work) if work is not None else None
        self.heat = heat_qimage(heat) if heat is not None else None
        self.outline = self._contours(lungs) if lungs is not None else []
        self.regions, self.lines = [], []
        self.zoom, self.offset = 1.0, QPointF(0, 0)
        self.update()

    @staticmethod
    def _contours(mask: np.ndarray) -> List[QPolygonF]:
        import cv2

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        return [QPolygonF([QPointF(float(x), float(y)) for [[x, y]] in c]) for c in contours
                if len(c) > 10]

    def set_overlays(self, regions: List[Dict[str, object]], lines) -> None:
        self.regions, self.lines = regions, lines
        self.update()

    def set_layer(self, name: str, on: bool) -> None:
        self.show[name] = on
        self.update()

    def set_opacity(self, value: float) -> None:
        self.opacity = value
        self.update()

    def _frame(self) -> Tuple[float, float, float]:
        size = 512.0
        scale = min(self.width(), self.height()) / size * self.zoom
        x = (self.width() - size * scale) / 2 + self.offset.x()
        y = (self.height() - size * scale) / 2 + self.offset.y()
        return x, y, scale

    def wheelEvent(self, event):
        old = self.zoom
        self.zoom = max(1.0, min(6.0, self.zoom * (1.15 if event.angleDelta().y() > 0 else 1 / 1.15)))
        if self.zoom == 1.0:
            self.offset = QPointF(0, 0)
        else:
            pos = event.position()
            centre = QPointF(self.width() / 2, self.height() / 2)
            self.offset = (self.offset - (pos - centre)) * (self.zoom / old) + (pos - centre)
        self.update()

    def mousePressEvent(self, event):
        self._drag = event.position()

    def mouseMoveEvent(self, event):
        if self._drag is not None and self.zoom > 1:
            self.offset += event.position() - self._drag
            self._drag = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        self._drag = None

    def mouseDoubleClickEvent(self, event):
        self.zoom, self.offset = 1.0, QPointF(0, 0)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#0B0B0D"))
        if self.base is None:
            p.setPen(QColor(G[500]))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            return
        x, y, s = self._frame()
        target = QRectF(x, y, 512 * s, 512 * s)
        p.drawImage(target, self.base)
        if self.blind:
            p.setPen(QColor("#E4E4E7"))
            p.fillRect(QRectF(x, y + 512 * s - 30, 512 * s, 30), QColor(0, 0, 0, 160))
            p.drawText(QRectF(x, y + 512 * s - 30, 512 * s, 30), Qt.AlignmentFlag.AlignCenter,
                       "Blinded first read - AI findings hidden until you record your impression")
            p.end()
            return
        if self.show["heat"] and self.heat is not None:
            p.setOpacity(self.opacity)
            p.drawImage(target, self.heat)
            p.setOpacity(1.0)
        p.translate(x, y)
        p.scale(s, s)
        if self.show["lungs"]:
            p.setPen(QPen(QColor(56, 189, 248, 200), 1.2 / s))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for poly in self.outline:
                p.drawPolygon(poly)
        if self.show["measure"]:
            for x0, y0, x1, y1, colour in self.lines:
                p.setPen(QPen(QColor(colour), 1.6 / s, Qt.PenStyle.SolidLine))
                p.drawLine(QPointF(x0, y0), QPointF(x1, y1))
                for xx in (x0, x1):
                    p.drawLine(QPointF(xx, y0 - 5), QPointF(xx, y0 + 5))
        if self.show["regions"]:
            for region in self.regions:
                x0, y0, x1, y1 = region["bbox"]
                selected = region.get("key") == self.selected
                colour = QColor(region.get("colour", theme.RED))
                p.setPen(QPen(colour, (2.4 if selected else 1.3) / s,
                              Qt.PenStyle.SolidLine if selected else Qt.PenStyle.DashLine))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))
                text = region.get("text", "")
                if text:
                    font = QFont("Inter", max(5, int(9 / s)))
                    font.setWeight(QFont.Weight.DemiBold)
                    p.setFont(font)
                    width = p.fontMetrics().horizontalAdvance(text) + 8 / s
                    p.fillRect(QRectF(x0, y0 - 14 / s, width, 14 / s), colour)
                    p.setPen(QColor("#FFFFFF"))
                    p.drawText(QRectF(x0 + 4 / s, y0 - 14 / s, width, 14 / s),
                               Qt.AlignmentFlag.AlignVCenter, text)
        p.resetTransform()
        p.setPen(QColor(G[400]))
        p.setFont(QFont("JetBrains Mono", 8))
        p.drawText(QRectF(8, self.height() - 20, 300, 16), Qt.AlignmentFlag.AlignLeft,
                   f"zoom {self.zoom:.1f}x - wheel to zoom, drag to pan, double-click to reset")
        p.end()
