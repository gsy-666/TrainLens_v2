"""Lightweight QPainter-based line chart for training metrics."""

from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import (
    QColor, QFont, QFontMetrics, QMouseEvent, QPainter, QPainterPath, QPen,
    QWheelEvent,
)
from PyQt6.QtWidgets import QSizePolicy, QToolTip, QWidget

_COLORS = [
    QColor(0x1F, 0x77, 0xB4), QColor(0xFF, 0x7F, 0x0E), QColor(0x2C, 0xA0, 0x2C),
    QColor(0xD6, 0x27, 0x28), QColor(0x94, 0x67, 0xBD), QColor(0x8C, 0x56, 0x4B),
    QColor(0xE3, 0x77, 0xC2), QColor(0x7F, 0x7F, 0x7F), QColor(0xBC, 0xBD, 0x22),
    QColor(0x17, 0xBE, 0xCF),
]


class MetricsChartWidget(QWidget):
    """QPainter-based line chart for a group of metric series.

    Supports: legend with click-to-toggle, hover tooltip, auto axis scaling,
    and efficient repaints with data downsampling for display.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self._series: List[Tuple[str, str, List[Tuple[float, float]], QColor, bool]] = []
        self._margin = (60, 20, 20, 40)  # left, top, right, bottom

        self._x_min = 0.0
        self._x_max = 1.0
        self._y_min = 0.0
        self._y_max = 1.0

    def set_series(self, data: List[Tuple[str, str, List[Tuple[float, float]]]]):
        """Set chart data: list of (name, display_name, [(x, y), ...])."""
        self._series = []
        for i, (name, display, points) in enumerate(data):
            color = _COLORS[i % len(_COLORS)]
            self._series.append((name, display, points, color, True))

        if self._series:
            all_x = [x for _, _, pts, _, _ in self._series for x, _ in pts]
            all_y = [y for _, _, pts, _, _ in self._series for _, y in pts]
            self._x_min = min(all_x) if all_x else 0
            self._x_max = max(all_x) if all_x else 1
            self._y_min = min(all_y) if all_y else 0
            self._y_max = max(all_y) if all_y else 1
            if self._y_min == self._y_max:
                self._y_min -= 0.1
                self._y_max += 0.1
        else:
            self._x_min, self._x_max, self._y_min, self._y_max = 0, 1, 0, 1

        self.update()

    def clear(self):
        self._series.clear()
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        ml, mt, mr, mb = self._margin
        plot_rect = QRectF(ml, mt, w - ml - mr, h - mt - mb)

        # Background
        p.fillRect(self.rect(), Qt.GlobalColor.white)
        p.fillRect(plot_rect, QColor(250, 250, 250))
        p.setPen(QPen(QColor(200, 200, 200), 1))
        p.drawRect(plot_rect)

        if not self._series:
            p.setPen(QColor(150, 150, 150))
            p.drawText(plot_rect, Qt.AlignmentFlag.AlignCenter, "No data")
            return

        # Grid + axes
        self._draw_grid(p, plot_rect)
        self._draw_axes_labels(p, plot_rect)

        # Series
        for name, display, pts, color, visible in self._series:
            if not visible or len(pts) < 1:
                continue
            self._draw_series(p, plot_rect, pts, color)

        # Legend
        self._draw_legend(p, plot_rect)

    def _draw_grid(self, p: QPainter, r: QRectF):
        p.setPen(QPen(QColor(220, 220, 220), 1, Qt.PenStyle.DotLine))
        for i in range(5):
            y = r.top() + r.height() * i / 4.0
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        for i in range(6):
            x = r.left() + r.width() * i / 5.0
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))

    def _draw_axes_labels(self, p: QPainter, r: QRectF):
        font = QFont("Segoe UI", 8)
        p.setFont(font)
        p.setPen(QColor(80, 80, 80))
        fm = QFontMetrics(font)

        # Y axis labels
        for i in range(5):
            val = self._y_min + (self._y_max - self._y_min) * (4 - i) / 4.0
            label = f"{val:.3g}"
            y = r.top() + r.height() * i / 4.0
            p.drawText(QRectF(2, y - 10, self._margin[0] - 6, 20),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label)

        # X axis labels
        for i in range(6):
            val = self._x_min + (self._x_max - self._x_min) * i / 5.0
            label = f"{val:.4g}" if isinstance(val, float) else str(int(val))
            x = r.left() + r.width() * i / 5.0
            tw = fm.horizontalAdvance(label)
            p.drawText(QPointF(x - tw / 2, r.bottom() + 15), label)

    def _draw_series(self, p: QPainter, r: QRectF, pts: List[Tuple[float, float]], color: QColor):
        if len(pts) < 1:
            return

        # Downsample for display
        max_pts = max(2, self.width() // 2)
        if len(pts) > max_pts:
            step = max(1, len(pts) // max_pts)
            pts = pts[::step]

        x_range = self._x_max - self._x_min or 1
        y_range = self._y_max - self._y_min or 1

        def tx(x):
            return r.left() + (x - self._x_min) / x_range * r.width()

        def ty(y):
            return r.bottom() - (y - self._y_min) / y_range * r.height()

        path = QPainterPath()
        path.moveTo(tx(pts[0][0]), ty(pts[0][1]))
        for x, y in pts[1:]:
            path.lineTo(tx(x), ty(y))

        pen = QPen(color, 1.5)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

        # Dots
        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)
        for x, y in pts:
            p.drawEllipse(QPointF(tx(x), ty(y)), 2.5, 2.5)

    def _draw_legend(self, p: QPainter, r: QRectF):
        font = QFont("Segoe UI", 8)
        p.setFont(font)
        fm = QFontMetrics(font)
        x = r.right() - 8
        y = r.top() + 4

        for name, display, _, color, visible in reversed(self._series):
            label = display if display else name
            tw = fm.horizontalAdvance(label) + 18
            x -= tw
            if x < r.left():
                break

            lr = QRectF(x, y, tw, 16)
            if visible:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(color)
                p.drawRect(QRectF(x + 2, y + 4, 10, 8))
                p.setPen(QColor(40, 40, 40))
                p.drawText(QRectF(x + 14, y, tw - 14, 16), Qt.AlignmentFlag.AlignVCenter, label)
            else:
                p.setPen(QColor(180, 180, 180))
                p.drawText(QRectF(x + 2, y, tw - 2, 16), Qt.AlignmentFlag.AlignVCenter, label)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        r = QRectF(self._margin[0], self._margin[1],
                    self.width() - self._margin[0] - self._margin[2],
                    self.height() - self._margin[1] - self._margin[3])

        # ── check legend items first (higher priority) ──
        x_cursor = r.right() - 8
        y_cursor = r.top() + 4
        legend_hit = False
        for name, display, pts, color, visible in reversed(self._series):
            label = display if display else name
            tw = QFontMetrics(QFont("Segoe UI", 8)).horizontalAdvance(label) + 18
            x_cursor -= tw
            lr = QRectF(x_cursor, y_cursor, tw, 16)
            if lr.contains(pos):
                nearest = self._find_nearest(r, pts, pos.x())
                if nearest:
                    QToolTip.showText(
                        event.globalPosition().toPoint(),
                        f"{display}\nEpoch {nearest[0]:.4g}: {nearest[1]:.6g}"
                    )
                legend_hit = True
                break
            if x_cursor < r.left():
                break

        if legend_hit:
            return

        # ── plot-area hover: show crosshair-style tooltip for nearest point across all visible series ──
        if r.contains(pos):
            best_series_name = None
            best_point = None
            best_dist = float("inf")
            for name, display, pts, color, visible in self._series:
                if not visible or not pts:
                    continue
                nearest = self._find_nearest(r, pts, pos.x())
                if nearest is None:
                    continue
                nx, ny = nearest
                px = r.left() + (nx - self._x_min) / (self._x_max - self._x_min or 1) * r.width()
                dist = abs(px - pos.x())
                if dist < best_dist and dist < 25:
                    best_dist = dist
                    best_series_name = display
                    best_point = (nx, ny)

            if best_point:
                QToolTip.showText(
                    event.globalPosition().toPoint(),
                    f"{best_series_name}\nEpoch {best_point[0]:.4g}: {best_point[1]:.6g}"
                )
                return

        QToolTip.hideText()

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position()
        r = QRectF(self._margin[0], self._margin[1],
                    self.width() - self._margin[0] - self._margin[2],
                    self.height() - self._margin[1] - self._margin[3])
        x_cursor = r.right() - 8
        y_cursor = r.top() + 4
        for i, (name, display, pts, color, visible) in enumerate(reversed(self._series)):
            label = display if display else name
            tw = QFontMetrics(QFont("Segoe UI", 8)).horizontalAdvance(label) + 18
            x_cursor -= tw
            lr = QRectF(x_cursor, y_cursor, tw, 16)
            if lr.contains(pos):
                idx = len(self._series) - 1 - i
                name2, d2, p2, c2, vis = self._series[idx]
                self._series[idx] = (name2, d2, p2, c2, not vis)
                self.update()
                return
            if x_cursor < r.left():
                break

    def _find_nearest(self, r, pts, mouse_x):
        if not pts:
            return None
        x_range = self._x_max - self._x_min or 1
        best = None
        best_dist = float("inf")
        for x, y in pts:
            px = r.left() + (x - self._x_min) / x_range * r.width()
            dist = abs(px - mouse_x)
            if dist < best_dist and dist < 20:
                best_dist = dist
                best = (x, y)
        return best
