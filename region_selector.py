"""Выбор области экрана мышью. Снимок экрана рисуется как фон — без хаков с прозрачностью."""
from __future__ import annotations

import cv2
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget


def bgr_to_pixmap(img) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


class RegionSelector(QWidget):
    selected = Signal(list)      # [x, y, w, h] в физических пикселях экрана
    cancelled = Signal()

    def __init__(self, title: str, screenshot_bgr, monitor: dict,
                 current: list[int] | None = None) -> None:
        super().__init__()
        self._title = title
        self._mon = monitor
        screen = QGuiApplication.primaryScreen()
        self._dpr = float(screen.devicePixelRatio())
        self._pix = bgr_to_pixmap(screenshot_bgr)
        self._pix.setDevicePixelRatio(self._dpr)
        self._current = current
        self._origin = QPoint()
        self._cursor = QPoint()
        self._dragging = False
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool)
        self.setGeometry(screen.geometry())
        self.setCursor(Qt.CursorShape.CrossCursor)

    def start(self) -> None:
        self.showFullScreen()
        self.activateWindow()
        self.setFocus()

    def _sel(self) -> QRect:
        return QRect(self._origin, self._cursor).normalized()

    def _to_logical(self, region: list[int]) -> QRect:
        x, y, w, h = region
        return QRect(int((x - self._mon["left"]) / self._dpr), int((y - self._mon["top"]) / self._dpr),
                     int(w / self._dpr), int(h / self._dpr))

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.drawPixmap(0, 0, self._pix)
        p.fillRect(self.rect(), QColor(0, 0, 0, 120))
        if self._current and not self._dragging:
            r = self._to_logical(self._current)
            p.setPen(QPen(QColor(255, 200, 60), 2, Qt.PenStyle.DashLine))
            p.drawRect(r)
        if self._dragging or not self._origin.isNull():
            r = self._sel()
            p.drawPixmap(r, self._pix, QRect(int(r.x() * self._dpr), int(r.y() * self._dpr),
                                             int(r.width() * self._dpr), int(r.height() * self._dpr)))
            p.setPen(QPen(QColor(70, 220, 140), 2))
            p.drawRect(r)
            p.setPen(Qt.GlobalColor.white)
            p.setFont(QFont("Segoe UI", 10))
            p.drawText(r.x(), max(14, r.y() - 6),
                       f"{int(r.width() * self._dpr)} × {int(r.height() * self._dpr)} px")
        p.setPen(Qt.GlobalColor.white)
        p.setFont(QFont("Segoe UI", 14))
        p.drawText(30, 45, f"{self._title} — выделите область мышью. ESC — отмена.")

    def mousePressEvent(self, e) -> None:
        self._origin = e.position().toPoint()
        self._cursor = self._origin
        self._dragging = True
        self.update()

    def mouseMoveEvent(self, e) -> None:
        if self._dragging:
            self._cursor = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        self._cursor = e.position().toPoint()
        self._dragging = False
        r = self._sel()
        w, h = int(r.width() * self._dpr), int(r.height() * self._dpr)
        if w >= 30 and h >= 30:
            self.selected.emit([int(r.x() * self._dpr) + self._mon["left"],
                                int(r.y() * self._dpr) + self._mon["top"], w, h])
        else:
            self.cancelled.emit()
        self.close()

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            self.close()
