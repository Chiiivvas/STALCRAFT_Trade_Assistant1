"""Поиск сетки слотов обмена и определение занятых слотов.

Линии сетки в игре — тёмные тонкие линии с постоянным шагом. Находим их по доле тёмных
пикселей вдоль каждой оси, оцениваем шаг и строим арифметическую прогрессию.
Если линии не найдены — запасной вариант: 8 равных колонок по ширине области.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import layout


@dataclass
class Cell:
    row: int
    col: int
    x: int
    y: int
    w: int
    h: int
    partial: bool = False   # слот виден не целиком (обрезан областью или интерфейсом игры)
    filled: bool = False    # в слоте есть предмет

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass
class Grid:
    cells: list[Cell] = field(default_factory=list)
    pitch_x: float = 0.0
    pitch_y: float = 0.0
    method: str = "lines"          # "lines" | "fallback"
    notes: list[str] = field(default_factory=list)

    def filled_cells(self) -> list[Cell]:
        return [c for c in self.cells if c.filled]


def _dips(profile: np.ndarray) -> np.ndarray:
    """Глубина узких (<5 px) тёмных провалов в профиле яркости: линии сетки."""
    p = profile.astype(np.float32).reshape(1, -1)
    closed = cv2.morphologyEx(p, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    return (closed - p).ravel()


def _fit_lines(dip: np.ndarray, min_pitch: int = 30):
    """Периодическая сетка линий по сигналу провалов. Возвращает (c0, pitch) или None.

    Шаг ищется автокорреляцией (устойчиво к ложным провалам внутри карточек и к
    сильным линиям на краях области), фаза — по максимуму суммы провалов,
    точный шаг — регрессией по найденным линиям.
    """
    n = dip.size
    if n < 2 * min_pitch or float(dip.max()) < 10.0:
        return None
    d = dip.astype(np.float64)
    ac = np.correlate(d, d, mode="full")[n - 1:]
    max_lag = min(n - 1, 600)
    lags = np.arange(min_pitch, max_lag)
    if lags.size == 0:
        return None
    vals = ac[lags]
    peak_floor = 0.5 * float(vals.max())
    pitch0 = None
    for i in range(1, len(vals) - 1):
        if vals[i] >= peak_floor and vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]:
            pitch0 = int(lags[i])
            break
    if pitch0 is None:
        return None
    # фаза
    best_phase, best_score = 0, -1.0
    for ph in range(pitch0):
        idx = np.arange(ph, n, pitch0)
        score = float(sum(d[max(0, j - 1):j + 2].max() for j in idx))
        if score > best_score:
            best_phase, best_score = ph, score
    # уточнение по линиям
    ks, ps = [], []
    floor = 0.3 * float(d.max())
    for k in range(-1, int(n / pitch0) + 2):
        j = best_phase + k * pitch0
        lo, hi = max(0, j - 3), min(n, j + 4)
        if hi - lo < 2:
            continue
        m = int(np.argmax(d[lo:hi])) + lo
        if d[m] >= floor:
            ks.append(k)
            ps.append(float(m))
    if len(ks) < 2:
        return float(best_phase), float(pitch0)
    pitch, c0 = np.polyfit(ks, ps, 1)
    if abs(pitch - pitch0) > 0.1 * pitch0:
        return float(best_phase), float(pitch0)
    return float(c0), float(pitch)


def _spans(c0: float, pitch: float, size: int, max_count: int | None):
    """[(a, b, visible_fraction)] для ячеек, пересекающих [0, size]."""
    out = []
    k0 = int(np.floor((0 - c0) / pitch)) - 1
    k1 = int(np.ceil((size - c0) / pitch)) + 1
    for k in range(k0, k1 + 1):
        a, b = c0 + k * pitch, c0 + (k + 1) * pitch
        vis = (min(b, size) - max(a, 0)) / pitch
        cut_at_start = a < -0.12 * pitch
        if vis >= 0.88:                         # целая ячейка (допуск на дрожание мыши)
            out.append((max(a, 0.0), min(b, float(size)), vis))
        elif vis >= 0.35 and not cut_at_start:  # обрезана снизу/справа — «частичная»
            out.append((max(a, 0.0), min(b, float(size)), vis))
    if max_count:
        out = out[:max_count]
    return out


def _best_phase(dip: np.ndarray, pitch: float) -> float:
    """Смещение первой линии (0 <= c0 < pitch), при котором сумма провалов максимальна."""
    n = dip.size
    best_ph, best_score = 0.0, -1.0
    for ph in np.arange(0.0, pitch, 1.0):
        idx = np.round(np.arange(ph, n, pitch)).astype(int)
        idx = idx[idx < n]
        score = float(sum(dip[max(0, j - 1):j + 2].max() for j in idx))
        if score > best_score:
            best_ph, best_score = float(ph), score
    return best_ph


def detect_grid(region_bgr: np.ndarray) -> Grid:
    h, w = region_bgr.shape[:2]
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    col_dip = _dips(gray.mean(axis=0))
    row_dip = _dips(gray.mean(axis=1))

    grid = Grid()
    fit_x = _fit_lines(col_dip)
    # в игре 8 колонок: шаг заметно меньше w/9.5 физически невозможен (ложное срабатывание)
    if fit_x is not None and fit_x[1] < w / (layout.GRID_COLUMNS + 1.5):
        fit_x = None
    if fit_x is None:
        pitch_x = w / layout.GRID_COLUMNS
        c0x = _best_phase(col_dip, pitch_x) if col_dip.max() >= 10.0 else 0.0
        if c0x > pitch_x / 2:
            c0x -= pitch_x
        grid.method = "fallback"
        grid.notes.append("линии сетки по горизонтали не найдены надёжно: 8 колонок по ширине области")
    else:
        c0x, pitch_x = fit_x

    expected_y = pitch_x * layout.CELL_ASPECT
    fit_y = _fit_lines(row_dip)
    if fit_y is not None and abs(fit_y[1] - expected_y) <= 0.08 * expected_y:
        c0y, pitch_y = fit_y
    else:
        pitch_y = expected_y
        c0y = _best_phase(row_dip, pitch_y) if row_dip.max() >= 10.0 else 0.0
        if c0y > pitch_y / 2:
            c0y -= pitch_y

    grid.pitch_x, grid.pitch_y = pitch_x, pitch_y
    xs = _spans(c0x, pitch_x, w, layout.GRID_COLUMNS)
    ys = _spans(c0y, pitch_y, h, None)

    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    for r, (ya, yb, vy) in enumerate(ys):
        for c, (xa, xb, vx) in enumerate(xs):
            x, y = int(round(xa)) + 1, int(round(ya)) + 1
            cw, ch = int(round(xb)) - 1 - x, int(round(yb)) - 1 - y
            if cw < 8 or ch < 8:
                continue
            cell = Cell(r, c, x, y, cw, ch, partial=(vx < 0.88 or vy < 0.88))
            v = hsv[y:y + ch, x:x + cw, 2]
            cell.filled = float((v > layout.FILLED_BRIGHT_V).mean()) >= layout.FILLED_MIN_FRACTION
            grid.cells.append(cell)
    return grid
