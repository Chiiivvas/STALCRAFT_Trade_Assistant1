"""Фоновый поток сканирования.

Гарантии:
  * одновременно выполняется не более одного скана (один поток, один цикл);
  * интерфейс не блокируется;
  * результаты старых сканов отбрасываются по номеру «поколения» (после сброса);
  * если кадр не изменился — тяжёлое распознавание не запускается.
"""
from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from .config import Config
from .engine import ScanEngine
from .models import ScanResult
from .screen_capture import CaptureError, ScreenCapture

log = logging.getLogger(__name__)


def _signature(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (96, 48), interpolation=cv2.INTER_AREA).astype(np.float32)


def _differs(a: np.ndarray | None, b: np.ndarray, threshold: float) -> bool:
    return a is None or a.shape != b.shape or float(np.abs(a - b).mean()) > threshold


class ScanWorker(QThread):
    resultReady = Signal(object)     # ScanResult
    failed = Signal(str)

    def __init__(self, engine: ScanEngine, cfg: Config, parent=None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.cfg = cfg
        self.generation = 0
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._paused = False
        self._force = True
        self._last: ScanResult | None = None
        self._last_error = ""

    # ---- управление из GUI-потока ------------------------------------------
    def request_reset(self) -> None:
        """Сброс: старые результаты станут недействительными, кэш очищается."""
        self.generation += 1
        self._force = True
        self._last = None
        self.engine.clear_cache()
        self._wake.set()

    def force_rescan(self) -> None:
        self._force = True
        self._wake.set()

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        if not paused:
            self._force = True
            self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if not self.wait(8000):
            log.warning("Поток сканирования не завершился вовремя")

    # ---- цикл ---------------------------------------------------------------
    def _sleep(self, seconds: float) -> None:
        self._wake.wait(max(0.0, seconds))
        self._wake.clear()

    def run(self) -> None:
        cap = ScreenCapture()
        sig_r = sig_l = None
        last_full = 0.0
        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                interval = max(0.15, self.cfg.scan_interval_ms / 1000.0)
                try:
                    if self._paused or not self.cfg.right_region:
                        self._sleep(interval)
                        continue
                    gen = self.generation
                    right = cap.grab(self.cfg.right_region)
                    left = cap.grab(self.cfg.left_region) if self.cfg.left_region else None
                    s_r = _signature(right)
                    s_l = _signature(left) if left is not None else None
                    changed = (_differs(sig_r, s_r, 1.0)
                               or (s_l is not None and _differs(sig_l, s_l, 6.0)))
                    stale = (time.perf_counter() - last_full) > self.cfg.force_rescan_sec
                    if self._force or changed or stale or self._last is None:
                        self._force = False
                        res = self.engine.scan(right, left)
                        res.generation = gen
                        sig_r, sig_l = s_r, s_l
                        last_full = time.perf_counter()
                        self._last = res
                        self.resultReady.emit(res)
                    elif self._last is not None:
                        self.resultReady.emit(self._last)      # «ничего не изменилось»
                    self._last_error = ""
                except CaptureError as e:
                    self._report(str(e))
                except Exception as e:                           # защита цикла от любых сбоев
                    log.exception("Ошибка сканирования")
                    self._report(f"Ошибка сканирования: {e}")
                self._sleep(interval - (time.perf_counter() - t0))
        finally:
            cap.close()

    def _report(self, msg: str) -> None:
        if msg != self._last_error:
            self._last_error = msg
            self.failed.emit(msg)
