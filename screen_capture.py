"""Захват экрана (mss). Экземпляр нельзя делить между потоками — у каждого потока свой."""
from __future__ import annotations

import mss
import numpy as np


class CaptureError(RuntimeError):
    pass


class ScreenCapture:
    def __init__(self) -> None:
        self._sct: mss.base.MSSBase | None = None

    def _get(self):
        if self._sct is None:
            self._sct = mss.mss()
        return self._sct

    def grab(self, region: list[int] | tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = [int(v) for v in region]
        if w < 8 or h < 8:
            raise CaptureError("Область слишком мала — выберите её заново")
        try:
            shot = self._get().grab({"left": x, "top": y, "width": w, "height": h})
        except Exception as e:
            self.close()
            raise CaptureError(f"Не удалось захватить экран: {e}") from e
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])   # BGRA -> BGR

    def grab_primary(self) -> tuple[np.ndarray, dict]:
        """Весь основной монитор (BGR) и его геометрия в физических пикселях."""
        try:
            mon = self._get().monitors[1]
            shot = self._get().grab(mon)
        except Exception as e:
            self.close()
            raise CaptureError(f"Не удалось сделать снимок экрана: {e}") from e
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3]), dict(mon)

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
