"""Обёртка над Tesseract: явный путь к exe, проверка языков, варианты предобработки."""
from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field

import cv2
import numpy as np

try:  # pytesseract может отсутствовать — приложение должно объяснить это, а не упасть
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

DEFAULT_WIN_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class OcrError(RuntimeError):
    """Ошибка работы Tesseract (нет exe, нет языка, таймаут...)."""


@dataclass
class OcrStatus:
    ok: bool
    message: str
    version: str = ""
    languages: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def resolve_tesseract(path: str | None) -> str | None:
    """Путь из настроек -> путь Windows по умолчанию -> PATH."""
    candidates = [path, DEFAULT_WIN_PATH, shutil.which("tesseract")]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


class OcrEngine:
    def __init__(self, tesseract_path: str | None = None, required_langs=("eng", "rus"),
                 timeout: float = 6.0) -> None:
        self.required_langs = tuple(required_langs)
        self.timeout = timeout
        self.path: str | None = None
        self.status = OcrStatus(False, "Tesseract не проверен")
        self.configure(tesseract_path)

    # ---- настройка / проверка -------------------------------------------------
    def configure(self, tesseract_path: str | None) -> OcrStatus:
        if pytesseract is None:
            self.status = OcrStatus(False, "Не установлен пакет pytesseract (pip install pytesseract)")
            return self.status
        self.path = resolve_tesseract(tesseract_path)
        if not self.path:
            self.status = OcrStatus(
                False,
                "Tesseract не найден. Укажите путь к tesseract.exe в настройках "
                f"(обычно {DEFAULT_WIN_PATH}).")
            return self.status
        pytesseract.pytesseract.tesseract_cmd = self.path
        try:
            version = str(pytesseract.get_tesseract_version())
            langs = sorted(pytesseract.get_languages(config=""))
        except Exception as e:
            self.status = OcrStatus(False, f"Не удалось запустить Tesseract ({self.path}): {e}")
            return self.status
        missing = [l for l in self.required_langs if l not in langs]
        if missing:
            self.status = OcrStatus(
                False, "Не хватает языков Tesseract: " + ", ".join(missing)
                + ". Переустановите Tesseract и отметьте эти языки.", version, langs, missing)
        else:
            self.status = OcrStatus(True, f"Tesseract {version}: языки {', '.join(langs)}",
                                    version, langs)
        return self.status

    @property
    def ready(self) -> bool:
        return self.status.ok

    # ---- распознавание -------------------------------------------------------
    def image_to_text(self, img: np.ndarray, lang: str, psm: int, whitelist: str | None = None) -> str:
        if not self.ready:
            raise OcrError(self.status.message)
        cfg = f"--psm {psm} --oem 1 -c load_system_dawg=0 -c load_freq_dawg=0"
        if whitelist:
            cfg += f" -c tessedit_char_whitelist={whitelist}"
        try:
            return pytesseract.image_to_string(img, lang=lang, config=cfg, timeout=self.timeout)
        except RuntimeError as e:           # таймаут pytesseract
            raise OcrError(f"Tesseract завис (таймаут {self.timeout} c): {e}") from e
        except Exception as e:
            raise OcrError(f"Ошибка Tesseract: {e}") from e


# ---- предобработка -----------------------------------------------------------
def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def upscale(gray: np.ndarray, factor: float) -> np.ndarray:
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def light_text_variants(img: np.ndarray, factor: float = 4.0) -> dict[str, np.ndarray]:
    """Светлый текст на тёмном фоне -> чёрный на белом, несколькими способами."""
    g = upscale(to_gray(img), factor)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    out: dict[str, np.ndarray] = {}
    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out["otsu"] = 255 - otsu
    hi = float(np.percentile(g, 99))
    _, fixed = cv2.threshold(g, max(60.0, 0.6 * hi), 255, cv2.THRESH_BINARY)
    out["p99"] = 255 - fixed
    out["gray_inv"] = 255 - cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    return {k: cv2.copyMakeBorder(v, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
            for k, v in out.items()}


_NON_DIGIT = re.compile(r"\D")


def digits_only(s: str) -> str:
    return _NON_DIGIT.sub("", s)
