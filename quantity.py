"""Отдельное распознавание количества («7x», «5423x») и веса стопки («105 г»).

Количество читается независимо от названия и иконки. Используется голосование по
нескольким вариантам предобработки: быстрый проход, а при разногласии — расширенный.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import layout
from .icons import crop_fraction
from .ocr import OcrEngine, OcrError, to_gray, upscale

_QTY_RE = re.compile(r"^[^\d]*(\d{1,7})[xXхХ×*]?[^\d]*$")
_WEIGHT_RE = re.compile(r"^\D*(\d{1,7})")
_KG_HINT = re.compile(r"[.,]\d|[kKкК]")


@dataclass
class ReadResult:
    value: int | None = None
    raw: list[str] = field(default_factory=list)
    votes: dict[int, int] = field(default_factory=dict)
    share: float = 0.0
    mode: str = "none"          # none | fast | extended
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None


def _binarize(gray_up: np.ndarray, ratio: float) -> np.ndarray:
    hi = float(np.percentile(gray_up, 99))
    _, b = cv2.threshold(gray_up, max(50.0, ratio * hi), 255, cv2.THRESH_BINARY)
    return cv2.copyMakeBorder(255 - b, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def _variants(img: np.ndarray, extended: bool):
    g0 = to_gray(img)
    if not extended:
        g = cv2.GaussianBlur(upscale(g0, 4.0), (3, 3), 0)
        for r in (0.6, 0.5, 0.7):
            yield f"p99@{r}x4", _binarize(g, r)
        return
    for f in (3.0, 5.0):
        g = cv2.GaussianBlur(upscale(g0, f), (3, 3), 0)
        for r in (0.45, 0.55, 0.65, 0.75, 0.85):
            yield f"p99@{r}x{f:g}", _binarize(g, r)
        _, o = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield f"otsu x{f:g}", cv2.copyMakeBorder(255 - o, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def _decide(values: list[int], mode: str, min_votes: int, min_share: float) -> ReadResult:
    res = ReadResult(mode=mode)
    if not values:
        res.reason = "число не прочитано"
        return res
    counts = Counter(values)
    top, n = counts.most_common(1)[0]
    res.votes = dict(counts)
    res.share = n / len(values)
    runner_up = counts.most_common(2)[1][1] if len(counts) > 1 else 0
    clear_majority = mode == "extended" and n >= 3 and n >= 2 * runner_up and res.share >= 0.45
    if (n >= min_votes and res.share >= min_share) or clear_majority:
        res.value = top
    else:
        res.reason = f"варианты чтения расходятся: {dict(counts)}"
    return res


def read_quantity(ocr: OcrEngine, cell_bgr: np.ndarray, extended: bool = False) -> ReadResult:
    """Количество в правом верхнем углу карточки. Не подставляет «1» по умолчанию."""
    crop = crop_fraction(cell_bgr, layout.QTY_BOX)
    values: list[int] = []
    raw: list[str] = []
    for _name, v in _variants(crop, extended):
        txt = ocr.image_to_text(v, "eng", 7, "0123456789xX").strip()
        raw.append(txt)
        m = _QTY_RE.match(txt.replace(" ", ""))
        if m:
            values.append(int(m.group(1)))
        if not extended and len(values) >= 2 and len(set(values)) == 1:
            break
    res = _decide(values, "extended" if extended else "fast",
                  min_votes=3 if extended else 2, min_share=0.6 if extended else 0.66)
    res.raw = raw
    return res


def read_weight(ocr: OcrEngine, cell_bgr: np.ndarray, extended: bool = False) -> ReadResult:
    """Вес стопки в граммах (левый нижний угол). Вес в кг считается нечитаемым."""
    crop = crop_fraction(cell_bgr, layout.WEIGHT_BOX)
    values: list[int] = []
    raw: list[str] = []
    for _name, v in _variants(crop, extended):
        txt = ocr.image_to_text(v, "eng", 7).strip()
        raw.append(txt)
        t = txt.replace(" ", "")
        if _KG_HINT.search(t):
            continue
        m = _WEIGHT_RE.match(t)
        if m:
            values.append(int(m.group(1)))
    res = _decide(values, "extended" if extended else "fast",
                  min_votes=2, min_share=0.6)
    res.raw = raw
    return res
