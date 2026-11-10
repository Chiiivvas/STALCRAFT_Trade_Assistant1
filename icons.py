"""Распознавание иконок предметов (OpenCV).

Для каждой иконки считаются признаки, не зависящие от масштаба и яркости:
  * цветовая гистограмма (оттенок x насыщенность) по пикселям самого предмета;
  * форма (маска предмета);
  * внешний вид (корреляция яркости внутри маски).
Допускается несколько эталонов на один предмет.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import layout

SIZE = 40
HUE_BINS = 12
SAT_EDGES = (40, 110)   # граница «серый / бледный / насыщенный»


def crop_fraction(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    xa, ya = int(round(x1 * w)), int(round(y1 * h))
    xb, yb = max(xa + 1, int(round(x2 * w))), max(ya + 1, int(round(y2 * h)))
    return img[ya:yb, xa:xb]


def extract_icon_crop(cell_bgr: np.ndarray) -> np.ndarray:
    """Единая функция вырезания иконки — используется и для эталонов, и для кадров."""
    return crop_fraction(cell_bgr, layout.ICON_BOX)


@dataclass
class IconFeatures:
    hist: np.ndarray
    mask: np.ndarray       # bool SIZE x SIZE
    gray: np.ndarray       # float32 SIZE x SIZE (нормирована внутри маски)
    area: float


def extract_features(icon_bgr: np.ndarray) -> IconFeatures:
    img = cv2.resize(icon_bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2].astype(np.float32)

    p98 = float(np.percentile(v, 98))
    thr = max(90.0, 0.5 * p98)
    mask = v > thr
    ex = layout.SHIELD_EXCLUDE
    mask[int(ex[1] * SIZE):int(ex[3] * SIZE), int(ex[0] * SIZE):int(ex[2] * SIZE)] = False
    area = float(mask.mean())

    # гистограмма: оттенок (круговой) x 2 уровня насыщенности + отдельный «серый» бин
    hist = np.zeros(HUE_BINS * 2 + 1, np.float32)
    hm, sm = h[mask].astype(np.int32), s[mask].astype(np.int32)
    gray_bin = sm < SAT_EDGES[0]
    hist[-1] = gray_bin.sum()
    col = ~gray_bin
    hb = np.minimum(hm[col] * HUE_BINS // 180, HUE_BINS - 1)
    lvl = (sm[col] >= SAT_EDGES[1]).astype(np.int32)
    np.add.at(hist, lvl * HUE_BINS + hb, 1.0)
    for off in (0, HUE_BINS):   # круговое сглаживание по оттенку
        seg = hist[off:off + HUE_BINS].copy()
        hist[off:off + HUE_BINS] = 0.5 * seg + 0.25 * np.roll(seg, 1) + 0.25 * np.roll(seg, -1)
    total = float(hist.sum())
    if total > 0:
        hist /= total

    g = v.copy()
    if mask.any():
        vals = g[mask]
        g = (g - vals.mean()) / (vals.std() + 1e-3)
    g[~mask] = 0.0
    return IconFeatures(hist, mask, g.astype(np.float32), area)


def color_similarity(a: IconFeatures, b: IconFeatures) -> float:
    return float(np.minimum(a.hist, b.hist).sum())


def similarity(a: IconFeatures, b: IconFeatures) -> float:
    color = color_similarity(a, b)
    inter = float((a.mask & b.mask).sum())
    union = float((a.mask | b.mask).sum())
    shape = inter / union if union else 0.0
    both = a.mask | b.mask
    if both.sum() > 10:
        x, y = a.gray[both], b.gray[both]
        ncc = float(np.dot(x - x.mean(), y - y.mean()) /
                    (np.linalg.norm(x - x.mean()) * np.linalg.norm(y - y.mean()) + 1e-6))
        ncc = max(0.0, ncc)
    else:
        ncc = 0.0
    return 0.45 * color + 0.30 * shape + 0.25 * ncc


@dataclass
class IconMatch:
    best_id: str | None = None
    best_score: float = 0.0
    second_id: str | None = None
    second_score: float = 0.0
    best_color: float = 0.0      # совпадение цвета с лучшим кандидатом
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def margin(self) -> float:
        return self.best_score - self.second_score

    def confident(self, min_score: float, min_margin: float) -> bool:
        return (self.best_id is not None and self.best_score >= min_score
                and self.margin >= min_margin)


class IconMatcher:
    def __init__(self) -> None:
        self._templates: dict[str, list[IconFeatures]] = {}

    def set_templates(self, images_by_item: dict[str, list[np.ndarray]]) -> None:
        """images_by_item: item_id -> список ЦЕЛЫХ ячеек (карточек) с этим предметом.
        Иконка вырезается здесь той же функцией, что и при распознавании кадра."""
        self._templates = {
            item_id: [extract_features(extract_icon_crop(im)) for im in imgs
                      if im is not None and im.size]
            for item_id, imgs in images_by_item.items()
        }
        self._templates = {k: v for k, v in self._templates.items() if v}

    def has_templates(self) -> bool:
        return bool(self._templates)

    def match(self, cell_bgr: np.ndarray) -> IconMatch:
        feats = extract_features(extract_icon_crop(cell_bgr))
        res = IconMatch()
        if feats.area < 0.03:        # в ячейке нет заметного предмета
            return res
        scores = {iid: max(similarity(feats, t) for t in tpls)
                  for iid, tpls in self._templates.items()}
        res.scores = scores
        colors = {iid: max(color_similarity(feats, t) for t in tpls)
                  for iid, tpls in self._templates.items()}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if ranked:
            res.best_id, res.best_score = ranked[0]
            res.best_color = colors[res.best_id]
        if len(ranked) > 1:
            res.second_id, res.second_score = ranked[1]
        return res
