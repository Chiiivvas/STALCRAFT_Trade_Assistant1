"""Чтение текстового списка «Вы получите: Название x N» (левая область)."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..item_matcher import NameIndex
from .ocr import OcrEngine, to_gray, upscale

# «Название х 11»: x бывает латинской/кириллической, пробелы внутри числа, х прилипает к слову
_LINE_QTY = re.compile(r"^(?P<name>.*?)\s*[xXхХ×]\s*(?P<qty>(?:\d\s*){1,7})$")
_HEADER_GET = re.compile(r"получит|получ", re.I)
_HEADER_GIVE = re.compile(r"отда[дё]|отдае|предлож|вы\s*отд", re.I)
_WEIGHT_IN_HEADER = re.compile(r"\(?\s*(\d[\d\s]*)\s*[гrg]\s*[\)\]]?\s*:?\s*$")


@dataclass
class ListLine:
    raw: str
    name: str
    qty: int | None
    item_id: str | None
    match_kind: str            # exact | near | none
    section: str               # receive | give | unknown


@dataclass
class ListReading:
    totals: dict[str, int] = field(default_factory=dict)       # item_id -> сумма из списка
    ambiguous: set[str] = field(default_factory=set)           # варианты OCR разошлись
    unknown_lines: list[str] = field(default_factory=list)     # строки, не сопоставленные с базой
    lines: list[ListLine] = field(default_factory=list)
    raw_texts: dict[str, str] = field(default_factory=dict)
    header_weight: int | None = None
    section_seen: bool = False
    read_ok: bool = False      # прочитана хотя бы одна строка предмета или заголовок


def _variants(img: np.ndarray) -> dict[str, np.ndarray]:
    g = upscale(to_gray(img), 3.0)
    out: dict[str, np.ndarray] = {}
    bg = cv2.GaussianBlur(g, (0, 0), 9)
    d = cv2.subtract(g, bg)
    _, t = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out["dog_otsu"] = 255 - t
    _, t2 = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out["otsu"] = 255 - t2 if t2.mean() < 127 else t2
    return {k: cv2.copyMakeBorder(v, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
            for k, v in out.items()}


def text_mask(img: np.ndarray) -> np.ndarray:
    """Бинарная маска текста (устойчива к изменению размытого фона) — для кэша списка."""
    return _variants(img)["dog_otsu"]


def parse_text(text: str, index: NameIndex, missing_x_means_one: bool = True):
    """Разбор одного OCR-текста -> (lines, header_weight, section_seen)."""
    lines: list[ListLine] = []
    section = "unknown"
    section_seen = False
    header_weight = None
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if _HEADER_GIVE.search(s) and not _LINE_QTY.match(s):
            section, section_seen = "give", True
            continue
        if _HEADER_GET.search(s) and not _LINE_QTY.match(s):
            section, section_seen = "receive", True
            m = _WEIGHT_IN_HEADER.search(s)
            if m:
                header_weight = int(re.sub(r"\D", "", m.group(1)) or 0) or None
            continue
        m = _LINE_QTY.match(s)
        if m:
            name, qty = m.group("name").strip(" .,:;|-"), int(re.sub(r"\D", "", m.group("qty")))
        else:
            # строка без «x N» — по наблюдению в игре это ровно один предмет
            name, qty = s.strip(" .,:;|-"), (1 if missing_x_means_one else None)
        nm = index.match(name)
        if nm.item_id is None and m is None:
            lines.append(ListLine(raw, name, None, None, "none", section))   # мусор, не предмет
            continue
        lines.append(ListLine(raw, name, qty, nm.item_id, nm.kind, section))
    return lines, header_weight, section_seen


def read_list(ocr: OcrEngine, region_bgr: np.ndarray, index: NameIndex,
              missing_x_means_one: bool = True) -> ListReading:
    res = ListReading()
    per_variant: list[dict[str, int]] = []
    for name, v in _variants(region_bgr).items():
        txt = ocr.image_to_text(v, "rus+eng", 6)
        res.raw_texts[name] = txt
        lines, hw, seen = parse_text(txt, index, missing_x_means_one)
        res.section_seen = res.section_seen or seen
        if hw and res.header_weight is None:
            res.header_weight = hw
        tot: dict[str, int] = defaultdict(int)
        for ln in lines:
            if ln.section == "give":
                continue
            if ln.item_id and ln.qty is not None:
                tot[ln.item_id] += ln.qty
        per_variant.append(dict(tot))
        if name == "dog_otsu" or not res.lines:
            res.lines = lines
    ids = {i for d in per_variant for i in d}
    for item_id in ids:
        votes = Counter(d[item_id] for d in per_variant if item_id in d)
        top, n = votes.most_common(1)[0]
        if len(votes) == 1:
            res.totals[item_id] = top          # все варианты, где строка найдена, согласны
        elif n > len(per_variant) / 2:
            res.totals[item_id] = top
        else:
            res.ambiguous.add(item_id)
    res.unknown_lines = [ln.raw for ln in res.lines
                         if ln.item_id is None and ln.section != "give" and ln.qty is not None]
    res.read_ok = bool(res.totals) or res.header_weight is not None
    return res
