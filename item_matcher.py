"""Строгое сопоставление названий с базой предметов (без агрессивного нечёткого поиска)."""
from __future__ import annotations

import re
from dataclasses import dataclass

# латинские «двойники» кириллических букв, которые OCR любит путать
_HOMOGLYPHS = str.maketrans({
    "a": "а", "b": "в", "c": "с", "e": "е", "h": "н", "k": "к", "m": "м", "o": "о",
    "p": "р", "t": "т", "x": "х", "y": "у", "i": "и", "u": "и", "n": "н", "r": "г",
})
_NON_LETTER = re.compile(r"[^а-я0-9 ]+")


def normalize_name(s: str) -> str:
    s = s.lower().replace("ё", "е").translate(_HOMOGLYPHS)
    s = _NON_LETTER.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class NameMatch:
    item_id: str | None
    kind: str = "none"      # exact | near | none
    distance: int = 0


class NameIndex:
    """Индекс названий и алиасов. exact — точное совпадение после нормализации;
    near — допускается 1–2 ошибки OCR ТОЛЬКО если кандидат единственный и
    следующий по близости заметно дальше (используется лишь для сверки списка)."""

    def __init__(self, names_by_item: dict[str, list[str]]) -> None:
        self._exact: dict[str, str] = {}
        self._all: list[tuple[str, str]] = []
        for item_id, names in names_by_item.items():
            for n in names:
                key = normalize_name(n)
                if key:
                    self._exact.setdefault(key, item_id)
                    self._all.append((key, item_id))

    def match(self, text: str, allow_near: bool = True) -> NameMatch:
        key = normalize_name(text)
        if not key:
            return NameMatch(None)
        if key in self._exact:
            return NameMatch(self._exact[key], "exact", 0)
        if not allow_near or len(key) < 8:
            return NameMatch(None)
        best: dict[str, int] = {}
        for cand, item_id in self._all:
            d = levenshtein(key, cand)
            if d < best.get(item_id, 99):
                best[item_id] = d
        ranked = sorted(best.items(), key=lambda kv: kv[1])
        if not ranked:
            return NameMatch(None)
        item_id, d = ranked[0]
        limit = 1 if len(key) < 12 else 2
        second = ranked[1][1] if len(ranked) > 1 else 99
        if d <= limit and second - d >= 3:
            return NameMatch(item_id, "near", d)
        return NameMatch(None)
