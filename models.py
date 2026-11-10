"""Структуры данных результатов распознавания."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .vision.grid import Grid
from .vision.icons import IconMatch
from .vision.list_reader import ListReading
from .vision.quantity import ReadResult

OK, CHECK, UNRECOGNIZED = "ok", "check", "unrecognized"
STATUS_TEXT = {OK: "уверенно распознано", CHECK: "требуется проверка", UNRECOGNIZED: "не распознано"}


@dataclass
class RawRead:
    """Результат чтения одной карточки (кэшируется по содержимому)."""
    icon: IconMatch
    qty: ReadResult
    weight: ReadResult | None
    extended: bool = False


@dataclass
class CardReading:
    row: int
    col: int
    rect: tuple[int, int, int, int]
    partial: bool
    raw: RawRead | None = None
    icon_id: str | None = None          # опознанный предмет (даже неуверенно)
    icon_confident: bool = False
    weight_consistent: bool | None = None
    status: str = OK
    reasons: list[str] = field(default_factory=list)
    from_cache: bool = False
    cell: np.ndarray | None = None

    @property
    def qty(self) -> int | None:
        return self.raw.qty.value if self.raw else None


@dataclass
class ItemResult:
    key: str                            # item_id или "card:r:c"
    name: str
    item_id: str | None                 # подтверждённый предмет
    candidate_id: str | None = None     # предполагаемый предмет (для ручного принятия)
    qty: int | None = None
    price: int | None = None
    status: str = OK
    reasons: list[str] = field(default_factory=list)
    confirmed_by: list[str] = field(default_factory=list)
    cards: list[tuple[int, int]] = field(default_factory=list)
    needs_extended: bool = False
    unknown: bool = False               # предмета нет в прайс-листе / базе иконок

    @property
    def amount(self) -> int | None:
        if self.qty is None or self.price is None:
            return None
        return self.qty * self.price


@dataclass
class ScanResult:
    items: list[ItemResult] = field(default_factory=list)
    cards: list[CardReading] = field(default_factory=list)
    grid: Grid | None = None
    list_reading: ListReading | None = None
    empty: bool = False
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    generation: int = 0
    right_img: np.ndarray | None = None      # исходные кадры — для диагностики
    left_img: np.ndarray | None = None
