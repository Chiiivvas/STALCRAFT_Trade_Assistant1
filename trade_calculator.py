"""Расчёт стоимости: считаются только уверенно распознанные позиции (и принятые вручную)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import OK, ItemResult


@dataclass
class Row:
    item: ItemResult
    counted: bool
    manual: bool = False

    @property
    def amount(self) -> int | None:
        return self.item.amount


@dataclass
class Calculation:
    rows: list[Row] = field(default_factory=list)
    total: int = 0
    excluded: int = 0          # сколько позиций не вошло в сумму


def can_accept_manually(item: ItemResult) -> bool:
    """Ручное принятие возможно, только если известны предмет, количество и цена."""
    target = item.item_id or item.candidate_id
    return bool(target and item.qty is not None and item.price is not None)


def calculate(items: list[ItemResult], accepted: set[str] | None = None) -> Calculation:
    accepted = accepted or set()
    calc = Calculation()
    for it in items:
        manual = it.status != OK and it.key in accepted and can_accept_manually(it)
        counted = (it.status == OK and it.amount is not None) or manual
        calc.rows.append(Row(it, counted, manual))
        if counted:
            calc.total += int(it.amount or 0)
        else:
            calc.excluded += 1
    return calc
