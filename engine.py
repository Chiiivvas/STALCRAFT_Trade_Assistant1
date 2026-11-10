"""Движок распознавания: карточки -> иконка/количество/вес -> сверка со списком -> позиции.

Принцип: позиция попадает в сумму ТОЛЬКО если иконка опознана уверенно, количество
прочитано и результат независимо подтверждён (список слева или вес стопки).
Любое противоречие -> «требуется проверка», позиция в сумму не входит.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import Config
from .item_matcher import NameIndex
from .models import (CHECK, OK, UNRECOGNIZED, CardReading, ItemResult, RawRead, ScanResult)
from .price_database import Catalog
from .vision.grid import detect_grid
from .vision.icons import IconMatcher
import cv2

from .vision.list_reader import read_list, text_mask
from .vision.ocr import OcrEngine, OcrError
from .vision.quantity import read_quantity, read_weight

UNKNOWN_ICON_SCORE = 0.55     # ниже — предмет явно не из базы


class ScanEngine:
    def __init__(self, config: Config, ocr: OcrEngine, cache_size: int = 512) -> None:
        self.cfg = config
        self.ocr = ocr
        self.catalog = Catalog({}, 0)
        self._matcher = IconMatcher()
        self._index = NameIndex({})
        self._cache: OrderedDict[str, RawRead] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ocr")
        self._list_cache: tuple[np.ndarray, object] | None = None

    # ---- настройка ----------------------------------------------------------
    def set_catalog(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self._matcher.set_templates(catalog.template_images())
        self._index = catalog.name_index()
        self.clear_cache()

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
            self._list_cache = None

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---- чтение карточки ----------------------------------------------------
    @staticmethod
    def _key(cell: np.ndarray) -> str:
        return hashlib.md5(cell.tobytes() + bytes(str(cell.shape), "ascii")).hexdigest()

    def _read_cell(self, cell: np.ndarray, extended: bool = False,
                   want_weight: bool = False) -> tuple[RawRead, bool]:
        """Чтение карточки с кэшем. Количество — всегда, вес — только по запросу (ленивое)."""
        key = self._key(cell)
        with self._lock:
            raw = self._cache.get(key)
        from_cache = raw is not None
        if raw is None:
            icon = self._matcher.match(cell)
            raw = RawRead(icon, read_quantity(self.ocr, cell, extended), None, extended)
        elif extended and not raw.extended:
            raw.qty = read_quantity(self.ocr, cell, True)
            raw.weight = None
            raw.extended = True
            from_cache = False
        if want_weight and self.cfg.use_weight_check and raw.weight is None:
            raw.weight = read_weight(self.ocr, cell, raw.extended)
            from_cache = False
        with self._lock:
            self._cache[key] = raw
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return raw, from_cache

    def _name(self, item_id: str | None) -> str:
        it = self.catalog.items.get(item_id) if item_id else None
        return it.name if it else "?"

    def _judge_card(self, card: CardReading) -> None:
        """Выставляет card.status / reasons / icon_id / weight_consistent."""
        card.reasons = []
        card.status = OK
        raw = card.raw
        if raw is None:
            card.status = CHECK
            return
        icon, cfg = raw.icon, self.cfg
        card.icon_id = icon.best_id
        card.icon_confident = icon.confident(cfg.icon_min_score, cfg.icon_min_margin)
        if (icon.best_id is None or icon.best_score < UNKNOWN_ICON_SCORE
                or icon.best_color < cfg.icon_min_color):
            card.status = UNRECOGNIZED
            card.icon_id, card.icon_confident = None, False
            card.reasons.append("иконка не найдена в базе предметов"
                                + (f" (лучшее совпадение {icon.best_score:.0%})" if icon.best_id else ""))
        elif not card.icon_confident:
            card.status = CHECK
            second = f", второй «{self._name(icon.second_id)}» {icon.second_score:.0%}" if icon.second_id else ""
            card.reasons.append(f"иконка опознана неуверенно: «{self._name(icon.best_id)}» "
                                f"{icon.best_score:.0%}{second}")
        if card.partial:
            card.status = CHECK if card.status == OK else card.status
            card.reasons.append("карточка видна не полностью — расширьте область или прокрутите сетку")
        if not raw.qty.ok:
            card.status = CHECK if card.status == OK else card.status
            card.reasons.append(f"количество не прочитано ({raw.qty.reason or 'нет данных'})")
        # сверка веса стопки
        card.weight_consistent = None
        it = self.catalog.items.get(icon.best_id) if icon.best_id else None
        if (raw.weight and raw.weight.ok and raw.qty.ok and card.icon_confident
                and it and it.unit_weight):
            expected = raw.qty.value * it.unit_weight
            got = raw.weight.value
            if got == expected:
                card.weight_consistent = True
            elif str(got).startswith(str(expected)) and len(str(got)) == len(str(expected)) + 1:
                card.weight_consistent = None     # буква «г» прочитана как цифра: не подтверждает и не мешает
            else:
                card.weight_consistent = False
            if card.weight_consistent is False:
                card.status = CHECK if card.status == OK else card.status
                card.reasons.append(f"вес {raw.weight.value} г не сходится: {raw.qty.value} × "
                                    f"{it.unit_weight} г = {expected} г")

    # ---- агрегирование ------------------------------------------------------
    def _aggregate(self, cards: list[CardReading], lr) -> list[ItemResult]:
        items: list[ItemResult] = []
        groups: dict[str, list[CardReading]] = {}
        for c in cards:
            if c.icon_id and c.icon_confident:
                groups.setdefault(c.icon_id, []).append(c)
            else:
                label = (f"{self._name(c.icon_id)}? (неуверенно)" if c.icon_id
                         else "Неизвестный предмет")
                items.append(ItemResult(
                    key=f"card:{c.row}:{c.col}", name=label, item_id=None,
                    candidate_id=c.icon_id, qty=c.qty, status=c.status,
                    reasons=list(c.reasons), cards=[(c.row, c.col)],
                    unknown=(c.status == UNRECOGNIZED),
                    price=self.catalog.items[c.icon_id].price if c.icon_id in self.catalog.items else None,
                    needs_extended=not (c.raw and c.raw.qty.ok)))

        list_ok = bool(lr and lr.read_ok)
        for item_id, cs in groups.items():
            rec = self.catalog.items.get(item_id)
            res = ItemResult(key=item_id, name=rec.name if rec else item_id, item_id=item_id,
                             price=rec.price if rec else None,
                             cards=[(c.row, c.col) for c in cs])
            bad = [c for c in cs if c.status != OK]
            qtys = [c.qty for c in cs]
            res.qty = sum(qtys) if all(q is not None for q in qtys) else None
            for c in bad:
                res.reasons += [f"слот {c.row + 1}:{c.col + 1} — {r}" for r in c.reasons]
            if bad:
                res.status = CHECK
                res.needs_extended = True
            # сверка со списком слева
            list_confirms = False
            if list_ok:
                if item_id in lr.ambiguous:
                    res.reasons.append("список слева: строка прочитана неоднозначно")
                elif item_id in lr.totals:
                    if res.qty is not None and lr.totals[item_id] == res.qty:
                        list_confirms = True
                        res.confirmed_by.append("список")
                    elif res.qty is not None:
                        res.status = CHECK
                        res.needs_extended = True
                        res.reasons.append(f"список слева: {lr.totals[item_id]} шт., карточки: {res.qty} шт.")
                else:
                    res.status = CHECK
                    res.needs_extended = True
                    res.reasons.append("предмета нет в списке слева (или строка не прочитана)")
            weight_confirms = bool(cs) and all(c.weight_consistent is True for c in cs)
            if weight_confirms:
                res.confirmed_by.append("вес")
            if res.status == OK and self.cfg.require_confirmation and not (list_confirms or weight_confirms):
                res.status = CHECK
                res.needs_extended = True
                res.reasons.append("нет независимого подтверждения количества (список слева / вес)")
            if res.price is None:
                res.status = CHECK
                res.reasons.append("не задана цена в прайс-листе")
            items.append(res)

        # предметы из списка, которых нет среди карточек (скрытые строки, прокрутка)
        if list_ok:
            seen = {c.icon_id for c in cards if c.icon_id}
            for item_id, q in lr.totals.items():
                if item_id not in seen:
                    rec = self.catalog.items.get(item_id)
                    items.append(ItemResult(
                        key=item_id, name=rec.name if rec else item_id, item_id=item_id, qty=q,
                        price=rec.price if rec else None, status=CHECK,
                        reasons=["есть в списке слева, но не найден среди карточек "
                                 "(проверьте область / прокрутку сетки)"]))
            for raw in lr.unknown_lines:
                items.append(ItemResult(key=f"list:{raw}", name=raw, item_id=None, status=UNRECOGNIZED,
                                        unknown=True,
                                        reasons=["название из списка слева не найдено в прайс-листе"]))
        return items

    # ---- основной вызов -----------------------------------------------------
    def scan(self, right: np.ndarray, left: np.ndarray | None = None) -> ScanResult:
        t0 = time.perf_counter()
        out = ScanResult(right_img=right, left_img=left)
        if not self.ocr.ready:
            out.errors.append(self.ocr.status.message)
            return out
        if not self._matcher.has_templates():
            out.errors.append("В базе нет эталонных иконок — добавьте их (меню → Иконки).")
            return out
        grid = detect_grid(right)
        out.grid = grid
        out.notes += grid.notes

        # список слева (только сверка, в сумму сам по себе предметы не добавляет)
        lr = None
        list_future = None
        if left is not None:
            mask = text_mask(left)
            cached = self._list_cache
            if (cached is not None and cached[0].shape == mask.shape
                    and float((cv2.absdiff(cached[0], mask) > 0).mean()) < 0.004):
                lr = cached[1]
            else:
                list_future = self._pool.submit(
                    read_list, self.ocr, left, self._index, self.cfg.missing_x_means_one_in_list)

        cells = [c for c in grid.filled_cells()]
        futures = []
        for c in cells:
            img = right[c.y:c.y + c.h, c.x:c.x + c.w]
            futures.append((c, img, None if c.partial else self._pool.submit(self._read_cell, img)))
        try:
            for c, img, fut in futures:
                card = CardReading(c.row, c.col, c.rect, c.partial, cell=img.copy())
                if fut is not None:
                    card.raw, card.from_cache = fut.result()
                else:   # частичные карточки не читаем — они всё равно будут «требует проверки»
                    from .vision.icons import IconMatch
                    from .vision.quantity import ReadResult
                    card.raw = RawRead(self._matcher.match(img), ReadResult(reason="карточка обрезана"), None)
                out.cards.append(card)
            if list_future is not None:
                lr = list_future.result()
                self._list_cache = (mask, lr)
        except OcrError as e:
            out.errors.append(str(e))
            return out
        out.list_reading = lr

        def rejudge() -> list[ItemResult]:
            for card in out.cards:
                self._judge_card(card)
            return self._aggregate(out.cards, lr)

        def reread(cards: list[CardReading], **kw) -> None:
            futs = [(c, self._pool.submit(self._read_cell, c.cell, **kw)) for c in cards]
            for c, f in futs:
                c.raw, c.from_cache = f.result()

        try:
            items = rejudge()
            # шаг 2: вес стопки — только там, где количество не подтверждено списком слева
            if self.cfg.use_weight_check:
                need_w = {pos for it in items if "список" not in it.confirmed_by for pos in it.cards}
                todo = [c for c in out.cards if (c.row, c.col) in need_w and not c.partial
                        and c.raw is not None and c.raw.weight is None]
                if todo:
                    reread(todo, want_weight=True)
                    items = rejudge()
            # шаг 3: расширенный режим только для проблемных позиций
            retry = {pos for it in items if it.needs_extended for pos in it.cards}
            redo = [c for c in out.cards if (c.row, c.col) in retry and not c.partial
                    and c.raw is not None and not c.raw.extended]
            if redo:
                reread(redo, extended=True, want_weight=self.cfg.use_weight_check)
                items = rejudge()
        except OcrError as e:
            out.errors.append(str(e))
            return out

        if lr and lr.header_weight is not None:
            ws = [c.raw.weight.value for c in out.cards
                  if c.raw and c.raw.weight and c.raw.weight.ok]
            if len(ws) == len(out.cards) and out.cards and sum(ws) != lr.header_weight:
                out.notes.append(f"сумма весов карточек {sum(ws)} г ≠ {lr.header_weight} г в заголовке списка")

        out.items = sorted(items, key=lambda i: (i.status != OK, i.name))
        out.empty = (not cells) and not (lr and lr.totals)
        out.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return out
