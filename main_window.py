"""Компактное главное окно (≈ 360×330): таблица, итог, 4 кнопки; остальное — в меню."""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QGuiApplication
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QMenu,
                               QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
                               QToolButton, QVBoxLayout, QWidget)

from ..config import ConfigStore
from ..engine import ScanEngine
from ..models import CHECK, OK, STATUS_TEXT, UNRECOGNIZED, ScanResult
from ..price_database import PriceDatabase
from ..region_selector import RegionSelector
from ..screen_capture import CaptureError, ScreenCapture
from ..trade_calculator import Calculation, calculate, can_accept_manually
from ..vision.ocr import OcrEngine
from ..worker import ScanWorker
from . import theme
from .dialogs import DiagnosticsDialog, IconsDialog, PriceDialog, SettingsDialog

log = logging.getLogger(__name__)
PREFIX = {OK: "✔ ", CHECK: "⚠ ", UNRECOGNIZED: "✖ "}
COLOR = {OK: theme.COLOR_OK, CHECK: theme.COLOR_CHECK, UNRECOGNIZED: theme.COLOR_BAD}


def fmt(n: int | None) -> str:
    return "—" if n is None else f"{n:,}".replace(",", " ")


class MainWindow(QWidget):
    def __init__(self, store: ConfigStore, db: PriceDatabase, ocr: OcrEngine,
                 engine: ScanEngine) -> None:
        super().__init__()
        self.store, self.cfg = store, store.cfg
        self.db, self.ocr, self.engine = db, ocr, engine
        self._calc = Calculation()
        self._last_res: ScanResult | None = None
        self._diag_res: ScanResult | None = None
        self._empty_streak = 0
        self._accepted: set[str] = set()
        self._all_items: list = []
        self._selector: RegionSelector | None = None

        self.setWindowTitle("STALCRAFT Trade Assistant")
        self.setMinimumWidth(340)
        self.resize(360, 330)
        self._apply_flags()
        self.setStyleSheet(theme.STYLE)
        self._build_ui()
        if self.cfg.window_x is not None and self.cfg.window_y is not None:
            self.move(self.cfg.window_x, self.cfg.window_y)

        self.engine.set_catalog(self.db.catalog())
        self.worker = ScanWorker(self.engine, self.cfg, self)
        self.worker.resultReady.connect(self.on_result)
        self.worker.failed.connect(lambda m: self.set_status(m, bad=True))
        self.worker.start()
        self._initial_hint()

    # ---- UI -----------------------------------------------------------------
    def _apply_flags(self) -> None:
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, self.cfg.always_on_top)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        top = QHBoxLayout()
        title = QLabel("◈ TRADE ASSISTANT")
        title.setStyleSheet("font-weight:700;color:#b8f3d0;")
        self.status = QLabel("")
        self.status.setStyleSheet(f"color:{theme.COLOR_DIM};")
        self.status.setWordWrap(True)
        menu_btn = QToolButton()
        menu_btn.setText("☰")
        menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(menu_btn)
        menu.addAction("Прайс-лист…", self.open_prices)
        menu.addAction("Эталонные иконки…", self.open_icons)
        menu.addAction("Диагностика…", self.open_diagnostics)
        menu.addAction("Настройки…", self.open_settings)
        menu_btn.setMenu(menu)
        top.addWidget(title)
        top.addStretch()
        top.addWidget(menu_btn)
        root.addLayout(top)
        root.addWidget(self.status)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Предмет", "Кол-во", "Цена/шт.", "Сумма"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in (1, 2, 3):
            hh.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.cellDoubleClicked.connect(self._toggle_manual)
        root.addWidget(self.table, 1)

        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(f"color:{theme.COLOR_DIM};font-size:11px;")
        root.addWidget(self.hint)

        tot = QHBoxLayout()
        tot.addWidget(QLabel("ИТОГО:"))
        tot.addStretch()
        self.total = QLabel("0")
        self.total.setStyleSheet(f"font-size:22px;font-weight:800;color:{theme.COLOR_OK};")
        tot.addWidget(self.total)
        root.addLayout(tot)

        self.copy_btn = QPushButton("⧉  СКОПИРОВАТЬ СУММУ")
        self.copy_btn.setMinimumHeight(38)
        self._style_copy(False)
        self.copy_btn.clicked.connect(self.copy_total)
        root.addWidget(self.copy_btn)

        row = QHBoxLayout()
        self.area_btn = QPushButton("Область ▾")
        am = QMenu(self.area_btn)
        am.addAction("Правая: предметы партнёра…", lambda: self.choose_region("right"))
        am.addAction("Левая: список «Вы получите»…", lambda: self.choose_region("left"))
        am.addAction("Убрать левую область", self._clear_left)
        self.area_btn.setMenu(am)
        self.pause_btn = QPushButton("Пауза")
        self.pause_btn.clicked.connect(self.toggle_pause)
        reset_btn = QPushButton("Сброс")
        reset_btn.clicked.connect(self.reset)
        for b in (self.area_btn, self.pause_btn, reset_btn):
            row.addWidget(b)
        root.addLayout(row)

    def _style_copy(self, warn: bool) -> None:
        bg = "#9a6b16" if warn else "#237b51"
        self.copy_btn.setStyleSheet(f"background:{bg};color:white;font-weight:700;border-radius:5px;")

    def set_status(self, text: str, bad: bool = False) -> None:
        self.status.setText(text)
        self.status.setStyleSheet(f"color:{theme.COLOR_BAD if bad else theme.COLOR_DIM};")

    def _initial_hint(self) -> None:
        if not self.ocr.ready:
            self.set_status(self.ocr.status.message, bad=True)
        elif not self.cfg.right_region:
            self.set_status("Нажмите «Область ▾» и выделите сетку предметов партнёра.")
        else:
            self.set_status("Сканирование…")
        if any(i.price is None for i in self.db.items()):
            self.hint.setText("Цены не заданы: меню ☰ → Прайс-лист.")

    # ---- результаты -----------------------------------------------------------
    def on_result(self, res: ScanResult) -> None:
        if res.generation != self.worker.generation:
            return                                  # результат до сброса — игнорируем
        repeat = res is self._last_res
        if not repeat:
            self._last_res = res
            self._diag_res = res
            if res.errors:
                self._render([])
                self.set_status(" ".join(res.errors), bad=True)
                return
        if res.empty:
            self._empty_streak += 1
            if self._empty_streak >= self.cfg.empty_scans_to_clear and self._calc.rows:
                self._accepted.clear()
                self._render([])
            if self._empty_streak >= self.cfg.empty_scans_to_clear:
                self.set_status("Трейд пуст")
            return
        self._empty_streak = 0
        if repeat:
            return
        self._render(res.items)
        note = f" · {res.notes[0]}" if res.notes else ""
        self.set_status(f"Скан {res.elapsed_ms:.0f} мс{note}")

    def _render(self, items) -> None:
        self._all_items = list(items)
        hidden = 0
        if self.cfg.hide_unknown_items:
            # один чужой предмет виден и на карточке, и в списке слева — считаем его один раз
            by_card = sum(1 for i in items if i.unknown and i.key.startswith("card:"))
            by_list = sum(1 for i in items if i.unknown and i.key.startswith("list:"))
            hidden = max(by_card, by_list)
            items = [i for i in items if not i.unknown]
        self._accepted &= {i.key for i in items}
        calc = calculate(items, self._accepted)
        self._calc = calc
        self.table.setRowCount(len(calc.rows))
        for r, row in enumerate(calc.rows):
            it = row.item
            name = ("✎ " if row.manual else PREFIX[it.status]) + it.name
            vals = [name, "?" if it.qty is None else str(it.qty), fmt(it.price),
                    fmt(it.amount) if row.counted else "—"]
            color = QColor("#e8e8e8" if row.manual else COLOR[it.status])
            tip = [STATUS_TEXT[it.status] + (" (принято вручную)" if row.manual else "")]
            if it.confirmed_by:
                tip.append("подтверждено: " + ", ".join(it.confirmed_by))
            tip += it.reasons
            if it.status != OK and can_accept_manually(it):
                tip.append("Двойной клик — принять/отклонить вручную")
            for c, v in enumerate(vals):
                cell = QTableWidgetItem(v)
                cell.setToolTip("\n".join(tip))
                cell.setForeground(QBrush(color if c == 0 or row.counted else QColor(theme.COLOR_DIM)))
                if c:
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, cell)
        self.total.setText(fmt(calc.total))
        self._style_copy(calc.excluded > 0)
        if calc.excluded:
            self.hint.setText(f"Не учтено позиций: {calc.excluded} (⚠/✖) — наведите курсор для причины.")
            self.hint.setStyleSheet(f"color:{theme.COLOR_CHECK};font-size:11px;")
        else:
            self.hint.setText("Все позиции подтверждены." if calc.rows else "")
            self.hint.setStyleSheet(f"color:{theme.COLOR_DIM};font-size:11px;")
        if hidden:
            self.hint.setText((self.hint.text() + " " if self.hint.text() else "")
                              + f"Предметов не из прайс-листа (не учтены): {hidden}.")

    def _toggle_manual(self, row: int, _col: int) -> None:
        if not (0 <= row < len(self._calc.rows)):
            return
        it = self._calc.rows[row].item
        if it.status == OK or not can_accept_manually(it):
            return
        self._accepted ^= {it.key}
        self._render(self._all_items)

    # ---- кнопки ---------------------------------------------------------------
    def copy_total(self) -> None:
        calc = self._calc
        if calc.excluded:
            ans = QMessageBox.question(
                self, "Есть не учтённые позиции",
                f"Не учтено позиций: {calc.excluded}. Сумма {fmt(calc.total)} посчитана без них.\n"
                "Скопировать всё равно?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return
        QGuiApplication.clipboard().setText(str(calc.total))      # только число, без пробелов
        self.set_status(f"Скопировано: {calc.total}")

    def toggle_pause(self) -> None:
        paused = self.pause_btn.text() == "Пауза"
        self.worker.set_paused(paused)
        self.pause_btn.setText("Продолжить" if paused else "Пауза")
        self.set_status("Пауза" if paused else "Сканирование…")

    def reset(self) -> None:
        self.worker.request_reset()
        self._accepted.clear()
        self._empty_streak = 0
        self._last_res = None
        self._render([])
        self.set_status("Сброс: трейд перечитывается…")

    # ---- области --------------------------------------------------------------
    def choose_region(self, which: str) -> None:
        title = ("Правая область: сетка предметов, которые даёт партнёр" if which == "right"
                 else "Левая область: текст «Вы получите: …»")
        self.hide()
        QTimer.singleShot(350, lambda: self._start_selector(which, title))

    def _start_selector(self, which: str, title: str) -> None:
        cap = ScreenCapture()
        try:
            img, mon = cap.grab_primary()
        except CaptureError as e:
            self.show()
            QMessageBox.warning(self, "Ошибка", str(e))
            return
        finally:
            cap.close()
        sel = RegionSelector(title, img, mon, getattr(self.cfg, f"{which}_region"))
        sel.selected.connect(lambda r: self._region_done(which, r))
        sel.cancelled.connect(self._region_cancel)
        self._selector = sel
        sel.start()

    def _region_done(self, which: str, region: list[int]) -> None:
        setattr(self.cfg, f"{which}_region", region)
        self.store.save()
        self.worker.request_reset()
        self.show()
        self.set_status(f"Область «{'правая' if which == 'right' else 'левая'}» сохранена: {region[2]}×{region[3]}")

    def _region_cancel(self) -> None:
        self.show()

    def _clear_left(self) -> None:
        self.cfg.left_region = None
        self.store.save()
        self.worker.request_reset()
        self.set_status("Левая область убрана — сверка только по весу карточек.")

    # ---- диалоги --------------------------------------------------------------
    def reload_catalog(self) -> None:
        self.engine.set_catalog(self.db.catalog())
        self.worker.request_reset()
        self.hint.setText("")

    def open_prices(self) -> None:
        if PriceDialog(self.db, self).exec():
            self.reload_catalog()

    def open_icons(self) -> None:
        dlg = IconsDialog(self.db, self.cfg, self)
        dlg.exec()
        if dlg.changed:
            self.reload_catalog()

    def open_diagnostics(self) -> None:
        DiagnosticsDialog(lambda: self._diag_res, self).exec()

    def open_settings(self) -> None:
        if SettingsDialog(self.cfg, self.ocr, self).exec():
            self.store.save()
            self._apply_flags()
            self.show()
            self.engine.cfg = self.cfg
            self.engine.clear_cache()
            self.worker.force_rescan()
            self._initial_hint()

    # ---- завершение -----------------------------------------------------------
    def closeEvent(self, e) -> None:
        self.cfg.window_x, self.cfg.window_y = self.x(), self.y()
        try:
            self.store.save()
        finally:
            self.worker.stop()
            self.engine.close()
            self.db.close()
        super().closeEvent(e)
