"""Диалоги: прайс-лист, эталонные иконки, настройки, диагностика."""
from __future__ import annotations

import re
import sqlite3

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton,
                               QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout)

from .. import diagnostics
from ..item_matcher import normalize_name
from ..models import STATUS_TEXT
from ..price_database import ItemRecord, PriceDatabase, imread_unicode
from ..region_selector import bgr_to_pixmap
from ..screen_capture import CaptureError, ScreenCapture
from ..vision.grid import detect_grid
from ..vision.ocr import OcrEngine


def parse_int(text: str) -> int | None:
    digits = re.sub(r"\D", "", text or "")
    return int(digits) if digits else None


# ---------------------------------------------------------------- прайс-лист
class PriceDialog(QDialog):
    COLS = ["Название", "Цена за шт.", "Вес 1 шт., г", "Категория", "Алиасы (через ;)", "Иконок"]

    def __init__(self, db: PriceDatabase, parent=None) -> None:
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Прайс-лист")
        self.resize(720, 380)
        self._ids: list[str | None] = []
        self._deleted: set[str] = set()
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Цена — за одну штуку. «Вес 1 шт.» нужен для сверки количества с весом стопки."))
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(0, 190)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        lay.addWidget(self.table)
        for it in db.items():
            self._add_row(it)
        btns = QHBoxLayout()
        add, rm, ok, cancel = (QPushButton(t) for t in ("+ Предмет", "Удалить", "Сохранить", "Отмена"))
        add.clicked.connect(lambda: self._add_row(None))
        rm.clicked.connect(self._remove_row)
        ok.clicked.connect(self._save)
        cancel.clicked.connect(self.reject)
        for b in (add, rm):
            btns.addWidget(b)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        lay.addLayout(btns)

    def _add_row(self, rec: ItemRecord | None) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self._ids.append(rec.id if rec else None)
        vals = ([rec.name, "" if rec.price is None else str(rec.price),
                 "" if rec.unit_weight is None else str(rec.unit_weight), rec.category,
                 "; ".join(rec.aliases), str(len(rec.templates))] if rec else ["", "", "", "", "", "0"])
        for c, v in enumerate(vals):
            item = QTableWidgetItem(v)
            if c == 5:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, c, item)

    def _remove_row(self) -> None:
        r = self.table.currentRow()
        if r < 0:
            return
        if self._ids[r]:
            self._deleted.add(self._ids[r])
        self.table.removeRow(r)
        self._ids.pop(r)

    def _save(self) -> None:
        recs, seen = [], set()
        for r in range(self.table.rowCount()):
            def t(c): return (self.table.item(r, c).text() if self.table.item(r, c) else "").strip()
            name = t(0)
            if not name:
                QMessageBox.warning(self, "Прайс-лист", f"Строка {r + 1}: пустое название.")
                return
            key = normalize_name(name)
            if key in seen:
                QMessageBox.warning(self, "Прайс-лист", f"Название «{name}» повторяется.")
                return
            seen.add(key)
            recs.append(ItemRecord(self._ids[r] or self.db.new_id(), name, t(3), parse_int(t(1)),
                                   parse_int(t(2)), [a for a in t(4).split(";") if a.strip()]))
        try:
            for did in self._deleted:
                self.db.delete_item(did)
            for rec in recs:
                self.db.upsert_item(rec)
        except (sqlite3.IntegrityError, ValueError) as e:
            QMessageBox.warning(self, "Прайс-лист", f"Не удалось сохранить: {e}")
            return
        self.accept()


# ---------------------------------------------------------------- иконки
class CardPickerDialog(QDialog):
    def __init__(self, frame, cells, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Выберите карточку предмета")
        self.selected_cell = None
        self._cells = cells
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Выберите карточку, которую нужно сохранить как эталон:"))
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.ViewMode.IconMode)
        self.list.setIconSize(QSize(111, 114))
        for c in cells:
            img = frame[c.y:c.y + c.h, c.x:c.x + c.w]
            it = QListWidgetItem(QIcon(bgr_to_pixmap(img).scaled(111, 114)), f"слот {c.row + 1}:{c.col + 1}")
            it.setData(Qt.ItemDataRole.UserRole, img.copy())
            self.list.addItem(it)
        self.list.itemDoubleClicked.connect(lambda _i: self._ok())
        lay.addWidget(self.list)
        ok = QPushButton("Добавить как эталон")
        ok.clicked.connect(self._ok)
        lay.addWidget(ok)
        self.resize(560, 320)

    def _ok(self) -> None:
        it = self.list.currentItem()
        if it:
            self.selected_cell = it.data(Qt.ItemDataRole.UserRole)
            self.accept()


class IconsDialog(QDialog):
    def __init__(self, db: PriceDatabase, cfg, main_window, parent=None) -> None:
        super().__init__(parent)
        self.db, self.cfg, self._main = db, cfg, main_window
        self.changed = False
        self.setWindowTitle("Эталонные иконки")
        self.resize(640, 380)
        lay = QHBoxLayout(self)
        self.items = QListWidget()
        self.items.currentRowChanged.connect(self._refresh_templates)
        lay.addWidget(self.items, 1)
        right = QVBoxLayout()
        right.addWidget(QLabel("Эталоны выбранного предмета (можно несколько вариантов):"))
        self.tpls = QListWidget()
        self.tpls.setViewMode(QListWidget.ViewMode.IconMode)
        self.tpls.setIconSize(QSize(74, 76))
        right.addWidget(self.tpls, 1)
        for text, fn in (("Добавить из трейда…", self._from_trade), ("Добавить из файла…", self._from_file),
                         ("Удалить выбранный эталон", self._remove), ("Закрыть", self.accept)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            right.addWidget(b)
        lay.addLayout(right, 2)
        self._records: list[ItemRecord] = []
        self._reload()

    def _reload(self, keep: int = 0) -> None:
        self._records = self.db.items()
        self.items.clear()
        for r in self._records:
            self.items.addItem(f"{r.name}  ({len(r.templates)})")
        if self._records:
            self.items.setCurrentRow(min(keep, len(self._records) - 1))

    def _refresh_templates(self, row: int) -> None:
        self.tpls.clear()
        if not 0 <= row < len(self._records):
            return
        for t in self._records[row].templates:
            img = imread_unicode(t.path)
            if img is None:
                continue
            it = QListWidgetItem(QIcon(bgr_to_pixmap(img)), "")
            it.setData(Qt.ItemDataRole.UserRole, t.id)
            self.tpls.addItem(it)

    def _current(self) -> ItemRecord | None:
        r = self.items.currentRow()
        return self._records[r] if 0 <= r < len(self._records) else None

    def _from_trade(self) -> None:
        if not self._current():
            return
        if not self.cfg.right_region:
            QMessageBox.information(self, "Иконки", "Сначала выберите правую область (кнопка «Область»).")
            return
        self.hide()
        self._main.hide()
        QTimer.singleShot(450, self._capture)

    def _capture(self) -> None:
        cap, frame = ScreenCapture(), None
        try:
            frame = cap.grab(self.cfg.right_region)
        except CaptureError as e:
            QMessageBox.warning(self, "Иконки", str(e))
        finally:
            cap.close()
            self._main.show()
            self.show()
        if frame is None:
            return
        cells = [c for c in detect_grid(frame).filled_cells() if not c.partial]
        if not cells:
            QMessageBox.information(self, "Иконки", "В выбранной области нет предметов.")
            return
        dlg = CardPickerDialog(frame, cells, self)
        if dlg.exec() and dlg.selected_cell is not None:
            self._add(dlg.selected_cell)

    def _from_file(self) -> None:
        if not self._current():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Скриншот ОДНОЙ ячейки предмета", "",
                                              "Изображения (*.png *.jpg *.jpeg *.bmp)")
        img = imread_unicode(path) if path else None
        if img is not None:
            self._add(img)

    def _add(self, img) -> None:
        cur = self.items.currentRow()
        self.db.add_template(self._records[cur].id, img)
        self.changed = True
        self._reload(cur)

    def _remove(self) -> None:
        it = self.tpls.currentItem()
        if not it:
            return
        cur = self.items.currentRow()
        if len(self._records[cur].templates) <= 1:
            QMessageBox.information(self, "Иконки", "Нельзя удалить последний эталон предмета.")
            return
        self.db.remove_template(it.data(Qt.ItemDataRole.UserRole))
        self.changed = True
        self._reload(cur)


# ---------------------------------------------------------------- настройки
class SettingsDialog(QDialog):
    def __init__(self, cfg, ocr: OcrEngine, parent=None) -> None:
        super().__init__(parent)
        self.cfg, self.ocr = cfg, ocr
        self.setWindowTitle("Настройки")
        form = QFormLayout(self)
        self.path = QLineEdit(cfg.tesseract_path)
        browse = QPushButton("…")
        browse.clicked.connect(self._browse)
        check = QPushButton("Проверить")
        check.clicked.connect(self._check)
        row = QHBoxLayout()
        for w in (self.path, browse, check):
            row.addWidget(w)
        form.addRow("Tesseract (tesseract.exe):", row)
        self.check_lbl = QLabel(ocr.status.message)
        self.check_lbl.setWordWrap(True)
        form.addRow(self.check_lbl)
        self.interval = QSpinBox(); self.interval.setRange(150, 3000); self.interval.setSuffix(" мс")
        self.interval.setValue(cfg.scan_interval_ms)
        self.empty = QSpinBox(); self.empty.setRange(1, 20); self.empty.setValue(cfg.empty_scans_to_clear)
        self.score = QDoubleSpinBox(); self.score.setRange(0.5, 0.99); self.score.setSingleStep(0.01)
        self.score.setValue(cfg.icon_min_score)
        self.margin = QDoubleSpinBox(); self.margin.setRange(0.0, 0.5); self.margin.setSingleStep(0.01)
        self.margin.setValue(cfg.icon_min_margin)
        self.confirm = QCheckBox("Требовать подтверждение количества (список слева / вес)")
        self.confirm.setChecked(cfg.require_confirmation)
        self.weight = QCheckBox("Сверять количество с весом стопки")
        self.weight.setChecked(cfg.use_weight_check)
        self.one = QCheckBox("В списке слева строка без «x N» = 1 шт.")
        self.one.setChecked(cfg.missing_x_means_one_in_list)
        self.hide_unknown = QCheckBox("Не показывать предметы, которых нет в прайс-листе")
        self.hide_unknown.setChecked(cfg.hide_unknown_items)
        self.top = QCheckBox("Окно поверх всех окон")
        self.top.setChecked(cfg.always_on_top)
        form.addRow("Интервал сканирования:", self.interval)
        form.addRow("Пустых сканов до очистки:", self.empty)
        form.addRow("Порог совпадения иконки:", self.score)
        form.addRow("Мин. отрыв от 2-го кандидата:", self.margin)
        for w in (self.confirm, self.weight, self.one, self.hide_unknown, self.top):
            form.addRow(w)
        btns = QHBoxLayout()
        ok, cancel = QPushButton("Сохранить"), QPushButton("Отмена")
        ok.clicked.connect(self._save)
        cancel.clicked.connect(self.reject)
        btns.addStretch(); btns.addWidget(ok); btns.addWidget(cancel)
        form.addRow(btns)

    def _browse(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "tesseract.exe", "", "tesseract.exe (tesseract.exe);;Все (*)")
        if p:
            self.path.setText(p)

    def _check(self) -> None:
        self.check_lbl.setText(self.ocr.configure(self.path.text().strip()).message)

    def _save(self) -> None:
        c = self.cfg
        c.tesseract_path = self.path.text().strip()
        c.scan_interval_ms, c.empty_scans_to_clear = self.interval.value(), self.empty.value()
        c.icon_min_score, c.icon_min_margin = self.score.value(), self.margin.value()
        c.require_confirmation, c.use_weight_check = self.confirm.isChecked(), self.weight.isChecked()
        c.missing_x_means_one_in_list, c.always_on_top = self.one.isChecked(), self.top.isChecked()
        c.hide_unknown_items = self.hide_unknown.isChecked()
        self.ocr.configure(c.tesseract_path)
        self.accept()


# ---------------------------------------------------------------- диагностика
class DiagnosticsDialog(QDialog):
    def __init__(self, get_result, parent=None) -> None:
        super().__init__(parent)
        self.get_result = get_result
        self.setWindowTitle("Диагностика распознавания")
        self.resize(900, 560)
        lay = QVBoxLayout(self)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        self.shot = QLabel("Нет данных")
        sc = QScrollArea(); sc.setWidget(self.shot)
        self.cards = QTableWidget(0, 7)
        self.cards.setHorizontalHeaderLabels(["Слот", "Карточка", "Иконка", "Количество", "Вес", "Статус", "Причины"])
        self.cards.setIconSize(QSize(56, 58))
        self.cards.verticalHeader().setDefaultSectionSize(62)
        self.items = QTableWidget(0, 6)
        self.items.setHorizontalHeaderLabels(["Позиция", "Кол-во", "Статус", "Подтверждено", "Цена", "Причины"])
        self.text = QPlainTextEdit(); self.text.setReadOnly(True)
        for w, n in ((sc, "Область"), (self.cards, "Карточки"), (self.items, "Позиции"), (self.text, "Список / OCR")):
            self.tabs.addTab(w, n)
        row = QHBoxLayout()
        for label, fn in (("Обновить", self.refresh), ("Сохранить проблемный кадр", self._save), ("Закрыть", self.accept)):
            b = QPushButton(label); b.clicked.connect(fn); row.addWidget(b)
        lay.addLayout(row)
        self.refresh()

    def refresh(self) -> None:
        res = self.get_result()
        if res is None:
            return
        ov = diagnostics.render_overlay(res)
        if ov is not None:
            pm = bgr_to_pixmap(ov)
            self.shot.setPixmap(pm); self.shot.resize(pm.size())
        self.cards.setRowCount(len(res.cards))
        for r, c in enumerate(res.cards):
            ic = c.raw.icon if c.raw else None
            vals = [f"{c.row + 1}:{c.col + 1}", "",
                    f"{c.icon_id or '?'} {ic.best_score:.0%} (отрыв {ic.margin:.0%})" if ic else "",
                    f"{c.qty} ← {c.raw.qty.raw}" if c.raw else "",
                    (f"{c.raw.weight.value} ← {c.raw.weight.raw}" if c.raw and c.raw.weight else "—"),
                    STATUS_TEXT[c.status], "; ".join(c.reasons)]
            for k, v in enumerate(vals):
                self.cards.setItem(r, k, QTableWidgetItem(str(v)))
            if c.cell is not None:
                self.cards.item(r, 1).setIcon(QIcon(bgr_to_pixmap(c.cell)))
        self.items.setRowCount(len(res.items))
        for r, i in enumerate(res.items):
            vals = [i.name, i.qty, STATUS_TEXT[i.status], ", ".join(i.confirmed_by), i.price, "; ".join(i.reasons)]
            for k, v in enumerate(vals):
                self.items.setItem(r, k, QTableWidgetItem("" if v is None else str(v)))
        lr, lines = res.list_reading, []
        lines.append(f"Скан: {res.elapsed_ms:.0f} мс; ошибки: {res.errors or 'нет'}; заметки: {res.notes or 'нет'}")
        if lr:
            lines += [f"\nСписок слева: {lr.totals}", f"неоднозначно: {sorted(lr.ambiguous)}",
                      f"неизвестные строки: {lr.unknown_lines}", f"вес в заголовке: {lr.header_weight}"]
            for k, t in lr.raw_texts.items():
                lines.append(f"\n--- OCR [{k}] ---\n{t}")
        self.text.setPlainText("\n".join(lines))

    def _save(self) -> None:
        res = self.get_result()
        if res is None:
            return
        path = diagnostics.save_problem(res)
        QMessageBox.information(self, "Сохранено", f"Кадры и отчёт сохранены локально:\n{path}")
