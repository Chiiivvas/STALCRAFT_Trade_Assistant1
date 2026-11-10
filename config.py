"""Настройки приложения (JSON, атомарная запись, устойчивость к повреждённому файлу)."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .paths import data_dir
from .vision.ocr import DEFAULT_WIN_PATH


@dataclass
class Config:
    tesseract_path: str = DEFAULT_WIN_PATH
    right_region: list[int] | None = None      # [x, y, w, h] — сетка предметов партнёра (источник)
    left_region: list[int] | None = None       # [x, y, w, h] — список «Вы получите» (только сверка)
    scan_interval_ms: int = 400
    empty_scans_to_clear: int = 3              # столько пустых сканов подряд -> очистить список
    force_rescan_sec: float = 3.0              # полный перечит, даже если картинка «не менялась»
    require_confirmation: bool = True          # позиция считается только при независимом подтверждении
    missing_x_means_one_in_list: bool = True   # в списке слева строка без «x N» = 1 шт. (так в игре)
    use_weight_check: bool = True              # сверять количество с весом стопки на карточке
    icon_min_score: float = 0.80
    icon_min_margin: float = 0.10
    always_on_top: bool = True
    hide_unknown_items: bool = True            # не показывать предметы, которых нет в прайс-листе
    icon_min_color: float = 0.50               # ниже — цвет не тот, предмет считается чужим
    window_x: int | None = None
    window_y: int | None = None


class ConfigStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (data_dir() / "config.json")
        self.cfg = self.load()

    def load(self) -> Config:
        cfg = Config()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cfg
        except Exception:
            # повреждённый файл не должен ронять приложение — сохраняем копию и стартуем с нуля
            try:
                self.path.replace(self.path.with_suffix(".broken.json"))
            except OSError:
                pass
            return cfg
        known = {f.name for f in fields(Config)}
        for k, v in raw.items():
            if k in known:
                setattr(cfg, k, v)
        return cfg

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(asdict(self.cfg), f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
