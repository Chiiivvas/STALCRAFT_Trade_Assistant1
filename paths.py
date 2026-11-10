"""Пути проекта. Работает и из исходников, и из PyInstaller-сборки."""
from __future__ import annotations

import sys
from pathlib import Path


def bundle_dir() -> Path:
    """Папка с ресурсами, упакованными в EXE (assets/)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Пользовательские данные: рядом с EXE / main.py (не во временной _MEIPASS)."""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).resolve().parent.parent
    d = base / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


ASSETS_DIR = bundle_dir() / "assets"
SEED_ICONS_DIR = ASSETS_DIR / "seed_icons"
