"""STALCRAFT Trade Assistant — запуск приложения."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from PySide6.QtWidgets import QApplication, QMessageBox


def _setup_logging() -> None:
    from app.paths import data_dir
    h = RotatingFileHandler(data_dir() / "app.log", maxBytes=500_000, backupCount=2, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[h],
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    _setup_logging()
    app = QApplication(sys.argv)

    def hook(exc_type, exc, tb):
        logging.getLogger("crash").error("Необработанная ошибка", exc_info=(exc_type, exc, tb))
        QMessageBox.critical(None, "Ошибка", f"{exc_type.__name__}: {exc}\nПодробности: data/app.log")
    sys.excepthook = hook

    from app.config import ConfigStore
    from app.engine import ScanEngine
    from app.gui.main_window import MainWindow
    from app.price_database import PriceDatabase
    from app.vision.ocr import OcrEngine

    store = ConfigStore()
    db = PriceDatabase()
    ocr = OcrEngine(store.cfg.tesseract_path)
    engine = ScanEngine(store.cfg, ocr)
    win = MainWindow(store, db, ocr, engine)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
