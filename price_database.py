"""Локальная база предметов и цен (SQLite) + снимок каталога для фонового потока."""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .item_matcher import NameIndex
from .paths import SEED_ICONS_DIR, data_dir


@dataclass
class TemplateRec:
    id: int
    path: str


@dataclass
class ItemRecord:
    id: str
    name: str
    category: str = ""
    price: int | None = None
    unit_weight: int | None = None          # вес 1 шт. в граммах (для сверки с карточкой)
    aliases: list[str] = field(default_factory=list)
    templates: list[TemplateRec] = field(default_factory=list)


def imread_unicode(path: str | Path) -> np.ndarray | None:
    """cv2.imread не умеет не-ASCII пути в Windows — читаем через numpy."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: str | Path, img: np.ndarray) -> bool:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))
    return True


@dataclass
class Catalog:
    """Неизменяемый снимок базы — безопасно передаётся в рабочий поток."""
    items: dict[str, ItemRecord]
    version: int = 0

    def name_index(self) -> NameIndex:
        return NameIndex({i.id: [i.name, *i.aliases] for i in self.items.values()})

    def template_images(self) -> dict[str, list[np.ndarray]]:
        out: dict[str, list[np.ndarray]] = {}
        for it in self.items.values():
            imgs = [im for t in it.templates if (im := imread_unicode(t.path)) is not None]
            if imgs:
                out[it.id] = imgs
        return out


class PriceDatabase:
    def __init__(self, db_path: Path | None = None, icons_dir: Path | None = None,
                 seed: bool = True) -> None:
        base = data_dir()
        self.db_path = db_path or (base / "items.sqlite3")
        self.icons_dir = icons_dir or (base / "icons")
        self.icons_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.version = 0
        self._init_schema()
        if seed:
            self._seed_if_empty()

    # ---- схема --------------------------------------------------------------
    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS items(
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL DEFAULT '',
                price INTEGER,
                unit_weight INTEGER
            );
            CREATE TABLE IF NOT EXISTS aliases(
                item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
                alias TEXT NOT NULL,
                PRIMARY KEY(item_id, alias)
            );
            CREATE TABLE IF NOT EXISTS templates(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
                path TEXT NOT NULL
            );
            """)

    def _seed_if_empty(self) -> None:
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        seed_file = SEED_ICONS_DIR / "seed_items.json"
        if n or not seed_file.exists():
            return
        for it in json.loads(seed_file.read_text(encoding="utf-8")):
            rec = ItemRecord(id=it["id"], name=it["name"], category=it.get("category", ""),
                             price=None, unit_weight=it.get("unit_weight"),
                             aliases=it.get("aliases", []))
            self.upsert_item(rec)
            src = SEED_ICONS_DIR / it["icon"]
            img = imread_unicode(src)
            if img is not None:
                self.add_template(rec.id, img)

    # ---- чтение -------------------------------------------------------------
    def items(self) -> list[ItemRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM items ORDER BY category, name").fetchall()
            out = []
            for r in rows:
                aliases = [a["alias"] for a in self._conn.execute(
                    "SELECT alias FROM aliases WHERE item_id=? ORDER BY alias", (r["id"],))]
                tpls = [TemplateRec(t["id"], t["path"]) for t in self._conn.execute(
                    "SELECT id, path FROM templates WHERE item_id=? ORDER BY id", (r["id"],))]
                out.append(ItemRecord(r["id"], r["name"], r["category"], r["price"],
                                      r["unit_weight"], aliases, tpls))
            return out

    def catalog(self) -> Catalog:
        return Catalog({i.id: i for i in self.items()}, self.version)

    # ---- изменение ----------------------------------------------------------
    @staticmethod
    def new_id() -> str:
        return "i_" + uuid.uuid4().hex[:8]

    def upsert_item(self, rec: ItemRecord) -> None:
        name = rec.name.strip()
        if not name:
            raise ValueError("Название предмета не может быть пустым")
        if rec.price is not None and rec.price < 0:
            raise ValueError("Цена не может быть отрицательной")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO items(id,name,category,price,unit_weight) VALUES(?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET name=excluded.name, category=excluded.category, "
                "price=excluded.price, unit_weight=excluded.unit_weight",
                (rec.id, name, rec.category.strip(), rec.price, rec.unit_weight))
            self._conn.execute("DELETE FROM aliases WHERE item_id=?", (rec.id,))
            for a in {x.strip() for x in rec.aliases if x.strip()}:
                self._conn.execute("INSERT INTO aliases(item_id, alias) VALUES(?,?)", (rec.id, a))
        self.version += 1

    def set_price(self, item_id: str, price: int | None) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE items SET price=? WHERE id=?", (price, item_id))
        self.version += 1

    def delete_item(self, item_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM items WHERE id=?", (item_id,))
        shutil.rmtree(self.icons_dir / item_id, ignore_errors=True)
        self.version += 1

    def add_template(self, item_id: str, cell_bgr: np.ndarray) -> str:
        folder = self.icons_dir / item_id
        folder.mkdir(parents=True, exist_ok=True)
        n = 1
        while (folder / f"{n}.png").exists():
            n += 1
        path = folder / f"{n}.png"
        if not imwrite_unicode(path, cell_bgr):
            raise OSError("Не удалось сохранить эталонную иконку")
        with self._lock, self._conn:
            self._conn.execute("INSERT INTO templates(item_id, path) VALUES(?,?)",
                               (item_id, str(path)))
        self.version += 1
        return str(path)

    def remove_template(self, template_id: int) -> None:
        with self._lock, self._conn:
            row = self._conn.execute("SELECT path FROM templates WHERE id=?",
                                     (template_id,)).fetchone()
            self._conn.execute("DELETE FROM templates WHERE id=?", (template_id,))
        if row:
            try:
                Path(row["path"]).unlink(missing_ok=True)
            except OSError:
                pass
        self.version += 1

    def close(self) -> None:
        with self._lock:
            self._conn.close()
