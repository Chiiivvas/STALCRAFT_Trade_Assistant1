"""Диагностика: наложение на кадр, отчёт, сохранение проблемных кадров для разбора."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .models import CHECK, OK, STATUS_TEXT, UNRECOGNIZED, ScanResult
from .paths import data_dir
from .price_database import imwrite_unicode

_COLORS = {OK: (90, 217, 149), CHECK: (60, 170, 255), UNRECOGNIZED: (80, 80, 255)}


def render_overlay(result: ScanResult) -> np.ndarray | None:
    """Кадр правой области с выделенными карточками (подписи — латиницей: OpenCV без кириллицы)."""
    if result.right_img is None:
        return None
    img = result.right_img.copy()
    if result.grid:
        for c in result.grid.cells:
            cv2.rectangle(img, (c.x, c.y), (c.x + c.w, c.y + c.h), (90, 90, 90), 1)
    for card in result.cards:
        x, y, w, h = card.rect
        color = _COLORS.get(card.status, (200, 200, 200))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"{card.icon_id or '?'} x{card.qty if card.qty is not None else '?'}"
        cv2.putText(img, label, (x + 3, y + h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return img


def report_dict(result: ScanResult) -> dict:
    return {
        "elapsed_ms": round(result.elapsed_ms, 1),
        "errors": result.errors,
        "notes": result.notes,
        "grid": ({"method": result.grid.method, "pitch_x": result.grid.pitch_x,
                  "pitch_y": result.grid.pitch_y} if result.grid else None),
        "cards": [{
            "row": c.row, "col": c.col, "partial": c.partial, "status": c.status,
            "icon_id": c.icon_id, "icon_confident": c.icon_confident,
            "icon_score": round(c.raw.icon.best_score, 3) if c.raw else None,
            "icon_margin": round(c.raw.icon.margin, 3) if c.raw else None,
            "icon_scores": ({k: round(v, 3) for k, v in c.raw.icon.scores.items()} if c.raw else None),
            "qty": c.qty, "qty_raw": c.raw.qty.raw if c.raw else None,
            "qty_votes": c.raw.qty.votes if c.raw else None,
            "weight": c.raw.weight.value if c.raw and c.raw.weight else None,
            "weight_raw": c.raw.weight.raw if c.raw and c.raw.weight else None,
            "weight_consistent": c.weight_consistent, "reasons": c.reasons,
        } for c in result.cards],
        "items": [{
            "key": i.key, "name": i.name, "qty": i.qty, "price": i.price,
            "status": i.status, "status_text": STATUS_TEXT[i.status],
            "confirmed_by": i.confirmed_by, "reasons": i.reasons,
        } for i in result.items],
        "list": ({"totals": result.list_reading.totals,
                  "ambiguous": sorted(result.list_reading.ambiguous),
                  "unknown_lines": result.list_reading.unknown_lines,
                  "header_weight": result.list_reading.header_weight,
                  "raw_texts": result.list_reading.raw_texts}
                 if result.list_reading else None),
    }


def save_problem(result: ScanResult, note: str = "", root: Path | None = None) -> Path:
    """Сохраняет кадры + отчёт. Всё остаётся локально, наружу ничего не отправляется."""
    root = root or (data_dir() / "problems")
    folder = root / time.strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    if result.right_img is not None:
        imwrite_unicode(folder / "right_region.png", result.right_img)
    if result.left_img is not None:
        imwrite_unicode(folder / "left_region.png", result.left_img)
    for c in result.cards:
        if c.cell is not None:
            imwrite_unicode(folder / "cards" / f"r{c.row + 1}c{c.col + 1}_{c.status}.png", c.cell)
    rep = report_dict(result)
    rep["note"] = note
    (folder / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder
