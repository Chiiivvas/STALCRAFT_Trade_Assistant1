import threading, time
from app.item_matcher import NameIndex, normalize_name
from app.models import CHECK, OK, ItemResult
from app.trade_calculator import calculate
from app.vision.list_reader import parse_text

IDX = NameIndex({"a": ["Мякоть сластены"], "b": ["Мякоть куборбуза"], "c": ["Мякоть лимонника"],
                 "d": ["Мякоть солевика"], "e": ["Аномальная пыль"]})


def test_calculator_example_from_task():
    it = ItemResult("a", "Мякоть сластены", "a", qty=17, price=8300)
    assert calculate([it]).total == 141100


def test_manual_accept_and_exclusion():
    ok = ItemResult("a", "A", "a", qty=2, price=10, status=OK)
    bad = ItemResult("b", "B", "b", qty=3, price=100, status=CHECK)
    assert calculate([ok, bad]).total == 20
    c = calculate([ok, bad], {"b"})
    assert c.total == 320 and c.excluded == 0


def test_names_exact_and_no_aggressive_fuzzy():
    assert IDX.match("Мякоть сластены").kind == "exact"
    assert IDX.match("мякоть  СЛАСТЕНЫ").item_id == "a"
    assert IDX.match("Мякоть куборбузах").item_id == "b"      # 1 ошибка OCR, единственный кандидат
    for bad in ("Мякоть", "Мякоть арбуза", "Аномальная", "Мякоть соленика и лимона"):
        assert IDX.match(bad).item_id is None, bad
    assert normalize_name("Мякоть сластeны") == normalize_name("Мякоть сластены")  # латинская e


def test_similar_names_not_swapped():
    idx = NameIndex({"x": ["Мякоть лимонника"], "y": ["Мякоть лимонника редкая"]})
    assert idx.match("Мякоть лимонника").item_id == "x"


def test_list_parser_quirks():
    txt = ("Вы получите (550 г]:\nМякоть сластены х 7\nМякоть куборбузах 6\nМякоть лимонника x 1 1\n"
           "Аномальная пыль\nМякоть солевика x 12\nВы отдадите (10 г):\nМякоть сластены x 99")
    lines, hw, seen = parse_text(txt, IDX)
    got = {(l.item_id, l.qty, l.section) for l in lines if l.item_id}
    assert ("a", 7, "receive") in got and ("b", 6, "receive") in got
    assert ("c", 11, "receive") in got and ("e", 1, "receive") in got
    assert ("a", 99, "give") in got and hw == 550 and seen


def test_missing_x_rule_is_configurable():
    lines, *_ = parse_text("Аномальная пыль", IDX, missing_x_means_one=False)
    assert lines[0].qty is None


def test_worker_never_overlaps_and_drops_stale(monkeypatch):
    import numpy as np
    from PySide6.QtCore import QCoreApplication
    import app.worker as W
    from app.config import Config
    from app.models import ScanResult
    qa = QCoreApplication.instance() or QCoreApplication([])

    class FakeCap:
        n = 0
        def grab(self, r): FakeCap.n += 1; return np.full((50, 50, 3), FakeCap.n % 250, np.uint8)
        def close(self): pass

    class FakeEngine:
        active = peak = calls = 0
        def scan(self, r, l):
            FakeEngine.active += 1; FakeEngine.peak = max(FakeEngine.peak, FakeEngine.active)
            FakeEngine.calls += 1; time.sleep(0.25); FakeEngine.active -= 1
            return ScanResult()
        def clear_cache(self): pass

    monkeypatch.setattr(W, "ScreenCapture", FakeCap)
    cfg = Config(right_region=[0, 0, 50, 50], scan_interval_ms=150)
    w = W.ScanWorker(FakeEngine(), cfg)
    got = []
    w.resultReady.connect(lambda r: got.append(r.generation))
    w.start()
    end = time.time() + 1.2
    while time.time() < end: qa.processEvents(); time.sleep(0.02)
    w.request_reset()
    end = time.time() + 0.8
    while time.time() < end: qa.processEvents(); time.sleep(0.02)
    w.stop()
    assert FakeEngine.peak == 1 and FakeEngine.calls >= 3
    assert 0 in got and 1 in got and not w.isRunning()
