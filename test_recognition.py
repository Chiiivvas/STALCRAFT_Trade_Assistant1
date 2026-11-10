import cv2, numpy as np, pytest
from .conftest import LEFT, RIGHT, crop
from app.models import OK, CHECK, UNRECOGNIZED
from app.trade_calculator import calculate
from app.vision.grid import detect_grid

EXPECTED = {"Мякоть сластены": 7, "Мякоть куборбуза": 6, "Мякоть лимонника": 11,
            "Аномальная пыль": 1, "Мякоть солевика": 12}
EXPECTED_TOTAL = 7 * 8300 + 6 * 1000 + 11 * 1000 + 280 + 12 * 1500


def qtys(res):
    return {i.name: i.qty for i in res.items if i.status == OK}


# ---- сетка ---------------------------------------------------------------
@pytest.mark.parametrize("box", [(323, 598, 593, 261), (310, 585, 640, 290), (326, 601, 590, 257),
                                 (300, 560, 660, 330)])
def test_grid_robust_to_region_selection(shot, box):
    g = detect_grid(crop(shot, box))
    assert g.method == "lines"
    assert len(g.filled_cells()) == 6 and all(not c.partial for c in g.filled_cells())


@pytest.mark.parametrize("scale", [0.75, 1.25, 1.5])
def test_grid_scales(shot, scale):
    im = cv2.resize(shot, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    box = [int(v * scale) for v in RIGHT]
    assert len(detect_grid(crop(im, box)).filled_cells()) == 6


def test_my_own_offer_grid_is_empty(shot):
    assert detect_grid(crop(shot, (323, 190, 593, 260))).filled_cells() == []


# ---- основной сценарий -----------------------------------------------------
def test_full_trade_total(engine, shot):
    res = engine.scan(crop(shot, RIGHT), crop(shot, LEFT))
    assert not res.errors and qtys(res) == EXPECTED
    assert calculate(res.items).total == EXPECTED_TOTAL


def test_without_left_region_uses_weight(engine, shot):
    res = engine.scan(crop(shot, RIGHT), None)
    assert qtys(res) == EXPECTED
    assert all(i.confirmed_by == ["вес"] for i in res.items)


def test_cache_makes_rescan_fast(engine, shot):
    r, l = crop(shot, RIGHT), crop(shot, LEFT)
    engine.scan(r, l)
    assert engine.scan(r, l).elapsed_ms < 300


# ---- устойчивость к качеству кадра -----------------------------------------
@pytest.mark.parametrize("scale", [1.25, 1.5])
def test_ui_scale_up(engine, shot, scale):
    im = cv2.resize(shot, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    res = engine.scan(crop(im, [int(v * scale) for v in RIGHT]), crop(im, [int(v * scale) for v in LEFT]))
    assert not any(i.status == OK and i.qty != EXPECTED.get(i.name) for i in res.items)  # неверного нет
    assert qtys(res) == EXPECTED


def test_blur_never_gives_wrong_value(engine, shot):
    """Главное правило: при плохом кадре допустимо «не уверен», но недопустимо неверное число."""
    r, l = (cv2.GaussianBlur(crop(shot, b), (5, 5), 1.8) for b in (RIGHT, LEFT))
    res = engine.scan(r, l)
    for i in res.items:
        if i.status == OK:
            assert i.qty == EXPECTED[i.name], f"неверное количество выдано как уверенное: {i}"


def test_brightness_shift(engine, shot):
    r = cv2.convertScaleAbs(crop(shot, RIGHT), alpha=0.8, beta=-6)
    res = engine.scan(r, crop(shot, LEFT))
    for i in res.items:
        if i.status == OK:
            assert i.qty == EXPECTED[i.name]


# ---- сомнительные случаи ---------------------------------------------------
def test_unreadable_quantity_is_excluded(engine, shot):
    r = crop(shot, RIGHT)
    r[8:24, 40:73] = 30            # закрасили «7x» у первой карточки (сластена)
    res = engine.scan(r, crop(shot, LEFT))
    sl = next(i for i in res.items if i.name == "Мякоть сластены")
    assert sl.status != OK
    calc = calculate(res.items)
    assert calc.total == EXPECTED_TOTAL - 7 * 8300 and calc.excluded >= 1


def test_list_mismatch_marks_item_for_check(engine, shot, monkeypatch):
    from app.vision import list_reader
    real = list_reader.read_list

    def fake(*a, **k):
        r = real(*a, **k)
        r.totals["kuborbuz"] = 9      # список «врёт»
        return r
    monkeypatch.setattr("app.engine.read_list", fake)
    res = engine.scan(crop(shot, RIGHT), crop(shot, LEFT))
    ku = next(i for i in res.items if i.name == "Мякоть куборбуза")
    assert ku.status == CHECK and any("список" in r for r in ku.reasons)
    assert calculate(res.items).total == EXPECTED_TOTAL - 6 * 1000


def test_unknown_icon_not_counted(engine, shot):
    r = crop(shot, RIGHT)
    cell = r[1:75, 75:147].copy()                       # красная карточка -> перекрасим в зелёную
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = np.where(hsv[..., 1] > 80, 60, hsv[..., 0])
    r[1:75, 75:147] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    res = engine.scan(r, None)
    assert not any(i.name == "Мякоть куборбуза" and i.status == OK for i in res.items)
    assert calculate(res.items).total <= EXPECTED_TOTAL - 6 * 1000


def test_empty_trade(engine, shot):
    res = engine.scan(crop(shot, (323, 190, 593, 260)), None)
    assert res.empty and res.items == []


def test_no_price_is_not_counted(db, engine, shot):
    db.set_price("slastena", None)
    engine.set_catalog(db.catalog())
    res = engine.scan(crop(shot, RIGHT), crop(shot, LEFT))
    sl = next(i for i in res.items if i.name == "Мякоть сластены")
    assert sl.status == CHECK and calculate(res.items).total == EXPECTED_TOTAL - 7 * 8300


# ---- предмет не из прайс-листа (реальный скриншот: зелёная «Мякоть мятноплода») ----
@pytest.fixture(scope="session")
def shot_green():
    import pathlib
    return cv2.imread(str(pathlib.Path(__file__).parent / "fixtures" / "trade_unknown_green.png"))


def test_foreign_item_is_flagged_unknown_and_never_counted(engine, shot_green):
    res = engine.scan(crop(shot_green, RIGHT), crop(shot_green, LEFT))
    known = {i.name: i.qty for i in res.items if i.status == OK}
    assert known == {"Мякоть куборбуза": 4, "Мякоть сластены": 6, "Мякоть солевика": 2}
    assert calculate(res.items).total == 4 * 1000 + 6 * 8300 + 2 * 1500      # 56 800
    unknown = [i for i in res.items if i.unknown]
    assert unknown and all(i.status == UNRECOGNIZED for i in unknown)
    # зелёная карточка не должна превращаться в «сластену?» — это был дубль в интерфейсе
    assert not any(i.status == CHECK and "сластены" in i.name for i in res.items)
    green = next(c for c in res.cards if c.col == 3)
    assert green.status == UNRECOGNIZED and green.icon_id is None


def test_known_items_keep_high_color_similarity(engine, shot):
    res = engine.scan(crop(shot, RIGHT), None)
    assert all(c.raw.icon.best_color > 0.9 for c in res.cards)
