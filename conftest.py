import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import cv2, pytest
from app.config import Config
from app.engine import ScanEngine
from app.price_database import PriceDatabase
from app.vision.ocr import OcrEngine

PRICES = {"slastena": 8300, "kuborbuz": 1000, "limonnik": 1000, "solevik": 1500, "dust": 280}
RIGHT = (323, 598, 594, 260)    # x, y, w, h на скриншоте 1920x1080
LEFT = (95, 565, 185, 160)


@pytest.fixture(scope="session")
def ocr():
    e = OcrEngine(None)
    if not e.ready:
        pytest.skip("Tesseract с языками eng+rus не установлен: " + e.status.message)
    return e


@pytest.fixture()
def db(tmp_path):
    d = PriceDatabase(tmp_path / "i.sqlite3", tmp_path / "icons")
    for it in d.items():
        d.set_price(it.id, PRICES[it.id])
    yield d
    d.close()


@pytest.fixture()
def engine(db, ocr):
    e = ScanEngine(Config(), ocr)
    e.set_catalog(db.catalog())
    yield e
    e.close()


@pytest.fixture(scope="session")
def shot():
    return cv2.imread(str(pathlib.Path(__file__).parent / "fixtures" / "trade_full_1920x1080.png"))


def crop(img, box):
    x, y, w, h = box
    return img[y:y + h, x:x + w].copy()
