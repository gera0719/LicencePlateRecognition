import detect_plate as dp
import detect_ocr as do
from pathlib import Path


if __name__ == "__main__":
    plates_path = dp.detect_plates()
    do.perform_ocr(plates_path / "crops" / "license_plate")