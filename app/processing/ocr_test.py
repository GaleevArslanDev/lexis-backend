import cv2
import numpy as np
from PIL import Image
import pytesseract
import os


def test_ocr_on_image(image_path: str):
    """Тестирует различные конфигурации OCR на изображении"""

    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Failed to load image"}

    results = {}

    # Различные конфигурации Tesseract
    configs = [
        ("basic", "--oem 3 --psm 6"),
        ("single_column", "--oem 3 --psm 4"),
        ("sparse", "--oem 3 --psm 11"),
        ("auto", "--oem 3 --psm 3"),
        ("single_line", "--oem 3 --psm 7"),
        ("single_word", "--oem 3 --psm 8"),
        ("single_char", "--oem 3 --psm 10"),
    ]

    # Различные языки
    languages = ["rus", "eng", "rus+eng", "equ", "rus+equ"]

    for lang in languages[:2]:  # Тестируем только rus и eng
        for config_name, config in configs:
            try:
                text = pytesseract.image_to_string(
                    image,
                    lang=lang,
                    config=config
                )

                key = f"{lang}_{config_name}"
                results[key] = {
                    "text": text.strip(),
                    "config": config,
                    "language": lang
                }

                print(f"\n=== {key} ===")
                print(f"Config: {config}")
                print(f"Text: {text[:200]}...")
            except Exception as e:
                results[key] = {"error": str(e)}

    return results
