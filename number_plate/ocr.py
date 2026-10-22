import argparse
import os
import sys
import cv2
import easyocr

from typing import Union
import numpy as np

_reader = None


def _get_reader() -> easyocr.Reader:
    """Lazily initialize and return the cached EasyOCR reader instance."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def read_plate_text(image_input: Union[str, np.ndarray]) -> list[dict]:
    """Read text from an image, cropped plate path, or numpy image array using EasyOCR on CPU.

    Args:
        image_input: Path to the image file (str) or loaded BGR/RGB image (np.ndarray).

    Returns:
        List of dicts with keys:
            - 'text': recognized text stripped of whitespace
            - 'confidence': confidence score rounded to 3 decimal places
            - 'box': bounding-box coordinates returned by EasyOCR

    Raises:
        FileNotFoundError: If the image file does not exist or cannot be decoded by OpenCV.
        TypeError: If image_input is not a string or numpy ndarray.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file does not exist: {image_input}")
        image = cv2.imread(image_input)
        if image is None:
            raise FileNotFoundError(f"Unable to read or decode image at: {image_input}")
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            return []
        image = image_input
    else:
        raise TypeError(
            f"Expected image path string or np.ndarray, got {type(image_input).__name__}"
        )

    reader = _get_reader()
    raw_results = reader.readtext(image)

    detections: list[dict] = []
    for bbox, text, confidence in raw_results:
        cleaned_text = str(text).strip()
        if not cleaned_text:
            continue

        box_coords = [[int(round(coord)) for coord in point] for point in bbox]

        detections.append(
            {
                "text": cleaned_text,
                "confidence": round(float(confidence), 3),
                "box": box_coords,
            }
        )

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract text from an image or cropped number plate."
    )
    parser.add_argument("image_path", help="Path to the image file")
    args = parser.parse_args()

    try:
        results = read_plate_text(args.image_path)
        if not results:
            print("No text detected.")
        else:
            for idx, item in enumerate(results, start=1):
                print(
                    f"[{idx}] Text: {item['text']} | Confidence: {item['confidence']} | Box: {item['box']}"
                )
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
