import argparse
import os
import sys
import urllib.request
from typing import Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
DEFAULT_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "license_plate_yolov8n.pt")
MODEL_URL = "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt"

_detector = None


def ensure_weights(weights_path: str = DEFAULT_WEIGHTS_PATH) -> str:
    """Ensure that the pretrained license plate detector weights are present locally."""
    if not os.path.exists(weights_path) or os.path.getsize(weights_path) < 1000000:
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        print(f"Downloading detector weights to {weights_path}...")
        urllib.request.urlretrieve(MODEL_URL, weights_path)
    return weights_path


def _get_detector(weights_path: Optional[str] = None) -> YOLO:
    """Lazily load and cache the YOLO license plate detector."""
    global _detector
    if _detector is None:
        target_path = ensure_weights(weights_path or DEFAULT_WEIGHTS_PATH)
        _detector = YOLO(target_path)
    return _detector


def detect_plates(
    image_input: Union[str, np.ndarray],
    conf_threshold: float = 0.25,
    weights_path: Optional[str] = None,
) -> list[dict]:
    """Detect license plates within an image or full traffic frame.

    Args:
        image_input: Path to the image file (str) or loaded BGR image (np.ndarray).
        conf_threshold: Detection confidence threshold (0.0 to 1.0).
        weights_path: Optional path to custom YOLO weights.

    Returns:
        List of dicts with:
            - 'box': [x1, y1, x2, y2] bounding box coordinates (integers)
            - 'confidence': detection confidence score rounded to 3 decimal places
            - 'crop': cropped license plate region as a numpy ndarray (BGR)

    Raises:
        FileNotFoundError: If the image path does not exist or cannot be read.
        TypeError: If image_input is not a path string or numpy ndarray.
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

    model = _get_detector(weights_path)
    img_h, img_w = image.shape[:2]

    # Run inference in CPU mode with specified confidence
    results = model.predict(source=image, conf=conf_threshold, device="cpu", verbose=False)

    detections: list[dict] = []
    if not results or len(results) == 0:
        return detections

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detections

    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])

        x1 = max(0, min(int(round(xyxy[0])), img_w - 1))
        y1 = max(0, min(int(round(xyxy[1])), img_h - 1))
        x2 = max(x1 + 1, min(int(round(xyxy[2])), img_w))
        y2 = max(y1 + 1, min(int(round(xyxy[3])), img_h))

        crop = image[y1:y2, x1:x2].copy()

        detections.append(
            {
                "box": [x1, y1, x2, y2],
                "confidence": round(conf, 3),
                "crop": crop,
            }
        )

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect license plates in an image using pretrained YOLOv8n."
    )
    parser.add_argument("image_path", help="Path to input traffic or vehicle image")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for plate detection (default: 0.25)",
    )
    args = parser.parse_args()

    try:
        plates = detect_plates(args.image_path, conf_threshold=args.conf)
        if not plates:
            print("No license plates detected.")
        else:
            for idx, p in enumerate(plates, start=1):
                crop_shape = p["crop"].shape
                print(
                    f"[{idx}] Plate Box: {p['box']} | Confidence: {p['confidence']} | Crop Size: {crop_shape[1]}x{crop_shape[0]} px"
                )
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
