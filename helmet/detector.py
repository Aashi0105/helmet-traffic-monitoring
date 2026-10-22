import argparse
import json
import os
import sys
import urllib.request
from typing import Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
DEFAULT_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "helmet_yolov8n.pt")
MODEL_URL = (
    "https://huggingface.co/iam-tsr/yolov8n-helmet-detection/resolve/main/best.pt"
)

_detector = None


def ensure_weights(weights_path: str = DEFAULT_WEIGHTS_PATH) -> str:
    """Ensure that the pretrained helmet detector weights are present locally."""
    if not os.path.exists(weights_path) or os.path.getsize(weights_path) < 1000000:
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        print(f"Downloading helmet detector weights to {weights_path}...")
        urllib.request.urlretrieve(MODEL_URL, weights_path)
    return weights_path


def _get_detector(weights_path: Optional[str] = None) -> YOLO:
    """Lazily load and cache the YOLO helmet detector on CPU."""
    global _detector
    if _detector is None:
        target_path = ensure_weights(weights_path or DEFAULT_WEIGHTS_PATH)
        _detector = YOLO(target_path)
    return _detector


def detect_helmets(
    image_input: Union[str, np.ndarray],
    conf_threshold: float = 0.25,
    weights_path: Optional[str] = None,
) -> list[dict]:
    """Detect helmets and bare heads within an image or video frame on CPU.

    Note:
        This model identifies helmet status for each detected person ('With Helmet'
        or 'Without Helmet'). It does NOT identify driver vs. pillion passenger roles.

    Args:
        image_input: Path to the image file (str) or loaded BGR image (np.ndarray).
        conf_threshold: Detection confidence threshold (0.0 to 1.0).
        weights_path: Optional path to custom YOLO weights.

    Returns:
        List of dicts with:
            - 'box': [x1, y1, x2, y2] bounding box coordinates (integers)
            - 'label': class label ('With Helmet' or 'Without Helmet')
            - 'confidence': confidence score rounded to 3 decimal places

    Raises:
        FileNotFoundError: If the image path does not exist or cannot be decoded.
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

    # CPU inference with specified confidence
    results = model.predict(
        source=image, conf=conf_threshold, device="cpu", verbose=False
    )

    detections: list[dict] = []
    if not results or len(results) == 0:
        return detections

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detections

    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names.get(cls_id, str(cls_id))

        x1 = max(0, min(int(round(xyxy[0])), img_w - 1))
        y1 = max(0, min(int(round(xyxy[1])), img_h - 1))
        x2 = max(x1 + 1, min(int(round(xyxy[2])), img_w))
        y2 = max(y1 + 1, min(int(round(xyxy[3])), img_h))

        detections.append(
            {
                "box": [x1, y1, x2, y2],
                "label": label,
                "confidence": round(conf, 3),
            }
        )

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect helmets and helmet violations using pretrained YOLOv8n."
    )
    parser.add_argument("image_path", help="Path to input traffic or rider image")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for helmet detection (default: 0.25)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results formatted as JSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to save annotated output image",
    )
    args = parser.parse_args()

    try:
        detections = detect_helmets(args.image_path, conf_threshold=args.conf)
        if args.json:
            print(json.dumps(detections, indent=2))
        else:
            if not detections:
                print("No helmets or riders detected.")
            else:
                for idx, d in enumerate(detections, start=1):
                    print(
                        f"[{idx}] Status: {d['label']} | Confidence: {d['confidence']} | Box: {d['box']}"
                    )

        if args.output and detections:
            annotated_img = cv2.imread(args.image_path)
            if annotated_img is not None:
                for d in detections:
                    x1, y1, x2, y2 = d["box"]
                    # Green for 'With Helmet', Red for 'Without Helmet'
                    color = (
                        (0, 200, 0)
                        if "with" in d["label"].lower()
                        and "without" not in d["label"].lower()
                        else (0, 0, 220)
                    )
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{d['label']} ({d['confidence']})"
                    cv2.putText(
                        annotated_img,
                        label_text,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )
                out_dir = os.path.dirname(os.path.abspath(args.output))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(args.output, annotated_img)
                print(f"Annotated image saved to: {args.output}")
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
