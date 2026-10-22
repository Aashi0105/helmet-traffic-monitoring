import os
import sys
import time
import uuid
from typing import Optional

import cv2
import numpy as np
from sqlalchemy.orm import Session

# Add project root directory to path to access helmet and number_plate modules
ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.models import Violation

EVIDENCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "evidence",
)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Path reserved for when final trained weights arrive
FINAL_TRAINED_WEIGHTS_PATH = os.path.join(
    ROOT_DIR, "helmet", "weights", "helmet_trained_final.pt"
)


# ---------------------------------------------------------------------------
# Model Inference Interface / Adapters
# ---------------------------------------------------------------------------

class BaseHelmetDetector:
    """Base interface for helmet detection models."""

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25) -> list[dict]:
        raise NotImplementedError


class MockHelmetDetector(BaseHelmetDetector):
    """Simulated detector for testing backend endpoints, upload validation,

    and database storage without falsely presenting results as genuine AI predictions.
    """

    def __init__(self, test_detections: Optional[list[dict]] = None):
        self.test_detections = test_detections or []

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25) -> list[dict]:
        """Returns an empty helmet detection list in mock mode unless valid test coordinates

        are explicitly supplied. Never implies a fabricated location is a genuine detection.
        """
        return list(self.test_detections)


class LiveHelmetDetector(BaseHelmetDetector):
    """Executes helmet detection using local YOLOv8 weights or final trained weights."""

    def __init__(self, weights_path: Optional[str] = None):
        from helmet.detector import detect_helmets
        self._detect_fn = detect_helmets
        self.weights_path = weights_path

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25) -> list[dict]:
        return self._detect_fn(
            image,
            conf_threshold=conf_threshold,
            weights_path=self.weights_path,
        )


def get_helmet_detector(
    force_mock: bool = False,
    test_detections: Optional[list[dict]] = None,
) -> tuple[BaseHelmetDetector, str, bool]:
    """Factory function returning the active detector, status message, and mock flag.

    - When final trained weights exist at FINAL_TRAINED_WEIGHTS_PATH, they are used.
    - If force_mock is requested or final model is not yet trained, mock/interim mode is used.
    """
    if force_mock:
        return MockHelmetDetector(test_detections=test_detections), "mock_inference_mode", True

    if os.path.exists(FINAL_TRAINED_WEIGHTS_PATH):
        return LiveHelmetDetector(FINAL_TRAINED_WEIGHTS_PATH), "final_trained_model_active", False

    # Default to available interim weights or fallback cleanly
    try:
        detector = LiveHelmetDetector()
        return detector, "interim_weights_unverified", False
    except Exception:
        return MockHelmetDetector(test_detections=test_detections), "mock_fallback_mode", True


# ---------------------------------------------------------------------------
# Visual Annotation & Pipeline
# ---------------------------------------------------------------------------

def annotate_frame(
    image: np.ndarray,
    helmet_detections: list[dict],
    plate_detections: list[dict],
    is_mock: bool = False,
) -> np.ndarray:
    """Draw bounding boxes and labels for detected helmets and license plates."""
    annotated = image.copy()

    # Draw helmet / head detections
    for item in helmet_detections:
        x1, y1, x2, y2 = item["box"]
        label = item["label"]
        conf = item["confidence"]
        is_compliant = "with" in label.lower() and "without" not in label.lower()

        color = (0, 210, 0) if is_compliant else (0, 0, 220)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        prefix = "[MOCK] " if is_mock else ""
        clean_label = "With Helmet" if is_compliant else "Without Helmet"
        tag = f"{prefix}Helmet: {clean_label} ({conf:.2f})"
        cv2.putText(
            annotated,
            tag,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    # Draw license plate detections
    for plate in plate_detections:
        x1, y1, x2, y2 = plate["plate_box"]
        plate_text = plate.get("plate_text") or "Plate"
        conf = plate.get("detector_confidence", 0.0)

        color = (255, 200, 0)  # Cyan/Yellow
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        tag = f"Number Plate: {plate_text} ({conf:.2f})"
        cv2.putText(
            annotated,
            tag,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return annotated


def analyze_traffic_image(
    image_bytes: bytes,
    original_filename: str,
    db: Session,
    detection_source: str = "camera_upload",
    conf_threshold: float = 0.2,
    force_mock: bool = False,
    test_detections: Optional[list[dict]] = None,
) -> dict:
    """Execute model inference workflow:

    1. Validates and decodes image bytes.
    2. Runs helmet detector (live or mock).
    3. Runs license-plate detector + EasyOCR.
    4. Evaluates compliance.
    5. Saves annotated evidence image.
    6. Persists structured record into PostgreSQL.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid or corrupted image format. Unable to decode image.")

    # 1. Helmet Detection
    detector, model_status, is_mock = get_helmet_detector(
        force_mock=force_mock,
        test_detections=test_detections,
    )
    helmet_detections = detector.predict(image, conf_threshold=conf_threshold)

    # 2. License Plate Detection & OCR
    plate_detections = []
    try:
        from number_plate.pipeline import process_traffic_image
        plate_detections = process_traffic_image(image, conf_threshold=conf_threshold)
    except Exception as err:
        # Graceful fallback if OCR is unavailable or fails
        plate_detections = []

    # 3. Compliance Decision Logic
    has_without_helmet = any(
        "without" in d["label"].lower() for d in helmet_detections
    )
    has_with_helmet = any(
        "with" in d["label"].lower() and "without" not in d["label"].lower()
        for d in helmet_detections
    )

    if has_without_helmet:
        helmet_detected = False
        review_status = "pending"
    elif has_with_helmet:
        helmet_detected = True
        review_status = "approved"
    else:
        helmet_detected = True
        review_status = "approved"

    # 4. Extract Primary License Plate Text
    recognized_plates = [
        p["plate_text"] for p in plate_detections if p.get("plate_text")
    ]
    primary_plate = recognized_plates[0] if recognized_plates else None

    # 5. Save Annotated Evidence File
    annotated = annotate_frame(image, helmet_detections, plate_detections, is_mock=is_mock)
    unique_suffix = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    evidence_filename = f"evidence_{unique_suffix}.jpg"
    evidence_disk_path = os.path.join(EVIDENCE_DIR, evidence_filename)
    cv2.imwrite(evidence_disk_path, annotated)

    evidence_url_path = f"/evidence/{evidence_filename}"

    # 6. Database Storage in existing PostgreSQL table
    source_tag = f"{detection_source}-mock" if is_mock else detection_source
    violation = Violation(
        number_plate=primary_plate,
        helmet_detected=helmet_detected,
        detection_source=source_tag,
        review_status=review_status,
        evidence_path=evidence_url_path,
    )
    db.add(violation)
    db.commit()
    db.refresh(violation)

    # 7. Summary Message
    if is_mock:
        if has_without_helmet:
            summary = f"[Simulated Test] Simulated violation (no helmet). Plate: {primary_plate or 'None'}"
        elif has_with_helmet:
            summary = f"[Simulated Test] Simulated compliant. Plate: {primary_plate or 'None'}"
        else:
            summary = f"[Simulated Test] Model training in progress (no helmet detected). Plate: {primary_plate or 'None'}"
    elif not helmet_detected:
        summary = f"Potential violation: Rider without helmet. Plate: {primary_plate or 'Unreadable'}"
    elif has_with_helmet:
        summary = f"Compliant: Rider wearing helmet. Plate: {primary_plate or 'Unreadable'}"
    else:
        summary = f"No riders identified in frame. Plate: {primary_plate or 'None'}"

    return {
        "violation": {
            "id": violation.id,
            "number_plate": violation.number_plate,
            "helmet_detected": violation.helmet_detected,
            "detection_source": violation.detection_source,
            "review_status": violation.review_status,
            "evidence_path": violation.evidence_path,
            "created_at": violation.created_at,
        },
        "helmet_classification": [
            {
                "box": d["box"],
                "label": d["label"],
                "confidence": d["confidence"],
            }
            for d in helmet_detections
        ],
        "number_plate_detection": [
            {
                "plate_box": p["plate_box"],
                "confidence": p["detector_confidence"],
                "plate_text": p["plate_text"],
            }
            for p in plate_detections
        ],
        "ocr_output": {
            "primary_plate": primary_plate,
            "all_plates": recognized_plates,
        },
        "evidence_path": evidence_url_path,
        "is_mock_inference": is_mock,
        "model_status": model_status,
        "summary": summary,
    }
