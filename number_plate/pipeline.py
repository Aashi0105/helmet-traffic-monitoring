import argparse
import json
import os
import sys
from typing import Union

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from number_plate.detector import detect_plates
    from number_plate.ocr import read_plate_text
else:
    from .detector import detect_plates
    from .ocr import read_plate_text


def process_traffic_image(
    image_input: Union[str, np.ndarray], conf_threshold: float = 0.25
) -> list[dict]:
    """Execute the end-to-end number plate pipeline: detection followed by OCR.

    Args:
        image_input: Path to full traffic image (str) or loaded BGR image (np.ndarray).
        conf_threshold: Detector confidence threshold.

    Returns:
        List of dicts, each containing:
            - 'plate_box': [x1, y1, x2, y2] bounding box coordinates in original image
            - 'detector_confidence': float confidence from detector
            - 'plate_text': consolidated recognized plate text (string)
            - 'ocr_details': list of raw detection dicts with confidence and local text boxes
    """
    detected_plates = detect_plates(image_input, conf_threshold=conf_threshold)

    results: list[dict] = []
    for plate in detected_plates:
        crop = plate["crop"]
        ocr_results = read_plate_text(crop)

        # Consolidate recognized text tokens into a single clean string
        recognized_tokens = [item["text"] for item in ocr_results if item.get("text")]
        consolidated_text = " ".join(recognized_tokens).strip()

        results.append(
            {
                "plate_box": plate["box"],
                "detector_confidence": plate["confidence"],
                "plate_text": consolidated_text,
                "ocr_details": ocr_results,
            }
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end traffic number-plate detector and OCR reader."
    )
    parser.add_argument("image_path", help="Path to traffic or vehicle image")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
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
        help="Optional path to save annotated output image with detected boxes and text",
    )
    args = parser.parse_args()

    try:
        pipeline_results = process_traffic_image(args.image_path, conf_threshold=args.conf)
        if args.json:
            print(json.dumps(pipeline_results, indent=2))
        else:
            if not pipeline_results:
                print("No number plates detected.")
            else:
                for idx, item in enumerate(pipeline_results, start=1):
                    text_display = item["plate_text"] if item["plate_text"] else "(no text read)"
                    print(
                        f"[{idx}] Plate Box: {item['plate_box']} | "
                        f"Detector Conf: {item['detector_confidence']} | "
                        f"Text: '{text_display}'"
                    )

        if args.output and pipeline_results:
            annotated_img = cv2.imread(args.image_path)
            if annotated_img is not None:
                for item in pipeline_results:
                    x1, y1, x2, y2 = item["plate_box"]
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = (
                        f"{item['plate_text']} ({item['detector_confidence']})"
                        if item["plate_text"]
                        else f"Plate ({item['detector_confidence']})"
                    )
                    cv2.putText(
                        annotated_img,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
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
