from pathlib import Path
import base64
import uuid

import cv2
import numpy as np
from ultralytics import YOLO

from damage_labels import get_label


MODEL_PATH = (
    Path(__file__).parent.parent
    / "models"
    / "best.pt"
)

RESULTS_DIR = (
    Path(__file__).parent.parent
    / "data"
    / "results"
)

model = YOLO(MODEL_PATH)


def draw_detections(image, detections):
    """Draw bounding boxes + friendly labels onto an image for user-facing evidence.

    Accepts a PIL Image, returns an annotated OpenCV (BGR) numpy array.
    """
    image_np = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        label_text = f'{det["label"]} {det["confidence"] * 100:.0f}%'

        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (text_w, text_h), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_y = max(y1, text_h + 8)
        cv2.rectangle(
            image_np,
            (x1, label_y - text_h - 8),
            (x1 + text_w + 6, label_y),
            (0, 255, 0),
            -1
        )
        cv2.putText(
            image_np, label_text, (x1 + 3, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

    return image_np


def save_annotated_image(image_np, original_filename):
    """Save an annotated image to backend/data/results and return its path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}_{Path(original_filename).stem}.jpg"
    output_path = RESULTS_DIR / unique_name

    cv2.imwrite(str(output_path), image_np)

    return output_path


def encode_image_base64(image_np):
    """Encode an OpenCV (BGR) image as a base64 JPEG string for JSON responses."""
    success, buffer = cv2.imencode(".jpg", image_np)
    if not success:
        raise ValueError("Failed to encode annotated image")
    return base64.b64encode(buffer).decode("utf-8")


def extract_detections(result):
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class": class_name,
            "label": get_label(class_name),
            "confidence": confidence,
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        })

    return detections


def detect_image(image, filename="image.jpg", conf=0.35):

    results = model.predict(
        source=image,
        conf=conf,
        device=0,
        verbose=True
    )

    detections = extract_detections(results[0])

    annotated_np = draw_detections(image, detections)
    evidence_path = save_annotated_image(annotated_np, filename)
    annotated_image_base64 = encode_image_base64(annotated_np)

    return {
        "type": "image",
        "detections": detections,
        "evidence_path": str(evidence_path),
        "annotated_image_base64": annotated_image_base64
    }


def detect_video(video_path, conf=0.35):

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise ValueError("Could not open video")

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    frame_number = 0

    while True:

        success, frame = video.read()

        if not success:
            break

        results = model.predict(
            source=frame,
            conf=conf,
            device=0,
            verbose=False
        )

        frames.append({
            "frame": frame_number,
            "timestamp": frame_number / fps,
            "detections": extract_detections(results[0])
        })

        frame_number += 1

    video.release()

    return {
        "type": "video",
        "fps": fps,
        "frame_count": frame_count,
        "frames": frames
    }