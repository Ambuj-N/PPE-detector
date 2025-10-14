# utils/detect.py
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Callable

import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# Optional huggingface download
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

# ----- CONFIG -----
HF_MODEL_REPO = "Anbhigya/ppe-detector-model"
HF_MODEL_FILENAME = "yolo9s.pt"
LOCAL_MODEL_PATH = Path("yolo9s.pt")

# SH17 dataset label list (indices -> label)
ALL_MODEL_LABELS = {
    0: "Person",
    1: "Head",
    2: "Face",
    3: "Glasses",
    4: "Face-mask-medical",
    5: "Face-guard",
    6: "Ear",
    7: "Earmuffs",
    8: "Hands",
    9: "Gloves",
    10: "Foot",
    11: "Shoes",
    12: "Safety-vest",
    13: "Tools",
    14: "Helmet",
    15: "Medical-suit",
    16: "Safety-suit"
}

# Which items the UI will show as options (default set)
SUPPORTED_ITEMS = [
    "Helmet",
    "Gloves",
    "Safety-vest",
    "Face-mask-medical",
    "Earmuffs",
    "Shoes"
]

# SH17 does not use NO- labels so keep mapping empty
POSITIVE_NEGATIVE_MAP = {}

# internal singletons
_model: YOLO = None
_model_path: str = ""


def _ensure_model() -> YOLO:
    """Load YOLO model from local path if present else download from HF hub."""
    global _model, _model_path
    if _model is not None:
        return _model

    # Prefer local uploaded model
    if LOCAL_MODEL_PATH.exists():
        _model_path = str(LOCAL_MODEL_PATH)
    else:
        if hf_hub_download is None:
            raise RuntimeError(
                "Local model not found and huggingface_hub is not available to download."
            )
        _model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME)

    _model = YOLO(_model_path)
    return _model


def _box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0


def _point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


def _normalize(s: str) -> str:
    return s.strip().lower()


def detect_ppe_image(
    uploaded_file_or_pil,
    selected_detection_items: List[str],
    selected_warning_items: List[str],
    confidence_threshold: float = 0.5,
    draw_all_detections: bool = False
) -> Tuple[Image.Image, Dict[str, int], int, int, Dict[str, int]]:
    """
    Detect PPE in a single image.

    Returns:
        annotated_pil, missing_counts, total_violators, person_count, detection_summary
    """
    model = _ensure_model()

    # Convert input to PIL Image then to numpy RGB
    if hasattr(uploaded_file_or_pil, "read"):
        uploaded_file_or_pil.seek(0)
        pil = Image.open(uploaded_file_or_pil).convert("RGB")
    elif isinstance(uploaded_file_or_pil, Image.Image):
        pil = uploaded_file_or_pil.convert("RGB")
    else:
        pil = Image.open(uploaded_file_or_pil).convert("RGB")

    img_np = np.array(pil)  # RGB

    # Run inference
    results = model(img_np, conf=confidence_threshold)
    res = results[0]

    model_names = ALL_MODEL_LABELS

    boxes = getattr(res, "boxes", None)
    if boxes is None:
        # Nothing detected
        return pil, {it: 0 for it in selected_warning_items}, 0, 0, {}

    # Extract detection arrays (handles different ultralytics versions)
    try:
        xyxy_arr = boxes.xyxy.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy().astype(int)
        conf_arr = boxes.conf.cpu().numpy()
    except Exception:
        data = boxes.data.cpu().numpy()
        xyxy_arr = data[:, :4]
        # class might be last column
        cls_arr = data[:, -1].astype(int)
        conf_arr = data[:, 4] if data.shape[1] >= 5 else np.ones(len(cls_arr))

    # Collect detections with confidence filtering
    persons = []  # list of (xyxy, confidence)
    other_dets = []  # list of (label, xyxy, confidence)

    for i, cls_idx in enumerate(cls_arr):
        label = model_names.get(int(cls_idx), str(int(cls_idx)))
        confidence = float(conf_arr[i])
        xyxy = xyxy_arr[i].astype(float)

        if confidence < confidence_threshold:
            continue

        if _normalize(label) == "person":
            persons.append((xyxy, confidence))
        else:
            other_dets.append((label, xyxy, confidence))

    person_count = len(persons)
    missing_counts = {it: 0 for it in selected_warning_items}
    violator_flags = [False] * max(0, person_count)
    detection_summary = {}

    # Count all detections for summary
    for label, _, _ in other_dets:
        detection_summary[label] = detection_summary.get(label, 0) + 1

    # If no persons, just draw selected PPE boxes and return
    if person_count == 0:
        annotated_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        for label, b, confidence in other_dets:
            if not draw_all_detections:
                draw_item = False
                for sel in selected_detection_items:
                    if sel.lower() in label.lower():
                        draw_item = True
                        break
                if not draw_item:
                    continue

            x1, y1, x2, y2 = map(int, b)
            # color for items
            color = (0, 255, 0) if label in selected_detection_items else (0, 165, 255)
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label} {confidence:.2f}"
            cv2.putText(annotated_bgr, label_text, (x1, max(16, y1 - 6)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_rgb), missing_counts, 0, 0, detection_summary

    # For each person, check each selected warning item
    for p_idx, (p_box, _) in enumerate(persons):
        px1, py1, px2, py2 = p_box

        for warning_item in selected_warning_items:
            present = False

            # Check positive detections for this item
            positive_label = warning_item

            for label, b, confidence in other_dets:
                # match by substring to be flexible with label formatting
                if positive_label.lower() in label.lower():
                    cx, cy = _box_center(b)
                    if _point_in_box(cx, cy, (px1, py1, px2, py2)):
                        present = True
                        break

            # If not present, mark missing (conservative)
            if not present:
                missing_counts[warning_item] += 1
                violator_flags[p_idx] = True

    total_violators = int(sum(1 for v in violator_flags if v))

    # Build annotated image
    annotated_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Draw PPE item boxes
    for label, b, confidence in other_dets:
        if not draw_all_detections:
            draw_item = False
            for sel in selected_detection_items:
                if sel.lower() in label.lower():
                    draw_item = True
                    break
            if not draw_item:
                continue

        x1, y1, x2, y2 = map(int, b)
        # Color coding
        if label in SUPPORTED_ITEMS:
            color = (0, 255, 0)  # Green for compliant PPE
        else:
            color = (0, 165, 255)  # Orange for other items

        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
        label_text = f"{label} {confidence:.2f}"
        cv2.putText(annotated_bgr, label_text, (x1, max(16, y1 - 6)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Draw person boxes
    for p_idx, (p_box, confidence) in enumerate(persons):
        x1, y1, x2, y2 = map(int, p_box)
        compliant = not violator_flags[p_idx]
        color = (0, 255, 0) if compliant else (0, 0, 255)  # Green or Red

        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 3)
        status = "COMPLIANT" if compliant else "VIOLATOR"
        label_text = f"Person {status} {confidence:.2f}"
        cv2.putText(annotated_bgr, label_text, (x1, max(16, y1 - 6)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), missing_counts, total_violators, person_count, detection_summary


def detect_ppe_video(
    input_video_path: str,
    output_video_path: str,
    selected_detection_items: List[str],
    selected_warning_items: List[str],
    confidence_threshold: float = 0.5,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Tuple[str, Dict[str, int], int, int, Dict[str, int]]:
    """
    Process a video frame-by-frame for PPE detection.
    """
    model = _ensure_model()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    missing_counts = {it: 0 for it in selected_warning_items}
    violator_events = 0
    persons_seen = 0
    detection_summary = {}

    frame_count = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_count += 1
        if progress_callback and total_frames > 0:
            progress_callback(int((frame_count / total_frames) * 100))

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, conf=confidence_threshold)
        res = results[0]

        model_names = ALL_MODEL_LABELS

        boxes = getattr(res, "boxes", None)
        if boxes is None:
            out.write(frame_bgr)
            continue

        # Extract detection arrays
        try:
            xyxy_arr = boxes.xyxy.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy().astype(int)
            conf_arr = boxes.conf.cpu().numpy()
        except Exception:
            data = boxes.data.cpu().numpy()
            xyxy_arr = data[:, :4]
            cls_arr = data[:, -1].astype(int)
            conf_arr = data[:, 4] if data.shape[1] >= 5 else np.ones(len(cls_arr))

        persons = []
        other_dets = []

        for i, cls_idx in enumerate(cls_arr):
            label = model_names.get(int(cls_idx), str(int(cls_idx)))
            confidence = float(conf_arr[i])
            xyxy = xyxy_arr[i].astype(float)

            if confidence < confidence_threshold:
                continue

            if _normalize(label) == "person":
                persons.append((xyxy, confidence))
            else:
                other_dets.append((label, xyxy, confidence))
                detection_summary[label] = detection_summary.get(label, 0) + 1

        persons_seen += len(persons)
        violator_flags = [False] * max(0, len(persons))

        # Process each person in frame
        for p_idx, (p_box, _) in enumerate(persons):
            px1, py1, px2, py2 = p_box

            for warning_item in selected_warning_items:
                present = False
                positive_label = warning_item

                for label, b, _ in other_dets:
                    if positive_label.lower() in label.lower():
                        cx, cy = _box_center(b)
                        if _point_in_box(cx, cy, (px1, py1, px2, py2)):
                            present = True
                            break

                if not present:
                    missing_counts[warning_item] += 1
                    violator_flags[p_idx] = True

        violator_events += sum(1 for v in violator_flags if v)

        # Annotate frame
        annotated_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Draw PPE items
        for label, b, confidence in other_dets:
            draw_item = False
            for sel in selected_detection_items:
                if sel.lower() in label.lower():
                    draw_item = True
                    break
            if not draw_item:
                continue

            x1, y1, x2, y2 = map(int, b)

            if label in SUPPORTED_ITEMS:
                color = (0, 255, 0)  # Green
            else:
                color = (0, 165, 255)  # Orange

            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label} {confidence:.2f}"
            cv2.putText(annotated_bgr, label_text, (x1, max(16, y1 - 6)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Draw persons
        for p_idx, (p_box, confidence) in enumerate(persons):
            x1, y1, x2, y2 = map(int, p_box)
            compliant = not violator_flags[p_idx]
            color = (0, 255, 0) if compliant else (0, 0, 255)

            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 3)
            status = "OK" if compliant else "VIOLATOR"
            label_text = f"Person {status}"
            cv2.putText(annotated_bgr, label_text, (x1, max(16, y1 - 6)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(annotated_bgr)

    cap.release()
    out.release()

    return output_video_path, missing_counts, violator_events, persons_seen, detection_summary

