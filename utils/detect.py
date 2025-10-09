# utils/detect.py
import os
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
from PIL import Image
import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ----- CONFIG -----
HF_MODEL_REPO = "Anbhigya/ppe-detector-model"   # change if different
HF_MODEL_FILENAME = "best.pt"
# default PPE labels (human-friendly). If your model labels differ (e.g. "hardhat"), add synonyms below.
DEFAULT_PPE_ITEMS = ["helmet", "vest", "gloves", "goggles", "mask"]

# synonyms mapping (optional, helps match different label names)
PPE_SYNONYMS = {
    "helmet": ["helmet", "hardhat", "safetyhelmet"],
    "vest": ["vest", "safety vest", "high-visibility vest", "hi-vis"],
    "gloves": ["gloves", "glove"],
    "goggles": ["goggles", "safety glasses", "glasses"],
    "mask": ["mask", "face mask"]
}

# ---- internal singletons ----
_model: YOLO = None
_model_path: str = ""


def _ensure_model() -> YOLO:
    """Download (if required) and return a YOLO model instance (singleton)."""
    global _model, _model_path
    if _model is not None:
        return _model

    # Try to download the model using hf_hub_download (this will use HF cache)
    try:
        downloaded = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME)
        _model_path = str(downloaded)
    except Exception as e:
        # If download failed, try to use a local fallback path (model_cache/best.pt)
        fallback = Path("model_cache") / HF_MODEL_FILENAME
        if fallback.exists():
            _model_path = str(fallback)
        else:
            raise RuntimeError(
                f"Could not download model from HF hub and no local fallback found: {e}"
            )

    # instantiate YOLO
    _model = YOLO(_model_path)
    return _model


def _box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _normalize_label(s: str) -> str:
    return s.strip().lower()


def _label_matches_item(label: str, item: str) -> bool:
    """
    Returns True if a detected label (from model) should be considered as the PPE item.
    Uses synonyms mapping and substring matching defensively.
    """
    lab = _normalize_label(label)
    item_low = _normalize_label(item)
    # direct match / substring
    if item_low in lab or lab in item_low:
        return True
    # synonyms
    syns = PPE_SYNONYMS.get(item_low, [])
    for s in syns:
        if s in lab or lab in s:
            return True
    return False


def detect_ppe_image(uploaded_file_or_pil, selected_items: List[str]) -> Tuple[Image.Image, Dict[str, int], int, int]:
    """
    Detect PPE on a single image.

    Args:
        uploaded_file_or_pil: file-like (stream), PIL.Image, or path
        selected_items: list of items (strings) user selected e.g. ["helmet","vest"]

    Returns:
        annotated_pil_image, missing_counts_dict, total_violators, person_count
    """
    model = _ensure_model()

    # Normalize input to PIL -> numpy (RGB)
    if hasattr(uploaded_file_or_pil, "read"):
        uploaded_file_or_pil.seek(0)
        image = Image.open(uploaded_file_or_pil).convert("RGB")
    elif isinstance(uploaded_file_or_pil, Image.Image):
        image = uploaded_file_or_pil.convert("RGB")
    else:
        image = Image.open(uploaded_file_or_pil).convert("RGB")

    img_np = np.array(image)  # RGB numpy

    # Run inference
    results = model(img_np)
    res = results[0]

    # get model names mapping (index -> label)
    model_names = getattr(res, "names", None) or getattr(model, "names", None) or {}

    boxes = getattr(res, "boxes", None)
    if boxes is None:
        # nothing detected
        return image, {it: 0 for it in selected_items}, 0, 0

    # extract xyxy and class ids robustly
    try:
        xyxy_arr = boxes.xyxy.cpu().numpy()  # (N,4)
        cls_arr = boxes.cls.cpu().numpy().astype(int)  # (N,)
    except Exception:
        # try boxes.data fallback: [x1,y1,x2,y2,score,class]
        data = boxes.data.cpu().numpy()
        xyxy_arr = data[:, :4]
        cls_arr = data[:, -1].astype(int)

    # collect persons and PPE boxes
    persons = []  # list of xyxy arrays
    ppe_boxes = {}  # label -> list of xyxy arrays

    for i, cls_idx in enumerate(cls_arr):
        label = str(model_names.get(int(cls_idx), str(int(cls_idx)))).lower()
        xyxy = xyxy_arr[i]
        if label in ("person", "people", "worker"):
            persons.append(xyxy)
        else:
            ppe_boxes.setdefault(label, []).append(xyxy)

    person_count = len(persons)
    # initialize counters
    missing_counts = {item: 0 for item in selected_items}
    violator_flags = [False] * max(0, person_count)

    # If no persons detected: draw only selected PPE boxes (if any), return zero persons/violators
    if person_count == 0:
        annotated_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        for label, blist in ppe_boxes.items():
            # draw only selected items
            matches_any = any(_label_matches_item(label, it) for it in selected_items)
            if not matches_any:
                continue
            for b in blist:
                x1, y1, x2, y2 = map(int, b)
                color = (0, 140, 255)  # BGR orange
                cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_bgr, label, (x1, max(16, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_rgb), missing_counts, 0, 0

    # For each person, check selected items presence (by seeing if any PPE box center lies inside person bbox)
    for p_idx, p_xyxy in enumerate(persons):
        px1, py1, px2, py2 = p_xyxy
        for item in selected_items:
            present = False
            # candidate PPE labels from model that roughly match 'item'
            for label, boxes_list in ppe_boxes.items():
                if not _label_matches_item(label, item):
                    continue
                for b in boxes_list:
                    cx, cy = _box_center(b)
                    if (cx >= px1) and (cx <= px2) and (cy >= py1) and (cy <= py2):
                        present = True
                        break
                if present:
                    break
            if not present:
                missing_counts[item] += 1
                violator_flags[p_idx] = True

    total_violators = int(sum(1 for v in violator_flags if v))

    # Build annotated image: draw only selected PPE boxes and persons (green if compliant, red if violator)
    annotated_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Draw PPE boxes (selected only)
    for label, blist in ppe_boxes.items():
        matches_any = any(_label_matches_item(label, it) for it in selected_items)
        if not matches_any:
            continue
        for b in blist:
            x1, y1, x2, y2 = map(int, b)
            color = (0, 165, 255)  # BGR orange
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_bgr, label, (x1, max(16, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Draw person boxes
    for p_idx, p_xyxy in enumerate(persons):
        x1, y1, x2, y2 = map(int, p_xyxy)
        compliant = not violator_flags[p_idx]
        color = (0, 255, 0) if compliant else (0, 0, 255)  # green / red (BGR)
        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
        label = "person (OK)" if compliant else "person (VIOLATOR)"
        cv2.putText(annotated_bgr, label, (x1, max(16, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), missing_counts, total_violators, person_count


def detect_ppe_video(input_video_path: str, output_video_path: str, selected_items: List[str]) -> Tuple[str, Dict[str, int], int, int]:
    """
    Process a video file frame-by-frame. Returns output path and aggregated counts.
    Note: counts are frame-approx (same person across frames will be counted multiple times).
    """
    model = _ensure_model()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    missing_counts = {it: 0 for it in selected_items}
    violator_events = 0
    total_persons_seen = 0

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        res = results[0]

        model_names = getattr(res, "names", None) or getattr(model, "names", None) or {}

        boxes = getattr(res, "boxes", None)
        if boxes is None:
            out.write(frame_bgr)
            continue

        try:
            xyxy_arr = boxes.xyxy.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            data = boxes.data.cpu().numpy()
            xyxy_arr = data[:, :4]
            cls_arr = data[:, -1].astype(int)

        persons = []
        ppe_boxes = {}
        for i, cls_idx in enumerate(cls_arr):
            label = str(model_names.get(int(cls_idx), str(int(cls_idx)))).lower()
            xyxy = xyxy_arr[i]
            if label in ("person", "people", "worker"):
                persons.append(xyxy)
            else:
                ppe_boxes.setdefault(label, []).append(xyxy)

        total_persons_seen += len(persons)
        violator_flags = [False] * max(0, len(persons))

        # per-frame per-person check
        for p_idx, p_xyxy in enumerate(persons):
            px1, py1, px2, py2 = p_xyxy
            for item in selected_items:
                present = False
                for label, blist in ppe_boxes.items():
                    if not _label_matches_item(label, item):
                        continue
                    for b in blist:
                        cx, cy = _box_center(b)
                        if (cx >= px1) and (cx <= px2) and (cy >= py1) and (cy <= py2):
                            present = True
                            break
                    if present:
                        break
                if not present:
                    missing_counts[item] += 1
                    violator_flags[p_idx] = True

        violator_events += sum(1 for v in violator_flags if v)

        # annotate frame: draw selected PPE boxes and person boxes colorful
        annotated_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        for label, blist in ppe_boxes.items():
            matches_any = any(_label_matches_item(label, it) for it in selected_items)
            if not matches_any:
                continue
            for b in blist:
                x1, y1, x2, y2 = map(int, b)
                color = (0, 165, 255)
                cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_bgr, label, (x1, max(16, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        for p_idx, p_xyxy in enumerate(persons):
            x1, y1, x2, y2 = map(int, p_xyxy)
            compliant = not violator_flags[p_idx]
            color = (0, 255, 0) if compliant else (0, 0, 255)
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
            lbl = "person (OK)" if compliant else "person (VIOLATOR)"
            cv2.putText(annotated_bgr, lbl, (x1, max(16, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        out.write(annotated_bgr)

    cap.release()
    out.release()

    return output_video_path, missing_counts, violator_events, total_persons_seen

