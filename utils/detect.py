# utils/detect.py

import os
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2
from shutil import copyfile

# --- Config ---
HF_MODEL_REPO = "Anbhigya/ppe-detector-model"
HF_MODEL_FILENAME = "best.pt"
MODEL_CACHE_DIR = Path("model_cache")
# Default labels — change these if your model uses different names
DEFAULT_PPE_ITEMS = ["helmet", "vest", "gloves", "goggles", "mask"]

_model: YOLO = None


def _ensure_model() -> YOLO:
    global _model
    if _model is not None:
        return _model

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = MODEL_CACHE_DIR / HF_MODEL_FILENAME
    if not local_path.exists():
        downloaded = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME)
        try:
            os.symlink(downloaded, str(local_path))
        except Exception:
            copyfile(downloaded, str(local_path))

    _model = YOLO(str(local_path))
    return _model


def _box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _to_int_xyxy(xy):
    return int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])


def _normalize_label(s: str) -> str:
    return s.strip().lower()


def detect_ppe_image(uploaded_file_or_pil, selected_items: List[str]) -> Tuple[Image.Image, Dict[str, int], int, int]:
    """
    Detect PPE in an image but:
      - annotate only the user-selected PPE classes
      - compute missing counts per selected item (per person)
      - return annotated PIL image, missing_counts dict, total_violators, person_count
    """
    model = _ensure_model()

    # Normalize input to numpy RGB
    if hasattr(uploaded_file_or_pil, "read"):
        uploaded_file_or_pil.seek(0)
        image = Image.open(uploaded_file_or_pil).convert("RGB")
    elif isinstance(uploaded_file_or_pil, Image.Image):
        image = uploaded_file_or_pil.convert("RGB")
    else:
        image = Image.open(uploaded_file_or_pil).convert("RGB")

    img_np = np.array(image)  # RGB

    # Run model inference
    results = model(img_np)
    res = results[0]

    # Get model label mapping (index -> name)
    model_names = getattr(model, "names", None) or getattr(res, "names", None) or {}

    boxes = getattr(res, "boxes", None)
    if boxes is None:
        # nothing to draw
        return image, {item: 0 for item in selected_items}, 0, 0

    # Extract coordinates and classes defensively
    try:
        xyxy_arr = boxes.xyxy.cpu().numpy()  # shape (N,4)
        cls_arr = boxes.cls.cpu().numpy().astype(int)  # shape (N,)
    except Exception:
        # fallback to boxes.data
        data = boxes.data.cpu().numpy()
        xyxy_arr = data[:, :4]
        cls_arr = data[:, -1].astype(int)

    # Build lists: persons and ppe boxes (dict label -> list of bboxes)
    persons = []  # list of xyxy arrays for person bboxes
    ppe_boxes = {}  # label -> list of xyxy arrays

    for i, cls_idx in enumerate(cls_arr):
        label = str(model_names.get(int(cls_idx), str(int(cls_idx)))).lower()
        xyxy = xyxy_arr[i]
        if label in ("person", "people", "worker"):
            persons.append(xyxy)
        else:
            ppe_boxes.setdefault(label, []).append(xyxy)

    person_count = len(persons)

    # Prepare result counters
    missing_counts = {item: 0 for item in selected_items}
    violator_flags = [False] * max(0, person_count)

    # If there are no persons detected, we treat person_count = 0 and return zero violators.
    # (Alternative behaviors could be implemented if desired.)
    if person_count == 0:
        # Still draw the selected PPE boxes if present (use original annotated picture but filtered)
        # We'll create a clean annotation showing only selected PPE boxes.
        img_draw = img_np.copy()
        # iterate through detected boxes and draw only those matching selected_items
        for i, cls_idx in enumerate(cls_arr):
            label = str(model_names.get(int(cls_idx), str(int(cls_idx)))).lower()
            # check if label matches any selected item (substring tolerant)
            matches = any((si.lower() in label or label in si.lower()) for si in selected_items)
            if matches:
                x1, y1, x2, y2 = map(int, xyxy_arr[i])
                color = (255, 165, 0)  # orange BGR later converted
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), (int(color[2]), int(color[1]), int(color[0])), 2)
                cv2.putText(img_draw, label, (x1, max(16, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (int(color[2]), int(color[1]), int(color[0])), 1)
        annotated_pil = Image.fromarray(img_draw)
        return annotated_pil, missing_counts, 0, 0

    # For each person, check for presence of each selected PPE item
    # We match PPE box to person if the PPE box center lies inside the person's bbox.
    for p_idx, p_xyxy in enumerate(persons):
        px1, py1, px2, py2 = p_xyxy
        for item in selected_items:
            item_lower = _normalize_label(item)
            present = False
            # candidate labels in ppe_boxes that roughly match item_lower
            candidate_names = [name for name in ppe_boxes.keys() if (item_lower in name) or (name in item_lower) or (item_lower.replace(" ", "") in name)]
            # check any box of candidate names
            for cname in candidate_names:
                for b in ppe_boxes.get(cname, []):
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

    # Now build annotated image: draw persons (green if compliant, red if violator),
    # and draw PPE boxes only for selected_items (and label them)
    annotated_rgb = img_np.copy()
    # Draw PPE boxes (selected only)
    for label, boxes_list in ppe_boxes.items():
        # decide if label matches any selected item
        matches_selected = any((si.lower() in label) or (label in si.lower()) or (si.lower().replace(" ", "") in label) for si in selected_items)
        if not matches_selected:
            continue
        for b in boxes_list:
            x1, y1, x2, y2 = map(int, b)
            # choose color (BGR) for PPE boxes
            color = (230, 130, 0)  # orange (BGR as used by cv2 when writing later)
            # annotated_rgb is RGB; cv2 expects BGR when drawing directly, so convert color accordingly below
            cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), (int(color[2]), int(color[1]), int(color[0])), 2)
            cv2.putText(annotated_rgb, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (int(color[2]), int(color[1]), int(color[0])), 1)

    # Draw persons with compliance color
    for p_idx, p_xyxy in enumerate(persons):
        x1, y1, x2, y2 = map(int, p_xyxy)
        compliant = not violator_flags[p_idx]
        color = (0, 255, 0) if compliant else (255, 0, 0)  # RGB: green/red
        # cv2 uses BGR when drawing, so reverse color order
        cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), (int(color[2]), int(color[1]), int(color[0])), 2)
        label = "person (OK)" if compliant else "person (VIOLATOR)"
        cv2.putText(annotated_rgb, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(color[2]), int(color[1]), int(color[0])), 2)

    # Convert back to PIL
    annotated_pil = Image.fromarray(annotated_rgb.astype("uint8"))

    return annotated_pil, missing_counts, total_violators, person_count


def detect_ppe_video(input_video_path: str, output_video_path: str, selected_items: List[str]) -> Tuple[str, Dict[str, int], int, int]:
    """
    Process a video: annotate frames and compute counts (frame-approx).
    Returns output path and counts similar to image function.
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

    missing_counts = {item: 0 for item in selected_items}
    violator_events = 0
    persons_seen = 0

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1
        # convert to RGB for model
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        res = results[0]

        model_names = getattr(model, "names", None) or getattr(res, "names", None) or {}
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

        # Collect persons and ppe boxes
        persons = []
        ppe_boxes = {}
        for i, cls_idx in enumerate(cls_arr):
            label = str(model_names.get(int(cls_idx), str(int(cls_idx)))).lower()
            xyxy = xyxy_arr[i]
            if label in ("person", "people", "worker"):
                persons.append(xyxy)
            else:
                ppe_boxes.setdefault(label, []).append(xyxy)

        persons_count = len(persons)
        persons_seen += persons_count

        # per-frame person-level check
        violator_flags_frame = [False] * persons_count
        for p_idx, p_xyxy in enumerate(persons):
            px1, py1, px2, py2 = p_xyxy
            for item in selected_items:
                item_lower = _normalize_label(item)
                present = False
                candidate_names = [name for name in ppe_boxes.keys() if (item_lower in name) or (name in item_lower) or (item_lower.replace(" ", "") in name)]
                for cname in candidate_names:
                    for b in ppe_boxes.get(cname, []):
                        cx, cy = _box_center(b)
                        if (cx >= px1) and (cx <= px2) and (cy >= py1) and (cy <= py2):
                            present = True
                            break
                    if present:
                        break
                if not present:
                    missing_counts[item] += 1
                    violator_flags_frame[p_idx] = True

        violator_events += sum(1 for v in violator_flags_frame if v)

        # Annotate frame manually: draw selected PPE boxes and person boxes colored
        annotated_rgb = frame_rgb.copy()
        # PPE boxes (selected only)
        for label, boxes_for_label in ppe_boxes.items():
            matches_selected = any((si.lower() in label) or (label in si.lower()) or (si.lower().replace(" ", "") in label) for si in selected_items)
            if not matches_selected:
                continue
            for b in boxes_for_label:
                x1, y1, x2, y2 = map(int, b)
                color = (230, 130, 0)  # RGB-ish; cv2 takes BGR so reversed
                cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), (int(color[2]), int(color[1]), int(color[0])), 2)
                cv2.putText(annotated_rgb, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (int(color[2]), int(color[1]), int(color[0])), 1)

        # Person boxes
        for p_idx, p_xyxy in enumerate(persons):
            x1, y1, x2, y2 = map(int, p_xyxy)
            compliant = not violator_flags_frame[p_idx]
            color = (0, 255, 0) if compliant else (255, 0, 0)
            cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), (int(color[2]), int(color[1]), int(color[0])), 2)
            label = "person (OK)" if compliant else "person (VIOLATOR)"
            cv2.putText(annotated_rgb, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (int(color[2]), int(color[1]), int(color[0])), 1)

        # write frame (convert RGB -> BGR)
        annotated_bgr = cv2.cvtColor(annotated_rgb.astype("uint8"), cv2.COLOR_RGB2BGR)
        out.write(annotated_bgr)

    cap.release()
    out.release()

    return output_video_path, missing_counts, violator_events, persons_seen

