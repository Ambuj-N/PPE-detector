# utils/detect.py
import os
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

# ----- CONFIG -----
HF_MODEL_REPO = "Anbhigya/ppe-detector-model"
HF_MODEL_FILENAME = "best.pt"
LOCAL_MODEL_PATH = Path("best.pt")  # if you uploaded best.pt to Colab or placed locally

# These are the friendly names shown to user (must match model's semantics)
SUPPORTED_ITEMS = ["Hardhat", "Mask", "Safety Vest"]

# Model labels you discovered:
# {0: 'Hardhat', 1: 'Mask', 2: 'NO-Hardhat', 3: 'NO-Mask', 4: 'NO-Safety Vest', 5: 'Person', ...}
PERSON_LABELS = {"person", "Person", "PERSON"}  # we compare case-insensitively

# singletons
_model: YOLO = None
_model_path: str = ""


def _ensure_model() -> YOLO:
    """Load YOLO model from local path if present else download from HF hub (if available)."""
    global _model, _model_path
    if _model is not None:
        return _model

    # prefer local uploaded model
    if LOCAL_MODEL_PATH.exists():
        _model_path = str(LOCAL_MODEL_PATH)
    else:
        # try HF download
        if hf_hub_download is None:
            raise RuntimeError("Local model not found and huggingface_hub is not available to download.")
        _model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME)

    # instantiate YOLO (ultralytics handles object unpickling etc.)
    # this will choose CPU by default; inference device set at call-time
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


def detect_ppe_image(uploaded_file_or_pil, selected_items: List[str]) -> Tuple[Image.Image, Dict[str, int], int, int]:
    """
    Detect PPE in a single image.

    Returns:
        annotated_pil, missing_counts (per selected item), total_violators, person_count
    """
    model = _ensure_model()
    device = "cuda" if model.device and str(model.device).lower().startswith("cuda") else "cpu"

    # normalize input to PIL.Image then to numpy RGB
    if hasattr(uploaded_file_or_pil, "read"):
        uploaded_file_or_pil.seek(0)
        pil = Image.open(uploaded_file_or_pil).convert("RGB")
    elif isinstance(uploaded_file_or_pil, Image.Image):
        pil = uploaded_file_or_pil.convert("RGB")
    else:
        pil = Image.open(uploaded_file_or_pil).convert("RGB")

    img_np = np.array(pil)  # RGB

    # run inference (pass device explicitly)
    results = model(img_np, device=device)
    res = results[0]

    # get mapping index->name (strings)
    model_names = getattr(res, "names", None) or getattr(model, "names", None) or {}
    # normalize keys to int->str
    model_names = {int(k): str(v) for k, v in model_names.items()}

    boxes = getattr(res, "boxes", None)
    if boxes is None:
        # nothing detected
        return pil, {it: 0 for it in selected_items}, 0, 0

    # extract arrays robustly
    try:
        xyxy_arr = boxes.xyxy.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy().astype(int)
    except Exception:
        data = boxes.data.cpu().numpy()
        xyxy_arr = data[:, :4]
        cls_arr = data[:, -1].astype(int)

    # collect detections
    persons = []  # list of (xyxy)
    other_dets = []  # list of (label, xyxy)
    for i, cls_idx in enumerate(cls_arr):
        label = model_names.get(int(cls_idx), str(int(cls_idx)))
        xyxy = xyxy_arr[i].astype(float)
        # treat 'person' (case-insensitive) as person
        if _normalize(label) == "person" or _normalize(label) == "people" or _normalize(label) == "person ":
            persons.append(xyxy)
        else:
            other_dets.append((label, xyxy))

    person_count = len(persons)
    missing_counts = {it: 0 for it in selected_items}
    violator_flags = [False] * max(0, person_count)

    # If no persons, still draw selected PPE boxes (positive and NO ones) and return zeros for persons
    if person_count == 0:
        annotated_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        for label, b in other_dets:
            # draw only boxes that match selected items or NO-selected items
            for sel in selected_items:
                sel_norm = _normalize(sel)
                # check if label corresponds (either positive or NO-)
                lab_norm = _normalize(label).replace("_", " ").replace("-", " ")
                sel_norm_comp = _normalize(sel).replace(" ", "")
                if sel_norm_comp in lab_norm.replace(" ", ""):
                    x1, y1, x2, y2 = map(int, b)
                    color = (0, 165, 255)  # orange (BGR)
                    cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_bgr, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    break
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_rgb), missing_counts, 0, 0

    # For each person, check each selected item
    for p_idx, p_box in enumerate(persons):
        px1, py1, px2, py2 = p_box
        for sel in selected_items:
            sel_norm = _normalize(sel)
            present = False
            explicit_no = False

            # check detections that correspond to this sel item
            for label, b in other_dets:
                lab = label.strip()
                lab_low = _normalize(lab)
                # unify no- prefix checks
                # Examples from your model: 'Hardhat', 'NO-Hardhat', 'Mask', 'NO-Mask', 'NO-Safety Vest', 'Safety Vest'
                # Normalize by removing spaces and punctuation for comparisons
                lab_comp = lab_low.replace(" ", "").replace("-", "").replace("_", "")
                sel_comp = sel_norm.replace(" ", "").replace("-", "").replace("_", "")
                if sel_comp in lab_comp:
                    # does this bbox lie inside current person bbox?
                    cx, cy = _box_center(b)
                    if _point_in_box(cx, cy, (px1, py1, px2, py2)):
                        # if label contains 'no' or startswith 'no', treat as explicit missing
                        if lab_low.startswith("no") or lab_low.startswith("no-") or lab_low.startswith("no_"):
                            explicit_no = True
                        else:
                            present = True
            # Decide status for this person & selected item
            if present:
                # person has the item → ok
                pass
            elif explicit_no:
                # explicit NO label inside person → missing
                missing_counts[sel] += 1
                violator_flags[p_idx] = True
            else:
                # neither positive nor explicit NO detected inside person → treat as missing (conservative)
                missing_counts[sel] += 1
                violator_flags[p_idx] = True

    total_violators = int(sum(1 for v in violator_flags if v))

    # Build annotated image: draw only selected items' boxes + person boxes (green if OK, red if violator)
    annotated_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # draw selected item boxes (both positive and NO variations)
    for label, b in other_dets:
        lab_low = _normalize(label)
        lab_comp = lab_low.replace(" ", "").replace("-", "").replace("_", "")
        draw_this = False
        for sel in selected_items:
            sel_comp = _normalize(sel).replace(" ", "").replace("-", "").replace("_", "")
            if sel_comp in lab_comp:
                draw_this = True
                break
        if not draw_this:
            continue
        x1, y1, x2, y2 = map(int, b)
        # color positive vs NO
        if lab_low.startswith("no"):
            color = (0, 0, 255)  # red for NO-* (BGR)
        else:
            color = (0, 165, 255)  # orange for positive (BGR)
        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_bgr, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # draw persons (green if fully compliant for selected items, red otherwise)
    for p_idx, p_box in enumerate(persons):
        x1, y1, x2, y2 = map(int, p_box)
        compliant = not violator_flags[p_idx]
        color = (0, 255, 0) if compliant else (0, 0, 255)  # BGR
        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
        label = "person (OK)" if compliant else "person (VIOLATOR)"
        cv2.putText(annotated_bgr, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), missing_counts, total_violators, person_count


def detect_ppe_video(input_video_path: str, output_video_path: str, selected_items: List[str]) -> Tuple[str, Dict[str, int], int, int]:
    """
    Process a video frame-by-frame. Counts are frame-approximate.
    """
    model = _ensure_model()
    device = "cuda" if model.device and str(model.device).lower().startswith("cuda") else "cpu"

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
    persons_seen = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, device=device)
        res = results[0]

        model_names = getattr(res, "names", None) or getattr(model, "names", None) or {}
        model_names = {int(k): str(v) for k, v in model_names.items()}

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
        other_dets = []
        for i, cls_idx in enumerate(cls_arr):
            label = model_names.get(int(cls_idx), str(int(cls_idx)))
            xyxy = xyxy_arr[i].astype(float)
            if _normalize(label) == "person":
                persons.append(xyxy)
            else:
                other_dets.append((label, xyxy))

        persons_seen += len(persons)
        violator_flags = [False] * max(0, len(persons))

        # per-frame check
        for p_idx, p_box in enumerate(persons):
            px1, py1, px2, py2 = p_box
            for sel in selected_items:
                present = False
                explicit_no = False
                for label, b in other_dets:
                    lab_low = _normalize(label)
                    lab_comp = lab_low.replace(" ", "").replace("-", "").replace("_", "")
                    sel_comp = _normalize(sel).replace(" ", "").replace("-", "").replace("_", "")
                    if sel_comp in lab_comp:
                        cx, cy = _box_center(b)
                        if _point_in_box(cx, cy, (px1, py1, px2, py2)):
                            if lab_low.startswith("no"):
                                explicit_no = True
                            else:
                                present = True
                if present:
                    pass
                elif explicit_no:
                    missing_counts[sel] += 1
                    violator_flags[p_idx] = True
                else:
                    missing_counts[sel] += 1
                    violator_flags[p_idx] = True

        violator_events += sum(1 for v in violator_flags if v)

        # annotate frame: draw selected boxes and person boxes
        annotated_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        for label, b in other_dets:
            lab_low = _normalize(label)
            lab_comp = lab_low.replace(" ", "").replace("-", "").replace("_", "")
            draw_this = False
            for sel in selected_items:
                sel_comp = _normalize(sel).replace(" ", "").replace("-", "").replace("_", "")
                if sel_comp in lab_comp:
                    draw_this = True
                    break
            if not draw_this:
                continue
            x1, y1, x2, y2 = map(int, b)
            color = (0, 0, 255) if lab_low.startswith("no") else (0, 165, 255)
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_bgr, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        for p_idx, p_box in enumerate(persons):
            x1, y1, x2, y2 = map(int, p_box)
            compliant = not violator_flags[p_idx]
            color = (0, 255, 0) if compliant else (0, 0, 255)
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
            lbl = "person (OK)" if compliant else "person (VIOLATOR)"
            cv2.putText(annotated_bgr, lbl, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        out.write(annotated_bgr)

    cap.release()
    out.release()
    return output_video_path, missing_counts, violator_events, persons_seen

