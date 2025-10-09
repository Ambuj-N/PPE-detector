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
# Default labels — you can change these to match your model's exact label strings
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


def _xyxy_to_int_tuple(xyxy):
    # xyxy is [x1, y1, x2, y2]
    return int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])


def _box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy


def detect_ppe_image(uploaded_file_or_pil, selected_items: List[str]) -> Tuple[Image.Image, Dict[str, int], int, int]:
    """
    Run PPE detection on an image.

    Args:
        uploaded_file_or_pil: file-like or PIL Image
        selected_items: list of PPE labels to check (strings)

    Returns:
        annotated_pil_image,
        missing_counts_per_item (dict),
        total_violators (int),
        person_count (int)
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

    # Get model label mapping
    model_names = getattr(model, "names", None) or getattr(res, "names", None) or {}

    # Extract boxes and classes
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        # fallback
        annotated_np = img_np
        return Image.fromarray(annotated_np), {k: 0 for k in selected_items}, 0, 0

    # Fetch arrays (be defensive)
    try:
        xyxy_arr = boxes.xyxy.cpu().numpy()  # shape (N,4)
        cls_arr = boxes.cls.cpu().numpy().astype(int)  # shape (N,)
    except Exception:
        # If attributes differ, try alternative
        try:
            data = boxes.data.cpu().numpy()  # last col = class
            xyxy_arr = data[:, :4]
            cls_arr = data[:, -1].astype(int)
        except Exception:
            xyxy_arr = np.array([])
            cls_arr = np.array([])

    # Build lists of boxes per class name
    persons = []  # list of (bbox_xyxy)
    ppe_boxes = {}  # class_name -> list of bbox_xyxy
    for i, cls_idx in enumerate(cls_arr):
        cls_name = str(model_names.get(int(cls_idx), str(int(cls_idx)))).lower()
        xyxy = xyxy_arr[i]
        if cls_name in ("person", "people", "worker"):
            persons.append(xyxy)
        else:
            ppe_boxes.setdefault(cls_name, []).append(xyxy)

    person_count = len(persons)

    # initialize missing counts
    missing_counts = {item: 0 for item in selected_items}
    total_violators = 0

    # If no persons detected, we'll attempt a fallback:
    if person_count == 0:
        # If there are any PPE detections then we cannot reliably associate to persons.
        # We'll consider there are zero persons for counting, and report zero violators.
        # (Alternative: treat whole image as one person — but that often gives false positives.)
        annotated = res.plot() if hasattr(res, "plot") else img_np
        annotated_pil = Image.fromarray(annotated.astype("uint8"))
        return annotated_pil, missing_counts, 0, 0

    # For each person, check presence of each selected PPE by seeing whether
    # any PPE bbox center lies inside the person bbox.
    violator_flags = [False] * person_count  # whether that person misses any selected PPE

    for p_idx, p_xyxy in enumerate(persons):
        px1, py1, px2, py2 = p_xyxy
        for item in selected_items:
            item_lower = item.lower()
            present = False
            # check known ppe labels in model
            # exact match or substring match (defensive)
            candidate_names = [k for k in ppe_boxes.keys() if item_lower in k or k in item_lower or item_lower in k]
            for cname in candidate_names:
                for b in ppe_boxes.get(cname, []):
                    cx, cy = _box_center(b)
                    if (cx >= px1) and (cx <= px2) and (cy >= py1) and (cy <= py2):
                        present = True
                        break
                if present:
                    break
            if not present:
                # this person missing this item
                missing_counts[item] += 1
                violator_flags[p_idx] = True

    # total violators = number of persons who miss at least one selected item
    total_violators = int(sum(1 for v in violator_flags if v))

    # Prepare annotated image: draw boxes, color persons based on compliance
    annotated = res.plot() if hasattr(res, "plot") else img_np
    # annotated is RGB numpy
    annotated_bgr = cv2.cvtColor(annotated.astype("uint8"), cv2.COLOR_RGB2BGR)

    # draw person boxes with compliance color and PPE boxes with labels
    # Re-extract boxes (we need coordinates as ints)
    for i, cls_idx in enumerate(cls_arr):
        cls_name = str(model_names.get(int(cls_idx), str(int(cls_idx)))).lower()
        xyxy = xyxy_arr[i]
        x1, y1, x2, y2 = map(int, xyxy)
        if cls_name in ("person", "people", "worker"):
            # find index of this person (match by bbox equality)
            match_idx = None
            for idx_p, p in enumerate(persons):
                p_int = tuple(map(int, p))
                if p_int == (x1, y1, x2, y2):
                    match_idx = idx_p
                    break
            # fallback: choose nearest person by center distance
            if match_idx is None:
                cx, cy = _box_center(xyxy)
                dists = [abs(cx - (p[0]+p[2])/2) + abs(cy - (p[1]+p[3])/2) for p in persons]
                match_idx = int(np.argmin(dists))
            color = (0, 255, 0) if not violator_flags[match_idx] else (0, 0, 255)  # BGR: green/compliant, red/violator
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_bgr, "person", (x1, max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # PPE box: blue
            color = (255, 165, 0)  # orange-ish (BGR)
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
            label = cls_name
            cv2.putText(annotated_bgr, label, (x1, max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    return annotated_pil, missing_counts, total_violators, person_count


def detect_ppe_video(input_video_path: str, output_video_path: str, selected_items: List[str], progress_callback=None) -> Tuple[str, Dict[str, int], int, int]:
    """
    Process a video, annotate frames and compute counts.

    Returns:
        output_video_path, missing_counts, total_violators, person_count
    """
    model = _ensure_model()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # accumulators
    missing_counts = {item: 0 for item in selected_items}
    violator_person_ids = set()
    total_persons_seen = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        res = results[0]
        model_names = getattr(model, "names", None) or getattr(res, "names", None) or {}

        boxes = getattr(res, "boxes", None)
        if boxes is None:
            out.write(frame)
            if progress_callback:
                progress_callback(frame_idx / total_frames)
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
            cls_name = str(model_names.get(int(cls_idx), str(int(cls_idx)))).lower()
            xyxy = xyxy_arr[i]
            if cls_name in ("person", "people", "worker"):
                persons.append(xyxy)
            else:
                ppe_boxes.setdefault(cls_name, []).append(xyxy)

        persons_count_this_frame = len(persons)
        total_persons_seen += persons_count_this_frame

        # per-frame missing increment: for each person in frame, count missing items
        violator_flags_frame = [False] * persons_count_this_frame
        for p_idx, p_xyxy in enumerate(persons):
            px1, py1, px2, py2 = p_xyxy
            for item in selected_items:
                item_lower = item.lower()
                present = False
                candidate_names = [k for k in ppe_boxes.keys() if item_lower in k or k in item_lower or item_lower in k]
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

        # count distinct violators roughly as number of violator persons in frame
        for idx, flag in enumerate(violator_flags_frame):
            if flag:
                violator_person_ids.add((frame_idx, idx))  # frame-scoped id; keep approximate

        # annotate frame similar to image
        annotated = res.plot() if hasattr(res, "plot") else frame_rgb
        annotated_bgr = cv2.cvtColor(annotated.astype("uint8"), cv2.COLOR_RGB2BGR)
        out.write(annotated_bgr)

        # Update progress
        if progress_callback:
            progress_callback(frame_idx / total_frames)

    cap.release()
    out.release()

    # approximate total_violators = number of unique frame-person pairs flagged
    total_violators = len(violator_person_ids)
    total_persons = total_persons_seen

    return output_video_path, missing_counts, total_violators, total_persons
