# utils/detect.py

import os
from pathlib import Path
from typing import Tuple, Dict, List, Set

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2

# --- Config ---
HF_MODEL_REPO = "Anbhigya/ppe-detector-model"
HF_MODEL_FILENAME = "best.pt"
MODEL_CACHE_DIR = Path("model_cache")
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
            from shutil import copyfile
            copyfile(downloaded, str(local_path))
    _model = YOLO(str(local_path))
    return _model


def _extract_classnames(result) -> List[str]:
    """Extract detected class names from YOLO result."""
    try:
        cls_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()
        return [result.names[c].lower() for c in cls_ids]
    except Exception:
        return []


def detect_ppe_image(uploaded_file_or_pil, required_items: List[str]) -> Tuple[Image.Image, Dict[str, int], int]:
    """
    Detect PPE items in an image.

    Returns:
        annotated_pil, missing_count_per_item, total_violators
    """
    model = _ensure_model()

    # Normalize image input
    if hasattr(uploaded_file_or_pil, "read"):
        uploaded_file_or_pil.seek(0)
        image = Image.open(uploaded_file_or_pil).convert("RGB")
    elif isinstance(uploaded_file_or_pil, Image.Image):
        image = uploaded_file_or_pil.convert("RGB")
    else:
        image = Image.open(uploaded_file_or_pil).convert("RGB")

    np_img = np.array(image)

    results = model(np_img)
    res = results[0]

    # Annotate
    annotated = res.plot()
    annotated_pil = Image.fromarray(annotated.astype("uint8"))

    detected_names = _extract_classnames(res)
    detected_lower = [n.lower() for n in detected_names]

    missing_counts = {item: 0 for item in required_items}
    total_violators = 0

    # simple heuristic: if person detected but missing required items
    persons = [n for n in detected_lower if n in ["person", "worker"]]
    person_count = len(persons) if persons else 1  # fallback = 1 person if not labeled

    # if model doesn’t separate per person, approximate
    for item in required_items:
        if item.lower() not in detected_lower:
            missing_counts[item] = person_count
    total_violators = int(any(missing_counts.values()))

    return annotated_pil, missing_counts, total_violators


def detect_ppe_video(input_video_path: str, output_video_path: str, required_items: List[str]) -> Tuple[str, Dict[str, int], int]:
    """
    Detect PPE in a video and return annotated path + counts.
    """
    model = _ensure_model()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")

    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    missing_counts = {item: 0 for item in required_items}
    total_violators = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        res = results[0]
        annotated = res.plot()

        detected = _extract_classnames(res)
        detected_lower = [n.lower() for n in detected]

        persons = [n for n in detected_lower if n in ["person", "worker"]]
        person_count = len(persons) if persons else 1

        any_missing = False
        for item in required_items:
            if item.lower() not in detected_lower:
                missing_counts[item] += person_count
                any_missing = True

        if any_missing:
            total_violators += person_count

        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        out.write(annotated_bgr)

    cap.release()
    out.release()

    return output_video_path, missing_counts, total_violators

