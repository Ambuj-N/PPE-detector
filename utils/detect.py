# utils/detect.py

import os
import tempfile
from pathlib import Path
from typing import Tuple, List, Set

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2

# --- Configuration ---
HF_MODEL_REPO = "Anbhigya/ppe-detector-model"  # <--- your HF model repo
HF_MODEL_FILENAME = "best.pt"
MODEL_CACHE_DIR = Path("model_cache")  # local cache folder for downloaded model
REQUIRED_PPE_ITEMS = ["helmet", "vest", "gloves", "goggles", "mask"]  # adjust to your labels

# internal singleton
_model: YOLO = None
_model_path: Path = None


def _ensure_model() -> YOLO:
    """Download model from HF hub if needed and return YOLO instance (singleton)."""
    global _model, _model_path
    if _model is not None:
        return _model

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # download model if not present
    local_path = MODEL_CACHE_DIR / HF_MODEL_FILENAME
    if not local_path.exists():
        # hf_hub_download will cache the file in the huggingface cache as well,
        # but we copy it into our model_cache for clarity
        downloaded = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME)
        # hf_hub_download returns a path to the cached file; copy to local_path (or just use it)
        # We'll create a symlink or copy to keep consistent path:
        try:
            os.symlink(downloaded, str(local_path))
        except Exception:
            # fallback to copying if symlink not allowed
            from shutil import copyfile
            copyfile(downloaded, str(local_path))

    _model_path = str(local_path)
    # Load model with ultralytics YOLO class
    _model = YOLO(_model_path)
    return _model


def _get_detected_class_names(result) -> List[str]:
    """Safely extract detected class names from a ultralytics result object."""
    names = []
    try:
        boxes = result.boxes  # Boxes object
        # boxes.cls should be a tensor-like with class indices
        cls_tensor = getattr(boxes, "cls", None)
        if cls_tensor is None:
            return []
        # convert to python ints
        cls_list = cls_tensor.cpu().numpy().astype(int).tolist()
        # get mapping from model names (model.names is a dict)
        model_names = result.names if hasattr(result, "names") else None
        if model_names is None:
            # try global model names via loaded model
            model_names = {i: n for i, n in enumerate(range(len(cls_list)))}
        for c in cls_list:
            # If model_names maps ints to label strings:
            try:
                label = model_names[int(c)]
            except Exception:
                label = str(c)
            names.append(label)
    except Exception:
        # fallback: try result.boxes.data if available (older/newer APIs)
        try:
            if hasattr(result, "boxes") and hasattr(result.boxes, "data"):
                # often boxes.data columns are [x1,y1,x2,y2,score,class]
                data = result.boxes.data.cpu().numpy()
                cls_indices = data[:, -1].astype(int).tolist()
                model_names = result.names if hasattr(result, "names") else {}
                names = [model_names.get(i, str(i)) for i in cls_indices]
        except Exception:
            names = []
    return names


def detect_ppe_image(uploaded_file_or_pil) -> Tuple[Image.Image, List[str]]:
    """
    Run PPE detection on an image.

    Args:
        uploaded_file_or_pil: either a Streamlit UploadedFile, file-like object,
                              a Pillow Image, or a numpy array (H x W x C RGB).

    Returns:
        (annotated_pil_image, missing_items_list)
    """
    model = _ensure_model()

    # normalize input to numpy RGB
    if hasattr(uploaded_file_or_pil, "read"):
        # file-like: read bytes and open with PIL
        uploaded_file_or_pil.seek(0)
        image = Image.open(uploaded_file_or_pil).convert("RGB")
    elif isinstance(uploaded_file_or_pil, Image.Image):
        image = uploaded_file_or_pil.convert("RGB")
    elif isinstance(uploaded_file_or_pil, np.ndarray):
        # assume RGB
        image = Image.fromarray(uploaded_file_or_pil)
    else:
        # try to open via PIL
        image = Image.open(uploaded_file_or_pil).convert("RGB")

    img_np = np.array(image)  # RGB

    # Run model (ultralytics YOLO)
    results = model(img_np)  # returns Results object
    res0 = results[0]

    # Annotated image as numpy array
    try:
        annotated = res0.plot()  # returns RGB numpy array with boxes drawn
    except Exception:
        # fallback: use results.render() if available
        try:
            results.render()
            annotated = results.ims[0]
        except Exception:
            annotated = img_np

    # Extract detected class names
    detected_names = _get_detected_class_names(res0)

    # Normalize names to lowercase for matching
    detected_lower = [n.lower() for n in detected_names]

    # Determine missing PPE
    missing = [item for item in REQUIRED_PPE_ITEMS if item.lower() not in detected_lower]

    # Convert annotated array to PIL
    annotated_pil = Image.fromarray(annotated.astype("uint8"))

    return annotated_pil, missing


def detect_ppe_video(input_video_path: str, output_video_path: str = "output_annotated.mp4") -> Tuple[str, List[str]]:
    """
    Run detection on a video file and write an annotated output video.

    Args:
        input_video_path: path to input video file
        output_video_path: path to write annotated video

    Returns:
        (output_video_path, missing_items_overall)
    """
    model = _ensure_model()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open input video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    missing_set: Set[str] = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # convert BGR->RGB for model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(frame_rgb)
        res0 = results[0]

        # annotated frame (RGB numpy)
        try:
            annotated = res0.plot()
        except Exception:
            try:
                results.render()
                annotated = results.ims[0]
            except Exception:
                annotated = frame_rgb

        # collect detected classes
        detected_names = _get_detected_class_names(res0)
        detected_lower = [n.lower() for n in detected_names]
        for item in REQUIRED_PPE_ITEMS:
            if item.lower() not in detected_lower:
                missing_set.add(item)

        # convert back to BGR and write
        annotated_bgr = cv2.cvtColor(annotated.astype("uint8"), cv2.COLOR_RGB2BGR)
        out.write(annotated_bgr)

    cap.release()
    out.release()

    missing_list = list(missing_set)
    return output_video_path, missing_list

