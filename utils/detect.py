# utils/detect.py
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Callable
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# Optional HF download
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

# Hugging Face repository configuration
HF_MODEL_REPO = "Anbhigya/ppe-detector-model"

# Model registry and cache
_LOADED_MODELS: Dict[str, YOLO] = {}
_MODEL_LABELS: Dict[str, Dict[int, str]] = {}

# Model files mapping: model_name -> HF filename
MODEL_FILES = {
    "yolo9s.pt": "yolo9s.pt"
}

# ----------------------
# MODEL MANAGEMENT
# ----------------------
def load_model(model_name: str) -> YOLO:
    """
    Load a YOLO model dynamically and cache it.
    First checks for local file, then downloads from Hugging Face if needed.
    Handles compatibility issues with models trained on older ultralytics versions.
    """
    global _LOADED_MODELS, _MODEL_LABELS
    
    # Return cached model if already loaded
    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    if model_name not in MODEL_FILES:
        raise ValueError(f"Model {model_name} not found in MODEL_FILES!")

    # Check for local file first
    local_path = Path(MODEL_FILES[model_name])
    if local_path.exists():
        model_path = str(local_path)
        print(f"✅ Loading model from local path: {model_path}")
    else:
        # Download from Hugging Face
        if hf_hub_download is None:
            raise RuntimeError(
                "huggingface_hub is not installed and local model not found! "
                "Install it with: pip install huggingface_hub"
            )
        print(f"⬇️  Downloading model '{model_name}' from Hugging Face repo: {HF_MODEL_REPO}")
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=MODEL_FILES[model_name]
        )
        print(f"✅ Model downloaded to: {model_path}")

    # Load the YOLO model with compatibility handling
    try:
        # Try to load normally first
        model = YOLO(model_path)
    except ModuleNotFoundError as e:
        if "ultralytics.yolo" in str(e):
            # Model was trained with older ultralytics version
            # We need to load it differently to handle the module path issue
            print(f"⚠️  Model uses older ultralytics format, applying compatibility fix...")
            import torch
            import sys
            
            # Temporarily add compatibility module path
            if 'ultralytics.yolo' not in sys.modules:
                import ultralytics
                sys.modules['ultralytics.yolo'] = ultralytics
                sys.modules['ultralytics.yolo.utils'] = ultralytics.utils
                sys.modules['ultralytics.yolo.v8'] = ultralytics
            
            # Now try loading again
            model = YOLO(model_path)
            print(f"✅ Loaded model with compatibility fix")
        else:
            raise
    
    _LOADED_MODELS[model_name] = model
    
    # Extract and store labels dynamically from the model
    _MODEL_LABELS[model_name] = extract_labels_from_model(model, model_name)
    
    return model

def extract_labels_from_model(model: YOLO, model_name: str) -> Dict[int, str]:
    """
    Extract class labels directly from the loaded YOLO model.
    Falls back to hardcoded labels if extraction fails.
    """
    try:
        # Try to get names from model
        if hasattr(model, 'names'):
            names = model.names
            if isinstance(names, dict):
                return names
            elif isinstance(names, list):
                return {i: name for i, name in enumerate(names)}
        
        # Try to get from model.model
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            names = model.model.names
            if isinstance(names, dict):
                return names
            elif isinstance(names, list):
                return {i: name for i, name in enumerate(names)}
        
        print(f"⚠️  Could not extract labels from model, using fallback for {model_name}")
    except Exception as e:
        print(f"⚠️  Error extracting labels: {e}, using fallback for {model_name}")
    
    # Fallback to hardcoded labels
    return get_fallback_labels(model_name)

def get_fallback_labels(model_name: str) -> Dict[int, str]:
    """
    Fallback label mappings if dynamic extraction fails.
    """
    if model_name == "yolo9s.pt":
        return {
            0: "Person",
            1: "Helmet",
            2: "Gloves",
            3: "Safety-vest",
            4: "Face-mask-medical",
            5: "Earmuffs",
            6: "Shoes"
        }
    elif model_name == "best.pt":
        return {
            0: "Person",
            1: "Helmet",
            2: "Gloves"
        }
    elif model_name == "good.pt":
        return {
            0: "Person",
            1: "Helmet"
        }
    elif model_name == "yolov8n.pt":
        return {
            0: "Person",
            1: "Helmet",
            2: "Gloves",
            3: "Mask"
        }
    else:
        # Generic fallback
        return {0: "Person"}

def get_model_labels(model_name: str) -> Dict[int, str]:
    """
    Return the label mapping for a specific model.
    If model is not loaded yet, load it first to extract labels.
    """
    if model_name not in _MODEL_LABELS:
        # Load the model to extract labels
        load_model(model_name)
    return _MODEL_LABELS[model_name]

# ----------------------
# HELPER FUNCTIONS
# ----------------------
def _normalize(s: str) -> str:
    return s.strip().lower()

def _matches_ppe_item(selected_item: str, detected_label: str) -> bool:
    """
    Check if a selected PPE item matches a detected label.
    Handles variations like:
    - "Safety-vest" matches "Safety-vest", "safety vest", "vest", etc.
    - "Face-mask-medical" matches "face mask", "mask", etc.
    """
    selected_norm = selected_item.lower().replace('-', ' ').replace('_', ' ')
    detected_norm = detected_label.lower().replace('-', ' ').replace('_', ' ')
    
    # Direct match
    if selected_norm == detected_norm:
        return True
    
    # Check if one contains the other (handles partial matches)
    if selected_norm in detected_norm or detected_norm in selected_norm:
        return True
    
    # Special case: "vest" should match "safety vest" or "safety-vest"
    if 'vest' in selected_norm and 'vest' in detected_norm:
        return True
    
    # Special case: "mask" should match "face mask" or "face-mask-medical"
    if 'mask' in selected_norm and 'mask' in detected_norm:
        return True
    
    return False

def _box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0

def _point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)

# ----------------------
# IMAGE DETECTION
# ----------------------
def detect_ppe_image(
    uploaded_file_or_pil,
    selected_detection_items: List[str],
    selected_warning_items: List[str],
    confidence_threshold: float = 0.5,
    draw_all_detections: bool = False,
    model: YOLO = None,
    model_labels: Dict[int, str] = None
) -> Tuple[Image.Image, Dict[str, int], int, int, Dict[str, int]]:
    """
    Detect PPE in a single image using a selected YOLO model.
    """
    if model is None or model_labels is None:
        raise ValueError("Model and model_labels must be provided!")

    # Convert to PIL
    if hasattr(uploaded_file_or_pil, "read"):
        uploaded_file_or_pil.seek(0)
        pil = Image.open(uploaded_file_or_pil).convert("RGB")
    elif isinstance(uploaded_file_or_pil, Image.Image):
        pil = uploaded_file_or_pil.convert("RGB")
    else:
        pil = Image.open(uploaded_file_or_pil).convert("RGB")

    img_np = np.array(pil)

    # Run detection
    results = model(img_np, conf=confidence_threshold)
    res = results[0]

    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return pil, {it: 0 for it in selected_warning_items}, 0, 0, {}

    # Extract arrays
    try:
        xyxy_arr = boxes.xyxy.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy().astype(int)
        conf_arr = boxes.conf.cpu().numpy()
    except Exception:
        data = boxes.data.cpu().numpy()
        xyxy_arr = data[:, :4]
        cls_arr = data[:, -1].astype(int)
        conf_arr = data[:, 4] if data.shape[1] >= 5 else np.ones(len(cls_arr))

    # Separate persons vs PPE
    persons, other_dets = [], []
    for i, cls_idx in enumerate(cls_arr):
        label = model_labels.get(int(cls_idx), str(int(cls_idx)))
        confidence = float(conf_arr[i])
        xyxy = xyxy_arr[i].astype(float)
        if confidence < confidence_threshold:
            continue
        if _normalize(label) == "person":
            persons.append((xyxy, confidence))
        else:
            other_dets.append((label, xyxy, confidence))

    # Missing counts and violators
    person_count = len(persons)
    missing_counts = {it: 0 for it in selected_warning_items}
    violator_flags = [False] * max(0, person_count)
    detection_summary = {label: sum(1 for l, *_ in other_dets if l==label) for label in selected_detection_items}

    # Person-PPE checking
    for p_idx, (p_box, _) in enumerate(persons):
        px1, py1, px2, py2 = p_box
        for warning_item in selected_warning_items:
            present = any(
                _matches_ppe_item(warning_item, label) and _point_in_box(*_box_center(b), (px1, py1, px2, py2))
                for label, b, _ in other_dets
            )
            if not present:
                missing_counts[warning_item] += 1
                violator_flags[p_idx] = True

    total_violators = sum(1 for v in violator_flags if v)

    # Annotate image
    annotated_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    for label, b, confidence in other_dets:
        draw_item = draw_all_detections or any(_matches_ppe_item(sel, label) for sel in selected_detection_items)
        if draw_item:
            color = (0,255,0) if label in selected_detection_items else (0,165,255)
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(annotated_bgr, (x1,y1), (x2,y2), color, 2)
            cv2.putText(annotated_bgr, f"{label} {confidence:.2f}", (x1,max(16,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Draw person boxes
    for p_idx, (p_box, confidence) in enumerate(persons):
        x1, y1, x2, y2 = map(int, p_box)
        compliant = not violator_flags[p_idx]
        color = (0,255,0) if compliant else (0,0,255)
        cv2.rectangle(annotated_bgr, (x1,y1),(x2,y2), color, 3)
        status = "COMPLIANT" if compliant else "VIOLATOR"
        cv2.putText(annotated_bgr, f"Person {status} {confidence:.2f}", (x1,max(16,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), missing_counts, total_violators, person_count, detection_summary

# ----------------------
# VIDEO DETECTION
# ----------------------
def detect_ppe_video(
    input_video_path: str,
    output_video_path: str,
    selected_detection_items: List[str],
    selected_warning_items: List[str],
    confidence_threshold: float = 0.5,
    progress_callback: Optional[Callable[[float], None]] = None,
    model: YOLO = None,
    model_labels: Dict[int, str] = None
) -> Tuple[str, Dict[str, int], int, int, Dict[str, int]]:
    """
    Detect PPE frame-by-frame in video using a selected YOLO model.
    """
    if model is None or model_labels is None:
        raise ValueError("Model and model_labels must be provided!")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

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

        boxes = getattr(res, "boxes", None)
        if boxes is None:
            out.write(frame_bgr)
            continue

        try:
            xyxy_arr = boxes.xyxy.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy().astype(int)
            conf_arr = boxes.conf.cpu().numpy()
        except Exception:
            data = boxes.data.cpu().numpy()
            xyxy_arr = data[:, :4]
            cls_arr = data[:, -1].astype(int)
            conf_arr = data[:, 4] if data.shape[1]>=5 else np.ones(len(cls_arr))

        persons, other_dets = [], []
        for i, cls_idx in enumerate(cls_arr):
            label = model_labels.get(int(cls_idx), str(int(cls_idx)))
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

        for p_idx, (p_box, _) in enumerate(persons):
            px1, py1, px2, py2 = p_box
            for warning_item in selected_warning_items:
                present = any(
                    _matches_ppe_item(warning_item, label) and _point_in_box(*_box_center(b), (px1,py1,px2,py2))
                    for label, b, _ in other_dets
                )
                if not present:
                    missing_counts[warning_item] += 1
                    violator_flags[p_idx] = True

        violator_events += sum(1 for v in violator_flags if v)

        annotated_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # Draw PPE
        for label, b, confidence in other_dets:
            draw_item = any(_matches_ppe_item(sel, label) for sel in selected_detection_items)
            if draw_item:
                color = (0,255,0) if label in selected_detection_items else (0,165,255)
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(annotated_bgr,(x1,y1),(x2,y2),color,2)
                cv2.putText(annotated_bgr,f"{label} {confidence:.2f}",(x1,max(16,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
        # Draw persons
        for p_idx, (p_box, _) in enumerate(persons):
            x1, y1, x2, y2 = map(int, p_box)
            compliant = not violator_flags[p_idx]
            color = (0,255,0) if compliant else (0,0,255)
            cv2.rectangle(annotated_bgr,(x1,y1),(x2,y2),color,3)
            status = "OK" if compliant else "VIOLATOR"
            cv2.putText(annotated_bgr,f"Person {status}",(x1,max(16,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
        out.write(annotated_bgr)

    cap.release()
    out.release()
    return output_video_path, missing_counts, violator_events, persons_seen, detection_summary
