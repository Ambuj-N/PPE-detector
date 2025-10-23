"""
Quick Start Guide: Hugging Face Model Loading
==============================================

This file demonstrates how the new HF model loading works.
"""

# ============================================
# 1. BASIC MODEL LOADING
# ============================================
from utils.detect import load_model, get_model_labels

# Load any model - it will auto-download from HF if needed
model = load_model("yolov8n.pt")  
labels = get_model_labels("yolov8n.pt")

print(f"Loaded model with labels: {labels}")


# ============================================
# 2. WHAT HAPPENS BEHIND THE SCENES
# ============================================
"""
When you call load_model("yolov8n.pt"):

Step 1: Check in-memory cache
    ↓ (if not found)
Step 2: Check local directory for yolov8n.pt
    ↓ (if not found)
Step 3: Download from Hugging Face
    - Repo: Anbhigya/ppe-detector-model
    - File: yolov8n.pt
    - Cache: ~/.cache/huggingface/hub/
    ↓
Step 4: Load with YOLO() and cache in memory
    ↓
Step 5: Return model object

Next time you call load_model("yolov8n.pt"):
→ Returns instantly from in-memory cache!
"""


# ============================================
# 3. AVAILABLE MODELS
# ============================================
from utils.detect import MODEL_FILES

print("\nAvailable models:")
for model_name, filename in MODEL_FILES.items():
    labels = get_model_labels(model_name)
    print(f"  {model_name}: {len(labels)} classes")


# ============================================
# 4. USE IN YOUR CODE
# ============================================
def detect_image(image_path, model_name="yolov8n.pt"):
    """Example function using HF model loading"""
    model = load_model(model_name)
    results = model(image_path)
    return results


# ============================================
# 5. INTEGRATION WITH STREAMLIT
# ============================================
"""
In your Streamlit app (app.py):

    import streamlit as st
    from utils.detect import load_model, get_model_labels, MODEL_FILES

    # User selects model
    model_name = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()))
    
    # Load model (auto-downloads if needed)
    model = load_model(model_name)
    labels = get_model_labels(model_name)
    
    # Now use model for detection
    results = model(uploaded_image)
"""


# ============================================
# 6. CONFIGURATION
# ============================================
"""
To change the Hugging Face repository:
    
    Edit utils/detect.py:
    HF_MODEL_REPO = "YourUsername/your-repo-name"

To add a new model:
    
    1. Upload .pt file to your HF repo
    2. Add to MODEL_FILES dict
    3. Add label mapping in get_model_labels()
"""


# ============================================
# 7. TROUBLESHOOTING
# ============================================
"""
Q: Model download is slow?
A: First download caches the file. Subsequent loads are fast.

Q: Want to use only local models?
A: Place .pt files in project root with exact names from MODEL_FILES

Q: Getting import error for huggingface_hub?
A: Run: pip install huggingface_hub

Q: Want to clear cache and re-download?
A: rm -rf ~/.cache/huggingface/hub/models--Anbhigya--ppe-detector-model
"""
