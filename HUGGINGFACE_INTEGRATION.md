# 🦺 Hugging Face Model Integration

## Overview
The PPE Detector now automatically downloads models from Hugging Face if they're not found locally. This eliminates the need to store large `.pt` files in your repository.

## Features
- ✅ Automatic model download from Hugging Face
- ✅ Local caching for faster subsequent loads
- ✅ Fallback to local files if available
- ✅ Support for multiple YOLO models
- ✅ Custom label mappings per model

## Available Models

The following models are available from the Hugging Face repository `Anbhigya/ppe-detector-model`:

| Model Name | HF Filename | Detects |
|------------|-------------|---------|
| `yolo9s.pt` | yolo9s.pt | Person, Helmet, Gloves, Safety-vest, Face-mask-medical, Earmuffs, Shoes |
| `best.pt` | best.pt | Person, Helmet, Gloves |
| `good.pt` | good.pt | Person, Helmet |
| `yolov8n.pt` | yolov8n.pt | Person, Helmet, Gloves, Mask |

## How It Works

### 1. Model Loading Flow
```python
from utils.detect import load_model, get_model_labels

# Load a model (will auto-download from HF if needed)
model = load_model("yolov8n.pt")
labels = get_model_labels("yolov8n.pt")
```

**The loading process:**
1. Check if model is already cached in memory → return cached model
2. Check if model exists locally in current directory → load from local file
3. If not found locally → download from Hugging Face to cache directory
4. Load the model with YOLO
5. Cache in memory for faster subsequent access

### 2. Hugging Face Integration

The implementation uses `huggingface_hub` library:

```python
from huggingface_hub import hf_hub_download

# Downloads model to HF cache (~/.cache/huggingface/hub/)
model_path = hf_hub_download(
    repo_id="Anbhigya/ppe-detector-model",
    filename="yolov8n.pt"
)
```

### 3. Label Mapping

Each model has a specific label mapping defined in `get_model_labels()`:

```python
# Example: yolo9s.pt labels
{
    0: "Person",
    1: "Helmet",
    2: "Gloves",
    3: "Safety-vest",
    4: "Face-mask-medical",
    5: "Earmuffs",
    6: "Shoes"
}
```

## Installation

Make sure you have the required dependencies:

```bash
pip install -r requirements.txt
```

The key dependency is:
```
huggingface-hub>=0.20.0
```

## Usage Examples

### Basic Usage
```python
from utils.detect import load_model, get_model_labels

# Load model
model = load_model("yolov8n.pt")
labels = get_model_labels("yolov8n.pt")

# Use for inference
results = model("image.jpg")
```

### Using in Streamlit App
The app automatically uses this functionality:

```python
# In app.py
from utils.detect import load_model, get_model_labels

# Model selection
selected_model = st.sidebar.selectbox("Select Model", ["yolo9s.pt", "best.pt", "good.pt", "yolov8n.pt"])
model = load_model(selected_model)
labels = get_model_labels(selected_model)
```

### Testing Model Loading
Run the test script to verify everything works:

```bash
python3 test_model_loading.py
```

## Configuration

### Change Hugging Face Repository
Edit `utils/detect.py`:

```python
HF_MODEL_REPO = "YourUsername/your-model-repo"
```

### Add New Models
1. Upload `.pt` file to your HF repository
2. Add to `MODEL_FILES` dict:
```python
MODEL_FILES = {
    "yolo9s.pt": "yolo9s.pt",
    "your_new_model.pt": "your_new_model.pt"  # Add here
}
```

3. Add label mapping in `get_model_labels()`:
```python
elif model_name == "your_new_model.pt":
    return {
        0: "Person",
        1: "Helmet",
        # ... your labels
    }
```

## Cache Location

Models are cached by `huggingface_hub` at:
- **Linux/Mac**: `~/.cache/huggingface/hub/`
- **Windows**: `%USERPROFILE%\.cache\huggingface\hub\`

You can clear the cache to force re-download:
```bash
rm -rf ~/.cache/huggingface/hub/models--Anbhigya--ppe-detector-model
```

## Troubleshooting

### Issue: "huggingface_hub not installed"
**Solution**: Install the package:
```bash
pip install huggingface_hub
```

### Issue: Download fails with authentication error
**Solution**: Login to Hugging Face (if repository is private):
```bash
huggingface-cli login
```

### Issue: Want to use local models only
**Solution**: Place `.pt` files in the project root directory with exact names from `MODEL_FILES`

### Issue: Slow downloads
**Solution**: 
- Models are only downloaded once and cached
- Subsequent loads use the cached version
- Consider using a local copy for development

## Performance Notes

- **First Load**: Downloads from HF (~5-200 MB depending on model) - may take 10-60 seconds
- **Subsequent Loads**: Uses cached file - loads in 1-3 seconds
- **In-Memory Cache**: Once loaded, model stays in memory for instant access

## Security

- Models are downloaded over HTTPS
- SHA256 checksums are verified by `huggingface_hub`
- Cached files are stored in user's home directory
- No elevation/admin privileges required

## Contributing

To add your own models to the Hugging Face repository:

1. Create a HF repository (e.g., `YourUsername/ppe-models`)
2. Upload your `.pt` files
3. Update `HF_MODEL_REPO` in `detect.py`
4. Add model entries to `MODEL_FILES` and `get_model_labels()`

## License

Models must comply with Hugging Face's terms of service and the licenses of the underlying YOLO architecture.
