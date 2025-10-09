import torch
import gradio as gr
from huggingface_hub import hf_hub_download
from utils.detect import detect_ppe  # your custom detection logic

# ----------------------------
# Download the YOLO model from HF Hub
# ----------------------------
model_path = hf_hub_download(
    repo_id="Anbhigya/ppe-detector-model",  # your model repo
    filename="best.pt"
)

# Load YOLOv5 custom model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# ----------------------------
# PPE Detection Function
# ----------------------------
def predict(image):
    """
    Input: image (PIL/ndarray)
    Output: annotated image + list of missing PPE items
    """
    results = model(image)
    
    # You can modify detect_ppe to return missing PPE items or flagged info
    flagged_info = detect_ppe(results)  # returns dict/list
    annotated_image = results.render()[0]  # YOLO returns list of images
    return annotated_image, str(flagged_info)

# ----------------------------
# Gradio UI
# ----------------------------
title = "PPE Detector"
description = """
Upload an image or video frame, and the system will detect PPE items.
If people are missing PPE items, they will be flagged.
"""

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="numpy"), gr.Textbox()],
    title=title,
    description=description,
    allow_flagging="never"
)

# Launch Space
if __name__ == "__main__":
    interface.launch()

