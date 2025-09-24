# Object Detection Pipeline

This application provides a web interface for object detection, tracking, and counting using YOLOv8.

## Features

- **Object Detection**: Detect objects using pre-trained YOLOv8 models
- **Object Tracking**: Track objects across video frames
- **Object Counting**: Count objects by class
- **User Interface**: Simple Gradio interface for image and video processing

## Requirements

- Python 3.10+
- PyTorch
- OpenCV
- Ultralytics YOLOv8
- Supervision (for tracking)
- Gradio (for web interface)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python app.py
```

Then open your browser at http://localhost:7860.

## Using the App

1. **Initialize Model**: Choose a YOLOv8 model size and click "Initialize Model"
2. **Process Images**: Upload an image, set confidence threshold, select classes, and click "Process Image"
3. **Process Videos**: Upload a video, configure settings, and click "Process Video"

## Supported Models

- YOLOv8n (default, fastest)
- YOLOv8s
- YOLOv8m
- YOLOv8l
- YOLOv8x (most accurate, slowest)

## Deployment on Hugging Face Spaces

This app is designed to be easily deployed on Hugging Face Spaces. Create a new Space, select "Gradio" as the SDK, and upload these files.

## Custom Training

To train custom models:

1. Prepare your dataset in YOLO format
2. Use the ultralytics package to train your model
3. Replace the model path in the app with your custom model

## License

MIT