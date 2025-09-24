import os
import urllib.request

def download_model(model_name="yolov8n.pt", models_dir="models"):
    """
    Download YOLOv8 model if it doesn't exist
    
    Args:
        model_name: Name of the model to download
        models_dir: Directory to store models
        
    Returns:
        Path to the downloaded model
    """
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_name)
    
    if not os.path.exists(model_path):
        print(f"Downloading {model_name}...")
        url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}"
        urllib.request.urlretrieve(url, model_path)
        print(f"Downloaded {model_name}")
    
    return model_path

def get_video_info(video_path):
    """
    Get video information
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return {
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': frame_count,
        'duration': frame_count / fps if fps > 0 else 0
    }