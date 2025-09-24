import cv2
import numpy as np
import torch
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize the YOLOv8 object detector
        
        Args:
            model_path: Path to the YOLOv8 model weights
        """
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Get class names
        self.class_names = self.model.names
    
    def detect(self, image, conf_threshold=0.5, selected_classes=None):
        """
        Detect objects in an image
        
        Args:
            image: Input image (numpy array or PIL Image)
            conf_threshold: Confidence threshold for detections
            selected_classes: List of class names to detect, None for all classes
            
        Returns:
            List of detections, each containing bbox, confidence and class_id
        """
        # Convert PIL Image to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Run detection
        results = self.model(image, conf=conf_threshold, device=self.device)[0]
        
        detections = []
        
        for i, det in enumerate(results.boxes.data.tolist()):
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            class_name = self.class_names[cls_id]
            
            # Filter by selected classes if specified
            if selected_classes is not None and class_name not in selected_classes:
                continue
                
            detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(conf),
                'class_id': cls_id,
                'class_name': class_name
            })
            
        return detections