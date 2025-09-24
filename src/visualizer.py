import cv2
import numpy as np
import random

class Visualizer:
    def __init__(self):
        """
        Initialize visualizer for drawing detection results
        """
        self.colors = {}
    
    def get_color(self, class_id):
        """
        Get a consistent color for a class ID
        
        Args:
            class_id: Class identifier
            
        Returns:
            Color tuple (B,G,R)
        """
        if class_id not in self.colors:
            self.colors[class_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        return self.colors[class_id]
    
    def draw_results(self, image, detections):
        """
        Draw detection results on an image
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Annotated image
        """
        # Convert PIL Image to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Make a copy to avoid modifying the original
        annotated_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            class_id = det['class_id']
            
            # Get color for this class
            color = self.get_color(class_id)
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            text = f"{class_name}: {confidence:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_image, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_image
    
    def draw_results_with_tracks(self, image, tracked_objects):
        """
        Draw detection results with tracking information
        
        Args:
            image: Input image
            tracked_objects: List of tracked objects
            
        Returns:
            Annotated image
        """
        # Convert PIL Image to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Make a copy to avoid modifying the original
        annotated_image = image.copy()
        
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            class_name = obj['class_name']
            confidence = obj['confidence']
            class_id = obj['class_id']
            track_id = obj['track_id']
            
            # Get color for this class
            color = self.get_color(class_id)
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            text = f"{class_name} #{track_id}: {confidence:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_image, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_image
    
    def draw_results_with_tracks_and_counts(self, image, tracked_objects, counts):
        """
        Draw detection results with tracking information and object counts
        
        Args:
            image: Input image
            tracked_objects: List of tracked objects
            counts: Dictionary of object counts by class
            
        Returns:
            Annotated image
        """
        # First draw the tracks
        annotated_image = self.draw_results_with_tracks(image, tracked_objects)
        
        # Add count information at the top
        y_pos = 30
        for class_name, count in counts.items():
            text = f"{class_name}: {count}"
            cv2.putText(annotated_image, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
        
        return annotated_image