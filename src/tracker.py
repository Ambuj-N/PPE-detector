import cv2
import numpy as np
import supervision as sv

class ObjectTracker:
    def __init__(self, tracker_type="bytetrack"):
        """
        Initialize the object tracker
        
        Args:
            tracker_type: Type of tracker to use
        """
        self.tracker_type = tracker_type
        if tracker_type == "bytetrack":
            self.tracker = sv.ByteTrack()
        else:
            raise ValueError(f"Unsupported tracker type: {tracker_type}")
        
        self.tracks = {}
        
    def reset(self):
        """Reset the tracker state"""
        self.tracks = {}
        if self.tracker_type == "bytetrack":
            self.tracker = sv.ByteTrack()
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dictionaries with bbox, confidence, class_id
            
        Returns:
            List of tracked objects with track_id added
        """
        # Extract bounding boxes and detection info
        bboxes = np.array([det['bbox'] for det in detections])
        confidences = np.array([det['confidence'] for det in detections])
        class_ids = np.array([det['class_id'] for det in detections])
        
        if len(bboxes) == 0:
            return []
        
        # Convert to supervision detection format
        sv_detections = sv.Detections(
            xyxy=bboxes,
            confidence=confidences,
            class_id=class_ids
        )
        
        # Update tracker
        tracked_detections = self.tracker.update(
            detections=sv_detections
        )
        
        # Create list of tracked objects
        tracked_objects = []
        for i in range(len(tracked_detections.xyxy)):
            x1, y1, x2, y2 = tracked_detections.xyxy[i].astype(int)
            track_id = tracked_detections.tracker_id[i]
            confidence = tracked_detections.confidence[i]
            class_id = tracked_detections.class_id[i]
            class_name = detections[i]['class_name']
            
            tracked_objects.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(confidence),
                'class_id': int(class_id),
                'class_name': class_name,
                'track_id': int(track_id)
            })
            
            # Store track info
            self.tracks[int(track_id)] = tracked_objects[-1]
        
        return tracked_objects