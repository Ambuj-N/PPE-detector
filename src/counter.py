class ObjectCounter:
    def __init__(self):
        """
        Initialize object counter for tracking object counts
        """
        self.counts = {}
        self.track_history = {}
    
    def reset(self):
        """Reset all counters"""
        self.counts = {}
        self.track_history = {}
    
    def update(self, tracked_objects):
        """
        Update counters based on tracked objects
        
        Args:
            tracked_objects: List of tracked objects with track_id and class_name
        """
        # Initialize counts for class if not present
        for obj in tracked_objects:
            class_name = obj['class_name']
            track_id = obj['track_id']
            
            if class_name not in self.counts:
                self.counts[class_name] = 0
            
            # If this is a new track, increment the counter
            if track_id not in self.track_history:
                self.track_history[track_id] = class_name
                self.counts[class_name] += 1
    
    def get_counts(self):
        """
        Get the current object counts
        
        Returns:
            Dictionary mapping class names to counts
        """
        return self.counts
