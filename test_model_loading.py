#!/usr/bin/env python3
"""
Test script for Hugging Face model loading functionality
"""
import sys
from utils.detect import load_model, get_model_labels, MODEL_FILES

def test_model_loading():
    """Test loading a model from Hugging Face"""
    print("=" * 60)
    print("Testing PPE Model Loading from Hugging Face")
    print("=" * 60)
    
    # List available models
    print("\n📋 Available models:")
    for model_name in MODEL_FILES.keys():
        print(f"  - {model_name}")
    
    # Test loading a model (will download from HF if not local)
    print("\n🔄 Testing model load (yolov8n.pt)...")
    try:
        model = load_model("yolov8n.pt")
        labels = get_model_labels("yolov8n.pt")
        
        print("✅ Model loaded successfully!")
        print(f"   Model type: {type(model)}")
        print(f"   Labels: {labels}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
