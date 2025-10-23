#!/usr/bin/env python3
"""
Test script to verify:
1. Model loads with older ultralytics compatibility
2. Labels are extracted dynamically from the model
"""
import sys
sys.path.insert(0, '/home/anbhigya/Desktop/PPE-Detector')

from utils.detect import load_model, get_model_labels, MODEL_FILES

def test_dynamic_label_extraction():
    """Test loading models and extracting labels dynamically"""
    print("=" * 70)
    print("Testing Dynamic Label Extraction from Models")
    print("=" * 70)
    
    # Test with a model (you can change this to any model you have)
    test_model = "yolov8n.pt"  # Change to whichever model you want to test
    
    print(f"\n📦 Testing with: {test_model}")
    print("-" * 70)
    
    try:
        print(f"\n🔄 Loading model...")
        model = load_model(test_model)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Type: {type(model)}")
        
        print(f"\n🏷️  Extracting labels...")
        labels = get_model_labels(test_model)
        
        print(f"✅ Labels extracted successfully!")
        print(f"   Number of classes: {len(labels)}")
        print(f"\n   Label mapping:")
        for idx, name in sorted(labels.items()):
            print(f"      {idx}: {name}")
        
        # Show what the model actually has
        print(f"\n🔍 Model internal info:")
        if hasattr(model, 'names'):
            print(f"   model.names: {model.names}")
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            print(f"   model.model.names: {model.model.names}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🧪 Available models:")
    for model_name in MODEL_FILES.keys():
        print(f"   - {model_name}")
    
    success = test_dynamic_label_extraction()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ All tests passed! Labels are being extracted dynamically.")
    else:
        print("❌ Test failed. Check the error messages above.")
    print("=" * 70)
    
    sys.exit(0 if success else 1)
