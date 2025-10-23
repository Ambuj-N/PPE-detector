#!/usr/bin/env python3
"""
Quick test to verify the ultralytics.yolo compatibility fix
"""
import sys
import os

# Ensure we're using the project's modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Testing Model Loading with Compatibility Fix")
print("=" * 70)

# Step 1: Import the functions
print("\n📦 Step 1: Importing functions...")
try:
    from utils.detect import load_model, get_model_labels, MODEL_FILES
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Step 2: Show available models
print(f"\n📋 Step 2: Available models:")
for model_name in MODEL_FILES.keys():
    print(f"   - {model_name}")

# Step 3: Try to load a model
print(f"\n🔄 Step 3: Loading model (this will download from HF if not local)...")
test_model = list(MODEL_FILES.keys())[0]  # Use first model
print(f"   Testing with: {test_model}")

try:
    model = load_model(test_model)
    print(f"✅ Model loaded successfully!")
    print(f"   Model type: {type(model).__name__}")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Extract labels
print(f"\n🏷️  Step 4: Extracting labels...")
try:
    labels = get_model_labels(test_model)
    print(f"✅ Labels extracted!")
    print(f"   Number of classes: {len(labels)}")
    print(f"\n   Class mapping:")
    for class_id, class_name in sorted(labels.items()):
        print(f"      {class_id}: {class_name}")
except Exception as e:
    print(f"❌ Label extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Verify model is usable
print(f"\n🔍 Step 5: Verifying model is usable...")
try:
    # Check if model has essential attributes
    assert hasattr(model, 'predict') or hasattr(model, '__call__'), "Model missing predict method"
    print(f"✅ Model has inference capability")
    
    # Try to access names
    if hasattr(model, 'names'):
        print(f"   model.names: {model.names}")
    else:
        print(f"   Note: model.names not available (using fallback labels)")
        
except Exception as e:
    print(f"❌ Model verification failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("🎉 SUCCESS! All checks passed!")
print("=" * 70)
print("\n✅ The compatibility fix is working correctly.")
print("✅ Dynamic label extraction is working.")
print("✅ You can now run: streamlit run app.py")
print("=" * 70)
