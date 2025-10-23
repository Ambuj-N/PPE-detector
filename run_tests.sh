#!/bin/bash
# Run all tests to verify the fixes

echo "=================================="
echo "PPE Detector - Testing All Fixes"
echo "=================================="
echo ""

# Activate virtual environment
source yoloenv/bin/activate

echo "1️⃣  Running quick verification test..."
echo "-----------------------------------"
python3 verify_fix.py
RESULT1=$?

echo ""
echo ""
echo "2️⃣  Testing dynamic label extraction..."
echo "-----------------------------------"
python3 test_dynamic_labels.py
RESULT2=$?

echo ""
echo ""
echo "=================================="
echo "Test Results Summary"
echo "=================================="

if [ $RESULT1 -eq 0 ]; then
    echo "✅ Quick verification: PASSED"
else
    echo "❌ Quick verification: FAILED"
fi

if [ $RESULT2 -eq 0 ]; then
    echo "✅ Label extraction: PASSED"
else
    echo "❌ Label extraction: FAILED"
fi

echo ""
if [ $RESULT1 -eq 0 ] && [ $RESULT2 -eq 0 ]; then
    echo "🎉 All tests passed! Ready to run:"
    echo "   streamlit run app.py"
else
    echo "⚠️  Some tests failed. Check the output above."
fi
echo "=================================="
