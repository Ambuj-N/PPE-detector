#!/bin/bash
# Quick Launch Script for PPE Detector Pro

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                                                          ║"
echo "║   🦺  PPE DETECTOR PRO - BEAUTIFUL UI EDITION  🦺        ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "✨ Features:"
echo "   • Modern glass morphism design"
echo "   • Smooth gradient animations"
echo "   • Select All checkbox"
echo "   • Simplified single-model setup"
echo "   • Merged detection & warnings"
echo ""
echo "🚀 Launching Streamlit app..."
echo ""

# Activate virtual environment
source yoloenv/bin/activate

# Launch Streamlit
streamlit run app.py

echo ""
echo "✅ App closed. Thank you for using PPE Detector Pro!"
echo ""
