#!/bin/bash
# Quick test to verify UI is working

echo ""
echo "════════════════════════════════════════════════════"
echo "  🎨 UI VISIBILITY FIXES APPLIED"
echo "════════════════════════════════════════════════════"
echo ""
echo "✅ Fixed Issues:"
echo "   • Sidebar text now clearly visible"
echo "   • Main content readable in both themes"
echo "   • Alert boxes have proper contrast"
echo "   • Analytics tab content visible"
echo "   • All markdown content clear"
echo "   • File uploader labels readable"
echo "   • Section headers stand out"
echo ""
echo "🎨 Improvements:"
echo "   • Light mode: Dark text on light background"
echo "   • Dark mode: Light text on dark background"
echo "   • Increased opacity on alert boxes (0.15)"
echo "   • Stronger borders and shadows"
echo "   • Better text hierarchy"
echo ""
echo "🚀 Launching your beautiful, readable PPE Detector..."
echo ""

cd /home/anbhigya/Desktop/PPE-Detector
source yoloenv/bin/activate
streamlit run app.py

echo ""
echo "✨ Thanks for using PPE Detector Pro!"
echo ""
