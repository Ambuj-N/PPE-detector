# app.py
import streamlit as st
from PIL import Image
import tempfile
import os
import io
import time
import numpy as np
import pandas as pd
import altair as alt
import base64

from utils.detect import detect_ppe_image, detect_ppe_video, SUPPORTED_ITEMS, ALL_MODEL_LABELS

# ==========================================
# 🌐 PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="PPE Detector Pro", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🦺"
)

# ==========================================
# 🌈 THEME TOGGLE
# ==========================================
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

with st.sidebar:
    st.markdown("### 🎨 Theme Mode")
    theme_choice = st.radio("Choose theme:", ["🌞 Light", "🌙 Dark"],
                            index=0 if st.session_state["theme"] == "light" else 1)
    st.session_state["theme"] = "light" if "Light" in theme_choice else "dark"

# Apply theme colors dynamically
if st.session_state["theme"] == "light":
    primary, secondary, bg, text = "#2563eb", "#10b981", "#f9fafb", "#1e293b"
else:
    primary, secondary, bg, text = "#3b82f6", "#34d399", "#0f172a", "#f8fafc"

# ==========================================
# 💅 CUSTOM STYLES
# ==========================================
st.markdown(f"""
<style>
body {{
    background: {bg};
    color: {text};
    font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
}}
.main-header {{
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 1.5rem;
    background: linear-gradient(90deg, {primary}, {secondary});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientFlow 6s linear infinite;
}}
@keyframes gradientFlow {{
    0% {{ background-position: 0% center; }}
    100% {{ background-position: 200% center; }}
}}
.section-header {{
    font-size: 1.8rem;
    font-weight: 700;
    color: {primary};
    border-bottom: 2px solid {primary};
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}}
.metric-card {{
    background: linear-gradient(145deg, {primary}, {secondary});
    padding: 1.2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}}
.violation-alert {{
    background: linear-gradient(90deg, #ef4444, #b91c1c);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    font-weight: 600;
}}
.compliant-status {{
    background: linear-gradient(90deg, #22c55e, #15803d);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    font-weight: 600;
}}
.stButton button {{
    background: linear-gradient(90deg, {primary}, {secondary});
    color: white;
    border-radius: 8px;
    font-weight: 600;
    border: none;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}}
.stButton button:hover {{
    transform: translateY(-2px);
    background: linear-gradient(90deg, {secondary}, {primary});
}}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧭 HEADER
# ==========================================
st.markdown('<h1 class="main-header">🦺 PPE Detection Pro</h1>', unsafe_allow_html=True)
st.markdown("### Advanced Personal Protective Equipment Monitoring System")

# ==========================================
# 🧩 SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.markdown("## ⚙️ Configuration")

st.sidebar.markdown("### Model Labels")
st.sidebar.info(f"Available in model: {', '.join(ALL_MODEL_LABELS.values())}")

# Detection settings
st.sidebar.markdown("### Detection Settings")
st.sidebar.markdown("#### Select PPE Items to Monitor")

detection_items = {}
for item in SUPPORTED_ITEMS:
    detection_items[item] = st.sidebar.checkbox(f"Detect {item}", value=True, key=f"detect_{item}")

st.sidebar.markdown("#### Select Items for Violation Warnings")
warning_items = {}
for item in SUPPORTED_ITEMS:
    warning_items[item] = st.sidebar.checkbox(f"Warn on missing {item}", value=True, key=f"warn_{item}")

confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

selected_detection_items = [i for i, v in detection_items.items() if v]
selected_warning_items = [i for i, v in warning_items.items() if v]

if not selected_detection_items:
    st.sidebar.warning("⚠️ Please select at least one PPE item to detect.")

st.sidebar.markdown("---")
if st.sidebar.button("Clear Session Stats"):
    st.session_state.clear()

# ==========================================
# 📊 METRIC DISPLAY
# ==========================================
def display_metrics(person_count, total_violators, missing_counts, selected_warning_items):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Persons Detected", person_count)
    with col2:
        rate = (total_violators / person_count * 100) if person_count else 0
        st.metric("🚨 Violators", f"{total_violators} ({rate:.1f}%)")
    with col3:
        compliance = ((person_count - total_violators) / person_count * 100) if person_count else 100
        st.metric("✅ Compliance", f"{compliance:.1f}%")

    if total_violators:
        st.markdown('<div class="violation-alert">', unsafe_allow_html=True)
        st.subheader("🚨 Violation Details")
        viol = [f"{i}: {missing_counts[i]} persons" for i in selected_warning_items if missing_counts.get(i, 0) > 0]
        if viol:
            st.write("Missing items: " + ", ".join(viol))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="compliant-status">✅ All Persons are Compliant!</div>', unsafe_allow_html=True)

# ==========================================
# 📈 DATAFRAME HELPER
# ==========================================
def df_from_counts(counts: dict, selected_warning_items: list) -> pd.DataFrame:
    df = pd.DataFrame(
        [{"PPE Item": k, "Missing Count": v} for k, v in counts.items() if k in selected_warning_items and v > 0]
    )
    return df.sort_values("Missing Count", ascending=False).reset_index(drop=True) if not df.empty else pd.DataFrame()

# ==========================================
# 🧩 TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "🎥 Video Analysis", "📊 Analytics"])

# ---------------- IMAGE TAB ----------------
with tab1:
    st.markdown('<div class="section-header">Image Analysis</div>', unsafe_allow_html=True)
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    show_conf = st.checkbox("Show Confidence Scores", True)
    draw_all = st.checkbox("Draw All Detections", False)

    if uploaded_img and selected_detection_items:
        with st.spinner("🔍 Analyzing image..."):
            try:
                img, missing_counts, violators, person_count, summary = detect_ppe_image(
                    uploaded_img, selected_detection_items, selected_warning_items, confidence_threshold, draw_all
                )

                col1, col2 = st.columns([0.7, 0.3])
                with col1:
                    st.image(img, caption="Detection Result", use_container_width=True)
                with col2:
                    display_metrics(person_count, violators, missing_counts, selected_warning_items)

                    df = df_from_counts(missing_counts, selected_warning_items)
                    if not df.empty:
                        chart = alt.Chart(df).mark_bar(color=primary).encode(
                            x='Missing Count:Q', y='PPE Item:N'
                        ).properties(height=300, title="PPE Violations by Item")
                        st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"❌ {str(e)}")

# ---------------- VIDEO TAB ----------------
with tab2:
    st.markdown('<div class="section-header">Video Analysis</div>', unsafe_allow_html=True)
    uploaded_vid = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    if uploaded_vid and selected_detection_items:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_vid.read())
            input_path = tmp.name

        st.info("🎬 Processing video... please wait.")
        progress = st.progress(0)

        def update_prog(p): progress.progress(p)
        try:
            out_path, missing_counts, violators, persons, summary = detect_ppe_video(
                input_path, "output.mp4", selected_detection_items, selected_warning_items, confidence_threshold, progress_callback=update_prog
            )
            st.video(out_path)
            display_metrics(persons, violators, missing_counts, selected_warning_items)
        except Exception as e:
            st.error(f"❌ {str(e)}")

# ---------------- ANALYTICS TAB ----------------
with tab3:
    st.markdown('<div class="section-header">Analytics & Documentation</div>', unsafe_allow_html=True)
    st.markdown("""
    **How It Works**
    1. Upload your image or video.
    2. Choose PPE items to detect.
    3. Set which ones trigger warnings.
    4. View results and download analysis.
    """)

# Footer
st.markdown("<hr><center>🦺 PPE Detection Pro | Powered by Streamlit</center>", unsafe_allow_html=True)

