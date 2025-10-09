# app.py

import streamlit as st
from PIL import Image
import tempfile
import os
from utils.detect import detect_ppe_image, detect_ppe_video, DEFAULT_PPE_ITEMS

st.set_page_config(page_title="PPE Detector", layout="wide")
st.title("🦺 PPE Detector")

st.markdown(
    "Upload an **image** or **video**, then select which PPE items should be checked. "
    "The app will detect missing PPE items and report violations."
)

# User-selected PPE
st.sidebar.header("Select PPE to Monitor")
selected_items = [
    item for item in DEFAULT_PPE_ITEMS if st.sidebar.checkbox(item.capitalize(), True)
]
if not selected_items:
    st.sidebar.warning("Select at least one PPE item.")
st.sidebar.info("Default items: helmet, vest, gloves, goggles, mask")

uploaded = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded:
    if uploaded.type.startswith("image"):
        st.write("🔍 Detecting PPE in image...")
        annotated, missing_counts, total_violators = detect_ppe_image(uploaded, selected_items)

        st.image(annotated, caption="Detection Result", use_column_width=True)

        st.subheader("📊 Detection Summary")
        st.table(
            {"Item": list(missing_counts.keys()), "Missing Count": list(missing_counts.values())}
        )

        if total_violators > 0:
            st.warning(f"⚠️ Total persons violating PPE rules: {total_violators}")
        else:
            st.success("✅ All required PPE detected properly.")
    else:
        st.write("🎥 Detecting PPE in video (please wait)...")
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded.name)[1], delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        out_path = "output_annotated.mp4"
        annotated_video, missing_counts, total_violators = detect_ppe_video(tmp_path, out_path, selected_items)

        st.video(annotated_video)
        st.subheader("📊 Detection Summary")
        st.table(
            {"Item": list(missing_counts.keys()), "Missing Count": list(missing_counts.values())}
        )

        if total_violators > 0:
            st.warning(f"⚠️ Total persons violating PPE rules: {total_violators}")
        else:
            st.success("✅ All required PPE detected properly.")

        os.remove(tmp_path)
else:
    st.info("Please upload a file to start.")

