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

from utils.detect import detect_ppe_image, detect_ppe_video, DEFAULT_PPE_ITEMS

# Page config
st.set_page_config(page_title="PPE Detector", layout="wide", initial_sidebar_state="expanded")
st.title("🦺 PPE Detector")

# Sidebar: select PPE
st.sidebar.header("Select PPE items to monitor")
selected_items = []
for item in DEFAULT_PPE_ITEMS:
    if st.sidebar.checkbox(item.capitalize(), value=True):
        selected_items.append(item)

if not selected_items:
    st.sidebar.warning("Select at least one PPE item to enable detection.")

st.sidebar.markdown("---")
st.sidebar.info("If your model uses different label names (e.g. 'hardhat' for helmet), add synonyms in utils/detect.PPE_SYNONYMS.")

# Tabs
tab_img, tab_vid = st.tabs(["Image", "Video"])

def df_from_counts(counts: dict) -> pd.DataFrame:
    df = pd.DataFrame({"Item": list(counts.keys()), "Missing Count": list(counts.values())})
    return df.sort_values("Missing Count", ascending=False).reset_index(drop=True)

# IMAGE TAB
with tab_img:
    st.header("Image inspector")
    uploaded_img = st.file_uploader("Upload image (jpg, png)", type=["jpg", "jpeg", "png"], key="img")

    if uploaded_img is not None:
        if not selected_items:
            st.error("Please select at least one PPE item in the sidebar.")
        else:
            with st.spinner("Running detection..."):
                start = time.time()
                annotated_pil, missing_counts, total_violators, person_count = detect_ppe_image(uploaded_img, selected_items)
                elapsed = time.time() - start

            st.success(f"Inference done in {elapsed:.2f}s")
            col1, col2 = st.columns([0.7, 0.3])

            with col1:
                st.image(annotated_pil, caption="Annotated result", use_column_width=True)
                # Download annotated image
                buf = io.BytesIO()
                annotated_pil.save(buf, format="PNG")
                buf.seek(0)
                st.download_button("⬇️ Download annotated image", data=buf.getvalue(), file_name="annotated_ppe.png", mime="image/png")

            with col2:
                st.subheader("Summary")
                st.metric("Persons detected (approx)", person_count)
                st.metric("Violators (persons missing ≥1 selected item)", total_violators)
                df = df_from_counts(missing_counts)
                st.table(df)

                if not df.empty:
                    chart = alt.Chart(df).mark_bar().encode(
                        x=alt.X("Missing Count:Q"),
                        y=alt.Y("Item:N", sort='-x'),
                        tooltip=["Item", "Missing Count"]
                    ).properties(height=250)
                    st.altair_chart(chart, use_container_width=True)

# VIDEO TAB
with tab_vid:
    st.header("Video inspector")
    uploaded_vid = st.file_uploader("Upload short video (mp4, mov, avi)", type=["mp4", "mov", "avi"], key="vid")
    if uploaded_vid is not None:
        if not selected_items:
            st.error("Please select at least one PPE item in the sidebar.")
        else:
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_vid.name)[1])
            tmpf.write(uploaded_vid.read())
            tmpf.flush()
            tmpf.close()

            st.info("Processing video — this may take some time for long clips.")
            out_path = "output_annotated.mp4"
            start = time.time()
            try:
                out_path, missing_counts, violator_events, persons_seen = detect_ppe_video(tmpf.name, out_path, selected_items)
                elapsed = time.time() - start
                st.success(f"Processed in {elapsed:.1f}s")
                st.video(out_path)

                st.subheader("Summary")
                st.metric("Persons seen (sum frames approx)", persons_seen)
                st.metric("Violator events (approx)", violator_events)
                df = df_from_counts(missing_counts)
                st.table(df)
                if not df.empty:
                    chart = alt.Chart(df).mark_bar().encode(
                        x=alt.X("Missing Count:Q"),
                        y=alt.Y("Item:N", sort='-x'),
                        tooltip=["Item", "Missing Count"]
                    ).properties(height=250)
                    st.altair_chart(chart, use_container_width=True)

                # downloads
                with open(out_path, "rb") as f:
                    st.download_button("⬇️ Download annotated video", f, file_name="annotated_ppe_video.mp4", mime="video/mp4")
                csv_buf = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download counts (CSV)", csv_buf, file_name="ppe_video_counts.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error processing video: {e}")
            finally:
                try:
                    os.remove(tmpf.name)
                except Exception:
                    pass

