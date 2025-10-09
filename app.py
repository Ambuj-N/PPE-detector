import streamlit as st
from PIL import Image
import tempfile
import os
import io
import time
import pandas as pd
import altair as alt

from utils.detect import detect_ppe_image, detect_ppe_video, DEFAULT_PPE_ITEMS

# ---------------------------
# Page config & CSS (clean modern look)
# ---------------------------
st.set_page_config(page_title="PPE Detector", layout="wide", initial_sidebar_state="expanded")

# inject a modern font + small CSS polish
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  {
      font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    .header {
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .logo {
      background: linear-gradient(90deg,#0ea5a4,#7c3aed);
      color: white;
      padding: 10px 14px;
      border-radius: 10px;
      font-weight: 700;
      font-size: 18px;
    }
    .subtitle { color: #6b7280; margin-top: -6px; }
    .metric { font-size: 18px; font-weight: 600; }
    .small { font-size: 13px; color:#6b7280; }
    .card { border-radius: 12px; padding: 10px; background: #ffffff; box-shadow: 0 6px 18px rgba(0,0,0,0.04); }
    .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 8px; margin: 10px 0; }
    .info-box { background-color: #e3f2fd; border: 1px solid #bbdefb; padding: 10px; border-radius: 8px; margin: 10px 0; }
    .success-box { background-color: #e8f5e9; border: 1px solid #c8e6c9; padding: 10px; border-radius: 8px; margin: 10px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
col1, col2 = st.columns([0.12, 0.88])
with col1:
    st.markdown('<div class="logo">PPE</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="header"><div style="font-size:22px; font-weight:700">PPE Detector</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Detect helmets, vests, gloves, goggles, and masks. Select what to enforce and get clear counts & downloads.</div>', unsafe_allow_html=True)

st.write("")  # spacer

# Sidebar: Controls
st.sidebar.header("Configuration & Controls")

st.sidebar.markdown("**Select PPE items to monitor** (only selected items will be checked):")
selected_items = []
for item in DEFAULT_PPE_ITEMS:
    if st.sidebar.checkbox(item.capitalize(), value=True):
        selected_items.append(item)

if len(selected_items) == 0:
    st.sidebar.warning("Select at least one PPE item to enable detection.")

show_annotated_bbox = st.sidebar.checkbox("Show annotated bounding boxes (on preview)", value=True)
show_counts_table = st.sidebar.checkbox("Show counts table", value=True)
auto_download_model_info = st.sidebar.checkbox("Show model download info", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Processing options**")
progress_mode = st.sidebar.selectbox("Progress indicator style", ["Minimal", "Detailed"], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick actions**")
if st.sidebar.button("Reset selections"):
    # simple client-side reset: refresh page
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Help**")
st.sidebar.info("Upload an image (jpg/png) or short video (mp4). The UI shows per-item missing counts and total violators. Use downloads to save results.")

# Tabs for Image / Video
tab_img, tab_vid = st.tabs(["Image Detection", "Video Detection"])

# A small area to show model status/time — will be populated after first detection
model_info_placeholder = st.empty()

# Helper: convert PIL -> bytes for download
def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

# Helper: format missing_counts dict to dataframe
def counts_to_df(missing_counts: dict) -> pd.DataFrame:
    items = list(missing_counts.keys())
    counts = [missing_counts[k] for k in items]
    df = pd.DataFrame({"item": items, "missing_count": counts})
    df = df.sort_values("missing_count", ascending=False).reset_index(drop=True)
    return df

# ---------------------------
# IMAGE TAB
# ---------------------------
with tab_img:
    st.subheader("Image Inspection")
    colA, colB = st.columns([0.6, 0.4])

    with colA:
        uploaded_img = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"], key="img_upload")
        st.caption("Tip: use images with people clearly visible. Best results when persons and PPE are not extremely small.")
    with colB:
        st.markdown("**Selected PPE**")
        if selected_items:
            st.markdown(", ".join([f"`{s}`" for s in selected_items]))
        else:
            st.markdown("_No items selected_")

    if uploaded_img is not None:
        if not selected_items:
            st.error("Select at least one PPE item in the sidebar to run detection.")
        else:
            start = time.time()
            processing_msg = st.empty()
            if progress_mode == "Detailed":
                processing_msg.info("Running model inference... (this may take a few seconds)")
            # run detection
            try:
                annotated_pil, missing_counts, total_violators, person_count = detect_ppe_image(uploaded_img, selected_items)
                duration = time.time() - start
                # model info
                if auto_download_model_info:
                    model_info_placeholder.info(f"Model loaded and inference finished in {duration:.2f}s")
                # layout for results
                res_col1, res_col2 = st.columns([0.65, 0.35])
                with res_col1:
                    st.markdown("**Annotated preview**")
                    if show_annotated_bbox:
                        st.image(annotated_pil, use_column_width=True)
                    else:
                        st.image(annotated_pil, use_column_width=True)

                    # downloads
                    buf = pil_to_bytes(annotated_pil, fmt="PNG")
                    st.download_button("⬇️ Download annotated image (PNG)", data=buf, file_name="annotated_ppe.png", mime="image/png")

                with res_col2:
                    st.markdown("**Key metrics**")
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("Persons (approx)", person_count)
                    col_m2.metric("Violators", total_violators)
                    st.markdown("**Violation rate**")
                    rate = 0.0
                    try:
                        rate = (total_violators / max(1, person_count)) * 100
                    except Exception:
                        rate = 0.0
                    st.progress(min(1.0, rate / 100.0))
                    st.caption(f"{rate:.1f}% violator rate (persons missing ≥1 selected item)")

                    # counts table & chart
                    df_counts = counts_to_df(missing_counts)
                    if show_counts_table:
                        st.markdown("**Missing counts**")
                        st.table(df_counts.rename(columns={"item":"Item", "missing_count":"Missing Count"}))
                    # altair chart
                    if not df_counts.empty:
                        chart = alt.Chart(df_counts).mark_bar().encode(
                            x=alt.X("missing_count:Q", title="Missing count"),
                            y=alt.Y("item:N", sort='-x', title="PPE Item"),
                            tooltip=["item", "missing_count"]
                        ).properties(height=220)
                        st.altair_chart(chart, use_container_width=True)

                    # CSV download
                    csv_buf = df_counts.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download counts (CSV)", csv_buf, file_name="ppe_missing_counts.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Error during detection: {e}")
            finally:
                if progress_mode == "Detailed":
                    processing_msg.empty()

# ---------------------------
# VIDEO TAB
# ---------------------------
with tab_vid:
    st.subheader("Video Inspection")
    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        uploaded_vid = st.file_uploader("Upload a short video (mp4, mov, avi) — keep clips small for testing", type=["mp4","mov","avi"], key="vid_upload")
        st.caption("Note: video processing is frame-by-frame and may be slow; small clips are recommended for quick testing.")
    with col2:
        st.markdown("**Selected PPE**")
        if selected_items:
            st.markdown(", ".join([f"`{s}`" for s in selected_items]))
        else:
            st.markdown("_No items selected_")

    if uploaded_vid is not None:
        if not selected_items:
            st.error("Select at least one PPE item in the sidebar to run detection.")
        else:
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_vid.name)[1])
            tmpf.write(uploaded_vid.read())
            tmpf.flush()
            tmpf.close()

            st.info("Processing video — this can take time. Please wait.")
            progress_bar = st.progress(0.0)
            start = time.time()
            
            # Define progress callback
            def update_progress(progress):
                progress_bar.progress(progress)
            
            try:
                # The detect_ppe_video writes output file and returns counts
                output_path = "output_annotated.mp4"
                out_path, missing_counts, total_violators, person_count = detect_ppe_video(
                    tmpf.name, output_path, selected_items, progress_callback=update_progress
                )
                duration = time.time() - start
                st.success(f"Video processed in {duration:.1f}s")
                
                # show video
                st.video(out_path)
                
                # show metrics
                metrics_col1, metrics_col2 = st.columns(2)
                metrics_col1.metric("Persons (approx total frames)", person_count)
                metrics_col2.metric("Violator events (approx)", total_violators)
                
                # counts table & chart
                df_counts = counts_to_df(missing_counts)
                if show_counts_table:
                    st.markdown("**Missing counts**")
                    st.table(df_counts.rename(columns={"item":"Item", "missing_count":"Missing Count"}))
                
                if not df_counts.empty:
                    chart = alt.Chart(df_counts).mark_bar().encode(
                        x=alt.X("missing_count:Q", title="Missing count"),
                        y=alt.Y("item:N", sort='-x', title="PPE Item"),
                        tooltip=["item", "missing_count"]
                    ).properties(height=220)
                    st.altair_chart(chart, use_container_width=True)

                # downloads
                with open(out_path, "rb") as f:
                    st.download_button("⬇️ Download annotated video", f, file_name="annotated_ppe_video.mp4", mime="video/mp4")
                
                # CSV
                csv_buf = df_counts.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download counts (CSV)", csv_buf, file_name="ppe_missing_counts_video.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Error while processing video: {e}")
            finally:
                progress_bar.progress(1.0)
                try:
                    os.remove(tmpf.name)
                except Exception:
                    pass

# ---------------------------
# Footer / extra tips
# ---------------------------
st.markdown("---")
st.markdown(
    """
    **Tips & Notes**  
    - This tool uses a YOLO model — detection quality depends on the model you trained.  
    - If your model uses different label names (e.g., `hardhat` instead of `helmet`), update `DEFAULT_PPE_ITEMS` in `utils/detect.py`.  
    - Video violator counts are frame-based and may overcount the same person across frames (tracking would be required for perfect unique-person counts).  
    """
)
