# app.py

import streamlit as st
from PIL import Image
import tempfile
import os
from utils.detect import detect_ppe_image, detect_ppe_video, DEFAULT_PPE_ITEMS

st.set_page_config(page_title="PPE Detector", layout="wide")
st.title("🦺 PPE Detector")

st.markdown(
    "Upload an **image** or **video**, then select which PPE items to check for in the sidebar. "
    "Only selected items are checked and reported."
)

# Sidebar PPE selection
st.sidebar.header("Select PPE to Monitor")
selected_items = []
for item in DEFAULT_PPE_ITEMS:
    if st.sidebar.checkbox(item.capitalize(), value=True):
        selected_items.append(item)

if not selected_items:
    st.sidebar.warning("Please select at least one PPE item to monitor.")

uploaded = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded is None:
    st.info("Upload a file to get started.")
else:
    file_type = uploaded.type
    if file_type.startswith("image"):
        if not selected_items:
            st.error("No PPE items selected. Please choose items in the sidebar.")
        else:
            st.write("🔍 Detecting selected PPE in image...")
            try:
                annotated_pil, missing_counts, total_violators, person_count = detect_ppe_image(uploaded, selected_items)
                st.image(annotated_pil, caption="Annotated result", use_column_width=True)

                st.subheader("📊 Detection Summary")
                # Show only selected items in table (missing_counts already only contains selected items)
                rows = [{"Item": k, "Missing Count": v} for k, v in missing_counts.items()]
                st.table(rows)

                st.write(f"👤 Persons detected (approx): {person_count}")
                if total_violators > 0:
                    st.warning(f"⚠️ Total persons missing at least one selected item: {total_violators}")
                else:
                    st.success("✅ No violations detected for selected items.")
            except Exception as e:
                st.exception(f"Error during detection: {e}")

    elif file_type.startswith("video"):
        if not selected_items:
            st.error("No PPE items selected. Please choose items in the sidebar.")
        else:
            st.write("🎥 Processing video (may take time)...")
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded.name)[1], delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            out_path = "output_annotated.mp4"
            try:
                out_video_path, missing_counts, total_violators, person_count = detect_ppe_video(tmp_path, out_path, selected_items)
                st.video(out_video_path)
                st.subheader("📊 Detection Summary")
                rows = [{"Item": k, "Missing Count": v} for k, v in missing_counts.items()]
                st.table(rows)
                st.write(f"👤 Persons detected across video (approx sum frames): {person_count}")
                if total_violators > 0:
                    st.warning(f"⚠️ Total persons missing at least one selected item (approx): {total_violators}")
                else:
                    st.success("✅ No violations detected for selected items.")
            except Exception as e:
                st.exception(f"Error processing video: {e}")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    else:
        st.error("Unsupported file type.")

