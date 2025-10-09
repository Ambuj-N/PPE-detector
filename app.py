# app.py

import streamlit as st
from PIL import Image
import tempfile
import os

from utils.detect import detect_ppe_image, detect_ppe_video

st.set_page_config(page_title="PPE Detector", layout="wide")
st.title("🦺 PPE Detector")

st.markdown(
    "Upload an **image** or **video**. The app will detect PPE items (helmet, vest, gloves, goggles, mask) "
    "and flag missing items. Model is downloaded automatically from the Hugging Face model repo."
)

uploaded = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded is None:
    st.info("Upload a file to get started.")
else:
    file_type = uploaded.type
    if file_type.startswith("image"):
        st.write("Processing image...")
        try:
            annotated_pil, missing = detect_ppe_image(uploaded)
            st.image(annotated_pil, caption="Annotated result", use_column_width=True)
            if missing:
                st.warning(f"⚠️ Missing PPE items detected: {', '.join(missing)}")
            else:
                st.success("✅ All required PPE items detected (for the detected persons).")
        except Exception as e:
            st.error(f"Error during detection: {e}")

    elif file_type.startswith("video"):
        st.write("Processing video (this may take time depending on video length)...")
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded.name)[1], delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        out_path = "output_annotated.mp4"
        try:
            out_video_path, missing = detect_ppe_video(tmp_path, output_video_path=out_path)
            st.video(out_video_path)
            if missing:
                st.warning(f"⚠️ Missing PPE items found in video frames: {', '.join(missing)}")
            else:
                st.success("✅ All required PPE items detected across video frames.")
        except Exception as e:
            st.error(f"Error processing video: {e}")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    else:
        st.error("Unsupported file type.")

