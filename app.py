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

# Page configuration
st.set_page_config(
    page_title="PPE Detector Pro", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🦺"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .violation-alert {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #cc0000;
    }
    .compliant-status {
        background-color: #00C851;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #007E33;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e86ab;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🦺 PPE Detection Pro</h1>', unsafe_allow_html=True)
st.markdown("### Advanced Personal Protective Equipment Monitoring System")

# Sidebar Configuration
st.sidebar.markdown("## ⚙️ Configuration")

# Model Information
st.sidebar.markdown("### Model Labels")
st.sidebar.info(f"Available in model: {', '.join(ALL_MODEL_LABELS.values())}")

# Detection Settings
st.sidebar.markdown("### Detection Settings")

# PPE Items to Detect
st.sidebar.markdown("#### Select PPE Items to Monitor")
detection_items = {}
for item in SUPPORTED_ITEMS:
    detection_items[item] = st.sidebar.checkbox(f"Detect {item}", value=True, key=f"detect_{item}")

# Warning Settings
st.sidebar.markdown("#### Select Items for Violation Warnings")
warning_items = {}
for item in SUPPORTED_ITEMS:
    warning_items[item] = st.sidebar.checkbox(f"Warn on missing {item}", value=True, key=f"warn_{item}")

# Confidence threshold
confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# Get selected items for detection and warnings
selected_detection_items = [item for item, selected in detection_items.items() if selected]
selected_warning_items = [item for item, selected in warning_items.items() if selected]

if not selected_detection_items:
    st.sidebar.warning("⚠️ Please select at least one PPE item to detect.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Statistics")
if st.sidebar.button("Clear Session Stats"):
    st.session_state.clear()

# Helper function for metrics display
def display_metrics(person_count, total_violators, missing_counts, selected_warning_items):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("👥 Persons Detected", person_count)
    
    with col2:
        violation_rate = (total_violators / person_count * 100) if person_count > 0 else 0
        st.metric("🚨 Violators", f"{total_violators} ({violation_rate:.1f}%)")
    
    with col3:
        compliance_rate = ((person_count - total_violators) / person_count * 100) if person_count > 0 else 100
        st.metric("✅ Compliance Rate", f"{compliance_rate:.1f}%")
    
    # Violation details
    if total_violators > 0:
        st.markdown('<div class="violation-alert">', unsafe_allow_html=True)
        st.subheader("🚨 Violation Details")
        
        violation_details = []
        for item in selected_warning_items:
            if missing_counts.get(item, 0) > 0:
                violation_details.append(f"{item}: {missing_counts[item]} persons")
        
        if violation_details:
            st.write("Missing items: " + ", ".join(violation_details))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="compliant-status">', unsafe_allow_html=True)
        st.subheader("✅ All Persons are Compliant!")
        st.markdown('</div>', unsafe_allow_html=True)

# Dataframe helper
def df_from_counts(counts: dict, selected_warning_items: list) -> pd.DataFrame:
    filtered_counts = {k: v for k, v in counts.items() if k in selected_warning_items}
    if not filtered_counts:
        return pd.DataFrame()
    df = pd.DataFrame({"PPE Item": list(filtered_counts.keys()), "Missing Count": list(filtered_counts.values())})
    return df.sort_values("Missing Count", ascending=False).reset_index(drop=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "🎥 Video Analysis", "📈 Analytics"])

with tab1:
    st.markdown('<div class="section-header">Image Analysis</div>', unsafe_allow_html=True)
    
    col_upload, col_settings = st.columns([2, 1])
    
    with col_upload:
        uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], 
                                      help="Upload an image for PPE compliance analysis")
    
    with col_settings:
        st.markdown("### Analysis Settings")
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        draw_all_detections = st.checkbox("Draw All Detections", value=False)
    
    if uploaded_img is not None:
        if not selected_detection_items:
            st.error("❌ Please select at least one PPE item to detect in the sidebar.")
        else:
            with st.spinner("🔍 Analyzing image for PPE compliance..."):
                start = time.time()
                try:
                    # FIXED: Added all required parameters
                    annotated_pil, missing_counts, total_violators, person_count, detection_summary = detect_ppe_image(
                        uploaded_img, 
                        selected_detection_items,
                        selected_warning_items,
                        confidence_threshold,
                        draw_all_detections
                    )
                    elapsed = time.time() - start
                    
                    st.success(f"✅ Analysis completed in {elapsed:.2f}s")
                    
                    # Display results
                    col1, col2 = st.columns([0.7, 0.3])
                    
                    with col1:
                        st.image(annotated_pil, caption="PPE Compliance Analysis Result", use_column_width=True)
                        
                        # Download button for annotated image
                        buf = io.BytesIO()
                        annotated_pil.save(buf, format="PNG", quality=95)
                        buf.seek(0)
                        
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            st.download_button(
                                "💾 Download Annotated Image",
                                data=buf.getvalue(),
                                file_name="ppe_analysis.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    
                    with col2:
                        st.markdown("### 📊 Summary")
                        display_metrics(person_count, total_violators, missing_counts, selected_warning_items)
                        
                        # Detailed counts
                        df = df_from_counts(missing_counts, selected_warning_items)
                        if not df.empty:
                            st.markdown("#### Missing PPE Items")
                            st.table(df)
                            
                            # Chart
                            chart = alt.Chart(df).mark_bar(color='#ff6b6b').encode(
                                x=alt.X('Missing Count:Q', title='Number of Violations'),
                                y=alt.Y('PPE Item:N', sort='-x', title='PPE Item'),
                                tooltip=['PPE Item', 'Missing Count']
                            ).properties(
                                height=300,
                                title='PPE Violations by Item'
                            )
                            st.altair_chart(chart, use_container_width=True)
                        
                        # Detection summary
                        if detection_summary:
                            st.markdown("#### Detection Summary")
                            for item, count in detection_summary.items():
                                if count > 0:
                                    st.write(f"• {item}: {count} detected")
                
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.exception(e)

with tab2:
    st.markdown('<div class="section-header">Video Analysis</div>', unsafe_allow_html=True)
    
    uploaded_vid = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], 
                                  help="Upload a video for PPE compliance analysis")
    
    if uploaded_vid is not None:
        if not selected_detection_items:
            st.error("❌ Please select at least one PPE item to detect in the sidebar.")
        else:
            # Save uploaded video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_vid.name)[1]) as tmpf:
                tmpf.write(uploaded_vid.read())
                input_path = tmpf.name
            
            st.info("🎬 Processing video - this may take some time depending on video length.")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            output_path = "annotated_video_output.mp4"
            
            try:
                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {progress:.0%}")
                
                start = time.time()
                # FIXED: Added all required parameters for video detection
                out_path, missing_counts, violator_events, persons_seen, detection_summary = detect_ppe_video(
                    input_path, 
                    output_path, 
                    selected_detection_items,
                    selected_warning_items,
                    confidence_threshold,
                    progress_callback=update_progress
                )
                elapsed = time.time() - start
                
                progress_bar.progress(100)
                status_text.text(f"✅ Video processing completed in {elapsed:.1f}s")
                
                # Display results
                st.video(out_path)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Video Summary")
                    display_metrics(persons_seen, violator_events, missing_counts, selected_warning_items)
                
                with col2:
                    # Download buttons
                    with open(out_path, "rb") as f:
                        st.download_button(
                            "💾 Download Annotated Video",
                            f,
                            file_name="ppe_video_analysis.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                    
                    df = df_from_counts(missing_counts, selected_warning_items)
                    if not df.empty:
                        csv_buf = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📊 Download Violation Data (CSV)",
                            csv_buf,
                            file_name="ppe_violations.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                # Detailed analysis
                if not df.empty:
                    st.markdown("#### Violation Analysis")
                    col_chart, col_table = st.columns([2, 1])
                    
                    with col_chart:
                        chart = alt.Chart(df).mark_bar(color='#ff6b6b').encode(
                            x=alt.X('Missing Count:Q', title='Violation Count'),
                            y=alt.Y('PPE Item:N', sort='-x', title='PPE Item'),
                            tooltip=['PPE Item', 'Missing Count']
                        ).properties(
                            height=300,
                            title='PPE Violations Throughout Video'
                        )
                        st.altair_chart(chart, use_container_width=True)
                    
                    with col_table:
                        st.table(df)
            
            except Exception as e:
                st.error(f"❌ Video processing failed: {str(e)}")
                st.exception(e)
            finally:
                # Cleanup
                try:
                    os.unlink(input_path)
                    if os.path.exists(output_path):
                        os.unlink(output_path)
                except Exception:
                    pass

with tab3:
    st.markdown('<div class="section-header">Analytics & Documentation</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 How It Works")
        st.markdown("""
        1. **Upload** images or videos from your worksite
        2. **Configure** which PPE items to monitor
        3. **Select** which violations trigger warnings
        4. **Analyze** automatic PPE compliance detection
        5. **Review** detailed violation reports
        """)
        
        st.markdown("### 🎯 Detection Capabilities")
        st.markdown(f"""
        - **Persons**: {ALL_MODEL_LABELS[5]}
        - **PPE Items**: {', '.join([ALL_MODEL_LABELS[0], ALL_MODEL_LABELS[1], ALL_MODEL_LABELS[7]])}
        - **Violation Indicators**: {', '.join([ALL_MODEL_LABELS[2], ALL_MODEL_LABELS[3], ALL_MODEL_LABELS[4]])}
        - **Additional Objects**: {', '.join([ALL_MODEL_LABELS[6], ALL_MODEL_LABELS[8], ALL_MODEL_LABELS[9]])}
        """)
    
    with col2:
        st.markdown("### ⚠️ Common Issues")
        st.markdown("""
        - **Low detection accuracy**: Adjust confidence threshold
        - **Missing violations**: Ensure correct PPE items are selected
        - **Poor video quality**: Use well-lit, clear footage
        - **Slow processing**: Reduce video resolution if needed
        """)
        
        st.markdown("### 📈 Performance Tips")
        st.markdown("""
        - Use images with clear visibility of persons
        - Ensure proper lighting conditions
        - For videos, keep camera stable
        - Higher confidence = fewer false positives
        - Lower confidence = more detections (may include false positives)
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "PPE Detection Pro | Safety Compliance Monitoring System"
    "</div>",
    unsafe_allow_html=True
)
