# app.py
import streamlit as st
from PIL import Image
import tempfile
import cv2
import numpy as np
import pandas as pd
import altair as alt

from utils.detect import detect_ppe_image, detect_ppe_video, load_model, get_model_labels

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
# THEME: Dark-only (compact) + palette
# ==========================================
# Force dark theme only and set a compact palette
st.session_state["theme"] = "dark"
primary = "#3b82f6"
secondary = "#8b5cf6"
bg = "#0b1020"        # deep navy background
text = "#e6eef8"      # soft off-white for text

# ==========================================
# 💅 CUSTOM STYLES - MODERN DESIGN
# ==========================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=PT+Sans:wght@400;700&display=swap');

/* Compact dark theme (PT Sans) */
.stApp {{
    background: linear-gradient(180deg, {bg} 0%, #071029 100%);
    font-family: 'PT Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    color: {text};
    padding: 0.5rem 0.75rem;
}}

/* Main content text visibility */
.main .block-container {{
    color: {'#1e293b' if st.session_state["theme"] == "light" else '#f1f5f9'};
}}

.main p, .main span, .main label {{
    color: {'#334155' if st.session_state["theme"] == "light" else '#e2e8f0'} !important;
}}

.main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {{
    color: {'#1e293b' if st.session_state["theme"] == "light" else '#f1f5f9'} !important;
}}

/* Main Header with Gradient Animation */
.main-header {{
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    margin: 0 0 0.6rem 0;
    padding: 0.35rem 0.6rem;
    letter-spacing: 0.4px;
    display: inline-block;
    border-radius: 8px;
    background: linear-gradient(90deg, rgba(59,130,246,0.06), rgba(139,92,246,0.04));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientFlow 5s linear infinite;
    text-shadow: 0 6px 20px rgba(59, 130, 246, 0.12);
}}

@keyframes gradientFlow {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

/* Subtitle */
.subtitle {{
    text-align: center;
    color: #9fb6d9;
    font-size: 0.95rem;
    font-weight: 400;
    margin-bottom: 0.8rem;
    opacity: 0.98;
}}

/* Compact main container padding */
.main .block-container {{
    padding-top: 0.6rem;
    padding-left: 0.9rem;
    padding-right: 0.9rem;
    padding-bottom: 0.9rem;
}}

/* Make checkbox label larger and higher contrast */
.stCheckbox > label, .stCheckbox .stMarkdown {{
    font-size: 0.98rem;
    color: #eaf4ff !important;
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(-10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

/* Section Headers with Glass Effect */
.section-header {{
    font-size: 1.2rem;
    font-weight: 700;
    background: rgba(255,255,255,0.03);
    padding: 0.6rem 1rem;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.04);
    margin-bottom: 1rem;
    color: {text};
    animation: slideIn 0.5s ease-out;
}}

@keyframes slideIn {{
    from {{ opacity: 0; transform: translateX(-20px); }}
    to {{ opacity: 1; transform: translateX(0); }}
}}

/* Sidebar Styling */
[data-testid="stSidebar"] {{
    background: rgba(255,255,255,0.02);
    border-right: 1px solid rgba(255,255,255,0.04);
    padding: 0.6rem;
    width: 300px;
}}

[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {{
    color: {text} !important;
    font-weight: 700;
    margin: 0.25rem 0 0.5rem 0;
}}

[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {{
    color: #9fb6d9 !important;
}}

[data-testid="stSidebar"] .stMarkdown {{
    color: {text} !important;
}}

/* Checkbox Styling */
.stCheckbox {{
    padding: 0.5rem;
    border-radius: 8px;
    transition: all 0.3s ease;
}}

.stCheckbox:hover {{
    background: rgba(59, 130, 246, 0.1);
    transform: translateX(5px);
}}

/* Select All Box - Special Styling */
.select-all-box {{
    background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(139,92,246,0.12));
    padding: 0.6rem;
    border-radius: 10px;
    margin-bottom: 0.6rem;
    box-shadow: 0 6px 22px rgba(59, 130, 246, 0.08);
    animation: pulse 2.5s ease-in-out infinite;
}}

@keyframes pulse {{
    0%, 100% {{ box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3); }}
    50% {{ box-shadow: 0 4px 25px rgba(59, 130, 246, 0.5); }}
}}

/* Buttons */
.stButton button {{
    background: linear-gradient(135deg, {primary}, {secondary});
    color: white;
    border-radius: 10px;
    font-weight: 600;
    border: none;
    padding: 0.5rem 1rem;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.12);
    transition: all 0.22s ease;
    text-transform: none;
    letter-spacing: 0.6px;
    font-size: 0.9rem;
}}

.stButton button:hover {{
    transform: translateY(-3px);
    box-shadow: 0 6px 25px rgba(59, 130, 246, 0.6);
    background: linear-gradient(135deg, #2563eb, #7c3aed);
}}

/* Metrics Cards */
.stMetric {{
    background: rgba(255,255,255,0.02);
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.03);
    box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    transition: all 0.22s ease;
}}

.stMetric:hover {{
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(59, 130, 246, 0.3);
}}

/* Violation Alert with Animation */
.violation-alert {{
    background: linear-gradient(135deg, rgba(239,68,68,0.22), rgba(220,38,38,0.18));
    color: white;
    padding: 1rem;
    border-radius: 12px;
    font-weight: 700;
    box-shadow: 0 6px 20px rgba(239, 68, 68, 0.12);
    animation: shake 0.45s ease-in-out;
    border: 1px solid rgba(255,255,255,0.03);
}}

@keyframes shake {{
    0%, 100% {{ transform: translateX(0); }}
    25% {{ transform: translateX(-5px); }}
    75% {{ transform: translateX(5px); }}
}}

/* Compliant Status */
.compliant-status {{
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    font-weight: 600;
    box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
    animation: fadeIn 0.8s ease-in;
    border: 2px solid rgba(255,255,255,0.2);
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 1rem;
    background: transparent;
}}

.stTabs [data-baseweb="tab"] {{
    background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
    backdrop-filter: blur(10px);
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    border: 1px solid rgba(255,255,255,0.18);
    color: {'#1e293b' if st.session_state["theme"] == "light" else '#f1f5f9'};
    font-weight: 600;
    transition: all 0.3s ease;
}}

.stTabs [data-baseweb="tab"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
}}

/* File Uploader */
.stFileUploader {{
    background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
    backdrop-filter: blur(10px);
    padding: 2rem;
    border-radius: 15px;
    border: 2px dashed rgba(59, 130, 246, 0.4);
    transition: all 0.3s ease;
}}

.stFileUploader:hover {{
    border-color: rgba(59, 130, 246, 0.8);
    background: rgba(59, 130, 246, 0.05);
}}

/* Progress Bar */
.stProgress > div > div > div > div {{
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
    background-size: 200% 100%;
    animation: progressFlow 1.5s linear infinite;
}}

@keyframes progressFlow {{
    0% {{ background-position: 0% 0%; }}
    100% {{ background-position: 200% 0%; }}
}}

/* Spinner */
.stSpinner > div {{
    border-top-color: #3b82f6 !important;
    animation: spin 1s linear infinite;
}}

/* Image Container */
.stImage {{
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}}

.stImage:hover {{
    transform: scale(1.02);
    box-shadow: 0 12px 48px rgba(59, 130, 246, 0.3);
}}

/* Slider */
.stSlider {{
    padding: 1rem 0;
}}

.stSlider > div > div > div > div {{
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
}}

/* Info Box */
.stInfo {{
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.15)) !important;
    backdrop-filter: blur(10px);
    border-left: 4px solid #3b82f6 !important;
    border-radius: 10px;
    animation: slideIn 0.5s ease-out;
    color: {'#1e293b' if st.session_state["theme"] == "light" else '#f1f5f9'} !important;
}}

.stInfo p, .stInfo span {{
    color: {'#1e293b' if st.session_state["theme"] == "light" else '#f1f5f9'} !important;
}}

/* Success Box */
.stSuccess {{
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.15)) !important;
    backdrop-filter: blur(10px);
    border-left: 4px solid #10b981 !important;
    border-radius: 10px;
    color: {'#1e293b' if st.session_state["theme"] == "light" else '#f1f5f9'} !important;
}}

.stSuccess p, .stSuccess span {{
    color: {'#1e293b' if st.session_state["theme"] == "light" else '#f1f5f9'} !important;
}}

/* Error Box */
.stError {{
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.15)) !important;
    backdrop-filter: blur(10px);
    border-left: 4px solid #ef4444 !important;
    border-radius: 10px;
    color: {'#1e293b' if st.session_state["theme"] == "light" else '#f1f5f9'} !important;
}}

.stError p, .stError span {{
    color: {'#1e293b' if st.session_state["theme"] == "light" else '#f1f5f9'} !important;
}}

/* Warning Box */
.stWarning {{
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.15)) !important;
    backdrop-filter: blur(10px);
    border-left: 4px solid #f59e0b !important;
    border-radius: 10px;
    color: {'#1e293b' if st.session_state["theme"] == "light" else '#f1f5f9'} !important;
}}

.stWarning p, .stWarning span {{
    color: {'#1e293b' if st.session_state["theme"] == "light" else '#f1f5f9'} !important;
}}

/* Footer */
hr {{
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.5), transparent);
    margin: 3rem 0 1rem 0;
}}

/* Better text visibility for all markdown content */
.stMarkdown {{
    color: {'#1e293b' if st.session_state["theme"] == "light" else '#f1f5f9'} !important;
}}

.stMarkdown p, .stMarkdown li, .stMarkdown span {{
    color: {'#334155' if st.session_state["theme"] == "light" else '#e2e8f0'} !important;
}}

.stMarkdown strong {{
    color: {'#1e293b' if st.session_state["theme"] == "light" else '#ffffff'} !important;
}}

.stMarkdown a {{
    color: #3b82f6 !important;
    text-decoration: none;
    font-weight: 600;
}}

.stMarkdown a:hover {{
    color: #2563eb !important;
    text-decoration: underline;
}}

/* Analytics section styling */
.main ul, .main ol {{
    color: {'#334155' if st.session_state["theme"] == "light" else '#e2e8f0'} !important;
}}

.main ul li, .main ol li {{
    color: {'#334155' if st.session_state["theme"] == "light" else '#e2e8f0'} !important;
    margin-bottom: 0.5rem;
}}

/* Floating Animation for Icons */
@keyframes float {{
    0%, 100% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-10px); }}
}}

.floating {{
    animation: float 3s ease-in-out infinite;
}}

/* Glow Effect */
.glow {{
    animation: glow 2s ease-in-out infinite;
}}

@keyframes glow {{
    0%, 100% {{ box-shadow: 0 0 5px rgba(59, 130, 246, 0.5), 0 0 10px rgba(59, 130, 246, 0.3); }}
    50% {{ box-shadow: 0 0 20px rgba(59, 130, 246, 0.8), 0 0 30px rgba(59, 130, 246, 0.5); }}
}}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🧭 HEADER
# ==========================================
st.markdown('<h1 class="main-header">🦺 PPE Detection Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-Powered Personal Protective Equipment Monitoring System</p>', unsafe_allow_html=True)

# ==========================================
# 🧩 SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.markdown("## ⚙️ Configuration")

# Auto-load yolo9s model
model = load_model("yolo9s.pt")
model_labels = get_model_labels("yolo9s.pt")

# Generate PPE items dynamically
supported_items = list(model_labels.values())
# Remove 'Person' from the list as we don't detect it as PPE
supported_items = [item for item in supported_items if item.lower() != 'person']

# Detection settings
st.sidebar.markdown("### 🎯 PPE Detection Settings")
st.sidebar.info("📋 Select which PPE items to detect and monitor for violations")

# Initialize session state for checkboxes if not exists
if 'select_all' not in st.session_state:
    st.session_state.select_all = False

# Select All checkbox with special styling
st.sidebar.markdown('<div class="select-all-box">', unsafe_allow_html=True)
select_all = st.sidebar.checkbox("✨ Select / Deselect All", value=st.session_state.select_all, key="select_all_main")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Update all items when select all changes
if select_all != st.session_state.select_all:
    st.session_state.select_all = select_all

st.sidebar.markdown("#### 🛡️ PPE Items to Monitor")
detection_items = {}
for item in supported_items:
    # Use select_all state for default value
    detection_items[item] = st.sidebar.checkbox(
        f"🔍 {item}", 
        value=st.session_state.select_all, 
        key=f"detect_{item}"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎚️ Detection Parameters")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    0.1, 1.0, 0.5, 0.05,
    help="Higher values = more confident detections, but may miss some objects"
)

selected_items = [i for i, v in detection_items.items() if v]

if not selected_items:
    st.sidebar.warning("⚠️ Please select at least one PPE item to monitor!")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Clear Session & Reset"):
    st.session_state.clear()
    st.rerun()

# ==========================================
# 📊 METRIC DISPLAY FUNCTION
# ==========================================
def display_metrics(person_count, total_violators, missing_counts, selected_items):
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
        st.markdown('<div class="violation-alert">⚠️ <strong>Safety Violations Detected</strong><br>', unsafe_allow_html=True)
        viol = [f"<li>{i}: {missing_counts[i]} persons missing</li>" for i in selected_items if missing_counts.get(i,0)>0]
        if viol:
            st.markdown("<ul style='margin: 0.5rem 0 0 0; padding-left: 1.5rem;'>" + "".join(viol) + "</ul>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="compliant-status">✅ <strong>All Persons Compliant!</strong><br>Everyone is wearing required PPE</div>', unsafe_allow_html=True)

# ==========================================
# 📈 DATAFRAME HELPER
# ==========================================
def df_from_counts(counts: dict, selected_items: list) -> pd.DataFrame:
    df = pd.DataFrame(
        [{"PPE Item": k, "Missing Count": v} for k,v in counts.items() if k in selected_items and v>0]
    )
    return df.sort_values("Missing Count", ascending=False).reset_index(drop=True) if not df.empty else pd.DataFrame()

# ==========================================
# 🧩 TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "🎥 Video Analysis", "📊 Analytics & Info"])

# ---------------- IMAGE TAB ----------------
with tab1:
    st.markdown('<div class="section-header">📷 Image Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_img = st.file_uploader("📤 Upload Image", type=["jpg","jpeg","png"], help="Upload an image to detect PPE")
    with col2:
        draw_all = st.checkbox("🎨 Draw All Detections", False, help="Show all detected items, not just selected ones")

    if uploaded_img and selected_items:
        with st.spinner("🔍 Analyzing image with AI..."):
            try:
                img, missing_counts, violators, person_count, summary = detect_ppe_image(
                    uploaded_img,
                    selected_items,
                    selected_items,  # Use same items for detection and warnings
                    confidence_threshold,
                    draw_all,
                    model=model,
                    model_labels=model_labels
                )
                col1, col2 = st.columns([0.65,0.35])
                with col1:
                    st.image(img, caption="🎯 Detection Result", use_column_width=True)
                with col2:
                    display_metrics(person_count, violators, missing_counts, selected_items)
                    df = df_from_counts(missing_counts, selected_items)
                    if not df.empty:
                        st.markdown("### 📊 Violation Breakdown")
                        chart = alt.Chart(df).mark_bar(
                            color='#ef4444',
                            cornerRadius=8
                        ).encode(
                            x=alt.X('Missing Count:Q', title='Number of Violations'),
                            y=alt.Y('PPE Item:N', title='PPE Item', sort='-x'),
                            tooltip=['PPE Item', 'Missing Count']
                        ).properties(
                            height=250,
                            title="PPE Violations by Item"
                        ).configure_axis(
                            labelFontSize=12,
                            titleFontSize=14
                        ).configure_title(
                            fontSize=16,
                            font='Inter',
                            anchor='start'
                        )
                        st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
    elif uploaded_img and not selected_items:
        st.warning("⚠️ Please select at least one PPE item from the sidebar to monitor!")
    elif not uploaded_img:
        st.info("👆 Upload an image to get started with PPE detection")

# ---------------- VIDEO TAB ----------------
with tab2:
    st.markdown('<div class="section-header">🎥 Video Analysis</div>', unsafe_allow_html=True)
    
    uploaded_vid = st.file_uploader("📤 Upload Video", type=["mp4","mov","avi"], help="Upload a video to detect PPE frame-by-frame")
    
    if uploaded_vid and selected_items:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_vid.read())
            input_path = tmp.name

        st.info("🎬 Processing video... This may take a few minutes depending on video length.")
        progress = st.progress(0)
        def update_prog(p): progress.progress(p / 100.0)
        
        try:
            out_path, missing_counts, violators, persons, summary = detect_ppe_video(
                input_path,
                "output.mp4",
                selected_items,
                selected_items,  # Use same items for detection and warnings
                confidence_threshold,
                progress_callback=update_prog,
                model=model,
                model_labels=model_labels
            )
            st.success("✅ Video processing complete!")
            st.video(out_path)
            display_metrics(persons, violators, missing_counts, selected_items)
        except Exception as e:
            st.error(f"❌ Error during video analysis: {str(e)}")
    elif uploaded_vid and not selected_items:
        st.warning("⚠️ Please select at least one PPE item from the sidebar to monitor!")
    elif not uploaded_vid:
        st.info("👆 Upload a video to get started with PPE detection")

# ---------------- ANALYTICS TAB ----------------
with tab3:
    st.markdown('<div class="section-header">📊 Analytics & Documentation</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 How It Works
        
        1. **📤 Upload** your image or video
        2. **✅ Select** PPE items to monitor from sidebar
        3. **🎚️ Adjust** confidence threshold if needed
        4. **🔍 Analyze** and view results with AI detection
        5. **📊 Review** compliance metrics and violations
        
        ### 🛡️ Detected PPE Items
        
        This system can detect the following PPE:
        """)
        for item in supported_items:
            st.markdown(f"- 🔹 **{item}**")
    
    with col2:
        st.markdown("""
        ### ⚙️ Model Information
        
        - **Model**: YOLOv9s (State-of-the-art object detection)
        - **Framework**: Ultralytics YOLO
        - **Source**: Hugging Face Hub
        - **Classes**: 7 PPE categories + Person detection
        
        ### 🎨 Features
        
        - ✨ Real-time detection
        - 🎯 High accuracy AI model
        - 📊 Detailed compliance metrics
        - 🎥 Video frame-by-frame analysis
        - 📈 Visual violation breakdown
        - 🔄 Automatic model updates
        
        ### 💡 Tips
        
        - Higher confidence = fewer false positives
        - Lower confidence = catch more items but may have false detections
        - Use "Draw All Detections" to see everything the model found
        """)
    
    st.markdown("---")
    st.markdown("""
    ### � Development Team
    
    **Created by:**
    - **Ambuj Nayak** - [GitHub](https://github.com/Ambuj-N) - [24074007]
    - **Paturi Hemanth Sai** - [24075065]
    - **Ankit Raj** - [24074011]
    - **Jalla Poojitha** - [24124022]
    
    **Institution**: Indian Institute of Technology (BHU) Varanasi
    
    **Purpose**: Enhancing workplace safety through AI-powered PPE compliance monitoring
    """)

# Footer with enhanced styling
st.markdown(f"""
<hr>
<center style='padding: 1rem 0; animation: fadeIn 1s ease-in;'>
    <div style='font-size: 1.2rem; margin-bottom: 0.5rem;'>
        🦺 <strong style='background: linear-gradient(90deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        PPE Detector Pro
        </strong> 🦺
    </div>
    <div style='color: #64748b; font-size: 0.9rem;'>
        Built with ❤️ using AI & Streamlit | 
        <a href="https://github.com/Ambuj-N" target="_blank" style="color: #3b82f6; text-decoration: none; font-weight: 600;">
        @Ambuj-N
        </a> & Team | 
        IIT BHU Varanasi 🎓
    </div>
</center>
""", unsafe_allow_html=True)
