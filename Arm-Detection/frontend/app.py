import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import torch
from pathlib import Path
import sys
from torchvision import transforms
import warnings
import logging
import os

# Ensure the project root is on sys.path so `backend` can be imported when
# Streamlit runs from the `frontend` directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference import load_models, preprocess_image, predict_ultrasound, predict_arm, get_gradcam_visualization, create_pdf_report

# Suppress specific warnings and errors
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Suppress PyTorch class registration errors
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

# Populate model download URLs from Streamlit secrets if provided (useful in deployment)
model_url_secrets = st.secrets.get("model_urls", {})
secret_key_map = {
    "MODEL_ULTRASOUND_URL": "ultrasound",
    "MODEL_ARM_URL": "arm"
}
for env_key, secret_key in secret_key_map.items():
    secret_value = model_url_secrets.get(secret_key)
    if secret_value and not os.environ.get(env_key):
        os.environ[env_key] = secret_value

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent))

# Set page config
st.set_page_config(
    page_title="🏥 Pediatric Ultrasound ARM Detection",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for professional medical theme
st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background-color: #FFFFFF;
        color: #2C3E50;
        font-family: 'Segoe UI', sans-serif;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F8F9FA;
        border-right: 1px solid #E9ECEF;
    }
    /* Header styling */
    h1 {
        color: #2C3E50;
        text-align: center;
        font-size: 2.5em;
        font-weight: 600;
    }
    h3 {
        color: #2C3E50;
        font-size: 1.5em;
        font-weight: 500;
    }
    /* Upload area */
    .stFileUploader label {
        color: #2C3E50;
        font-weight: 500;
    }
    .stFileUploader div {
        border: none;
        border-radius: 8px;
        background-color: #F8F9FA;
        padding: 10px;
    }
    /* Button styling */
    .stButton>button {
        background-color: #3498DB;
        color: white;
        font-weight: 500;
        border-radius: 6px;
        padding: 10px 20px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980B9;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Results and insights box */
    .result-box, .insights-box {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #E9ECEF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Progress circle */
    .progress-circle {
        text-align: center;
        font-size: 1.2em;
        color: #3498DB;
    }
    /* Spinner */
    .spinner {
        text-align: center;
        font-size: 1.5em;
        color: #3498DB;
    }
    /* Links */
    a {
        color: #3498DB;
        text-decoration: none;
    }
    a:hover {
        color: #2980B9;
        text-decoration: underline;
    }
    /* Subheaders */
    .stSubheader {
        color: #2C3E50;
        font-weight: 500;
    }
    /* Info boxes */
    .stInfo {
        background-color: #E3F2FD;
        border-color: #BBDEFB;
    }
    /* Warning boxes */
    .stWarning {
        background-color: #FFF3E0;
        border-color: #FFE0B2;
    }
    /* Success boxes */
    .stSuccess {
        background-color: #E8F5E9;
        border-color: #C8E6C9;
    }
    /* Image container for side-by-side alignment */
    .image-container img {
        max-width: 300px;
        height: auto;
        display: block;
        margin-left: auto;
        margin-right: auto;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .image-container {
        text-align: center;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📘 About")
    st.info("""
    This tool helps clinicians and researchers screen pediatric ultrasound images for Anorectal Malformations (ARMs) using AI.
    """)

    st.title("📝 Instructions")
    st.markdown("""
    1. Upload a pediatric ultrasound image (`.jpg`, `.jpeg`, `.png`)
    2. AI model will classify it as **Normal** or **ARM**
    3. You'll see a heatmap highlighting important regions
    """)

    st.title("⚠️ Disclaimer")
    st.warning("""
    This tool is for educational/screening purposes only.
    Not a substitute for professional medical diagnosis.
    """)

# Main Title
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🩻 Pediatric Ultrasound ARM Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Empowering early diagnosis with AI 🧠</p>", unsafe_allow_html=True)
st.markdown("---")

# Layout with columns
col1, col2 = st.columns([1, 1])

# Left column: Image upload and Diagnose button
with col1:
    st.subheader("Upload Ultrasound Image")
    uploaded_file = st.file_uploader(
        "Upload pediatric ultrasound image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    # Store uploaded image in session state immediately if a new file is uploaded
    if uploaded_file is not None and ('uploaded_image' not in st.session_state or uploaded_file.getvalue() != st.session_state.get('last_uploaded_file', io.BytesIO()).getvalue()):
        st.session_state['uploaded_image'] = Image.open(uploaded_file)
        st.session_state['last_uploaded_file'] = uploaded_file
        if 'prediction_results' in st.session_state:
            del st.session_state['prediction_results']
        if 'gradcam_visualization' in st.session_state:
            del st.session_state['gradcam_visualization']
        st.rerun()

    diagnose_button = st.button("Diagnose", key="diagnose")

# Right column: Results
with col2:
    st.subheader("Diagnosis Results")
    
    if 'prediction_results' in st.session_state:
        results = st.session_state['prediction_results']
        
        arm_class_display = "Not Available"
        arm_class_color = "#6c757d" # Gray color for not available
        if results['is_ultrasound'] and results['arm_class'] is not None:
            arm_class_display = results['arm_class']
            if results['arm_class'] == 'ARM':
                arm_class_color = "#e74c3c" # Red color for ARM
            elif results['arm_class'] == 'Normal':
                arm_class_color = "#2ecc71" # Green color for Normal
            else:
                 arm_class_color = "#f39c12" # Orange for Uncertain

        st.markdown(f"""
            <div class='result-box'>
                <h3>Prediction: {results['image_type']}</h3>
                <p>Confidence: <span class='progress-circle'>{(results['ultrasound_prob'] if results['is_ultrasound'] else results['non_ultrasound_prob']):.2%}</span></p>
                <p>ARM Probability: {(results['arm_prob'] if results['arm_prob'] is not None else 0.0):.2%}</p>
                <p>Normal Probability: {(results['normal_prob'] if results['normal_prob'] is not None else 0.0):.2%}</p>
                <h3 style='color: {arm_class_color};'>ARM Classification: {arm_class_display}</h3>
            </div>
        """, unsafe_allow_html=True)

    # Trigger analysis when diagnose button is clicked
    if diagnose_button and 'uploaded_image' in st.session_state:
        with st.spinner("🔄 Analyzing..."):
            try:
                ultrasound_model, arm_model = load_models()
                if ultrasound_model is None or arm_model is None:
                    st.error("Failed to load models. Please check if model files are present.")
                    st.stop()
                
                processed_image, original_image = preprocess_image(st.session_state['uploaded_image'])
                
                # Stage 1: Ultrasound Detection
                ultrasound_result, ultrasound_conf, non_ultrasound_prob, ultrasound_prob = predict_ultrasound(
                    processed_image, ultrasound_model
                )
                
                is_ultrasound = ultrasound_result == 'ultrasound'
                
                # Stage 2: ARM Classification (only if ultrasound detected)
                arm_result = None
                arm_conf = None
                arm_prob = None
                normal_prob = None
                
                if is_ultrasound and arm_model is not None:
                    arm_result, arm_conf, arm_prob, normal_prob = predict_arm(
                        processed_image, arm_model
                    )
                
                st.session_state['prediction_results'] = {
                    'image_type': 'Ultrasound' if is_ultrasound else 'Non-Ultrasound',
                    'is_ultrasound': is_ultrasound,
                    'non_ultrasound_prob': non_ultrasound_prob,
                    'ultrasound_prob': ultrasound_prob,
                    'arm_class': arm_result if is_ultrasound else None,
                    'arm_confidence': arm_conf if is_ultrasound else None,
                    'arm_prob': arm_prob if is_ultrasound else None,
                    'normal_prob': normal_prob if is_ultrasound else None
                }
                
                # Generate Grad-CAM visualization for the appropriate model
                target_model = arm_model if is_ultrasound and arm_model is not None else ultrasound_model
                st.session_state['gradcam_visualization'] = get_gradcam_visualization(
                    target_model, processed_image, original_image
                )
                
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.error("Please try again with a different image.")

# Display Images side-by-side after analysis
if 'prediction_results' in st.session_state and 'uploaded_image' in st.session_state:
    st.markdown("---")
    st.subheader("Image Visualization")
    
    # ✅ Use uploaded image's original size
    original_size = st.session_state['uploaded_image'].size
    uploaded_img_resized = st.session_state['uploaded_image'].resize(original_size)

    gradcam_img_resized = (
        st.session_state['gradcam_visualization'].resize(original_size)
        if 'gradcam_visualization' in st.session_state and st.session_state['gradcam_visualization'] is not None
        else None
    )

    img_col1, img_col2 = st.columns([1, 1])
    
    with img_col1:
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(uploaded_img_resized, caption="Original Image", use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with img_col2:
        show_heatmap = st.checkbox("Show Grad-CAM Heatmap", value=True, key="show_heatmap_compare")
        if show_heatmap and gradcam_img_resized is not None:
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(gradcam_img_resized, caption="Grad-CAM Heatmap - Red areas indicate regions important for the prediction", use_container_width=False)
            st.markdown("</div>", unsafe_allow_html=True)
        elif show_heatmap:
            st.warning("Grad-CAM visualization could not be generated for this image.")

    # Add Download Report button
    if 'prediction_results' in st.session_state and 'uploaded_image' in st.session_state:
        pdf_data = create_pdf_report(
            st.session_state['uploaded_image'],
            st.session_state.get('gradcam_visualization'),
            st.session_state['prediction_results']
        )
        st.download_button(
            label="Download Report (PDF)",
            data=pdf_data,
            file_name="arm_detection_report.pdf",
            mime="application/pdf"
        )

# Move the 'Understanding the Heatmap' section outside of the columns
if 'prediction_results' in st.session_state and 'uploaded_image' in st.session_state and st.session_state.get('gradcam_visualization') is not None and st.session_state.get('show_heatmap_compare', True):
    st.markdown("""
        <div style="display: block; margin: 20px auto; background-color: #f8f9fa; border-radius: 10px; padding: 20px; border-left: 4px solid #3498db; width: 70%; max-width: 700px; text-align: left;">
          <h4 style="color:#2c3e50; margin-top: 0; text-align: center;">🧠 Understanding the Heatmap:</h4>
          <ul style="text-align: left; margin-bottom: 0;">
            <li>Red areas indicate regions that the model focused on for making its prediction</li>
            <li>Brighter red indicates higher importance</li>
            <li>This helps understand which parts of the image influenced the model's decision</li>
          </ul>
        </div>
    """, unsafe_allow_html=True)

# ARM Insights in the middle
st.markdown("---")
st.subheader("ARM Insights")
st.markdown("""
    <div class='insights-box'>
        <p><b>Did You Know?</b></p>
        <ul>
            <li>ARMs occur in ~1 in 5,000 births, with anal atresia being a common type.</li>
            <li>The "target sign" is key: absent in high-type ARMs, altered in low-type.</li>
            <li>Early prenatal diagnosis improves surgical outcomes.</li>
        </ul>
        <p><b>Screening Tip:</b> Optimal visualization at 28–33 weeks gestation.</p>
    </div>
""", unsafe_allow_html=True)

# Contact Details at the end
st.markdown("---")
st.subheader("📬 Contact Us")
st.markdown("""
    <div style='text-align: center;'>
        <p>For support and inquiries:</p>
        <p>📧 Email: pavansaibudur2008@gmail.com</p>
        <p>🌐 GitHub: <a href='https://github.com/Pavansai2008/ARM-Detection'>github.com/Pavansai2008</a></p>
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style='border-color: #E9ECEF'>
    <p style='text-align: center; color: #6C757D;'>© 2025 MediVision | Developed for Pediatric Radiology</p>
""", unsafe_allow_html=True)