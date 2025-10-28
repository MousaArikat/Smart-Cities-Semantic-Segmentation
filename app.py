from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import streamlit as st
from utils.segmentation import load_model, segment_image, decode_segmap, segment_video, build_model
from PIL import Image
import numpy as np
import torch
import io
from streamlit.components.v1 import html
import time
from utils.ui import render_model_cards, render_advanced_controls, render_image_results
from huggingface_hub import hf_hub_download
from utils.segmentation import build_model  # we’ll call your model builder

@st.cache_resource
def load_remote_model(model_name, architecture):
    repo_id = "MousaAricat/semantic-segmentation-models"
    model_path = hf_hub_download(repo_id=repo_id, filename=model_name)
    state_dict = torch.load(model_path, map_location="cpu")

    # rebuild model architecture
    model = build_model(architecture)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model



st.set_page_config(
    page_title="Smart City Segmentation AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with open("templates/hero.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)


AVAILABLE_MODELS = {
    "🚀 DeepLabV3 ResNet101 (Best)": {
        "filename": "deeplab3(best).pth",
        "name": "deeplabv3_resnet101_customhead",
        "description": "ResNet-101 backbone, Adam optimizer, lr 1e-4 + scheduler",
        "mIoU": "64.7%",
    },
    "🏙️ HRNetV2-W48 (Best)": {
        "filename": "hrnet1(best).pth",
        "name": "hrnetv2_w48",
        "description": "Adam optimizer, lr 1e-4, weight decay 1e-5 + scheduler",
        "mIoU": "74.6%",
    }
}



# @st.cache_resource
# def get_model(model_path, model_name):
#     return load_model(model_path, model_name)
st.markdown("""
<div class="fade-in-up" style='margin-bottom: 2rem;'>
    <h2 style='
        color: #2c3e50; 
        margin-bottom: 1.5rem;
        font-family: "Inter", sans-serif;
        font-weight: 700;
        font-size: 2rem;
    '>🤖 AI Model Selection</h2>
</div>
""", unsafe_allow_html=True)

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = list(AVAILABLE_MODELS.keys())[0]

render_model_cards(AVAILABLE_MODELS, st.session_state.selected_model)

selected_model_info = AVAILABLE_MODELS[st.session_state.selected_model]
with st.spinner("🔄 Loading AI model..."):
    model = load_remote_model(
        selected_model_info["filename"],
        selected_model_info["name"]
    )

st.markdown("""
<div class="fade-in-up" style='margin: 3rem 0 2rem 0;'>
    <h2 style='
        color: #2c3e50; margin-bottom: 2rem;font-family: "Inter", sans-serif; font-weight: 700; font-size: 2rem; text-align: center;
    '>📁 Upload Your Content</h2>
</div>
""", unsafe_allow_html=True)

upload_col1, upload_col2 = st.columns(2, gap="large")
with upload_col1:
    with open("templates/upload_image_card.html", "r", encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📤 Drop your image here or click to browse", 
        type=["jpg", "png", "jpeg"], 
        label_visibility="collapsed",
        key="image_uploader",
        help="Upload an image of a city scene for AI-powered semantic segmentation"
    )

with upload_col2:
    with open("templates/upload_video_card.html", "r", encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)
    video_file = st.file_uploader(
        "📤 Drop your video here or click to browse", 
        type=["mp4"],
        label_visibility="collapsed",
        key="video_uploader",
        help="Upload a video for frame-by-frame semantic segmentation analysis"
    )

alpha, hidden_class_ids = render_advanced_controls()

if video_file is not None:
    st.markdown("""
    <div class="glass-card fade-in-up" style=' padding: 3rem; margin: 3rem 0; text-align: center;'>
        <h3 style='color: #667eea; margin-bottom: 2rem;font-size: 2rem;font-weight: 700;'>🎬 AI Video Processing</h3>
        <p style='color: #6c757d; font-size: 1.1rem; margin-bottom: 2rem;'>
            Our advanced AI is analyzing your video frame by frame...
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    progress = st.progress(0)
    status_text = st.empty()
    all_frames = []

    start_time = time.time()
    for percent, frames in segment_video(model, video_file, alpha=alpha):
        progress.progress(percent)
        elapsed_time = time.time() - start_time
        eta = (elapsed_time / percent) * (1 - percent) if percent > 0 else 0
        status_text.markdown(f"""
        <div style='text-align: center; color: #667eea; font-weight: 600;'>
            Processing: {int(percent * 100)}% complete • ETA: {int(eta)}s
        </div>
        """, unsafe_allow_html=True)
        all_frames = frames

    st.success("✅ Video segmentation completed successfully!")
    status_text.empty()

    clip = ImageSequenceClip(all_frames, fps=10)
    output_path = "output_video.mp4"
    clip.write_videofile(output_path, codec="libx264", audio=False, verbose=False, logger=None)

    st.markdown("""
    <div class="glass-card" style='padding: 2rem; margin: 2rem 0; text-align: center;'>
        <h3 style='color: #667eea; margin-bottom: 1rem; font-weight: 700;'>🎥 Segmented Video Result</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.video(output_path)

    with open(output_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Segmented Video",
            data=f,
            file_name="ai_segmented_video.mp4",
            mime="video/mp4",
            use_container_width=True,
            type="primary"
        )

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    render_image_results(image, model, alpha, hidden_class_ids)

with open("templates/footer.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)