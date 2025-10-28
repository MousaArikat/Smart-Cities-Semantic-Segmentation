import streamlit as st
from PIL import Image
import io
from utils.segmentation import segment_image, decode_segmap

def render_image_results(image, model, alpha, hidden_class_ids):
    with st.spinner("🧠 AI is analyzing your image..."):
        start_time = st.session_state.get("start_time_image", None) or st.session_state.setdefault("start_time_image", __import__("time").time())
        mask = segment_image(model, image)
        decoded_mask = decode_segmap(mask, hidden_classes=hidden_class_ids)
        overlay = Image.blend(image.resize(decoded_mask.size), decoded_mask, alpha=alpha)
        processing_time = __import__("time").time() - start_time

    st.markdown(f"""
    <div class="fade-in-up" style='margin: 4rem 0 3rem 0;'>
        <h2 style='color: #2c3e50; text-align: center; margin-bottom: 1rem;
                   font-family: "Inter", sans-serif; font-weight: 700; font-size: 2.5rem;'>
            🧪 AI Analysis Results
        </h2>
        <p style='text-align: center; color: #6c757d; font-size: 1.1rem;
                  margin-bottom: 2rem;'>Processing completed in {processing_time:.2f} seconds</p>
    </div>
    """, unsafe_allow_html=True)

    display_size = (600, 400)
    resized_image = image.resize(display_size)
    resized_mask = decoded_mask.resize(display_size)
    resized_overlay = overlay.resize(display_size)

    col1, col2, col3 = st.columns([1, 1, 0.35], gap="large")

    with col1:
        st.markdown("""<div class="glass-card" style='padding: 1.5rem; margin-bottom: 1rem; height : 65px'>
            <h4 style='color: #1565c0; text-align: center; margin-top: -13px;
                       font-weight: 700; font-size: 1.2rem;'>🖼️ Original Image</h4>
        </div>""", unsafe_allow_html=True)
        tab1 = st.tabs(["Original Image"])
        st.image(resized_image, use_column_width=True)
        

    with col2:
        st.markdown("""<div class="glass-card" style='padding: 1.5rem; margin-bottom: 1rem; height: 65px'>
            <h4 style='color: #6a1b9a; text-align: center; margin-top: -13px;
                       font-weight: 700; font-size: 1.2rem;'>🎯 AI Segmentation</h4>
        </div>""", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🎯 Segmentation Mask", "🧪 Overlay Visualization"])

        with tab1:
            st.image(resized_mask, use_column_width=True)
            mask_bytes = io.BytesIO()
            resized_mask.save(mask_bytes, format="PNG")
            st.download_button("⬇️ Download Segmentation Mask", data=mask_bytes.getvalue(),
                               file_name="ai_segmentation_mask.png", mime="image/png",
                               use_container_width=True, type="primary")

        with tab2:
            st.image(resized_overlay, use_column_width=True)
            overlay_bytes = io.BytesIO()
            resized_overlay.save(overlay_bytes, format="PNG")
            st.download_button("⬇️ Download Overlay Result", data=overlay_bytes.getvalue(),
                               file_name="ai_overlay_result.png", mime="image/png",
                               use_container_width=True, type="primary")

    with col3:
        st.markdown("""<div class="glass-card" style='padding: 1.5rem; margin-bottom: 1rem;'>
            <h4 style='color: #2e7d32; text-align: center; margin-bottom: 1rem;
                       font-weight: 700; font-size: 1.2rem;'>🎨 Class Legend</h4>
        </div>""", unsafe_allow_html=True)
        st.image("assets/legend.png", use_column_width=True)

def render_model_cards(models: dict, selected_model_key: str):
    # Separate DeepLab models from HRNet models
    deeplab_models = {}
    hrnet_models = {}
    
    for display_name, info in models.items():
        if "HRNet" in display_name:
            hrnet_models[display_name] = info
        else:
            deeplab_models[display_name] = info
    
    # Create two columns for the model sections with equal proportions (50% each)
    left_col, right_col = st.columns([1, 1], gap="large")
    
    # DeepLab Models Section (Left)
    with left_col:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #00FFFF, #50c7c7);
            color: white;
            padding: 1rem;
            border-radius: 15px;
            margin-bottom: 1.5rem;
            text-align: center;
            box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3);
        '>
            <h3 style='margin: 0; font-weight: 700;'>🔬 DeepLab Models</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for DeepLab models
        deeplab_cols = st.columns(len(deeplab_models), gap="medium")
        
        for i, (display_name, info) in enumerate(deeplab_models.items()):
            is_selected = selected_model_key == display_name

            card_style = f"""
            background: {'linear-gradient(135deg, #3b82f6, #1d4ed8)' if is_selected else 'linear-gradient(145deg, #ffffff, #f8f9fa)'};
            color: {'white' if is_selected else '#2c3e50'};
            border: 2px solid {'#3b82f6' if is_selected else '#e9ecef'};
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: {'0 15px 40px rgba(59, 130, 246, 0.3)' if is_selected else '0 8px 25px rgba(0, 0, 0, 0.1)'};
            transform: {'translateY(-5px)' if is_selected else 'translateY(0)'};
            margin-bottom: 1rem;
            height: 320px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            """

            with deeplab_cols[i]:
                st.markdown(f"""
                <div style="{card_style}">
                    <h3 style='margin-bottom: 1rem; font-size: 1.3rem; font-weight: 700;'>{display_name}</h3>
                    <p style='margin-bottom: 1rem; opacity: 0.9; font-size: 0.95rem;'>{info['description']}</p>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 0;'>
                        <span style='font-size: 0.85rem; font-weight: 600;'>mIoU: {info['mIoU']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(
                    f"Select {display_name.split(' ')[1]}",
                    key=f"model_btn_deeplab_{i}",
                    use_container_width=True,
                    help=f"Click to select {display_name}",
                    type="secondary" if not is_selected else "primary"
                ):
                    st.session_state.selected_model = display_name
                    st.rerun()
    
    # HRNet Models Section (Right)
    with right_col:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #d4909b, #facfd6);
            color: white;
            padding: 1rem;
            border-radius: 15px;
            margin-bottom: 1.5rem;
            text-align: center;
            box-shadow: 0 8px 25px rgba(156, 39, 176, 0.3);
        '>
            <h3 style='margin: 0; font-weight: 700;'>🏙️ HRNet Models</h3>
        </div>
        """, unsafe_allow_html=True)
        
        hrnet_cols = st.columns(len(hrnet_models), gap="medium")
        
        for i, (display_name, info) in enumerate(hrnet_models.items()):
            is_selected = selected_model_key == display_name

            card_style = f"""
            background: {'linear-gradient(135deg, #9C27B0, #6A1B9A)' if is_selected else 'linear-gradient(145deg, #ffffff, #f8f9fa)'};
            color: {'white' if is_selected else '#2c3e50'};
            border: 2px solid {'#9C27B0' if is_selected else '#e9ecef'};
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: {'0 15px 40px rgba(156, 39, 176, 0.3)' if is_selected else '0 8px 25px rgba(0, 0, 0, 0.1)'};
            transform: {'translateY(-5px)' if is_selected else 'translateY(0)'};
            margin-bottom: 1rem;
            height: 320px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            """

            with hrnet_cols[i]:
                st.markdown(f"""
                <div style="{card_style}">
                    <h3 style='margin-bottom: 1rem; font-size: 1.3rem; font-weight: 700;'>{display_name}</h3>
                    <p style='margin-bottom: 1rem; opacity: 0.9; font-size: 0.95rem;'>{info['description']}</p>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 0;'>
                        <span style='font-size: 0.85rem; font-weight: 600;'>mIoU: {info['mIoU']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(
                    f"Select {display_name.split(' ')[1]}",
                    key=f"model_btn_hrnet_{i}",
                    use_container_width=True,
                    help=f"Click to select {display_name}",
                    type="secondary" if not is_selected else "primary"
                ):
                    st.session_state.selected_model = display_name
                    st.rerun()

def render_advanced_controls():
    st.markdown("""
    <div class="fade-in-up" style='margin: 3rem 0 2rem 0;'>
        <h2 style='
            color: #2c3e50; 
            margin-bottom: 2rem;
            font-family: "Inter", sans-serif;
            font-weight: 700;
            font-size: 2rem;
            text-align: center;
        '>Advanced Controls</h2>
    </div>
    """, unsafe_allow_html=True)

    settings_col1, settings_col2 = st.columns([1, 1], gap="large")

    with settings_col1:
        st.markdown("""
        <div class="glass-card" style='
            padding: 2rem;
            margin-bottom: 2rem;
        '>
            <div style='
                background: linear-gradient(135deg, #ffc107, #ff8f00);
                color: white;
                padding: 1rem;
                border-radius: 15px;
                margin-bottom: 1.5rem;
                text-align: center;
                box-shadow: 0 8px 25px rgba(255, 193, 7, 0.3);
            '>
                <h4 style='margin: 0; font-weight: 700;'>🎨 Overlay Settings</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)

        alpha = st.slider(
            "🔍 Overlay Opacity", 
            0.0, 1.0, 0.5, 
            step=0.05, 
            help="Adjust the transparency of the segmentation overlay on the original image"
        )

    with settings_col2:
        st.markdown("""
        <div class="glass-card" style='
            padding: 2rem;
            margin-bottom: 2rem;
        '>
            <div style='
                background: linear-gradient(135deg, #17a2b8, #138496);
                color: white;
                padding: 1rem;
                border-radius: 15px;
                margin-bottom: 1.5rem;
                text-align: center;
                box-shadow: 0 8px 25px rgba(23, 162, 184, 0.3);
            '>
                <h4 style='margin: 0; font-weight: 700;'>👁️ Class Visibility</h4>
            </div>
        </div>
        """, unsafe_allow_html=True)

        class_labels = [
            "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole", "Traffic Light", "Traffic Sign", "Vegetation", "Terrain",
            "Sky", "Person", "Rider", "Car", "Truck", "Bus", "Train", "Motorcycle", "Bicycle"
        ]
        id_to_label = {i: label for i, label in enumerate(class_labels)}
        label_to_id = {label: i for i, label in id_to_label.items()}

        hidden_class_labels = st.multiselect(
            "🎯 Select classes to hide from visualization:",
            options=class_labels,
            default=[],
            help="Choose which segmentation classes to hide from the overlay visualization"
        )

        hidden_class_ids = [label_to_id[label] for label in hidden_class_labels]

    return alpha, hidden_class_ids
