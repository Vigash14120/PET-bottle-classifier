import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
from core.visual_preprocessing import preprocess_image
from core.loadcell_preprocessing import LoadCellProcessor

# --- PAGE CONFIG ---
st.set_page_config(page_title="PET Classifier AI", page_icon="♻️", layout="wide")

# --- MODEL LOADING (Cached for performance) ---
@st.cache_resource
def load_all_models():
    return {
        'fused': tf.keras.models.load_model('models/fused_model.h5'),
        'vision': tf.keras.models.load_model('models/vision_model.h5'),
        'weight': tf.keras.models.load_model('models/weight_model.h5')
    }

try:
    models = load_all_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

loadcell = LoadCellProcessor()

# --- SIDEBAR ---
st.sidebar.title("🧠 AI Control Panel")
mode = st.sidebar.selectbox("Select Classification Engine", ["Fused (Vision + Weight)", "Vision Only", "Weight Only"])
mode_key = 'fused' if "Fused" in mode else 'vision' if "Vision" in mode else 'weight'

st.sidebar.divider()
st.sidebar.write("### Instructions")
st.sidebar.info("1. Upload a bottle image\n2. Provide measured weight\n3. View AI result")

# --- MAIN UI ---
st.title("♻️ AI-Powered Multi-Modal PET Sorter")
st.markdown("---")

col1, col2 = st.columns([1, 1])

# Placeholder for results data to share between columns
inference_result = None

with col1:
    st.subheader("📸 Sensor Inputs")
    uploaded_file = st.file_uploader("Upload Bottle Image", type=['jpg', 'jpeg', 'png'])
    
    raw_weight = st.slider("Live Weight Reading (grams)", 0.0, 100.0, 22.5, step=0.1)
    
    if uploaded_file is not None:
        # Load and display
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Bottle", use_container_width=True)
        
        # Prepare for processing
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_p = preprocess_image(opencv_image)
        clean_weight = loadcell.preprocess_weight(raw_weight)
        
        if img_p is not None:
            # RUN INFERENCE
            img_batch = np.expand_dims(img_p, axis=0)
            weight_batch = np.array([[clean_weight / 100.0]])
            
            with st.spinner('AI is thinking...'):
                if mode_key == 'fused':
                    res = models['fused'].predict({'image_input': img_batch, 'weight_input': weight_batch}, verbose=0)[0][0]
                elif mode_key == 'vision':
                    res = models['vision'].predict(img_batch, verbose=0)[0][0]
                else:
                    res = models['weight'].predict(weight_batch, verbose=0)[0][0]
                
                label = "PET" if res < 0.5 else "HDPE"
                conf = (1 - res) if res < 0.5 else res
                inference_result = {
                    "label": label,
                    "confidence": float(conf),
                    "weight": clean_weight
                }
    else:
        st.warning("Please upload an image to start classification.")

with col2:
    st.subheader("🤖 AI Analytics")
    
    if inference_result:
        label = inference_result['label']
        conf = inference_result['confidence']
        weight = inference_result['weight']
        
        # Display Results
        color = "#00FF00" if label == "PET" else "#FFA500"
        st.markdown(f"## Result: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        
        st.metric("Model Confidence", f"{conf*100:.2f}%")
        st.progress(conf)
        
        # System Decision
        if conf > 0.85:
            st.success(f"✅ SYSTEM DECISION: ACCEPT {label}")
        else:
            st.warning("⚠️ SYSTEM DECISION: REJECT (Uncertain Match)")
            
        # Details Table
        st.write("#### Technical Breakdown")
        st.table({
            "Metric": ["Detected Material", "AI Confidence", "Input Weight", "Engine Mode"],
            "Value": [label, f"{conf*100:.1f}%", f"{weight:.1f}g", mode.upper()]
        })
    else:
        st.info("Waiting for complete sensor data (Image + Weight)...")

st.markdown("---")
st.caption("Developed by Antigravity AI for PET Bottle Classification.")
