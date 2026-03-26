import tensorflow as tf
import numpy as np
import cv2
from visual_preprocessing import preprocess_image
from loadcell_preprocessing import LoadCellProcessor

# 1. Load the Fused Neural Brain
try:
    fused_model = tf.keras.models.load_model('fused_model.h5')
    print("✅ Fused Multi-Modal Model Loaded.")
except:
    print("⚠️ Fused Model not found. Start training with train_fused_model.py!")

# Initialize weight processor
weight_processor = LoadCellProcessor()

def classify_material(image_frame, raw_weight):
    """
    Final implementation: Uses the Deep Learning Model Level Fusion
    to decide material based on both image and loadcell data.
    """
    # --- STEP 1: Preprocessing (Modular + Shared) ---
    img_processed = preprocess_image(image_frame)
    clean_weight = weight_processor.preprocess_weight(raw_weight)
    
    if img_processed is None: return "REJECT: No Image"
    
    # --- STEP 2: Inference via Fused Model ---
    # Prepare batch data for the two model inputs
    img_batch = np.expand_dims(img_processed, axis=0) # (1, 150, 150, 3)
    weight_batch = np.array([[clean_weight / 100.0]]) # Normalized (1, 1)

    # Get the Fused AI prediction 
    try:
        prediction = fused_model.predict([img_batch, weight_batch], verbose=0)[0][0]
    except:
        return "ERROR: Model Incompatibility"

    # 0 = PET, 1 = HDPE
    material = "PET" if prediction < 0.5 else "HDPE"
    confidence = (1 - prediction) if prediction < 0.5 else prediction

    # --- STEP 3: Final Model-Level Decision ---
    # The neural network now handles the correlation between weight and vision.
    if confidence > 0.90:
        return f"ACCEPT: {material} (W: {clean_weight:.1f}g, Conf: {confidence:.2f})"
    else:
        return f"REJECT: Uncertain Match (Conf: {confidence:.2f})"

# --- Example Test ---
# result = classify_material(your_camera_frame, 20.5)
