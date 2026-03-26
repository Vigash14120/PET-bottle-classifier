import tensorflow as tf
import numpy as np
import cv2
import os
from visual_preprocessing import preprocess_image

# 1. Load the Fused Model
model_path = 'fused_model.h5'
if not os.path.exists(model_path):
    print(f"❌ Error: {model_path} not found. Train it first!")
    exit(1)

model = tf.keras.models.load_model(model_path)
print("✅ Fused Model Loaded Successfully.")

# 2. Setup Test Samples
test_samples = [
    # (image_path, simulated_weight, expected_label)
    ('data/train/PET_Bottle/PET1,000.jpg', 22.5, "PET"),
    ('data/train/PET_Bottle/PET1,010.jpg', 18.2, "PET"),
    ('data/train/HDPE_Bottle/HDPEM100.jpg', 35.0, "HDPE"),
    ('data/train/HDPE_Bottle/HDPEM105.jpg', 42.1, "HDPE"),
]

def run_test_inference(image_path, raw_weight, expected):
    full_path = os.path.join(os.getcwd(), image_path)
    if not os.path.exists(full_path):
        print(f"⚠️ Warning: {image_path} not found.")
        return

    # Load and Preprocess
    img = cv2.imread(full_path)
    img_processed = preprocess_image(img)
    clean_weight = raw_weight # Simplified for test
    
    if img_processed is None: return
    
    # Prepare Inputs
    img_batch = np.expand_dims(img_processed, axis=0) # (1, 150, 150, 3)
    weight_batch = np.array([[clean_weight / 100.0]]) # Normalized (1, 1)

    # Predict using the Model (using the dict format for stability)
    res = model.predict({'image_input': img_batch, 'weight_input': weight_batch}, verbose=0)[0][0]
    
    label = "PET" if res < 0.5 else "HDPE"
    confidence = (1 - res) if res < 0.5 else res
    
    status = "✅ MATCH" if label == expected else "❌ MISMATCH"
    print(f"[{status}] File: {os.path.basename(image_path)} | W: {raw_weight}g | Pred: {label} ({confidence*100:.1f}%) | Expected: {expected}")

print("\n--- 🧠 Fused Multi-Modal Inference Test ---")
for img_p, w, expected in test_samples:
    run_test_inference(img_p, w, expected)
