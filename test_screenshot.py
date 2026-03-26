import tensorflow as tf
import numpy as np
import cv2
import os
from visual_preprocessing import preprocess_image

# 1. Load the Fused Model
model = tf.keras.models.load_model('fused_model.h5')
print("✅ Fused Model Loaded.")

# 2. File and Param Setup
screenshot_path = 'Screenshot 2026-03-26 214424.png'
# A typical PET bottle weighs between 15g and 35g. Let's use 22.5g.
simulated_weight = 22.5 

def test_screenshot():
    if not os.path.exists(screenshot_path):
        print(f"❌ Error: {screenshot_path} not found.")
        return

    # Load and Preprocess
    img = cv2.imread(screenshot_path)
    img_p = preprocess_image(img)
    
    if img_p is None:
        print("❌ Preprocessing failed.")
        return
        
    # Prepare Inputs
    img_batch = np.expand_dims(img_p, axis=0) # (1, 150, 150, 3)
    weight_batch = np.array([[simulated_weight / 100.0]]) # Normalized

    # Predict
    res = model.predict({'image_input': img_batch, 'weight_input': weight_batch}, verbose=0)[0][0]
    
    label = "PET" if res < 0.5 else "HDPE"
    confidence = (1 - res) if res < 0.5 else res
    
    print("\n--- 🔍 INFERENCE RESULT ---")
    print(f"File: {screenshot_path}")
    print(f"Simulated Weight: {simulated_weight}g")
    print(f"AI Prediction: {label} ({confidence*100:.2f}% confidence)")
    print(f"RESULT: {'✅ SUCCESS' if label == 'PET' else '❌ MISMATCH'}")
    print("--------------------------")

if __name__ == "__main__":
    test_screenshot()
