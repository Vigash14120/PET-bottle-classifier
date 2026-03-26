import cv2
import tensorflow as tf
import numpy as np
import os
from core.visual_preprocessing import preprocess_image
from core.loadcell_preprocessing import LoadCellProcessor

# 1. Load All Models
print("--- 🧠 Loading AI Brain Suite ---")
try:
    models = {
        'fused': tf.keras.models.load_model('models/fused_model.h5'),
        'vision': tf.keras.models.load_model('models/vision_model.h5'),
        'weight': tf.keras.models.load_model('models/weight_model.h5')
    }
    print("✅ All 3 Models (Fused, Vision, Weight) Loaded.")
except Exception as e:
    print(f"⚠️ Error loading models: {e}")
    exit(1)

# 2. Setup Config
loadcell = LoadCellProcessor()
current_mode = 'fused' # Default mode
modes_info = {
    'fused': ("FUSED (Vision + Weight)", (0, 255, 255)), # Yellow
    'vision': ("VISION ONLY", (255, 100, 0)), # Blue
    'weight': ("WEIGHT ONLY", (100, 0, 255)) # Purple
}

def get_prediction(frame, raw_weight, mode):
    """
    Passes data to the selected model.
    """
    img = preprocess_image(frame)
    clean_weight = loadcell.preprocess_weight(raw_weight)
    
    if img is None: return "ERROR", 0, clean_weight
    
    # Prepare batch inputs
    img_batch = np.expand_dims(img, axis=0) # (1, 150, 150, 3)
    weight_batch = np.array([[clean_weight / 100.0]]) # Normalized (1, 1)

    # Inference based on selected mode
    if mode == 'fused': 
        res = models['fused'].predict({'image_input': img_batch, 'weight_input': weight_batch}, verbose=0)[0][0]
    elif mode == 'vision':
        res = models['vision'].predict(img_batch, verbose=0)[0][0]
    elif mode == 'weight':
        res = models['weight'].predict(weight_batch, verbose=0)[0][0]
        
    label = "PET" if res < 0.5 else "HDPE"
    conf = (1 - res) if res < 0.5 else res
    return label, conf, clean_weight

# 3. Start the Visual Stream
cap = cv2.VideoCapture(0)
raw_weight = 22.5 # Simulated initial weight

print("\n--- 🖥️ MULTI-MODAL PET SORTER LIVE ---")
print("Controls: \n [f] Fused Mode \n [v] Vision Mode \n [w] Weight Mode")
print(" [t] Tare \n [q] Quit \n [+/-] Change simulated weight")

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- INFERENCE ---
    material, confidence, clean_weight = get_prediction(frame, raw_weight, current_mode)

    # Decision logic
    status = "REJECT"
    color = (0, 0, 255) # Red
    if confidence > 0.85:
        status = "ACCEPT"
        color = (0, 255, 0) # Green

    # 4. UI Overlay
    mode_text, mode_color = modes_info[current_mode]
    cv2.rectangle(frame, (0, 0), (640, 40), mode_color, -1)
    cv2.putText(frame, f"MODE: {mode_text}", (20, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    cv2.putText(frame, f"Material: {material} ({confidence*100:.1f}%)", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Load Cell: {clean_weight:.1f}g", (20, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.rectangle(frame, (15, 170), (300, 240), color, -1)
    cv2.putText(frame, status, (40, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)

    cv2.imshow("Multi-Modal PET Classifier", frame)

    # Handle Keypresses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('f'): current_mode = 'fused'
    elif key == ord('v'): current_mode = 'vision'
    elif key == ord('w'): current_mode = 'weight'
    elif key == ord('+'): raw_weight += 5.0 
    elif key == ord('-'): raw_weight -= 5.0 
    elif key == ord('t'): loadcell.set_tare(raw_weight) 

cap.release()
cv2.destroyAllWindows()