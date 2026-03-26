import cv2
import tensorflow as tf
import numpy as np
import os
from core.visual_preprocessing import preprocess_image
from core.loadcell_preprocessing import LoadCellProcessor

# 1. Load All Models
print("\n--- 🧠 Loading AI Brain Suite for Dashboard ---")
try:
    models = {
        'fused': tf.keras.models.load_model('models/fused_model.h5'),
        'vision': tf.keras.models.load_model('models/vision_model.h5'),
        'weight': tf.keras.models.load_model('models/weight_model.h5')
    }
    print("✅ All Models Loaded.")
except Exception as e:
    print(f"⚠️ Error loading models: {e}")
    exit(1)

loadcell = LoadCellProcessor()

def create_dashboard_image(original_image, result_data):
    """
    Creates a combined image: [Source Image] | [AI Results Dashboard]
    """
    # Resize source image for display
    display_img = cv2.resize(original_image, (400, 400))
    
    # Create the Results Panel (Background)
    panel = np.zeros((400, 400, 3), dtype=np.uint8) + 30 # Dark grey background
    
    # Text Setup
    font = cv2.FONT_HERSHEY_SIMPLEX
    material = result_data['label']
    conf = result_data['confidence']
    weight = result_data['weight']
    mode = result_data['mode'].upper()
    
    # Decision Color
    color = (0, 255, 0) if material == "PET" else (0, 165, 255) # Green for PET, Orange for HDPE
    
    # Draw Header
    cv2.putText(panel, "AI CLASSIFIER DASHBOARD", (40, 50), font, 0.7, (255, 255, 255), 2)
    cv2.line(panel, (30, 65), (370, 65), (100, 100, 100), 1)
    
    # Draw Mode Info
    cv2.putText(panel, f"MODE: {mode}", (30, 100), font, 0.6, (200, 200, 200), 1)
    
    # Draw Results
    cv2.putText(panel, "CLASSIFICATION:", (30, 150), font, 0.6, (150, 150, 150), 1)
    cv2.putText(panel, material, (30, 190), font, 1.5, color, 4)
    
    # Confidence Bar
    cv2.putText(panel, f"CONFIDENCE: {conf*100:.1f}%", (30, 230), font, 0.6, (150, 150, 150), 1)
    cv2.rectangle(panel, (30, 245), (370, 265), (50, 50, 50), -1) # Track
    w_bar = int(340 * conf)
    cv2.rectangle(panel, (30, 245), (30 + w_bar, 265), color, -1) # Progress
    
    # Sensor Data
    cv2.putText(panel, f"Sensed Weight: {weight:.1f}g", (30, 310), font, 0.7, (255, 255, 255), 1)
    
    # Draw Footer
    status = "ACCEPT" if conf > 0.85 else "UNSURE"
    cv2.putText(panel, f"SYSTEM STATUS: {status}", (30, 360), font, 0.6, (100, 255, 100) if status == "ACCEPT" else (100, 100, 255), 2)
    
    # Concatenate side-by-side
    combined = np.hstack((display_img, panel))
    return combined

def run_dashboard():
    print("\n" + "="*40)
    print("Welcome to the PET Classifier Dashboard")
    print("="*40)
    
    # 1. Select Model
    mode = input("Select Model (f=Fused, v=Vision, w=Weight) [Default=f]: ").lower() or 'f'
    mode_key = 'fused' if mode == 'f' else 'vision' if mode == 'v' else 'weight'
    
    # 2. Get Image
    img_path = input("Enter path to bottle image: ").strip()
    if not os.path.exists(img_path):
        print(f"❌ Error: {img_path} not found.")
        return
        
    # 3. Get Weight
    try:
        raw_weight = float(input("Enter sensed weight (grams): "))
    except ValueError:
        print("❌ Invalid weight. Using default 0.0g")
        raw_weight = 0.0

    # 4. Inferences
    img = cv2.imread(img_path)
    img_p = preprocess_image(img)
    clean_weight = loadcell.preprocess_weight(raw_weight)
    
    if img_p is None:
        print("❌ Error preprocessing image.")
        return

    # Prep inputs
    img_batch = np.expand_dims(img_p, axis=0)
    weight_batch = np.array([[clean_weight / 100.0]])

    # Predicted value
    if mode_key == 'fused':
        res = models['fused'].predict({'image_input': img_batch, 'weight_input': weight_batch}, verbose=0)[0][0]
    elif mode_key == 'vision':
        res = models['vision'].predict(img_batch, verbose=0)[0][0]
    else:
        res = models['weight'].predict(weight_batch, verbose=0)[0][0]

    label = "PET" if res < 0.5 else "HDPE"
    conf = (1 - res) if res < 0.5 else res

    # 5. Show Results
    print(f"\n--- Result: {label} ({conf*100:.1f}%) ---")
    
    result_data = {
        'label': label,
        'confidence': conf,
        'weight': clean_weight,
        'mode': mode_key
    }
    
    dashboard = create_dashboard_image(img, result_data)
    
    # Save the dashboard for viewing
    output_name = "dashboard_output.png"
    cv2.imwrite(output_name, dashboard)
    print(f"📊 Dashboard generated and saved as '{output_name}'")
    
    # If on a machine with display, show it
    try:
        cv2.imshow("AI Classification Dashboard", dashboard)
        print("Press any key in the dashboard window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("⚠️ No display detected. Please check dashboard_output.png to see the result.")

if __name__ == "__main__":
    while True:
        run_dashboard()
        cont = input("\nClassify another one? (y/n): ").lower()
        if cont != 'y': break
    print("Dashboard closed.")
