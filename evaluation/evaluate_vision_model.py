import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from core.visual_preprocessing import preprocess_image

# 1. Load Data Metadata
df = pd.read_csv("metadata_fused.csv")
val_df = df[df['split'] == 'val'].copy()

# 2. Load the Standalone Vision Model
model_path = 'models/vision_model.h5'
if not os.path.exists(model_path):
    print(f"❌ Error: {model_path} not found. Train it first!")
    exit(1)

model = tf.keras.models.load_model(model_path)
print("✅ Standalone Vision Model Loaded.")

def evaluate_vision():
    y_true = []
    y_pred = []
    
    print(f"--- 📸 Evaluating Vision Only on {len(val_df)} validation samples ---")
    
    for _, row in val_df.iterrows():
        img_path = row['image_path']
        true_label = row['label']
        
        # Preprocess
        img = cv2.imread(img_path)
        img_p = preprocess_image(img)
        
        if img_p is not None:
            # Batch input (Image Only)
            img_batch = np.expand_dims(img_p, axis=0)
            
            # Predict
            res = model.predict(img_batch, verbose=0)[0][0]
            
            # Binary threshold
            pred_label = 1 if res >= 0.5 else 0
            
            y_true.append(true_label)
            y_pred.append(pred_label)

    # 3. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["PET (0)", "HDPE (1)"])
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "="*40)
    print(f"🖼️ VISION-ONLY EVALUATION REPORT")
    print("="*40)
    print(f"Overall Accuracy: {acc*100:.2f}%")
    print("\nDetailed Metrics:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    print("="*40)

if __name__ == "__main__":
    evaluate_vision()
