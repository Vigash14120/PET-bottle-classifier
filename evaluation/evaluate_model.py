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

# 2. Load the trained Fused Model
model = tf.keras.models.load_model('models/fused_model.h5')
print("✅ Fused Model Loaded for Evaluation.")

def evaluate():
    y_true = []
    y_pred = []
    
    print(f"--- 🧪 Evaluating on {len(val_df)} validation samples ---")
    
    for _, row in val_df.iterrows():
        img_path = row['image_path']
        raw_weight = row['weight_grams']
        true_label = row['label']
        
        # Preprocess
        img = cv2.imread(img_path)
        img_p = preprocess_image(img)
        
        if img_p is not None:
            # Batch inputs
            img_batch = np.expand_dims(img_p, axis=0)
            weight_batch = np.array([[raw_weight / 100.0]])
            
            # Predict
            res = model.predict({'image_input': img_batch, 'weight_input': weight_batch}, verbose=0)[0][0]
            
            # Binary threshold
            pred_label = 1 if res >= 0.5 else 0
            
            y_true.append(true_label)
            y_pred.append(pred_label)

    # 3. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["PET (0)", "HDPE (1)"])
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "="*40)
    print(f"📊 EVALUATION REPORT")
    print("="*40)
    print(f"Overall Accuracy: {acc*100:.2f}%")
    print("\nDetailed Metrics:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    print("="*40)

if __name__ == "__main__":
    evaluate()
