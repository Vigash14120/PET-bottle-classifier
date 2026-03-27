import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import sys

# 1. Project Setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.visual_preprocessing import preprocess_image

# 2. Load Metadata and Vision-Only Model
df = pd.read_csv("metadata_fused.csv")
val_df = df[df['split'] == 'val'].copy()
model_path = 'models/vision_model.h5'
model = tf.keras.models.load_model(model_path)

# 3. Collect Predictions (Image Only)
y_true = []
y_probs = []

print("--- 📸 Generating Vision-Only Evaluation Data ---")
for _, row in val_df.iterrows():
    img_path = row['image_path']
    true_label = row['label']
    
    img = cv2.imread(img_path)
    img_p = preprocess_image(img)
    
    if img_p is not None:
        img_batch = np.expand_dims(img_p, axis=0)
        
        # Predict (No weight input needed)
        prob = model.predict(img_batch, verbose=0)[0][0]
        
        y_true.append(true_label)
        y_probs.append(prob)

y_true = np.array(y_true)
y_probs = np.array(y_probs)
y_pred = (y_probs >= 0.5).astype(int)

# 4. Generate Visuals
if not os.path.exists("Diagrams"):
    os.makedirs("Diagrams")

# --- CHART 1: VISION CONFUSION MATRIX ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', annot_kws={"size": 16},
            xticklabels=["PET (0)", "HDPE (1)"], 
            yticklabels=["PET (0)", "HDPE (1)"])
plt.title('Vision-Only Confusion Matrix (No Sensors)', fontsize=14)
plt.ylabel('Actual Label')
plt.xlabel('AI Predicted Label')
plt.savefig("Diagrams/Vision Confusion Matrix.png", dpi=300)
print("✅ Created: Vision Confusion Matrix.png")

# --- CHART 2: VISION ROC CURVE AND AUC ---
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Vision ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Vision Component Performance (ROC)', fontsize=14)
plt.legend(loc="lower right")
plt.savefig("Diagrams/Vision ROC Curve and AUC.png", dpi=300)
print("✅ Created: Vision ROC Curve and AUC.png")

# --- CHART 3: VISION PERFORMANCE FRAMEWORK ---
report = classification_report(y_true, y_pred, output_dict=True)
metrics = ['precision', 'recall', 'f1-score']
scores = [report['weighted avg'][m] for m in metrics]

plt.figure(figsize=(10, 6))
plt.bar(metrics, scores, color=['#fb5607', '#ffbe0b', '#ff006e'], width=0.5)
for i, v in enumerate(scores):
    plt.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontsize=12)

plt.ylim(0, 1.15)
plt.title('Stand-alone Vision Evaluation Framework', fontsize=14)
plt.ylabel('Score (Percent)')
plt.savefig("Diagrams/Vision Performance Framework.png", dpi=300)
print("✅ Created: Vision Performance Framework.png")

print("\n🚀 Vision-Only Evaluation Visuals Completed.")
