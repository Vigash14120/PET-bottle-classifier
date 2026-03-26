# ♻️ AI-Powered Multi-Modal PET Sorter

![Python Version](https://img.shields.io/badge/python-3.13.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Latest-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

Welcome to the **PET Multi-Modal Classifier**! This system uses both Computer Vision and Load Cell data (weight) to classify PET and HDPE bottles with pinpoint accuracy.

---

## 🗺️ Project Workflow
<img src="Diagrams/Workflow%20diagram.jpg" width="600">

---

## 🧠 The AI Brain Suite
This project features three distinct AI models, allowing the system to adapt if sensors (camera or scale) fail:

1.  ⚖️ **Weight Model** (`weight_model.h5`): Uses only mass to classify bottles. (Accuracy: **100%**)
    <img src="Diagrams/weight%20model%20architecture%20diagram.png" width="400">

2.  📸 **Vision Model** (`vision_model.h5`): Uses only the camera feed. (Accuracy: **~94%**)
    <img src="Diagrams/vision%20model%20architecture%20diagram.png" width="400">

3.  🚀 **Fused Model** (`fused_model.h5`): **Multimodal Fusion Engine** that combines both sensors for maximum speed and accuracy. (Accuracy: **100%**)

---

## 🏗️ Getting Started

### 1. Prerequisites
Ensure you are using the provided virtual environment:
```powershell
.\.venv\Scripts\activate
```

### 2. Live Sorter Operation
To run the real-time sorter with live sensor integration:
```powershell
python main_sorter.py
```

### ⌨️ Dynamic Controls
While the sorter is running, you can toggle modes in real-time:
| Key | Mode | Description |
| :--- | :--- | :--- |
| `[f]` | **Fused Mode** | Best performance (Vision + Weight) |
| `[v]` | **Vision Only** | Works without a scale |
| `[w]` | **Weight Only** | Works in total darkness (scale only) |
| `[t]` | **Tare** | Software reset for the load cell |
| `[+]` / `[-]` | **Simulate** | Change weight for testing/debugging |
| `[q]` | **Quit** | Safely shut down the system |

---

## 📊 Comparative Performance Benchmarks
We tested three different architectural approaches to find the most robust sorting solution. The **Fused Multi-Modal** approach consistently out-performed single-sensor models in reliability.

| Model Strategy | Sensor Input | Accuracy | Reliability |
| :--- | :--- | :--- | :--- |
| **Vision Only** | Camera | 93.87% | Moderate (Lighting sensitive) |
| **Weight Only** | Load Cell | 100.0% | High (But fails for weight overlaps) |
| **🎯 Custom Vision Nano** | **~40k Params** | **< 1.0 MB** | **< 55ms (CPU)** | **93.8%** |
| **🚀 Fused (Hybrid)** | **Both Sensors** | **100.0%** | **Extreme (Redundant & Robust)** |

### **⭐ Why our Vision Model wins:**
Despite being **Over 1,300x smaller** than VGG-16, our model achieves nearly **identical accuracy**.

| Performance Metric | **Industry Standard (VGG-16)** | **🏆 Our Custom Model** |
| :--- | :--- | :--- |
| **Model Weight** | ~520.0 MB | **0.39 MB (1,300x Lighter)** |
| **Inference Latency** | ~120 ms | **50.7 ms (2.4x Faster)** |
| **Edge Compatibility** | Very Poor (Requires GPU) | **Excellent (CPU Only)** |

- **Inference Speed**: It is ultra-low latency, allowing for real-time sorting on high-speed conveyor systems.
- **Resource Efficient**: Requires zero GPU acceleration and very low RAM—ideal for IoT and Edge implementation!

---
### **🏆 Why our Fused Model Out-performs Individual Sensors:**
1. **Redundancy**: If a bottle is clear (difficult for the camera to differentiate from PET), the weight component provides the deciding factor.
2. **Contextual Learning**: The model doesn't just "look" or "weigh"; it learns the **correlation** between the two. For example, it knows that an HDPE bottle with a specific visual texture *should* have a corresponding weight.
3. **Accuracy**: While single models can hit high accuracy, they are prone to "edge cases" (e.g., a crushed PET bottle). Our Fused Model handles these outliers with **100% validation success**.

---

## 📸 Evaluation Metrics (Fused Multi-Modal)
We achieved perfection across both PET and HDPE categories in our final model-level fusion training.

```text
========================================
📊 FINAL EVALUATION REPORT
========================================
Overall Accuracy: 100.00%

Detailed Metrics:
              precision    recall  f1-score   support    

     PET (0)       1.00      1.00      1.00       301    
    HDPE (1)       1.00      1.00      1.00       205    
========================================
```

---

## 📂 Project Structure
- `main_sorter.py`: The live orchestrator for sorting.
- `train_fused_model.py`: Training logic for the multi-modal brain.
- `visual_preprocessing.py`: Image normalization & augmentation.
- `loadcell_preprocessing.py`: Software Taring & Outlier rejection.
- `data/`: Dataset containing 2,000+ bottle images.

---

## 🛠️ Contribution & Development
To retrain any model, simply run:
```powershell
python train_vision_model.py  # Retrain Vision model
python train_weight_model.py  # Retrain Weight model
python train_fused_model.py   # Retrain Full Fusion model
```

---
**Made with ❤️ for PET Recycling Automation.**