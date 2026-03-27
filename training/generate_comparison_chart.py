import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Prepare Benchmark Data
# [Name, F1-Score, Latency (seconds)]
approaches = [
    "AWS Rekognition (Cloud SaaS)",
    "VGG-16 (On-Device DL)",
    "MobileNetV2 (On-Device DL)",
    "🎯 Our Fused Model (Custom DL)"
]

f1_scores = [0.93, 0.94, 0.91, 1.00]
latencies = [2.10, 0.12, 0.08, 0.05]

# 2. Setup Plot
x = np.arange(len(approaches))
width = 0.35 # Width of bars

fig, ax = plt.subplots(figsize=(10, 6))

# Primary Axis: F1-Score (Blue)
rects1 = ax.bar(x - width/2, f1_scores, width, label='Performance (F1-Score)', color='#3a86ff')

# Secondary Axis: Latency (Green)
ax2 = ax.twinx()
rects2 = ax2.bar(x + width/2, latencies, width, label='Latency (Seconds)', color='#06d6a0')

# Labels and Styling
ax.set_ylabel('F1-Score (Higher is Better)', color='#3a86ff', fontsize=12)
ax2.set_ylabel('Inference Latency in Sec (Lower is Better)', color='#06d6a0', fontsize=12)
ax.set_title('Comparative Analysis: PET Bottle Classifier Performance', pad=20, fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(approaches, rotation=15)
ax.set_ylim(0, 1.2) # To give space for labels
ax2.set_ylim(0, 3.0)

# Add Legend
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)

# Annotation Function
def autolabel(rects, axis, is_latency=False):
    for rect in rects:
        height = rect.get_height()
        axis.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1, ax)
autolabel(rects2, ax2, is_latency=True)

# Layout Optimization
plt.tight_layout()

# 3. Save the Comparison Chart
save_path = "Diagrams/Comparative Analysis Report.png"
if not os.path.exists("Diagrams"):
    os.makedirs("Diagrams")

plt.savefig(save_path, dpi=300)
print(f"✅ Comparison Chart generated and saved as '{save_path}'")
