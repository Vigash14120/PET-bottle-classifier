import os
import pandas as pd
import random

# Root data path
DATA_DIR = "data"
CATEGORIES = {
    "PET_Bottle": {"min": 15.0, "max": 25.0, "label_int": 0},
    "HDPE_Bottle": {"min": 35.0, "max": 65.0, "label_int": 1}
}

data_list = []

print("--- 🛠️ Generating Multi-Modal Metadata (Train + Val) ---")

for split in ["train", "val"]:
    split_path = os.path.join(DATA_DIR, split)
    if not os.path.exists(split_path):
        continue
        
    for category, config in CATEGORIES.items():
        category_path = os.path.join(split_path, category)
        if not os.path.exists(category_path):
            continue
            
        images = os.listdir(category_path)
        print(f"Processing {len(images)} images in {split}/{category}...")

        for img in images:
            simulated_weight = round(random.uniform(config['min'], config['max']), 2)
            data_list.append({
                "image_path": os.path.join(DATA_DIR, split, category, img),
                "split": split,
                "label": config['label_int'], # Use integers directly for training
                "weight_grams": simulated_weight
            })

# Save to a CSV file
df = pd.DataFrame(data_list)
df.to_csv("metadata_fused.csv", index=False)
print("-" * 40)
print(f"✅ Success! 'metadata_fused.csv' created with {len(df)} entries.")
