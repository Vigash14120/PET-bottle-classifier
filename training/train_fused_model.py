import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import cv2
from core.visual_preprocessing import preprocess_image

# 1. Load Metadata
df = pd.read_csv("metadata_fused.csv")

import random

def data_generator(dataframe, batch_size=32, target_size=(150, 150), mode='train'):
    """
    Custom generator with basic augmentation for training.
    """
    df_subset = dataframe[dataframe['split'] == mode].copy()
    
    while True:
        # Shuffle if training
        if mode == 'train':
            df_subset = df_subset.sample(frac=1).reset_index(drop=True)
            
        for i in range(0, len(df_subset), batch_size):
            batch_df = df_subset.iloc[i : i + batch_size]
            
            images = []
            weights = []
            labels = []
            
            for _, row in batch_df.iterrows():
                # Load and preprocess image
                img_path = row['image_path']
                img = cv2.imread(img_path)
                
                # --- SIMPLE AUGMENTATION (Training only) ---
                if mode == 'train':
                   if random.random() > 0.5:
                       img = cv2.flip(img, 1) # Horizontal Flip
                   if random.random() > 0.8:
                       img = cv2.flip(img, 0) # Vertical Flip (sometimes bottles are upside down)
                
                processed_img = preprocess_image(img, target_size=target_size)
                
                if processed_img is not None:
                    images.append(processed_img)
                    weights.append(row['weight_grams'] / 100.0) 
                    labels.append(row['label'])
            
            if len(images) > 0:
                # Direct dict yielding is more stable for multi-input models
                yield ({
                    "image_input": np.array(images),
                    "weight_input": np.array(weights).reshape(-1, 1)
                }, np.array(labels))

# 2. Define the "Fused Nano" Architecture
def build_fused_model():
    # --- Vision Branch ---
    image_input = Input(shape=(150, 150, 3), name='image_input')
    v = Conv2D(16, (3,3), padding='same', activation='relu')(image_input)
    v = BatchNormalization()(v)
    v = MaxPooling2D((2,2))(v)

    v = Conv2D(32, (3,3), padding='same', activation='relu')(v)
    v = BatchNormalization()(v)
    v = MaxPooling2D((2,2))(v)

    v = Conv2D(64, (3,3), padding='same', activation='relu')(v)
    v = BatchNormalization()(v)
    v = GlobalAveragePooling2D()(v)
    
    visual_latent = Dense(64, activation='relu')(v)
    visual_latent = Dropout(0.2)(visual_latent)

    # --- Weight Branch (MLP) ---
    weight_input = Input(shape=(1,), name='weight_input')
    w = Dense(16, activation='relu')(weight_input) # Increased capacity
    w = Dense(8, activation='relu')(w)
    weight_latent = w

    # --- Fusion & Decision ---
    fused = Concatenate()([visual_latent, weight_latent])
    fused = Dense(32, activation='relu')(fused)
    fused = Dropout(0.3)(fused) # Slightly more dropout
    output = Dense(1, activation='sigmoid', name='output')(fused)

    model = Model(inputs=[image_input, weight_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Setup Generators
train_gen = data_generator(df, mode='train')
val_gen = data_generator(df, mode='val')

train_steps = len(df[df['split'] == 'train']) // 32
val_steps = len(df[df['split'] == 'val']) // 32

# 4. Train the Model
model = build_fused_model()
print("--- 🚀 Retraining (Vision + Weight) with 25 Epochs + Augmentation ---")
model.summary()

model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=25
)

# 5. Save the Fused Model
model.save("models/fused_model.h5")
print("✅ Fused Model Saved as models/fused_model.h5")
