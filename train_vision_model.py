import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import cv2
import random
from visual_preprocessing import preprocess_image

# 1. Load Metadata
df = pd.read_csv("metadata_fused.csv")

def data_generator(dataframe, batch_size=32, target_size=(150, 150), mode='train'):
    df_subset = dataframe[dataframe['split'] == mode].copy()
    while True:
        if mode == 'train':
            df_subset = df_subset.sample(frac=1).reset_index(drop=True)
            
        for i in range(0, len(df_subset), batch_size):
            batch_df = df_subset.iloc[i : i + batch_size]
            images = []
            labels = []
            
            for _, row in batch_df.iterrows():
                img_path = row['image_path']
                img = cv2.imread(img_path)
                
                if mode == 'train':
                   if random.random() > 0.5:
                       img = cv2.flip(img, 1) # Augmentation
                
                processed_img = preprocess_image(img, target_size=target_size)
                if processed_img is not None:
                    images.append(processed_img)
                    labels.append(row['label'])
            
            if len(images) > 0:
                yield np.array(images), np.array(labels)

# 2. Define the Single-Modal "Vision Brain"
def build_vision_model():
    image_input = Input(shape=(150, 150, 3), name='vision_input')
    v = Conv2D(16, (3,3), padding='same', activation='relu')(image_input)
    v = BatchNormalization()(v)
    v = MaxPooling2D((2,2))(v)

    v = Conv2D(32, (3,3), padding='same', activation='relu')(v)
    v = BatchNormalization()(v)
    v = MaxPooling2D((2,2))(v)

    v = Conv2D(64, (3,3), padding='same', activation='relu')(v)
    v = BatchNormalization()(v)
    v = GlobalAveragePooling2D()(v)
    
    v = Dense(64, activation='relu')(v)
    v = Dropout(0.2)(v)
    output = Dense(1, activation='sigmoid', name='output')(v)

    model = Model(inputs=image_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Setup Generators
train_gen = data_generator(df, mode='train')
val_gen = data_generator(df, mode='val')

train_steps = len(df[df['split'] == 'train']) // 32
val_steps = len(df[df['split'] == 'val']) // 32

# 4. Train the Model
model = build_vision_model()
print("--- 📸 Training STANDALONE VISION MODEL (15 Epochs) ---")
model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=15
)

# 5. Save the Vision Model
model.save("vision_model.h5")
print("✅ Vision Model Saved as vision_model.h5")
