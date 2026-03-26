import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import cv2
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
            weights = []
            labels = []
            for _, row in batch_df.iterrows():
                # Correct image path if needed (handle backslashes if on Windows)
                img_path = row['image_path']
                img = cv2.imread(img_path)
                processed_img = preprocess_image(img, target_size=target_size)
                if processed_img is not None:
                    images.append(processed_img)
                    weights.append(row['weight_grams'] / 100.0)
                    labels.append(row['label'])
            if len(images) > 0:
                yield ([np.array(images), np.array(weights).reshape(-1, 1)], np.array(labels))

# ... rest of the architecture ...
def build_fused_model():
    image_input = Input(shape=(150, 150, 3), name='image_input')
    v = Conv2D(16, (3,3), padding='same', activation='relu')(image_input)
    v = BatchNormalization()(v)
    v = MaxPooling2D((2,2))(v)
    v = GlobalAveragePooling2D()(v)
    visual_latent = Dense(64, activation='relu')(v)

    weight_input = Input(shape=(1,), name='weight_input')
    w = Dense(8, activation='relu')(weight_input)
    weight_latent = w

    fused = Concatenate()([visual_latent, weight_latent])
    fused = Dense(32, activation='relu')(fused)
    output = Dense(1, activation='sigmoid', name='output')(fused)

    model = Model(inputs=[image_input, weight_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_fused_model()
train_gen = data_generator(df, mode='train')
val_gen = data_generator(df, mode='val')

train_steps = len(df[df['split'] == 'train']) // 32
val_steps = len(df[df['split'] == 'val']) // 32

try:
    model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=1
    )
except Exception as e:
    import traceback
    traceback.print_exc()
