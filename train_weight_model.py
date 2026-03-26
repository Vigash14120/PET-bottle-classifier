import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np

# 1. Load Data Metadata
df = pd.read_csv("metadata_fused.csv")

def prepare_data(dataframe, mode='train'):
    df_subset = dataframe[dataframe['split'] == mode].copy()
    
    # Scale from 0.0 to 1.0 (Normalizing by 100g max)
    X = df_subset['weight_grams'].values / 100.0
    y = df_subset['label'].values
    
    # Reshape for input layer
    return X.reshape(-1, 1), y

# 2. Define the "Weight Brain"
def build_weight_model():
    input_layer = Input(shape=(1,), name='weight_input')
    w = Dense(32, activation='relu')(input_layer) # Higher capacity for weight logic
    w = Dense(16, activation='relu')(w)
    w = Dropout(0.2)(w)
    w = Dense(8, activation='relu')(w)
    output_layer = Dense(1, activation='sigmoid', name='output')(w)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Training & Validation
X_train, y_train = prepare_data(df, mode='train')
X_val, y_val = prepare_data(df, mode='val')

# 4. Train the Model
model = build_weight_model()
print("--- ⚖️ Training STANDALONE WEIGHT MODEL (25 Epochs) ---")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32
)

# 5. Save the Weight Model
model.save("weight_model.h5")
print("✅ Weight Model Saved as weight_model.h5")
