"""
test_cnn_lstm.py
----------------
Builds the CNN-LSTM with Soft Attention and tests it with dummy data.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_branch(input_shape=(64, 64, 12)):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    out = layers.Dense(128, activation="relu")(x)
    return models.Model(inp, out, name="CNN_Branch")

def build_lstm_branch(input_shape=(1460, 5)):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.2)(x)
    
    out = layers.Dense(128, activation="relu")(x)
    return models.Model(inp, out, name="LSTM_Branch")

def build_fc_branch(input_shape=(50,)):
    inp = layers.Input(shape=input_shape)
    x = layers.Dense(64)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)
    
    out = layers.Dense(128, activation="relu")(x)
    return models.Model(inp, out, name="FC_Branch")

def build_soc_model():
    sat_input = layers.Input(shape=(64, 64, 12), name="satellite")
    ts_input = layers.Input(shape=(1460, 5), name="timeseries")
    agri_input = layers.Input(shape=(50,), name="agronomic")
    
    cnn_out = build_cnn_branch()(sat_input)
    lstm_out = build_lstm_branch()(ts_input)
    fc_out = build_fc_branch()(agri_input)
    
    # Soft Attention Fusion
    stacked = layers.Stack(name="stack_branches")([cnn_out, lstm_out, fc_out])
    attention_weights = layers.Dense(1, activation="softmax", name="attention")(stacked)
    weighted = layers.Multiply()([stacked, attention_weights])
    fused = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name="fused_features")(weighted)
    
    output = layers.Dense(1, activation="linear", name="soc_prediction")(fused)
    
    return models.Model(inputs=[sat_input, ts_input, agri_input], outputs=output, name="SOC_CNN_LSTM")

if __name__ == "__main__":
    print("Building CNN-LSTM model...")
    model = build_soc_model()
    
    print("\n--- Model Architecture ---")
    model.summary()
    
    print("\nGenerating dummy data...")
    # 1 batch of fake data
    dummy_sat = np.random.rand(1, 64, 64, 12).astype(np.float32)
    dummy_ts = np.random.rand(1, 1460, 5).astype(np.float32)
    dummy_agri = np.random.rand(1, 50).astype(np.float32)
    
    print("Testing forward pass (prediction)...")
    prediction = model.predict([dummy_sat, dummy_ts, dummy_agri])
    
    print(f"\nSuccess! Predicted SOC: {prediction[0][0]:.3f} %")
