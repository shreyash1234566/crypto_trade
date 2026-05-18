
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 3]) # Predict 'close' price
    return np.array(x), np.array(y)

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv("/home/ubuntu/btc_usdt_1h.csv", index_col=0)
    
    # Use OHLCV features
    features = ['open', 'high', 'low', 'close', 'volume']
    data = df[features].values
    
    # Split data FIRST to prevent leakage
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Scale data based ONLY on training set
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Create sequences
    seq_length = 24
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)
    
    # Build Bi-LSTM model
    print("Building Bi-LSTM model...")
    model = Sequential([
        Input(shape=(seq_length, len(features))),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train model
    print("Training model...")
    # Using a smaller epoch count for verification, can be increased
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    predictions = model.predict(X_test)
    
    # Inverse transform for real prices
    dummy_test = np.zeros((len(y_test), len(features)))
    dummy_test[:, 3] = y_test
    y_test_real = scaler.inverse_transform(dummy_test)[:, 3]
    
    dummy_pred = np.zeros((len(predictions), len(features)))
    dummy_pred[:, 3] = predictions.flatten()
    predictions_real = scaler.inverse_transform(dummy_pred)[:, 3]
    
    # 1. Regression Metrics
    mape = np.mean(np.abs((y_test_real - predictions_real) / y_test_real)) * 100
    
    # 2. Directional Accuracy (The "Real" Trading Accuracy)
    # Did we predict the price would go up or down relative to the PREVIOUS close?
    # Previous close for y_test[i] is test_data[i + seq_length - 1, 3]
    y_prev_real = test_data[seq_length:-1, 3]
    y_curr_real = y_test_real[1:]
    pred_curr_real = predictions_real[1:]
    
    actual_direction = (y_curr_real > y_prev_real).astype(int)
    predicted_direction = (pred_curr_real > y_prev_real).astype(int)
    
    dir_acc = accuracy_score(actual_direction, predicted_direction)
    precision = precision_score(actual_direction, predicted_direction)
    recall = recall_score(actual_direction, predicted_direction)
    f1 = f1_score(actual_direction, predicted_direction)
    
    print(f"\n--- Refined Results (No Leakage) ---")
    print(f"Price Prediction MAPE: {mape:.2f}%")
    print(f"Directional Accuracy: {dir_acc*100:.2f}%")
    print(f"Directional Precision: {precision*100:.2f}%")
    print(f"Directional Recall: {recall*100:.2f}%")
    print(f"Directional F1-Score: {f1*100:.2f}%")
    
    # Save results
    with open("/home/ubuntu/refined_model_results.txt", "w") as f:
        f.write(f"Refined Bi-LSTM Model Results (No Leakage)\n")
        f.write(f"==========================================\n")
        f.write(f"Price Prediction MAPE: {mape:.2f}%\n")
        f.write(f"Directional Accuracy: {dir_acc*100:.2f}%\n")
        f.write(f"Directional Precision: {precision*100:.2f}%\n")
        f.write(f"Directional Recall: {recall*100:.2f}%\n")
        f.write(f"Directional F1-Score: {f1*100:.2f}%\n")

if __name__ == "__main__":
    main()
